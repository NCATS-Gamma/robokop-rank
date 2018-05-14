"""
ProtocopRank is the base class for ranking subgraphs for a given query.
The constructor takes a networkx multi-digraph.
The primary methods are rank(subgraphs) and score(subgraph)
The most typical use case is:
    pr = ProtocopRank(G) # Where G is a networkX MultiDiGraph()
    score_struct = pr.report_scores_dict(subgraphs) #  subgraphs is a list of networkX MultiDiGraph()
"""

import random
import logging
import networkx as nx
import numpy as np
import json
import os
import time
import scipy.sparse as sparse
from scipy.sparse.linalg.eigen.arpack import ArpackNoConvergence
import numpy.linalg
import scipy.linalg

logger = logging.getLogger(__name__)

class Ranker:
    G = nx.MultiDiGraph() # a networkx Digraph() with weights
    graphInfo = {} # will be populated on construction
    naga_parameters = {'alpha':.9, 'beta':.9}
    prescreen_count = 2000 # only look at this many graphs more in depth
    teleport_weight = 0.001 # probability to teleport along graph (make random inference)

    def __init__(self, G=nx.MultiDiGraph()):
        self.G = G
        self._evaluated_templates = {}
        self._result_count = -1

    def set_weights(self):
        self.set_weights_pubs()

    def set_weights_pubs(self):
        """ Initialize weights on the graph based on metadata.
            Currently just counts # of publications and applies a hand tuned logistic.
        """

        pmids = nx.get_edge_attributes(self.G, 'publications')
        # pub_counts = {edge:max(len(pmids[edge]),1) for edge in pmids}
        pub_counts = {edge:len(pmids[edge]) for edge in pmids}
        
        # apply logistic function to publications to get weights
        weights = {edge:1/(1 + np.exp((5-pub_counts[edge])/2)) for edge in pub_counts}
        nx.set_edge_attributes(self.G, values=weights, name='weight')
        
        # initialize scoring info
        scoring_info = {edge:{'num_pubs':pub_counts[edge], 'pub_weight': weights[edge]} for edge in pub_counts}
        nx.set_edge_attributes(self.G, values=scoring_info, name = 'scoring')
        
    def sum_edge_weights(self, sub_graph):
        sub_graph_update = self.G.subgraph([s['id'] for s in sub_graph])
        edges = sub_graph_update.edges(data='weight')
        return sum([edge[-1] for edge in edges])

    def rank(self, sub_graph_list):
        """ Primary method to generate a sorted list and scores for a set of subgraphs """
        # sub_graph_list is a list of lists of dicts with fields 'id' and 'bound'

        if not sub_graph_list:
            return ([],[],[])

        min_nodes = max(0, min([len(sg) for sg in sub_graph_list])-1)
        
        logger.debug('set_weights()... ')
        start = time.time()
        self.set_weights()
        logger.debug(f"{time.time()-start} seconds elapsed.")

        logger.debug("Prescreening sub_graph_list... ")
        start = time.time()
        prescreen_scores = [self.sum_edge_weights(sg) for sg in sub_graph_list]
        prescreen_sorting, prescreen_scores_sorted = zip(*sorted(enumerate(prescreen_scores), key = lambda elem: elem[1], reverse=True))
        
        if len(prescreen_sorting) > self.prescreen_count:
            prescreen_sorting = prescreen_sorting[0:self.prescreen_count]

        sub_graph_list = [sub_graph_list[i] for i in prescreen_sorting]
        logger.debug(f"{time.time()-start} seconds elapsed.")

        logger.debug("compute_hitting_times()... ")
        start = time.time()
        hitting_times = [self.compute_hitting_time(sg) for sg in sub_graph_list]
        logger.debug(f"{time.time()-start} seconds elapsed.")

        logger.debug("score(method='naga')... ")
        start = time.time()
        scores_naga = [self.score(sg, method='naga')**(1/min_nodes) for sg in sub_graph_list]
        logger.debug(f"{time.time()-start} seconds elapsed.")
        
        min_hit = min(hitting_times)
        scores_hit = [min_hit/h for h in hitting_times]
        
        ranking_scores = [np.sqrt(sn*sh) for (sn,sh) in zip(scores_naga, scores_hit)]
        
        ranking_sorting, ranking_scores_sorted = zip(*sorted(enumerate(ranking_scores), key = lambda elem: elem[1], reverse=True))
        sub_graph_list = [sub_graph_list[i] for i in ranking_sorting]

        # add extra computed metadata in self.G to subgraph for display
        logger.debug("Extracting subgraphs... ")
        start = time.time()
        sub_graphs_meta = [self.G.subgraph([s['id'] for s in sub_graph]) for sub_graph in sub_graph_list]
        logger.debug(f"{time.time()-start} seconds elapsed.")
        
        scoring_info = [{'rank_score':ranking_scores[i],'pre_score':prescreen_scores_sorted[i]} for i in ranking_sorting]
        
        sorted_inds = [prescreen_sorting[i] for i in ranking_sorting]
        return (scoring_info, sub_graphs_meta, sorted_inds)

    def compute_hitting_time(self, sub_graph):
        # sub_graph is a list of dicts with fields 'id' and 'bound'

        # get updated weights
        sub_graph_update = self.G.subgraph([s['id'] for s in sub_graph])
        
        sg_nodes = sub_graph_update.nodes(data=True)

        # get updated weights
        sub_graph_update = sub_graph_update.subgraph([s[0] for s in sg_nodes])
        
        # calculate hitting time of last node from first node
        L = nx.laplacian_matrix(sub_graph_update.to_undirected())
        L = np.array(L.todense())
        L[-1,:] = 0
        L[-1,-1] = 1
        b = np.transpose(1 - L[-1,:])
        x = np.linalg.solve(L,b)
        hitting_time = max(x)
        
        return hitting_time

    def score(self, sub_graph, method='naga'):
        """ Assign a score to a specific subGraph
        sub_graph is a sub_graph of self.G with specification of bounded and unbounded nodes."""
        # sub_graph is a list of dicts with fields 'id' and 'bound'
        
        sub_graph_update = self.G.subgraph([s['id'] for s in sub_graph])
        edges = list(sub_graph_update.edges(keys=True,data=True))

        # nodes_bound = nx.get_node_attributes(self.G,'bound')
        nodes_bound = {n['id']:n['bound'] for n in sub_graph}
        
        result_count = self.count_result_templates()
        
        sub_graph_score = 0.0
        for edge in edges:
            
            from_node = edge[0]
            to_node = edge[1]
            
            # support edges influence weights only in full graph transition matrix
            if edge[-1]['type']=='Support' or edge[-1]['type']=='Lookup':
                continue

            # sum weights along other edges connecting these nodes
            edge_weight = sum([e['weight'] if 'weight' in e else 0 for (key, e) in self.G[from_node][to_node].items()])

            if method=='prod':
                edge_proba = edge_weight

            elif method=='naga':
                
                # confidence probability determined by edge weight
                proba_conf = edge_weight
                
                # return the count of instances of this template, as well as the total weights of all
                # facts corresponding to this template

                # TODO: deal with bound edges, whatever that means
                # False below was once edge[-1]['bound']
                edge_bound = (nodes_bound[from_node], nodes_bound[to_node], False)
                edge_template = self.factTemplate(edge, edge_bound)
                [template_count, template_weight] = self.eval_fact_template(edge_template,field='num_pubs')
                
                # query importance is lower for those with more options
                proba_query = 1 - template_count/result_count
                
                # informativeness probability depends upon which parts of the fact are bound/unbound
                informativeness_weight = edge[-1]['scoring']['num_pubs']
                if template_weight is 0:
                    proba_info = 1
                else:
                    proba_info = informativeness_weight/template_weight
                
                edge_proba = (1-self.naga_parameters['alpha']) * proba_query + self.naga_parameters['alpha'] * (
                    (1-self.naga_parameters['beta']) * proba_info + self.naga_parameters['beta'] * proba_conf)
                
            else:
                raise ValueError

            sub_graph_score = sub_graph_score + np.log(edge_proba)

        sub_graph_score = np.exp(sub_graph_score)

        return sub_graph_score

    def count_result_templates(self):
        if self._result_count < 0:
            edge_types = nx.get_edge_attributes(self.G,'type')
            self._result_count = len([t for t in edge_types.values() if not t == 'Support'])

        return self._result_count

    def eval_fact_template(self,template,field):
        ''' Evaluate the counts and total weights of all facts in the graph matching this template.
        Evaluated template combinations are stored in a dict to avoid repeated computations.
        '''
        if template not in self._evaluated_templates:
            # need to compute counts and weights for this template on the graph
            # enumerate the facts (edges) matching the given template
            edges = self.G.edges(data=True,keys=True)
            cond = lambda e: ((template[0] is None or template[0] == e[0])
                and  (template[1] is None or template[1] == e[1])
                and  (template[2] is None or template[2] == e[2])
                and not e[-1]['type']=='Support')

            edge_weights = [e[-1]['scoring'][field] for e in edges if cond(e)]

            self._evaluated_templates[template] = (len(edge_weights), max(edge_weights) if edge_weights else 0)

        return self._evaluated_templates[template]

    def plot(self):
        pos = nx.nx_agraph.graphviz_layout(self.G, prog='dot')
        nx.draw(self.G, pos=pos, with_labels=True, font_weight='normal', node_color=[0,.5,1], node_size=700, node_shape='s')
        plot_weights = [e[3]['weight'] for e in self.G.edges(data=True,keys=True)]
        nx.draw_networkx_edges(self.G, pos, alpha=np.array(plot_weights))

        plt.show()

    def report_scores_dict(self, node_sets):
        scoring_info, sub_graphs, idx = self.rank(node_sets)

        report = []
        for i, sg in enumerate(sub_graphs):
            sgr = dict()
            sgr['score'] = scoring_info[i]
            sgr['nodes'] = list(sg.nodes(data=True))
            sgr['edges'] = list(sg.edges(data=True))

            report.append(sgr)

        sorted_node_sets = [node_sets[i] for i in idx]

        return (report, sorted_node_sets)

    def report_scores(self, sub_graphs):
        return json.dumps(self.report_scores_dict(sub_graphs))

    def report_scores_write(self, sub_graphs, output_file):
        json_dump = self.report_scores(sub_graphs)

        with open(output_file, 'wt') as json_out:
            json_out.write(json_dump)

    def factTemplate(self, fact, bound):
        """ Convert queried fact into its corresponding template. A template is
        a tuple with unbound parts of the fact replaced with None. None is used since None
        cannot be used as keys in networkx.
        """
        return tuple([f if b else None for (f,b) in zip(fact,bound)])

class ProtocopFact:
    """ Holds a "fact" representing a connection between two nodes
        Everything is stored as indices referenced to another graph
    """
    node_from = 0
    node_to = 0
    key = ''

    from_bound = False
    to_bound = False
    edge_bound = False
    def __init__(self, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def template(self):
        """ Convert queried fact into its corresponding template. A template is
        a tuple with unbound parts of the fact replaced with None. None is used since None
        cannot be used as keys in networkx.
        """
        fact = [self.node_from, self.node_to, self.key]
        bound = [self.from_bound, self.to_bound, self.edge_bound]
        return tuple([f if b else None for (f,b) in zip(fact,bound)])

    def to_dict(self, G):
        fact = dict()
        fact['node_from'] = G.nodes(data=True)[self.node_from]
        fact['node_to'] = G.nodes(data=True)[self.node_to]
        fact['edge'] = G[self.node_from][self.node_to][self.key]

        return fact

    def to_json(self, G):

        return json.dumps(self.to_dict(G))

    def __str__(self):
        return "{0}--({1})-->{2}".format(self.node_from, self.key, self.node_to)

    def __repr__(self):
        return self.__str__()
