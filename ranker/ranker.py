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

logger = logging.getLogger(__name__)

class Ranker:
    G = nx.MultiDiGraph() # a networkx Digraph() with weights
    graphInfo = {} # will be populated on construction
    naga_parameters = {'alpha':.9, 'beta':.9}
    prescreen_count = 2000 # only look at this many graphs more in depth
    teleport_weight = 0.001 # probability to teleport along graph (make random inference) in hitting time calculation
    output_count = 250
    
    def __init__(self, G=nx.MultiDiGraph()):
        self.G = G
        self._evaluated_templates = {}
        self._result_count = -1

    def set_weights(self,method='ngd'):
        """ Initialize weights on the graph based on metadata.
            logistic just counts # of publications and applies a hand tuned logistic.
            ngd uses the omnicorp_article_count on the edges to generate normalized google distance weights.
        """
        # notation: edge contains metadata, e is edge without metadata
        edges = self.G.edges(data=True,keys=True)
        pub_counts = {edge[:-1]:len(edge[-1]['publications']) if 'publications' in edge[-1] else 0 for edge in edges}
        
        # initialize scoring info - ngd initializes to big number (np.inf not well-liked) so that weight becomes zero
        scoring_info = {e:{'num_pubs':pub_counts[e],'ngd':1e6} for e in pub_counts}
        nx.set_edge_attributes(self.G, values=scoring_info, name = 'scoring')

        if method == 'logistic':
            # apply logistic function to publications to get weights
            weights = {e:1/(1 + np.exp((5-pub_counts[e])/2)) for e in pub_counts}

        elif method == 'ngd':
            # this method tries to use omnicorp's article counts to normalize probabilities in a meaningful way
            nodes = self.G.nodes()
            node_pub_sum = {n:sum([edge[-1]['scoring']['num_pubs'] for edge in self.G.edges(n,data=True)]) for n in nodes}
            
            N = 1e8 # approximate number of articles in corpus * typical number of keywords
            default_article_node_count = 25000 # large but typical number of article counts for a node
            minimum_article_node_count = 1000 # assume every concept actually has at least this many publications we may or may not know about

            mean_ngd = 1e-8
            node_count = [minimum_article_node_count]*2
            for edge in edges:
                for i in range(2):
                    if 'omnicorp_article_count' in self.G.node[edge[i]]:
                        node_count[i] = int(self.G.node[edge[i]]['omnicorp_article_count'])
                    else:
                        node_count[i] = default_article_node_count
                
                    # make sure the node counts are at least as great as the sum of pubs along the edges we have
                    node_count[i] = max(node_count[i],node_pub_sum[edge[i]],minimum_article_node_count)
                
                edge_count = edge[-1]['scoring']['num_pubs'] + 1 # avoid log(0) problem
                
                # formula for normalized google distance
                ngd = (np.log(min(node_count)) - np.log(edge_count))/ \
                    (np.log(N) - np.log(max(node_count)))
                
                # this shouldn't happen but theoretically could
                if ngd < 0:
                    ngd = 0

                edge[-1]['scoring']['ngd'] = ngd
                mean_ngd = mean_ngd + ngd

            mean_ngd = mean_ngd/len(edges)
            logger.debug(f"Mean ngd: {mean_ngd}")

            # ngd is like a metric, we need a similarity
            # common way to do this is with a gaussian kernel
            weights = {edge[:-1]:np.exp(-(edge[-1]['scoring']['ngd']/mean_ngd)**2/2) for edge in edges}

        else:
            raise Exception('Method ' + method + ' has not been implemented.')

        # set the weights on the edges
        nx.set_edge_attributes(self.G, values=weights, name='weight')

    def sum_edge_weights(self, sub_graph):
        sub_graph_update = self.G.subgraph([s['id'] for s in sub_graph])
        edges = sub_graph_update.edges(data='weight')
        return sum([edge[-1] for edge in edges])

    def rank(self, sub_graph_list):
        """ Primary method to generate a sorted list and scores for a set of subgraphs """
        # sub_graph_list is a list of lists of dicts with fields 'id' and 'bound'

        if not sub_graph_list:
            return ([],[],[])
        
        logger.debug('set_weights()... ')
        start = time.time()
        self.set_weights(method='ngd')
        logger.debug(f"{time.time()-start} seconds elapsed.")

        # add the none node to G 
        if 'None' in self.G:
            logger.error("Node None already exists in G. This could cause incorrect ranking results.")
        else:
            self.G.add_node('None') # must add the none node to correspond to None id's
        
        # convert None nodes to string None and check that all the subgraph nodes are in G
        for sg in sub_graph_list:
            for node in sg:
                if node['id'] is None:
                    node['id'] = 'None'
                if node['id'] not in self.G:
                    raise KeyError('Node id:' + node['id'] + ' does not exist in the graph G')

        logger.debug("Prescreening sub_graph_list... ")
        start = time.time()
        prescreen_scores = [self.sum_edge_weights(sg) for sg in sub_graph_list]
        prescreen_sorting, prescreen_scores_sorted = zip(*sorted(enumerate(prescreen_scores), key = lambda elem: elem[1], reverse=True))
        
        if len(prescreen_sorting) > self.prescreen_count:
            prescreen_sorting = prescreen_sorting[0:self.prescreen_count]
            prescreen_scores_sorted = prescreen_scores_sorted[0:self.prescreen_count]

        sub_graph_list = [sub_graph_list[i] for i in prescreen_sorting]
        logger.debug(f"{time.time()-start} seconds elapsed.")

        logger.debug("Calculating subgraph statistics()... ")
        start = time.time()
        
        graph_stat = [self.subgraph_statistic(sg,type='mix') for sg in sub_graph_list]
        logger.debug(f"{time.time()-start} seconds elapsed.")

        graph_comparison = self.comparison_statistic(sub_graph_list,type='mix')
        logger.debug(f"Comparison graph statistic: {graph_comparison}")

        # larger hitting/mixing times are worse
        ranking_scores = [graph_comparison/s for s in graph_stat]
        
        ranking_sorting, ranking_scores_sorted = zip(*sorted(enumerate(ranking_scores), key = lambda elem: elem[1], reverse=True))
        sub_graph_list = [sub_graph_list[i] for i in ranking_sorting]
        
        sub_graph_scores = [{'rank_score':ranking_scores[i],'pre_score':prescreen_scores_sorted[i]} for i in ranking_sorting]
        
        sorted_inds = [prescreen_sorting[i] for i in ranking_sorting]

        # trim output
        if len(sub_graph_list) > self.output_count:
            sub_graph_list = sub_graph_list[:self.output_count]
            sub_graph_scores = sub_graph_scores[:self.output_count]
            
        return (sub_graph_scores, sub_graph_list)

    def report_ranking(self,sub_graph_list):
        # construct the output that question.py expects

        (sub_graph_scores, sub_graph_list) = self.rank(sub_graph_list)

        # add extra computed metadata in self.G to subgraph for display
        logger.debug("Extracting subgraphs... ")
        start = time.time()
        sub_graphs_meta = [self.G.subgraph([s['id'] if s['id'] is not None else 'None' for s in sub_graph]) for sub_graph in sub_graph_list]
        logger.debug(f"{time.time()-start} seconds elapsed.")
        
        report = []
        for i, sg in enumerate(sub_graphs_meta):

            # re-sort the nodes in the sub-graph according to the node_list and remove None nodes
            node_list = sub_graph_list[i]
            nodes = list(sg.nodes(data=True))
            ids = [n[0] for n in nodes]
            nodes = [nodes[ids.index(n['id'])][-1] for n in node_list if n['id'] is not 'None']
            
            edges = list(sg.edges(data=True))
            edges = [e[-1] for e in edges]

            sgr = dict()
            sgr['score'] = sub_graph_scores[i]
            sgr['nodes'] = nodes
            sgr['edges'] = edges
            
            report.append(sgr)

        return (report, sub_graph_list)

    def subgraph_statistic(self,sub_graph,type='hit'):
        L = self.graph_laplacian(sub_graph)
        if type=='hit':
            x = self.hitting_time_from_laplacian(L)
        elif type=='mix':
            x = self.mixing_time_from_laplacian(L)
        
        return x
        
    def comparison_statistic(self, sub_graph_list, type='hit'):
        """ create a prototypical graph to compare hitting times against.
        For now, we use a fully connected graph with weights 1/2
        """
        w = 0.5
        n = int(np.round(np.mean([len(s) for s in sub_graph_list])))
        L = w * (n*np.eye(n) - np.ones((n,n)))

        if type=='hit':
            x = self.hitting_time_from_laplacian(L)
        elif type=='mix':
            x = self.mixing_time_from_laplacian(L)
        
        return x

    def graph_laplacian(self, sub_graph):
        # sub_graph is a list of dicts with fields 'id' and 'bound'

        # get updated weights
        node_ids = [s['id'] for s in sub_graph]
        sub_graph_update = self.G.subgraph(node_ids)
        
        nodes = sub_graph_update.nodes(data=True)

        # get updated weights
        sub_graph_update = sub_graph_update.subgraph([s[0] for s in nodes])
        sub_graph_update = sub_graph_update.to_undirected()

        # calculate hitting time of last node from first node
        #logger.debug(json.dumps(node_ids))
        #L = nx.laplacian_matrix(sub_graph_update.to_undirected(),nodelist = node_ids)
        #L = np.array(L.todense())

        # compute graph laplacian for this case with potentially duplicated nodes (None may be duplicated)
        n = len(node_ids)
        L = np.zeros((n,n))
        index = {id:node_ids.index(id) for id in node_ids}
        for u,v,w in sub_graph_update.edges_iter(data='weight'):
            if u is not v and (u in node_ids) and (v in node_ids):
                i, j = index[u], index[v]
                L[i,j] = -w
                L[j,i] = -w
                L[i,i] = L[i,i] + w
                L[j,j] = L[j,j] + w

        # add teleportation to allow leaps of faith
        L = L + self.teleport_weight * (n*np.eye(n)-np.ones((n,n)))
        return L
        
    def hitting_time_from_laplacian(self,L):
        # assume L is square
        L[-1,:] = 0
        L[-1,-1] = 1
        b = np.zeros(L.shape[0])
        b[0] = 1
        x = np.linalg.solve(L,b)
        return x[1]

    def mixing_time_from_laplacian(self,L):
        # assume L is square
        n = L.shape[0]
        g = max(np.diag(L))
        if g < 1e-8 or not np.isfinite(g):
            return np.Inf

        # uniformitization into discrete time chain
        P = np.eye(n) - L/g
        try: # always put other people's code inside a try catch loop
            ev = numpy.linalg.eigvals(P)
        except:
            # this should never happen for nonzero teleportation
            logger.debug("Eigenvalue computation failed for P:")
            for i in range(n):
                logger.debug(P[i,:])
            return np.Inf

        ev = np.abs(ev)
        ev.sort() # sorts ascending

        return 1/(1 - ev[-2])/g
