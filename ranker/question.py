'''
Question definition
'''

# standard modules
import os
import sys
import json
import hashlib
import warnings
import logging

# our modules
from ranker.universalgraph import UniversalGraph
from ranker.knowledgegraph import KnowledgeGraph
from ranker.answer import Answer, Answerset

# robokop-rank modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'robokop-rank'))
from ranker.ranker import Ranker

logger = logging.getLogger(__name__)

class Question():
    '''
    Represents a question such as "What genetic condition provides protection against disease X?"

    methods:
    * answer() - a struct containing the ranked answer paths
    * cypher() - the appropriate Cypher query for the Knowledge Graph
    '''

    def __init__(self, *args, **kwargs):
        '''
        keyword arguments: id, user, notes, natural_question, nodes, edges
        q = Question(kw0=value, ...)
        q = Question(struct, ...)
        '''

        # initialize all properties
        self.user_id = None
        self.id = None
        self.notes = None
        self.name = None
        self.natural_question = None
        self.nodes = [] # list of nodes
        self.edges = [] # list of edges

        # apply json properties to existing attributes
        attributes = self.__dict__.keys()
        if args:
            struct = args[0]
            for key in struct:
                if key in attributes:
                    setattr(self, key, struct[key])
                else:
                    warnings.warn("JSON field {} ignored.".format(key))

        # override any json properties with the named ones
        for key in kwargs:
            if key in attributes:
                setattr(self, key, kwargs[key])
            else:
                warnings.warn("Keyword argument {} ignored.".format(key))

        # replace input node names with identifiers
        for n in self.nodes:
            if 'nodeSpecType' in n and n['nodeSpecType'] == 'Named Node':
                identifiers = [n['meta']['identifier']]
                n['identifiers'] = identifiers
            else:
                n['identifiers'] = None
        for e in self.edges:
            if not 'length' in e:
                e['length'] = [1, 1]
            if len(e['length'])==1:
                e['length'] += e['length']

    def compute_hash(self):
        '''
        Generate an MD5 hash of the machine readable question interpretation
        i.e. the nodes and edges attributes
        '''

        json_spec = {
            "nodes":self.nodes,
            "edges":self.edges
        }
        m = hashlib.md5()
        m.update(json.dumps(json_spec).encode('utf-8'))
        return m.hexdigest()

    def relevant_subgraph(self):
        # get the subgraph relevant to the question from the knowledge graph
        database = KnowledgeGraph()
        subgraph_networkx = database.queryToGraph(self.subgraph_with_support())
        del database
        subgraph = UniversalGraph(subgraph_networkx)
        return {"nodes":subgraph.nodes,\
            "edges":subgraph.edges}

    def answer(self):
        '''
        Answer the question.

        Returns the answer struct, something along the lines of:
        https://docs.google.com/document/d/1O6_sVSdSjgMmXacyI44JJfEVQLATagal9ydWLBgi-vE
        '''
        
        # get all subgraphs relevant to the question from the knowledge graph
        database = KnowledgeGraph()
        subgraphs = database.query(self) # list of lists of nodes with 'id' and 'bound'
        answer_set_subgraph = database.queryToGraph(self.subgraph_with_support(database))
        del database

        # compute scores with NAGA, export to json
        pr = Ranker(answer_set_subgraph)
        score_struct, subgraphs = pr.report_ranking(subgraphs) # returned subgraphs are sorted by rank
        
        question_info = {
            'question_hash': self.compute_hash(),
            'natural_question': self.natural_question
        }
        aset = Answerset(question_info=question_info)
        for substruct, subgraph in zip(score_struct, subgraphs):
            graph = UniversalGraph(nodes=substruct['nodes'], edges=substruct['edges'])
            graph.to_answer_walk(subgraph)

            answer = Answer(nodes=graph.nodes,\
                    edges=graph.edges,\
                    score=substruct['score'])
            # TODO: move node/edge details to AnswerSet
            # node_ids = [node['id'] for node in graph.nodes]
            # edge_ids = [edge['id'] for edge in graph.edges]
            # answer = Answer(nodes=node_ids,\
            #         edges=edge_ids,\
            #         score=0)
            aset += answer #substruct['score'])

        return aset

    def node_match_string(self, node_struct, var_name, db):
        concept = node_struct['type'] if not node_struct['type'] == 'biological_process' else 'biological_process_or_molecular_activity'
        if 'identifiers' in node_struct and node_struct['identifiers']:
            if db:
                id_map = db.get_map_for_type(concept)
                id = id_map[node_struct['identifiers'][0]]
            else:
                id = node_struct['identifiers'][0]
            prop_string = f" {{id:'{id}'}}"
        else:
            prop_string = ''
        return f"({var_name}:{concept}{prop_string})"

    def cypher_match_string(self, db=None):

        nodes, edges = self.nodes, self.edges

        node_count = len(nodes)
        edge_count = len(edges)

        # generate internal node and edge variable names
        node_names = ['n{:d}'.format(i) for i in range(node_count)]
        edge_names = ['r{0:d}{1:d}'.format(i, i+1) for i in range(edge_count)]

        node_strings = [self.node_match_string(node, name, db) for node, name in zip(nodes, node_names)]

        nodes_conditions = []
        for node in nodes:
            node_conditions = []
            if 'identifiers' in node and node['identifiers']:
                node_conditions.append([{'prop':'id', 'val':node_id, 'op':'=', 'cond':True} for node_id in node['identifiers']])
            if 'type' in node and node['type']:
                node_conditions.append([{'prop':'node_type', 'val':node['type'].replace(' ', ''), 'op':'=', 'cond':True}])
            nodes_conditions += [node_conditions]

        # generate MATCH command string to get paths of the appropriate size
        match_strings = [f"MATCH {node_strings[0]}"]
        match_strings += [
            f"OPTIONAL MATCH ({node_names[i]})-[{edge_names[i]}*{edges[i]['length'][0]}..{edges[i]['length'][-1]}]-{node_strings[i+1]}" if not edges[i]['length'][0]==edges[i]['length'][1] \
            else f"OPTIONAL MATCH ({node_names[i]})-[{edge_names[i]}]-{node_strings[i+1]}" 
            for i in range(edge_count)]

        # generate WHERE command string to prune paths to those containing the desired nodes/node types
        nodes_conditions = [
            [
                [
                    {
                        k:(node_condition[k] if k != 'cond'\
                        else '' if node_condition[k]\
                        else 'NOT ')\
                        for k in node_condition
                    } for node_condition in node_conditions_union
                ] for node_conditions_union in node_conditions_intersection
            ] for node_conditions_intersection in nodes_conditions
        ]
        node_cond_strings = [['('+' OR '.join([f"{node_condition['cond']}{node_names[node_idx]}.{node_condition['prop']}{node_condition['op']}'{node_condition['val']}'"\
            for node_condition in node_conditions_union])+')'\
            for node_conditions_union in node_conditions_intersection]\
            for node_idx, node_conditions_intersection in enumerate(nodes_conditions)]
        edge_cond_strings = [f"NOT {e}.predicate_id='omnicorp:1'" for e in edge_names]
        where_strings = [""] + [f"WHERE {e}" for e in edge_cond_strings]
        match_string = ' '.join([f"{m} {w}" for m, w in zip(match_strings, where_strings)])
        return match_string

    def cypher(self, db):
        '''
        Generate a Cypher query to extract the portion of the Knowledge Graph necessary to answer the question.

        Returns the query as a string.
        '''

        match_string = self.cypher_match_string(db)

        # generate internal node and edge variable names
        node_names = ['n{:d}'.format(i) for i in range(len(self.nodes))]
        edge_names = ['r{0:d}{1:d}'.format(i, i+1) for i in range(len(self.edges))]

        # define bound nodes (no edges are bound)
        node_bound = ['identifiers' in n and n['identifiers'] for n in self.nodes]
        node_bound = ["True" if b else "False" for b in node_bound]

        # add bound fields and return map
        answer_return_string = f"RETURN [{', '.join([f'{{id:{n}.id, bound:{b}}}' for n, b in zip(node_names, node_bound)])}] as nodes"

        # return subgraphs matching query
        query_string = ' '.join([match_string, answer_return_string])

        return query_string

    def subgraph_with_support(self, db):
        match_string = self.cypher_match_string(db)

        # generate internal node and edge variable names
        node_names = ['n{:d}'.format(i) for i in range(len(self.nodes))]

        collection_string = f"WITH {'+'.join([f'collect({n})' for n in node_names])} as nodes" + "\n" + \
            "UNWIND nodes as n WITH collect(distinct n) as nodes"
        support_string = 'CALL apoc.path.subgraphAll(nodes, {maxLevel:0}) YIELD relationships as rels' + "\n" +\
            "WITH [r in rels | r{.*, start:startNode(r).id, end:endNode(r).id, type:type(r), id:id(r)}] as rels, nodes"
        return_string = 'RETURN nodes, rels'
        query_string = "\n".join([match_string, collection_string, support_string, return_string])

        return query_string

    def subgraph(self):
        match_string = self.cypher_match_string()

        # generate internal node and edge variable names
        node_names = ['n{:d}'.format(i) for i in range(len(self.nodes))]
        edge_names = ['r{0:d}{1:d}'.format(i, i+1) for i in range(len(self.edges))]

        # just return a list of nodes and edges
        collection_string = f"WITH {'+'.join([f'collect({e})' for e in edge_names])} as rels, {'+'.join([f'collect({n})' for n in node_names])} as nodes"
        unique_string = 'UNWIND nodes as n WITH collect(distinct n) as nodes, rels UNWIND rels as r WITH nodes, collect(distinct r) as rels'
        return_string = "\n".join([collection_string, unique_string, 'RETURN nodes, rels'])

        query_string = "\n".join([match_string, return_string])

        return query_string
