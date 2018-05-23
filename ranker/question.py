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

class NoAnswersException(Exception):
    pass

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
            elif not 'identifiers' in n:
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
        subgraph_networkx = database.queryToGraph(self.subgraph_with_support(database))
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
        subgraphs_with_metadata, subgraphs = pr.report_ranking(subgraphs) # returned subgraphs are sorted by rank
        
        question_info = {
            'question_hash': self.compute_hash(),
            'natural_question': self.natural_question
        }
        aset = Answerset(question_info=question_info)
        #for substruct, subgraph in zip(score_struct, subgraphs):
        for subgraph in subgraphs_with_metadata:
            #graph = UniversalGraph(nodes=substruct['nodes'], edges=substruct['edges'])
            #graph.merge_multiedges()
            #graph.to_answer_walk(subgraph)
            
            answer = Answer(nodes=subgraph['nodes'],\
                    edges=subgraph['edges'],\
                    score=subgraph['score'])
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
                try:
                    id = id_map[node_struct['identifiers'][0].upper()]
                except KeyError:
                    raise NoAnswersException("Question answering complete, found 0 answers.")
            else:
                id = node_struct['identifiers'][0].upper()
            prop_string = f" {{id:'{id}'}}"
        else:
            prop_string = ''
        return f"{var_name}:{concept}{prop_string}"

    def edge_match_string(self, edge_struct, var_name):
        if not edge_struct['length'][0]==edge_struct['length'][1]:
            return f"[{var_name}*{edge_struct['length'][0]}..{edge_struct['length'][-1]}]"
        else:
            return f"[{var_name}]"

    def cypher_match_string(self, db=None):
        nodes, edges = self.nodes, self.edges

        node_count = len(nodes)
        edge_count = len(edges)

        # generate internal node and edge variable names
        node_names = ['n{:d}'.format(i) for i in range(node_count)]
        edge_names = ['r{0:d}{1:d}'.format(i, i+1) for i in range(edge_count)]

        node_strings = [self.node_match_string(node, name, db) for node, name in zip(nodes, node_names)]

        # generate MATCH command string to get paths of the appropriate size
        match_strings = []
        match_strings.append(f"MATCH ({node_strings[0]})")
        for i in range(edge_count):
            match_strings.append(f"MATCH ({node_names[i]})-{self.edge_match_string(edges[i], edge_names[i])}-({node_strings[i+1]})")
            match_strings.append(f"WHERE NOT {edge_names[i]}.predicate_id='omnicorp:1'")

        # optional matches are super slow, for some reason
        # match_strings = [f"MATCH ({node_strings[0]})"]
        # for i in range(edge_count):
        #     match_strings.append(f"OPTIONAL MATCH ({node_names[i]})-{self.edge_match_string(edges[i], edge_names[i])}-({node_strings[i+1] if i+1<edge_count else node_names[i+1]})")
        #     match_strings.append(f"WHERE NOT {edge_names[i]}.predicate_id='omnicorp:1'")
        # if 'identifiers' in nodes[-1] and nodes[-1]['identifiers']:
        #     node_strings2 = [self.node_match_string(node, name+'b', db) for node, name in zip(nodes, node_names)]
        #     match_strings.insert(1,f"MATCH ({node_strings[-1]})")
        #     for i in range(edge_count-1, -1, -1):
        #         match_strings.append(f"OPTIONAL MATCH ({node_names[i+1]+('b' if not i==edge_count-1 else '')})-{self.edge_match_string(edges[i], f'{edge_names[i]}b')}-({node_strings2[i]})")
        #         match_strings.append(f"WHERE NOT {edge_names[i]}b.predicate_id='omnicorp:1'")
        #     match_strings.append(f"AND {node_names[0]}={node_names[0]}b")
        #     case_strings = [f"CASE WHEN {n} IS null THEN {n}b WHEN {n}b IS null THEN {n} ELSE null END AS {n}" for n in node_names[:-1]] + ['n6']
        #     with_string = f"WITH {', '.join(case_strings)}"
        #     match_strings.append(with_string)

        match_string = ' '.join(match_strings)
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
