"""Question definition."""

# standard modules
import os
import warnings
import logging
from importlib import import_module

# our modules
from ranker.universalgraph import UniversalGraph
from ranker.knowledgegraph import KnowledgeGraph
from ranker.answer import Answer, Answerset
from ranker.ranker import Ranker
from ranker.cache import Cache

logger = logging.getLogger(__name__)


class NoAnswersException(Exception):
    """Exception when no answers are found."""
    pass


class NodeReference():
    """Node reference object."""
    def __init__(self, node, db=None):
        """Create a node reference."""
        name = f'n{node["id"]}'
        label = node['type'] if 'type' in node else None

        if label == 'biological_process':
            label = 'biological_process_or_activity'

        if 'curie' in node:
            if db:
                id_map = db.get_map_for_type(label)
                try:
                    curie = id_map[node['curie'].upper()]
                except KeyError:
                    raise NoAnswersException("Question answering complete, found 0 answers.")
            else:
                curie = node['curie'].upper()
            prop_string = f" {{id: \'{curie}\'}}"
        else:
            prop_string = ''

        self.name = name
        self.label = label
        self.prop_string = prop_string
        self._num = 0

    def __str__(self):
        """Return the cypher node reference."""
        self._num += 1
        if self._num == 1:
            return f'{self.name}' + \
                   f'{":" + self.label if self.label else ""}' + \
                   f'{self.prop_string}'
        return self.name


class EdgeReference():
    """Edge reference object."""

    def __init__(self, edge, db=None):
        """Create an edge reference."""
        name = f'e{edge["id"]}'
        label = edge['type'] if 'type' in edge else None

        if 'min_length' not in edge:
            edge['min_length'] = 1
        if 'max_length' not in edge:
            edge['max_length'] = 1
        if not edge['min_length'] == edge['max_length'] == 1:
            length_string = f"*{edge['min_length']}..{edge['max_length']}"
        else:
            length_string = ''

        self.name = name
        self.label = label
        self.length_string = length_string
        self._num = 0

    def __str__(self):
        """Return the cypher edge reference."""
        self._num += 1
        if self._num == 1:
            return f'{self.name}{":" + self.label if self.label else ""}{self.length_string}'
        else:
            return self.name


class Question():
    """Question object.
    
    Represents a question such as "What genetic condition provides protection against disease X?"

    methods:
    * answer() - a struct containing the ranked answer paths
    * cypher() - the appropriate Cypher query for the Knowledge Graph
    """

    def __init__(self, *args, **kwargs):
        """Create a question.

        keyword arguments: id, user, notes, natural_question, nodes, edges
        q = Question(kw0=value, ...)
        q = Question(struct, ...)
        """
        # initialize all properties
        self.user_id = None
        self.id = None
        self.notes = None
        self.name = None
        self.natural_question = None
        self.machine_question = {}

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

        # add ids to edges if necessary
        if not any(['id' in e for e in self.machine_question['edges']]):
            for i, e in enumerate(self.machine_question['edges']):
                e['id'] = chr(ord('a') + i)

    def relevant_subgraph(self):
        # get the subgraph relevant to the question from the knowledge graph
        database = KnowledgeGraph()
        subgraph_networkx = database.queryToGraph(self.subgraph_with_support(database))
        del database
        subgraph = UniversalGraph(subgraph_networkx)
        return {"nodes": subgraph.nodes,
                "edges": subgraph.edges}

    def answer(self):
        """Answer the question.

        Returns the answer struct, something along the lines of:
        https://docs.google.com/document/d/1O6_sVSdSjgMmXacyI44JJfEVQLATagal9ydWLBgi-vE
        """

        # get all subgraphs relevant to the question from the knowledge graph
        database = KnowledgeGraph()
        subgraphs = database.query(self)  # list of lists of nodes with 'id' and 'bound'
        answerset_subgraph = database.queryToGraph(self.subgraph_with_support(database))
        del database

        nodes = set()
        pairs = set()
        for subgraph in subgraphs:
            for node in list(subgraph['nodes'].values()):
                nodes.add(node)
            node_ids = list(subgraph['nodes'].keys())
            for i, node_id in enumerate(node_ids):
                node_i = subgraph['nodes'][node_id]
                for node_id in node_ids[i + 1:]:
                    node_j = subgraph['nodes'][node_id]
                    if node_i > node_j:
                        pairs.add((node_j, node_i))
                    else:
                        pairs.add((node_i, node_j))

        # get cache
        # redis_conf = self.config["cache"]
        cache = Cache(
            redis_host=os.environ['CACHE_HOST'],  # redis_conf.get ("host"),
            redis_port=os.environ['CACHE_PORT'],  # redis_conf.get ("port"),
            redis_db=os.environ['CACHE_DB'])  # redis_conf.get ("db"))

        # get supoorter
        support_module_name = 'ranker.support.omnicorp'
        supporter = import_module(support_module_name).get_supporter()

        # get all node supports
        for node in nodes:
            key = f"{supporter.__class__.__name__}({node})"
            support_dict = cache.get(key)
            if support_dict is not None:
                logger.info(f"cache hit: {key} {support_dict}")
            else:
                logger.info(f"exec op: {key}")
                support_dict = supporter.get_node_info(node)
                cache.set(key, support_dict)
            print(support_dict)
            # if support_dict is not None:
            #     node.properties.update(support_dict)

        # get all pair supports
        for pair in pairs:
            key = f"{supporter.__class__.__name__}({pair[0]},{pair[1]})"
            support_edge = cache.get(key)
            if support_edge is not None:
                logger.info(f"cache hit: {key} {support_edge}")
            else:
                logger.info(f"exec op: {key}")
                try:
                    support_edge = supporter.term_to_term(pair[0], pair[1])
                    cache.set(key, support_edge)
                except Exception as e:
                    raise e
                    # logger.debug('Support error, not caching')
                    # continue
            print(support_edge)
        # ... incorporate support into ranking

        # compute scores with NAGA, export to json
        pr = Ranker(answerset_subgraph)
        subgraphs_with_metadata, subgraphs = pr.report_ranking(subgraphs)  # returned subgraphs are sorted by rank

        misc_info = {
            'natural_question': self.natural_question,
            'num_total_paths': len(subgraphs)
        }
        aset = Answerset(misc_info=misc_info)
        # for substruct, subgraph in zip(score_struct, subgraphs):
        for subgraph in subgraphs_with_metadata:
            # graph = UniversalGraph(nodes=substruct['nodes'], edges=substruct['edges'])
            # graph.merge_multiedges()
            # graph.to_answer_walk(subgraph)

            answer = Answer(nodes=subgraph['nodes'],
                            edges=subgraph['edges'],
                            score=subgraph['score'])
            # TODO: move node/edge details to AnswerSet
            # node_ids = [node['id'] for node in graph.nodes]
            # edge_ids = [edge['id'] for edge in graph.edges]
            # answer = Answer(nodes=node_ids,\
            #         edges=edge_ids,\
            #         score=0)
            aset += answer  # substruct['score'])

        return aset

    def cypher_match_string(self, db=None):
        nodes, edges = self.machine_question['nodes'], self.machine_question['edges']

        # generate internal node and edge variable names
        node_references = {n['id']: NodeReference(n, db=db) for n in nodes}
        edge_references = [EdgeReference(e, db=db) for e in edges]

        # generate MATCH command string to get paths of the appropriate size
        match_strings = []
        for e, eref in zip(edges, edge_references):
            match_strings.append(f"MATCH ({node_references[e['source_id']]})-[{eref}]-({node_references[e['target_id']]})")

        match_string = ' '.join(match_strings)
        return match_string

    def cypher(self, db):
        '''
        Generate a Cypher query to extract the portion of the Knowledge Graph necessary to answer the question.

        Returns the query as a string.
        '''

        match_string = self.cypher_match_string(db)

        nodes, edges = self.machine_question['nodes'], self.machine_question['edges']

        # generate internal node and edge variable names
        node_names = [f"n{n['id']}" for n in nodes]
        edge_names = [f"e{e['id']}" for e in edges]

        # add bound fields and return map
        answer_return_string = f"RETURN {{{', '.join([f'{n}:{n}.id' for n in node_names])}}} as nodes, {{{', '.join([f'{e}:id({e})' for e in edge_names])}}} as edges"

        # return subgraphs matching query
        query_string = ' '.join([match_string, answer_return_string])

        return query_string

    def subgraph_with_support(self, db):
        match_string = self.cypher_match_string(db)

        nodes, edges = self.machine_question['nodes'], self.machine_question['edges']

        # generate internal node and edge variable names
        node_names = [f"n{n['id']}" for n in nodes]
        edge_names = [f"e{e['id']}" for e in edges]

        collection_string = f"""WITH {'+'.join([f'collect({n})' for n in node_names])} as nodes, {'+'.join([f'collect({e})' for e in edge_names])} as edges
            UNWIND nodes as n WITH collect(distinct n) as nodes, edges
            UNWIND edges as e WITH nodes, collect(distinct e) as edges"""
        support_string = """WITH
            [r in edges | r{.*, source_id:startNode(r).id, target_id:endNode(r).id, type:type(r), id:id(r)}] as edges,
            [n in nodes | n{.*, type:labels(n)[0]}] as nodes"""
        return_string = 'RETURN nodes, edges'
        query_string = "\n".join([match_string, collection_string, support_string, return_string])

        return query_string

    def subgraph(self):
        match_string = self.cypher_match_string()

        nodes, edges = self.machine_question['nodes'], self.machine_question['edges']

        # generate internal node and edge variable names
        node_names = ['n{:d}'.format(i) for i in range(len(nodes))]
        edge_names = ['r{0:d}{1:d}'.format(i, i + 1) for i in range(len(edges))]

        # just return a list of nodes and edges
        collection_string = f"WITH {'+'.join([f'collect({e})' for e in edge_names])} as rels, {'+'.join([f'collect({n})' for n in node_names])} as nodes"
        unique_string = 'UNWIND nodes as n WITH collect(distinct n) as nodes, rels UNWIND rels as r WITH nodes, collect(distinct r) as rels'
        return_string = "\n".join([collection_string, unique_string, 'RETURN nodes, rels'])

        query_string = "\n".join([match_string, return_string])

        return query_string
