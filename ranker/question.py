"""Question definition."""

# standard modules
import os
import sys
import warnings
import logging
import requests
from importlib import import_module
from uuid import uuid4
from collections import defaultdict
from itertools import combinations
import pickle
import resource

# 3rd-party modules
import networkx as nx

# our modules
from ranker.knowledgegraph import KnowledgeGraph
from ranker.answer import Answer, Answerset
from ranker.ranker import Ranker
from ranker.cache import Cache
from ranker.support.omnicorp import OmnicorpSupport


logger = logging.getLogger(__name__)


class NoAnswersException(Exception):
    """Exception when no answers are found."""
    pass


class NodeReference():
    """Node reference object."""
    def __init__(self, node):
        """Create a node reference."""
        name = f'{node["id"]}'
        label = node['type'] if 'type' in node else None

        if label == 'biological_process':
            label = 'biological_process_or_activity'

        if 'curie' in node and node['curie'] is not None:
            if isinstance(node['curie'], str):
                # synonymize/normalize curie
                if 'type' in node:
                    response = requests.post(f"http://{os.environ['BUILDER_HOST']}:6010/api/synonymize/{node['curie']}/{node['type']}/")
                    curie = response.json()['id']
                else:
                    curie = node['curie']
                prop_string = f" {{id: \'{curie}\'}}"
                conditions = ''
            elif isinstance(node['curie'], list):
                conditions = []
                for curie in node['curie']:
                    # synonymize/normalize curie
                    if 'type' in node:
                        response = requests.post(f"http://{os.environ['BUILDER_HOST']}:6010/api/synonymize/{curie}/{node['type']}/")
                        curie = response.json()['id']
                    # generate curie-matching condition
                    conditions.append(f"{name}.id = '{curie}'")
                # OR curie-matching conditions together
                prop_string = ''
                conditions = ' OR '.join(conditions)
            else:
                raise TypeError("Curie should be a string or list of strings.")
        else:
            prop_string = ''
            conditions = ''

        self.name = name
        self.label = label
        self.prop_string = prop_string
        self._conditions = conditions
        self._num = 0

    def __str__(self):
        """Return the cypher node reference."""
        self._num += 1
        if self._num == 1:
            return f'{self.name}' + \
                   f'{":" + self.label if self.label else ""}' + \
                   f'{self.prop_string}'
        return self.name

    @property
    def conditions(self):
        """Return conditions for the cypher node reference.

        To be used in a WHERE clause following the MATCH clause.
        """
        if self._num == 1:
            return self._conditions
        else:
            return ''

class EdgeReference():
    """Edge reference object."""

    def __init__(self, edge):
        """Create an edge reference."""
        name = f'{edge["id"]}'
        label = edge['type'] if 'type' in edge else None

        if 'type' in edge and edge['type'] is not None:
            if isinstance(edge['type'], str):
                label = edge['type']
                conditions = ''
            elif isinstance(edge['type'], list):
                conditions = []
                for predicate in edge['type']:
                    conditions.append(f'type({name}) = "{predicate}"')
                conditions = ' OR '.join(conditions)
                label = None
        else:
            label = None
            conditions = ''

        self.name = name
        self.label = label
        self._num = 0
        self._conditions = conditions

    def __str__(self):
        """Return the cypher edge reference."""
        self._num += 1
        if self._num == 1:
            return f'{self.name}{":" + self.label if self.label else ""}'
        else:
            return self.name

    @property
    def conditions(self):
        """Return conditions for the cypher node reference.

        To be used in a WHERE clause following the MATCH clause.
        """
        if self._num == 1:
            return self._conditions
        else:
            return ''


def record2networkx(records):
    """Return a networkx graph corresponding to the Neo4j Record.

    http://neo4j.com/docs/api/java-driver/current/org/neo4j/driver/v1/Record.html
    """
    graph = nx.MultiDiGraph()
    for record in records:
        if 'nodes' in record:
            for node in record["nodes"]:
                graph.add_node(node['id'], **node)
        if 'edges' in record:
            for edge in record["edges"]:
                graph.add_edge(edge['source_id'], edge['target_id'], **edge)
    return graph


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

    def relevant_knowledge_graph(self):
        # get the knowledge graph relevant to the question from the big knowledge graph in Neo4j
        with KnowledgeGraph() as database:
            with database.driver.session() as session:
                record = list(session.run(self.kg_query()))[0]
        knowledge_graph = {
            'nodes': record['nodes'],
            'edges': record['edges']
        }
        for node in knowledge_graph['nodes']:
            node['type'].remove('named_thing')
            node['type'] = node['type'][0]
        return knowledge_graph

    def fetch_answers(self):
        # get Neo4j connection
        with KnowledgeGraph() as database:

            # get knowledge graph
            logger.debug('Getting knowledge graph...')
            query_string = self.kg_query()
            logger.debug(query_string)
            with database.driver.session() as session:
                result = session.run(query_string)
            if result.peek() is None:
                logger.debug("No answers found. Returning None.")
                return None
            logger.debug('Converting Neo4j Result to dict...')
            result = list(result)

            knowledge_graph = {
                'nodes': result[0]['nodes'],
                'edges': result[0]['edges']
            }
            for node in knowledge_graph['nodes']:
                node['type'].remove('named_thing')
                node['type'] = node['type'][0]

            # get all answer maps relevant to the question from the knowledge graph
            logger.debug('Getting answer maps...')
            knowledge_maps = []
            options = {
                'limit': 1000000,
                'skip': 0
            }
            while True:
                answer_maps = database.query(self, options=options)
                options['skip'] += options['limit']
                answer_maps = [{'nodes': g['nodes'], 'edges': g['edges']} for g in answer_maps]
                knowledge_maps.extend(answer_maps)
                logger.debug(f'{len(knowledge_maps)} answer_maps: {int(sys.getsizeof(pickle.dumps(knowledge_maps)) / 1e6):d} MB')
                logger.debug(f'memory usage: {int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3):d} MB')
                if len(answer_maps) < options['limit']:
                    break

        return {
            'knowledge_graph': knowledge_graph,
            'knowledge_maps': knowledge_maps
        }

    def get_support(self, knowledge_graph, knowledge_maps, cache=None):
        with OmnicorpSupport() as supporter:
            # get all node supports
            logger.info('Getting individual node supports...')
            for node in knowledge_graph['nodes']:
                key = f"{supporter.__class__.__name__}({node['id']})"
                support_dict = cache.get(key)
                if support_dict is not None:
                    #logger.info(f"cache hit: {key} {support_dict}")
                    pass
                else:
                    #logger.info(f"exec op: {key}")
                    support_dict = supporter.get_node_info(node['id'])
                    cache.set(key, support_dict)
                # add omnicorp_article_count to nodes in networkx graph
                node.update(support_dict)

            logger.info('Getting node pair supports...')
            # generate a set of pairs of node curies
            pair_to_answer = defaultdict(list)  # a map of node pairs to answers
            for ans_idx, answer_map in enumerate(knowledge_maps):
                for combo in combinations(answer_map['nodes'], 2):
                    if isinstance(answer_map['nodes'][combo[0]], str):
                        sources = [answer_map['nodes'][combo[0]]]
                    else:
                        sources = answer_map['nodes'][combo[0]]
                    if isinstance(answer_map['nodes'][combo[1]], str):
                        targets = [answer_map['nodes'][combo[1]]]
                    else:
                        targets = answer_map['nodes'][combo[1]]
                    for source_id in sources:
                        for target_id in targets:
                            node_i, node_j = sorted([source_id, target_id])
                            pair_to_answer[(node_i, node_j)].append(ans_idx)

            cached_prefixes = cache.get('OmnicorpPrefixes')
            # get all pair supports
            for support_idx, pair in enumerate(pair_to_answer):
                #logger.info(pair)
                #The id's are in the cache sorted.
                ids = [pair[0],pair[1]]
                ids.sort()
                key = f"{supporter.__class__.__name__}({ids[0]},{ids[1]})"
                support_edge = cache.get(key)
                if support_edge is not None:
                    #logger.info(f"cache hit: {key} {support_edge}")
                    pass
                else:
                    #There are two reasons that we don't get anything back:
                    # 1. We haven't evaluated that pair
                    # 2. We evaluated, and found it to be zero, and it was part
                    #  of a prefix pair that we evaluated all of.  In that case
                    #  we can infer that getting nothing back means an empty list
                    #  check cached_prefixes for this...
                    prefixes = tuple([ ident.split(':')[0].upper() for ident in ids ])
                    if prefixes in cached_prefixes:
                        support_edge = []
                    else:
                        #logger.info(f"exec op: {key}")
                        try:
                            support_edge = supporter.term_to_term(pair[0], pair[1])
                            cache.set(key, support_edge)
                        except Exception as e:
                            raise e
                            # logger.debug('Support error, not caching')
                            # continue
                if not support_edge:
                    continue
                uid = str(uuid4())
                knowledge_graph['edges'].append({
                    'type': 'literature_co-occurrence',
                    'id': uid,
                    'num_publications': len(support_edge),
                    'publications': [],
                    'source_database': 'omnicorp',
                    'source_id': pair[0],
                    'target_id': pair[1],
                    'edge_source': 'omnicorp.term_to_term'
                })
                for sg in pair_to_answer[pair]:
                    knowledge_maps[sg]['edges'].update({f's{support_idx}': uid})
        return knowledge_graph
                

    def answer(self, max_results=250, use_support=True):
        """Answer the question.

        Returns the answer struct, something along the lines of:
        https://docs.google.com/document/d/1O6_sVSdSjgMmXacyI44JJfEVQLATagal9ydWLBgi-vE
        """

        # get cache
        cache = Cache(
            redis_host=os.environ['CACHE_HOST'],
            redis_port=os.environ['CACHE_PORT'],
            redis_db=os.environ['CACHE_DB'],
            redis_password=os.environ['CACHE_PASSWORD'])

        answers = self.fetch_answers()
        if answers is None:
            return None
        knowledge_graph = answers['knowledge_graph']
        knowledge_maps = answers['knowledge_maps']

        #We don't need this generality if everything is omnicorp
        # get supporter
        #support_module_name = 'ranker.support.omnicorp'
        #supporter = import_module(support_module_name).get_supporter()

        if use_support:
            knowledge_graph = self.get_support(knowledge_graph, knowledge_maps, cache=cache)

        logger.debug('Ranking...')
        # compute scores with NAGA, export to json
        pr = Ranker(knowledge_graph, self.machine_question)
        subgraphs_with_metadata, subgraphs = pr.report_ranking(knowledge_maps, max_results=max_results)  # returned subgraphs are sorted by rank

        misc_info = {
            'natural_question': self.natural_question,
            'num_total_paths': len(subgraphs)
        }
        aset = Answerset(misc_info=misc_info)
        # for substruct, subgraph in zip(score_struct, subgraphs):
        for subgraph in subgraphs_with_metadata:

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

    def cypher_match_string(self):
        nodes, edges = self.machine_question['nodes'], self.machine_question['edges']

        # generate internal node and edge variable names
        node_references = {n['id']: NodeReference(n) for n in nodes}
        edge_references = [EdgeReference(e) for e in edges]

        match_strings = []

        # match orphaned nodes
        def flatten(l):
            return [e for sl in l for e in sl]
        all_nodes = set([n['id'] for n in nodes])
        all_referenced_nodes = set(flatten([[e['source_id'], e['target_id']] for e in edges]))
        orphaned_nodes = all_nodes - all_referenced_nodes
        for n in orphaned_nodes:
            match_strings.append(f"MATCH ({node_references[n]})")

        # match edges
        for e, eref in zip(edges, edge_references):
            source_node = node_references[e['source_id']]
            target_node = node_references[e['target_id']]
            if 'type' in e and e['type']:
                match_strings.append(f"MATCH ({source_node})-[{eref}]->({target_node})")
            else:
                match_strings.append(f"MATCH ({source_node})-[{eref}]-({target_node})")
            conditions = [c for c in [source_node.conditions, target_node.conditions, eref.conditions] if c]
            if conditions:
                match_strings.append("WHERE " + " OR ".join(conditions))

        match_string = ' '.join(match_strings)
        return match_string

    def cypher(self, options=None):
        '''
        Generate a Cypher query to extract the portion of the Knowledge Graph necessary to answer the question.

        Returns the query as a string.
        '''

        match_string = self.cypher_match_string()

        nodes, edges = self.machine_question['nodes'], self.machine_question['edges']
        node_map = {n['id']: n for n in nodes}

        # generate internal node and edge variable names
        node_names = [f"{n['id']}" for n in nodes]
        edge_names = [f"{e['id']}" for e in edges]

        # deal with sets
        node_id_accessor = [f"collect(distinct {n['id']}.id) as {n['id']}" if 'set' in n and n['set'] else f"{n['id']}.id as {n['id']}" for n in nodes]
        edge_id_accessor = [f"collect(distinct toString(id({e['id']}))) as {e['id']}" for e in edges]
        with_string = f"WITH {', '.join(node_id_accessor+edge_id_accessor)}"

        # add bound fields and return map
        answer_return_string = f"RETURN {{{', '.join([f'{n}:{n}' for n in node_names])}}} as nodes, {{{', '.join([f'{e}:{e}' for e in edge_names])}}} as edges"

        # return answer maps matching query
        query_string = ' '.join([match_string, with_string, answer_return_string])
        if options is not None:
            if 'skip' in options:
                query_string += f' SKIP {options["skip"]}'
            if 'limit' in options:
                query_string += f' LIMIT {options["limit"]}'

        return query_string

    def kg_query(self):
        match_string = self.cypher_match_string()

        nodes, edges = self.machine_question['nodes'], self.machine_question['edges']

        # generate internal node and edge variable names
        node_names = [f"{n['id']}" for n in nodes]
        edge_names = [f"{e['id']}" for e in edges]

        collection_string = f"""WITH {' + '.join([f'collect(distinct {n})' for n in node_names])} as nodes, {'+'.join([f'collect(distinct {e})' for e in edge_names])} as edges
            UNWIND nodes as n WITH collect(distinct n) as nodes, edges
            UNWIND edges as e WITH nodes, collect(distinct e) as edges"""
        support_string = """WITH
            [r in edges | r{.*, source_id:startNode(r).id, target_id:endNode(r).id, type:type(r), id:toString(id(r))}] as edges,
            [n in nodes | n{.*, type:labels(n)}] as nodes"""
        return_string = 'RETURN nodes, edges'
        query_string = "\n".join([match_string, collection_string, support_string, return_string])

        return query_string
