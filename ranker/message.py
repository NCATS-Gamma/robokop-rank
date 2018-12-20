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
from ranker.ranker_obj import Ranker
from ranker.cache import Cache
from ranker.support.omnicorp import OmnicorpSupport

logger = logging.getLogger('ranker')

output_formats = ['APIStandard', 'Message']

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

class Message():
    """Message object.

    Represents a question and answer object such as "What genetic condition provides protection against disease X?"

    """

    def __init__(self, *args, **kwargs):
        """Create a question.

        keyword arguments: question_graph or machine_question, knowledge_graph, answer_maps
        """
        # initialize all properties
        self.natural_question = ''
        self.question_graph = {}
        self._knowledge_graph = None
        self._answer_maps = None

        # apply json properties to existing attributes
        attributes = self.__dict__.keys()
        if args:
            struct = args[0]
            for key in struct:
                if key in attributes:
                    setattr(self, key, struct[key])
                elif key is 'machine_question':
                    setattr(self, 'question_graph', struct[key])
                else:
                    warnings.warn("JSON field {} ignored.".format(key))

        # override any json properties with the named ones
        for key in kwargs:
            if key in attributes:
                setattr(self, key, kwargs[key])
            elif key is 'machine_question':
                setattr(self, 'question_graph', kwargs[key])
            else:
                warnings.warn("Keyword argument {} ignored.".format(key))

        # add ids to question graph edges if necessary ()
        if not any(['id' in e for e in self.question_graph['edges']]):
            for i, e in enumerate(self.question_graph['edges']):
                e['id'] = chr(ord('a') + i)

    @property
    def knowledge_graph(self):
        if self._knowledge_graph is None:
            self._knowledge_graph = self.fetch_knowledge_graph()
        return self._knowledge_graph

    @knowledge_graph.setter
    def set_knowledge_graph(self, value):
        self._knowledge_graph = value

    def fetch_knowledge_graph(self):
        # get the knowledge graph relevant to the question from the big knowledge graph in Neo4j
        with KnowledgeGraph() as database:
            query_string = self.cypher_query_knowledge_graph()

            logger.debug("Fetching knowledge graph")
            logger.debug(query_string)
            with database.driver.session() as session:
                result = session.run(query_string)
            if result.peek() is None:
                logger.debug("No relevent knowledge graph was found. Returning None.")
                return None
            
            logger.debug('Converting Neo4j Result to dict...')
            result = list(result)

            knowledge_graph = {
                'nodes': result[0]['nodes'],
                'edges': result[0]['edges']
            }
            # Remove neo4j oddities
            for node in knowledge_graph['nodes']:
                node['type'].remove('named_thing')
                node['type'] = node['type'][0]

        return knowledge_graph

    @property
    def answer_maps(self):
        if self._answer_maps is None:
            self._answer_maps = self.fetch_answer_maps()
        return self._answer_maps
    
    @answer_maps.setter
    def set_answer_maps(self, value):
        self._answer_maps = value

    def fetch_answer_maps(self):
        # get Neo4j connection
        with KnowledgeGraph() as database:

            knowledge_graph = self.fetch_knowledge_graph()

            # get knowledge graph
            logger.debug('Getting knowledge graph...')

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
                answer_maps = [{'node_bindings': g['nodes'], 'edge_bindings': g['edges']} for g in answer_maps]
                knowledge_maps.extend(answer_maps)
                logger.debug(f'{len(knowledge_maps)} answer_maps: {int(sys.getsizeof(pickle.dumps(knowledge_maps)) / 1e6):d} MB')
                logger.debug(f'memory usage: {int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3):d} MB')
                if len(answer_maps) < options['limit']:
                    break

        return {
            'knowledge_graph': knowledge_graph,
            'answers': knowledge_maps
        }

    def fetch_knowledge_graph_support(self):

        #We don't need this generality if everything is omnicorp
        # get supporter
        #support_module_name = 'ranker.support.omnicorp'
        #supporter = import_module(support_module_name).get_supporter()

        # get cache if possible
        try:
            cache = Cache(
                redis_host=os.environ['CACHE_HOST'],
                redis_port=os.environ['CACHE_PORT'],
                redis_db=os.environ['CACHE_DB'],
                redis_password=os.environ['CACHE_PASSWORD'])
        except Exception as e:
            logger.info('Could not connect to request cache for support...')
            cache = None


        with OmnicorpSupport() as supporter:
            # get all node supports
            logger.info('Getting individual node supports...')
            for node in self.knowledge_graph['nodes']:
                key = f"{supporter.__class__.__name__}({node['id']})"
                support_dict = cache.get(key) if cache else None
                if support_dict is not None:
                    #logger.info(f"cache hit: {key} {support_dict}")
                    pass
                else:
                    #logger.info(f"exec op: {key}")
                    support_dict = supporter.get_node_info(node['id'])
                    if cache:
                        cache.set(key, support_dict)
                # add omnicorp_article_count to nodes in networkx graph
                node.update(support_dict)

            logger.info('Getting support for answer connected node pairs...')
            # Generate a set of pairs of node curies
            pair_to_answer = defaultdict(list)  # a map of node pairs to answers
            for ans_idx, answer_map in enumerate(self.answer_maps):
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

            cached_prefixes = cache.get('OmnicorpPrefixes') if cache else None
            # get all pair supports
            for support_idx, pair in enumerate(pair_to_answer):
                #logger.info(pair)
                #The id's are in the cache sorted.
                ids = [pair[0],pair[1]]
                ids.sort()
                key = f"{supporter.__class__.__name__}_count({ids[0]},{ids[1]})"
                support_edge = cache.get(key) if cache else None
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
                    if cached_prefixes and prefixes in cached_prefixes:
                        support_edge = []
                    else:
                        #logger.info(f"exec op: {key}")
                        try:
                            support_edge = supporter.term_to_term(pair[0], pair[1])
                            if cache:
                                cache.set(key, support_edge)
                        except Exception as e:
                            raise e
                            # logger.debug('Support error, not caching')
                            # continue
                if not support_edge:
                    continue
                uid = str(uuid4())
                self.knowledge_graph['edges'].append({
                    'type': 'literature_co-occurrence',
                    'id': uid,
                    'num_publications': support_edge,
                    'publications': [],
                    'source_database': 'omnicorp',
                    'source_id': pair[0],
                    'target_id': pair[1],
                    'edge_source': 'omnicorp.term_to_term'
                })

                for sg in pair_to_answer[pair]:
                    self.answer_maps[sg]['edges'].update({f's{support_idx}': uid})
            # Next pair
        # Close the supporter

    def rank_answers(self, max_results=250):
        """Rank answers to the question
        
        This is mostly glue around the heavy lifting in ranker_obj.Ranker
        """

        # compute scores with ranker

        # Get local knowledge graph for this question
        local_kg = self.knowledge_graph # This will attempt to fetch from graph db if empty in a getter()
        if local_kg is None:
            self.answer_maps = None
            logger.debug('No possible answers found')
            return
        
        # Actually have a local knowledge graph
        answer_maps = self.answer_maps # This will attempt to fetch from the graph db if empty in a getter()
        if answer_maps is None:
            logger.debug('No possible answers found')
            return
        
        logger.debug('Ranking answers')
        pr = Ranker(local_kg, self.question_graph)
        (answer_scores, sorted_answer_maps) = pr.rank(answer_maps, max_results=max_results)
        
        # Add the scores to the answer_maps
        for i, answer in enumerate(sorted_answer_maps):
            sorted_answer_maps['score'] = answer_scores[i]

        self.answer_maps = sorted_answer_maps

    def dump(self):
        out = {
            'question_graph': self.question_graph,
            'knowledge_graph': self.knowledge_graph,
            'answer_maps': self.answer_maps
        }
        return out

    def dump_standard(self):

        def flatten_semilist(x):
            lists = [n if isinstance(n, list) else [n] for n in x]
            return [e for el in lists for e in el]

        misc_info = {
            'natural_question': self.natural_question,
            'num_total_paths': len(self.answer_maps)
        }
        aset = Answerset(misc_info=misc_info)

        kgraph_map = {n['id']: n for n in self.knowledge_graph['nodes'] + self.knowledge_graph['edges']}
        for subgraph in self.answer_maps:
            node_ids = flatten_semilist(subgraph['nodes'].values())
            nodes = [kgraph_map[n] for n in node_ids]
            edge_ids = flatten_semilist(subgraph['edges'].values())
            edges = [kgraph_map[e] for e in edge_ids]

            answer = Answer(nodes=nodes,
                        edges=edges,
                        score=subgraph['score'])
        
            aset += answer

        return aset.toStandard()

    def cypher_query_fragment_match(self): # cypher_match_string
        '''
        Generate a Cypher query fragment to match the nodes and edges that correspond to a question.

        This is used internally for cypher_query_answer_map and cypher_query_knowledge_graph

        Returns the query fragment as a string.
        '''

        nodes, edges = self.question_graph['nodes'], self.question_graph['edges']

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

    def cypher_query_answer_map(self, options=None):
        '''
        Generate a Cypher query to extract the answer maps for a question.

        Returns the query as a string.
        '''

        match_string = self.cypher_query_fragment_match()

        nodes, edges = self.question_graph['nodes'], self.question_graph['edges']
        # node_map = {n['id']: n for n in nodes}

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

    def cypher_query_knowledge_graph(self): #kg_query
        '''
        Generate a Cypher query to extract the knowledge graph for a question.

        Returns the query as a string.
        '''

        match_string = self.cypher_query_fragment_match()

        nodes, edges = self.question_graph['nodes'], self.question_graph['edges']

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
