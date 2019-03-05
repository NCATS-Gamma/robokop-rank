"""Question definition."""

# standard modules
import os
import sys
import time
from copy import deepcopy
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

output_formats = ['DENSE', 'MESSAGE', 'CSV', 'ANSWERS']


def cypher_prop_string(value):
    """Convert property value to cypher string representation."""
    if isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, str):
        return f"'{value}'"
    else:
        raise ValueError(f'Unsupported property type: {type(value).__name__}.')


class NodeReference():
    """Node reference object."""

    def __init__(self, node):
        """Create a node reference."""
        node = dict(node)
        name = f'{node.pop("id")}'
        label = node.pop('type', None)
        props = {}

        if label == 'biological_process':
            label = 'biological_process_or_activity'

        curie = node.pop("curie", None)
        if curie is not None:
            if isinstance(curie, str):
                # synonymize/normalize curie
                if label is not None:
                    response = requests.post(f"http://{os.environ['BUILDER_HOST']}:6010/api/synonymize/{curie}/{label}/")
                    curie = response.json()['id']
                props['id'] = curie
                conditions = ''
            elif isinstance(curie, list):
                conditions = []
                for ci in curie:
                    # synonymize/normalize curie
                    if label is not None:
                        response = requests.post(f"http://{os.environ['BUILDER_HOST']}:6010/api/synonymize/{ci}/{label}/")
                        ci = response.json()['id']
                    # generate curie-matching condition
                    conditions.append(f"{name}.id = '{ci}'")
                # OR curie-matching conditions together
                conditions = ' OR '.join(conditions)
            else:
                raise TypeError("Curie should be a string or list of strings.")
        else:
            conditions = ''

        node.pop('name', None)
        node.pop('set', False)
        props.update(node)

        self.name = name
        self.label = label
        self.prop_string = ' {' + ', '.join([f"`{key}`: {cypher_prop_string(props[key])}" for key in props]) + '}'
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

        keyword arguments: question_graph or machine_question, knowledge_graph, answers
        """
        # initialize all properties
        self.natural_question = ''
        self.question_graph = {}
        self.knowledge_graph = None
        self.answers = None

        # apply json properties to existing attributes
        attributes = self.__dict__.keys()
        if args:
            struct = args[0]
            for key in struct:
                if key in attributes:
                    setattr(self, key, struct[key])
                elif key == 'machine_question':
                    setattr(self, 'question_graph', struct[key])
                else:
                    warnings.warn("JSON field {} ignored.".format(key))

        # override any json properties with the named ones
        for key in kwargs:
            if key in attributes:
                setattr(self, key, kwargs[key])
            elif key == 'machine_question':
                setattr(self, 'question_graph', kwargs[key])
            else:
                warnings.warn("Keyword argument {} ignored.".format(key))

        # add ids to question graph edges if necessary ()
        if not any(['id' in e for e in self.question_graph['edges']]):
            for i, e in enumerate(self.question_graph['edges']):
                e['id'] = chr(ord('a') + i)

    def from_dense(self, denseAnswerset):
        # Check Q graph to ensure that there are no ambiguous nodes or edges
        qgraph = self.question_graph

        qnode_types = {}
        qnode_ids = {}
        for qn in qgraph['nodes']:
            qn_type = qn['type']
            if qn_type in qnode_types:
                raise RuntimeError('Invalid question graph for conversion. This question graph contains multiple nodes of the same type.')
            else:
                qnode_types[qn_type] = qn['id']
                qnode_ids[qn['id']] = qn_type

        # Edges are found by comparing the source_type and target_type
        qedge_types = {}
        qedge_ids = {}
        for qe in qgraph['edges']:
            n_source = [n for n in qgraph['nodes'] if n['id'] == qe['source_id']]
            if len(n_source) > 1:
                raise RuntimeError('Invalid question graph for conversion. This question graph contains multiple nodes with the id.')
            n_source = n_source[0]
            n_target = [n for n in qgraph['nodes'] if n['id'] == qe['target_id']]
            if len(n_target) > 1:
                raise RuntimeError('Invalid question graph for conversion. This question graph contains multiple nodes with the id.')
            n_target = n_target[0]

            qe_types = (n_source['type'], n_target['type'])

            if qe_types in qedge_types:
                raise RuntimeError('Invalid question graph for conversion. This question graph contains multiple edges connecting the same types.')
            else:
                qedge_types[qe_types] = qe['id']
                qedge_ids[qe['id']] = qe_types

        # If we made it to this point we know that,
        # 1) We only have question nodes with unique types
        # 2) We only have question edges with unique sets of source and target node types

        edge_id_fun = lambda e: f"({e['source_id']})-({e['type']};{e['provided_by']})-({e['target_id']})"
        knowledge_graph = {
            'nodes': [],
            'edges': []
        }
        answers = []
        node_ids = set()
        edge_ids = set()
        for r in denseAnswerset['result_list']:
            ans_nodes = r['result_graph']['node_list']
            ans_edges = r['result_graph']['edge_list']

            # Form knowledge_graph nodes
            for n in ans_nodes:
                if n['id'] not in node_ids:
                    node_ids.add(n['id'])
                    # n['curie'] = n['id']
                    knowledge_graph['nodes'].append(n)

            # Form knowledge_graph edge
            for e in ans_edges:
                if 'id' not in e:
                    # edge doesn't have an id
                    # We have to make up an id
                    # Using source_id, target_id, type, provided_by
                    e['id'] = edge_id_fun(e)
                
                edge_id = e['id']
                e['source'] = e['provided_by']
                
                if edge_id not in edge_ids:
                    edge_ids.add(edge_id)
                    knowledge_graph['edges'].append(e)

            # Form new answer
            # We need to figure out which answer node corresponds to which node in the q_graph
            # Similarly for edges
            # Above, we checked the the qgraph to ensure that nodes can be identified by type and
            # edges can be identified by the tuple of (source_type, target_type)
            new_answer = {
                'node_bindings': {},
                'edge_bindings': {},
                'score': []
            }
            
            for qn_id in qnode_ids:
                these_nodes = [n for n in ans_nodes if n['type'] == qnode_ids[qn_id]]
                new_answer['node_bindings'][qn_id] = [n['id'] for n in these_nodes]

                if isinstance(new_answer['node_bindings'][qn_id],list) and len(new_answer['node_bindings'][qn_id]):
                    new_answer['node_bindings'][qn_id] = new_answer['node_bindings'][qn_id][0]
            
            edge_id_tuples = {}
            edge_id_tuples2 = {}
            for e in ans_edges:
                n_source = next(n for n in ans_nodes if n['id'] == e['source_id'])
                n_target = next(n for n in ans_nodes if n['id'] == e['target_id'])
                
                edge_id_tuples[edge_id_fun(e)] = (n_source['type'], n_target['type'])
                edge_id_tuples2[edge_id_fun(e)] = (n_target['type'], n_source['type'])

            for qe_id in qedge_ids:
                these_edges = [e for e in ans_edges if edge_id_tuples[edge_id_fun(e)] == qedge_ids[qe_id] or edge_id_tuples2[edge_id_fun(e)] == qedge_ids[qe_id]]
                new_answer['edge_bindings'][qe_id] = [e['id'] for e in these_edges]
                
                if isinstance(new_answer['edge_bindings'][qe_id],list) and len(new_answer['edge_bindings'][qe_id]) == 1:
                    new_answer['edge_bindings'][qe_id] = new_answer['edge_bindings'][qe_id][0]

            new_answer['score'] = r['confidence']
            
            answers.append(new_answer)

        self.knowledge_graph = knowledge_graph
        self.answers = answers

    # @property
    # def knowledge_graph(self):
    #     if self.knowledge_graph is None:
    #         logger.debug('Auto-fetching knowledge graph')
    #         self.knowledge_graph = self.fetchknowledge_graph()
    #     return self.knowledge_graph

    # @knowledge_graph.setter
    # def setknowledge_graph(self, value):
    #     self.knowledge_graph = value

    def fetch_knowledge_graph(self, max_connectivity=0):
        # get the knowledge graph relevant to the question from the big knowledge graph in Neo4j
        with KnowledgeGraph() as database:
            options = {
                'max_connectivity': max_connectivity
            }
            query_string = self.cypher_query_knowledge_graph(options)

            logger.info("Fetching knowledge graph")
            # logger.debug(query_string)
            with database.driver.session() as session:
                result = session.run(query_string)
            if result.peek() is None:
                logger.info("No relevent knowledge graph was found. Returning None.")
                return None
            
            logger.info('Converting neo4j Result to dict')
            result = list(result)

            knowledge_graph = {
                'nodes': result[0]['nodes'],
                'edges': result[0]['edges']
            }
            # Remove neo4j oddities
            for node in knowledge_graph['nodes']:
                node['type'].remove('named_thing')
                # node['type'] = node['type'][0]

            logger.info('Knowledge graph obtained')
            self.knowledge_graph = knowledge_graph
        return

    # @property
    # def answers(self):
    #     if self._answers is None:
    #         logger.debug('Auto-fetching answer maps')
    #         self._answers = self.fetch_answers()
    #     return self._answers
    
    # @answers.setter
    # def set_answers(self, value):
    #     self._answers = value

    def fetch_answers(self, max_connectivity=0):
        # get Neo4j connection
        with KnowledgeGraph() as database:

            # # get knowledge graph
            # logger.debug('Getting knowledge graph...')

            # get all answer maps relevant to the question from the knowledge graph
            logger.info('Getting answers')
            answers = []
            options = {
                'limit': 1000000,
                'skip': 0,
                'max_connectivity': max_connectivity
            }
            
            logger.info('Running answer generation query')
            while True:
                
                query_string = self.cypher_query_answer_map(options=options)
                
                # logger.debug(query_string)
                start = time.time()
                with database.driver.session() as session:
                    result = session.run(query_string)
            
                these_answers = [{'node_bindings': r['nodes'], 'edge_bindings': r['edges']} for r in result]
                
                logger.info(f"{time.time()-start} seconds elapsed")
                logger.info(f"{len(these_answers)} subgraphs returned.")

                options['skip'] += options['limit']
                answers.extend(these_answers)

                logger.info(f'{len(answers)} answers: {int(sys.getsizeof(pickle.dumps(answers)) / 1e6):d} MB')
                logger.info(f'memory usage: {int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3):d} MB')
                if len(these_answers) < options['limit']:
                    # If this batch is less then the page size
                    # It must be the last page or you have reached the limit (below)
                    break
                elif len(these_answers) >= options['limit']:
                    logger.warning('Maximum number of potential answers reached. We are going to go forward with these, but there may be other potential answers.')
                    break
            self.answers = answers
            logger.info('Answers obtained')
        return

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
                    if cache and support_dict['omnicorp_article_count']:
                        cache.set(key, support_dict)
                # add omnicorp_article_count to nodes in networkx graph
                node.update(support_dict)
            logger.info(f'Finished support requests for {len(self.knowledge_graph["nodes"])} nodes')

            logger.info('Getting support for answer connected node pairs...')
            # Generate a set of pairs of node curies
            pair_to_answer = defaultdict(list)  # a map of node pairs to answers
            for ans_idx, answer_map in enumerate(self.answers):
                
                # Get all nodes that are not part of sets and densly connect them
                nodes = [nb for nb in answer_map['node_bindings'].values() if isinstance(nb, str)]
                for node_pair in combinations(nodes, 2):
                    pair_to_answer[node_pair].append(ans_idx)
                
                # For all nodes that are within sets, connect them to all nodes that are not in sets
                set_nodes_list_list = [nb for nb in answer_map['node_bindings'].values() if isinstance(nb, list)]
                set_nodes = [n for el in set_nodes_list_list for n in el]
                for set_node in set_nodes:
                    for node in nodes:
                        pair_to_answer[(node, set_node)].append(ans_idx)


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
                            support_edge = supporter.term_to_term_count(pair[0], pair[1])
                            if cache and support_edge:
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
                    self.answers[sg]['edge_bindings'].update({f's{support_idx}': uid})
            # Next pair

            logger.info(f'Finished support requests for {len(pair_to_answer)} pairs of nodes')

        # Close the supporter


    def fill(self, max_connectivity=0):
        """Determine potential answers to the question
        
        This primarily fectches the knowledge_graph and answers
        """

        # compute scores with ranker

        # Get local knowledge graph for this question
        self.fetch_knowledge_graph(max_connectivity=max_connectivity) # This will attempt to fetch from graph db if empty in a getter()
        if self.knowledge_graph is None:
            self.knowledge_graph = {'nodes':[], 'edges': []}
            self.answers = []
            logger.info('No possible answers found')
            return
        
        # Actually have a local knowledge graph
        self.fetch_answers(max_connectivity=max_connectivity) # This will attempt to fetch from the graph db if empty in a getter()
        if self.answers is None:
            logger.info('No possible answers found')
            return

        self.fetch_knowledge_graph_support()


    def rank(self, max_results=None):
        """Rank answers to the question
        
        This is mostly glue around the heavy lifting in ranker_obj.Ranker
        """
        logger.info('Ranking answers')
        if (not self.knowledge_graph) or (not self.answers) or ((('nodes' in self.knowledge_graph) and not self.knowledge_graph['nodes']) and (('edges' in self.knowledge_graph) and not self.knowledge_graph['edges'])):
            logger.info('   No answers to rank.')
            self.knowledge_graph = {'nodes':[], 'edges': []}
            self.answers = []
            return
        
        logger.info('Ranking answers')
        pr = Ranker(self.knowledge_graph, self.question_graph)
        (answer_scores, sorted_answers) = pr.rank(self.answers, max_results=max_results)

        # Add the scores to the answers
        for i, answer in enumerate(sorted_answers):
            answer['score'] = answer_scores[i]

        self.answers = sorted_answers

        # We may have pruned answers using max_results, in which case nodes and edges in the KG are not in answers
        # We should remove these unused nodes and edges
        def flatten_semilist(x):
            lists = [n if isinstance(n, list) else [n] for n in x]
            return [e for el in lists for e in el]

        node_ids = set()
        edge_ids = set()
        for answer in self.answers:
            these_node_ids = list(answer['node_bindings'].values())
            these_node_ids = flatten_semilist(these_node_ids)
            node_ids.update(these_node_ids)

            these_edge_ids = list(answer['edge_bindings'].values())
            these_edge_ids = flatten_semilist(these_edge_ids)
            edge_ids.update(these_edge_ids)

        self.knowledge_graph['edges'] = [e for e in self.knowledge_graph['edges'] if e['id'] in edge_ids]
        self.knowledge_graph['nodes'] = [n for n in self.knowledge_graph['nodes'] if n['id'] in node_ids]

    def select_knowledge_graph_types(self):
        """
        Look at question_graph to assign a single best type to each node in the kg
        """
        for n in self.knowledge_graph['nodes']:
            if ('type' not in n) or (('type' in n) and isinstance(n['type'], list)):
                # n['type'] is a list
                # We need to pick one
                # We will look through all answers
                # We will count the number of times it is used in each qNode
                # Then take the max to pick the best one
                # The type is then the type of that qNode
                qnode_counts = {}
                for qn in self.question_graph['nodes']:
                    qnode_counts[qn['id']] = 0

                for a in self.answers:
                    # Go through answers and look for this node
                    for qnode_id, qnode_vals in a['node_bindings'].items():
                        if isinstance(qnode_vals, str) and qnode_vals == n['id']:
                            qnode_counts[qnode_id] += 1
                        elif isinstance(qnode_vals, list) and n['id'] in qnode_vals:
                            qnode_counts[qnode_id] += 1

                # See what question node this was mapped to most
                max_count = 0
                max_id = ''
                for qn_id, qn_count in qnode_counts.items():
                    if qn_count >= max_count:
                        max_id = qn_id
                        max_count = qn_count
                
                best_qnode = [qn for qn in self.question_graph['nodes'] if qn['id'] == max_id]
                if len(best_qnode) < 1:
                    n['type'] = ''
                else:
                    n['type'] = best_qnode[0]['id']

    def dump(self):
        out = {
            'question_graph': self.question_graph,
            'knowledge_graph': self.knowledge_graph,
            'answers': self.answers
        }
        return out

    def dump_answers(self):
        # remove extra (support) edges from answer maps
        answers = deepcopy(self.answers)
        eids = [edge['id'] for edge in self.question_graph['edges']]
        for answer_map in answers:
            edge_bindings = answer_map['edge_bindings']
            answer_map['edge_bindings'] = {key: edge_bindings[key] for key in edge_bindings if key in eids}

        out = {
            'answers': answers
        }
        return out

    def dump_csv(self):
        nodes = self.question_graph['nodes']
        answers = self.answers
        csv_header = [n['id'] for n in nodes]
        csv_lines = []
        for a in answers:
            this_line = [[] for i in range(len(csv_header))]
            for n in a['node_bindings']:
                header_ind = [i for i, s in enumerate(csv_header) if s == n]

                if len(header_ind) < 1:
                    raise Exception(f'Unspecified question nodes found within an answer')
                elif len(header_ind) > 1:
                    raise Exception(f'Too many question nodes identified within an answer')
                
                n_values = a['node_bindings'][n]
                if isinstance(n_values, list):
                    n_values = n_values.join('|')

                this_line[header_ind[0]] = n_values
            csv_lines.append(this_line)

        csv = [','.join(csv_header)]
        csv.extend([','.join(l) for l in csv_lines])
        csv = '\n'.join(csv)

        return csv

    def dump_dense(self):
        self.select_knowledge_graph_types()

        def flatten_semilist(x):
            lists = [n if isinstance(n, list) else [n] for n in x]
            return [e for el in lists for e in el]

        misc_info = {
            'natural_question': self.natural_question,
            'num_total_paths': len(self.answers)
        }
        aset = Answerset(misc_info=misc_info)

        if self.answers:
            kgraph_map = {n['id']: n for n in self.knowledge_graph['nodes'] + self.knowledge_graph['edges']}
        
            for answer in self.answers:
                node_ids = flatten_semilist(answer['node_bindings'].values())
                nodes = [kgraph_map[n] for n in node_ids]
                edge_ids = flatten_semilist(answer['edge_bindings'].values())
                edges = [kgraph_map[e] for e in edge_ids]

                answer = Answer(nodes=nodes,
                            edges=edges,
                            score=answer['score'])
            
                aset += answer

        return aset.toStandard()

    def cypher_query_fragment_match(self, max_connectivity=0): # cypher_match_string
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
            if node_references[n].conditions:
                match_strings.append("WHERE " + node_references[n].conditions)

        # match edges
        include_size_constraints = bool(max_connectivity)
        for e, eref in zip(edges, edge_references):
            source_node = node_references[e['source_id']]
            target_node = node_references[e['target_id']]
            has_type = 'type' in e and e['type']
            is_directed = e.get('directed', has_type)
            if is_directed:
                match_strings.append(f"MATCH ({source_node})-[{eref}]->({target_node})")
            else:
                match_strings.append(f"MATCH ({source_node})-[{eref}]-({target_node})")
            conditions = [c for c in [source_node.conditions, target_node.conditions, eref.conditions] if c]
            if conditions:
                match_strings.append("WHERE " + " AND ".join(conditions))
                if include_size_constraints:
                    match_strings.append(f"AND size( ({target_node})-[]-() ) < {max_connectivity}")
            else:
                if include_size_constraints:
                    match_strings.append(f"WHERE size( ({target_node})-[]-() ) < {max_connectivity}")



        match_string = ' '.join(match_strings)
        # logger.debug(match_string)
        return match_string

    def cypher_query_answer_map(self, options=None):
        '''
        Generate a Cypher query to extract the answer maps for a question.

        Returns the query as a string.
        '''

        max_connectivity = 0
        if options and 'max_connectivity' in options:
            max_connectivity = options['max_connectivity']
        
        match_string = self.cypher_query_fragment_match(max_connectivity)

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

    def cypher_query_knowledge_graph(self, options=None): #kg_query
        '''
        Generate a Cypher query to extract the knowledge graph for a question.

        Returns the query as a string.
        '''

        max_connectivity = 0
        if options and 'max_connectivity' in options:
            max_connectivity = options['max_connectivity']

        match_string = self.cypher_query_fragment_match(max_connectivity)

        nodes, edges = self.question_graph['nodes'], self.question_graph['edges']

        # generate internal node and edge variable names
        node_names = [f"{n['id']}" for n in nodes]
        edge_names = [f"{e['id']}" for e in edges]

        collection_string = "WITH "
        collection_string += ' + '.join([f'collect(distinct {n})' for n in node_names]) + ' as nodes, '
        if edge_names:
            collection_string += ' + '.join([f'collect(distinct {e})' for e in edge_names]) + ' as edges'
        else:
            collection_string += '[] as edges'
        collection_string += "\nUNWIND nodes as n WITH collect(distinct n) as nodes, edges"
        if edge_names:
            collection_string += "\nUNWIND edges as e WITH nodes, collect(distinct e) as edges"""
        support_string = """WITH
            [r in edges | r{.*, source_id:startNode(r).id, target_id:endNode(r).id, type:type(r), id:toString(id(r))}] as edges,
            [n in nodes | n{.*, type:labels(n)}] as nodes"""
        return_string = 'RETURN nodes, edges'
        query_string = "\n".join([match_string, collection_string, support_string, return_string])

        return query_string
