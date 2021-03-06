#!/usr/bin/env python

"""Flask REST API server for ranker."""

import os
import logging
import json
import sys

import redis
import requests
from flask_restful import Resource
from flask import request

from ranker.api.setup import app, api
from ranker.message import Message, output_formats
from ranker.tasks import answer_question, celery
from ranker.knowledgegraph import KnowledgeGraph
from ranker.support.omnicorp import OmnicorpSupport
from ranker.util import flatten_semilist
from ranker.api.compliance import message2std, std2message
from ranker.api.node_lookup import get_nodes_by_name
import ranker.definitions
import ranker.api.utilities
from ranker.core import run, dense, to_robokop, strip_kg, csv

logger = logging.getLogger("ranker")


redis_client = redis.Redis(
    host=os.environ['RESULTS_HOST'],
    port=os.environ['RESULTS_PORT'],
    db=os.environ['RANKER_RESULTS_DB'],
    password=os.environ['RESULTS_PASSWORD'])

class InvalidUsage(Exception):
    pass


def parse_args_output_format(req_args):
    output_format = req_args.get('output_format', default=output_formats[1])
    if output_format.upper() not in output_formats:
        raise InvalidUsage(f'output_format must be one of [{" ".join(output_formats)}]')
    
    return output_format

def parse_args_max_results(req_args):
    max_results = req_args.get('max_results', default=None)
    max_results = max_results if max_results is not None else 250
    max_results = int(max_results) if isinstance(max_results, str) else max_results
    return max_results

def parse_args_max_connectivity(req_args):
    max_connectivity = req_args.get('max_connectivity', default=None)
    
    if max_connectivity and isinstance(max_connectivity, str):
        if max_connectivity.lower() == 'none':
            max_connectivity = None
        else:
            try:
                max_connectivity = int(max_connectivity)
            except ValueError:
                raise RuntimeError(f'max_connectivity should be an integer')
            except:
                raise
            if max_connectivity < 0:
                max_connectivity = None

    return max_connectivity


class NodeLookup(Resource):
    def post(self):
        """
        Look up biomedical entities by name.
        ---
        tags: [util]
        requestBody:
            description: A string contained in the entity name.
            content:
                text/plain:
                    schema:
                        type: string
                    examples:
                        ebola:
                            summary: "Find the identifier for ebola"
                            value: "ebola"
                        cog:
                            summary: "Find anything containing 'cog'"
                            value: "cog"
            required: true
        responses:
            200:
                description: A list of entities and scores.
                content:
                    application/json:
                        schema:
                            type: array
                            items:
                                type: object
                                properties:
                                    curie:
                                        type: string
                                    name:
                                        type: string
                                    type:
                                        type: array
                                        items:
                                            type: string
        """
        
        results = get_nodes_by_name(request.data.decode('utf-8'))
        return results, 200

api.add_resource(NodeLookup, '/entity_lookup/')

class NodeLookupFilter(Resource):
    def post(self,node_type):
        """
        Look up biomedical entities by name.
        ---
        tags: [util]
        parameters:
          - in: path
            name: node_type
            description: type of search term of interest
            schema:
                type: string
            required: true
            example: "disease"
        requestBody:
            description: A string contained in the entity name.
            content:
                text/plain:
                    schema:
                        type: string
                    examples:
                        ebola:
                            summary: "Find the identifier for ebola"
                            value: "ebola"
            required: true
        responses:
            200:
                description: A list of entities and scores.
                content:
                    application/json:
                        schema:
                            type: array
                            items:
                                type: object
                                properties:
                                    curie:
                                        type: string
                                    name:
                                        type: string
                                    type:
                                        type: array
                                        items:
                                            type: string
        """
        
        results = get_nodes_by_name(request.data.decode('utf-8'), node_type=node_type)
        return results, 200

api.add_resource(NodeLookupFilter, '/entity_lookup/<node_type>/')


class AnswerQuestionStd(Resource):
    def post(self):
        """
        Get answers to a question
        ---
        tags: [answer]
        requestBody:
            description: A message with question graph.
            content:
                application/json:
                    schema:
                        type: object
                        properties:
                            message:
                                $ref: '#/definitions/Message'
                    example:
                        message:
                            query_graph:
                                nodes:
                                  - id: "n00"
                                    type: "disease"
                                    curie: "MONDO:0005737"
                                  - id: "n01"
                                    type: "phenotypic_feature"
                                edges:
                                  - id: "e00"
                                    source_id: "n00"
                                    target_id: "n01"
            required: true
        parameters:
          - in: query
            name: max_results
            description: Maximum number of results to return. Provide -1 to indicate no maximum.
            schema:
                type: integer
            default: -1
          - in: query
            name: output_format
            description: Requested output format. DENSE, MESSAGE, CSV, ANSWERS, or STD
            schema:
                type: string
            default: STD
          - in: query
            name: max_connectivity
            description: Max connectivity of nodes considered in the answers, Use 0 for no restriction
            schema:
                type: integer
            default: -1
          - in: query
            name: use_novelty
            description: Novelty weighting helps prevent over-weighting answers with popular nodes
            schema:
                type: boolean
            default: false
        responses:
            200:
                description: A message with knowledge graph and answers.
                content:
                    application/json:
                        schema:
                            $ref: '#/definitions/Message'
        """
        max_results = int(request.args.get('max_results', -1))
        output_format = request.args.get('output_format', 'STD')
        max_connectivity = int(request.args.get('max_connectivity', -1))
        use_novelty = request.args.get('use_novelty', 'false').lower() == 'true'

        message_json = {
            'query_graph': request.json['message']['query_graph'],
        }
        message_json = run(
            message_json,
            max_results=max_results,
            max_connectivity=max_connectivity,
            use_novelty=use_novelty,
        )

        # convert output format
        if output_format.upper() == 'DENSE':
            output = dense(message_json)
        elif output_format.upper() == 'ANSWERS':
            output = to_robokop(strip_kg(message_json))
        elif output_format.upper() == 'MESSAGE':
            output = to_robokop(message_json)
        elif output_format.upper() == 'STD':
            output = message_json
        elif output_format.upper() == 'CSV':
            output = csv(message_json)
        else:
            raise ValueError(f'Unrecognized output format "{output_format}"')

        return output, 200

api.add_resource(AnswerQuestionStd, '/query/')

class AnswerQuestion(Resource):
    def post(self):
        """
        Get answers to a question
        ---
        tags: [answer]
        requestBody:
            description: A message with question graph.
            content:
                application/json:
                    schema:
                        $ref: '#/definitions/Message'
                    example:
                        question_graph:
                            nodes:
                              - id: "n00"
                                type: "disease"
                                curie: "MONDO:0005737"
                              - id: "n01"
                                type: "gene"
                                set: true
                              - id: "n02"
                                type: genetic_condition
                            edges:
                              - id: "e00"
                                source_id: "n00"
                                target_id: "n01"
                              - id: "e01"
                                source_id: "n01"
                                target_id: "n02"
                        knowledge_graph:
                            nodes: []
                            edges: []
                        answers: []
            required: true
        parameters:
          - in: query
            name: max_results
            description: Maximum number of results to return. Provide -1 to indicate no maximum.
            schema:
                type: integer
            default: -1
          - in: query
            name: output_format
            description: Requested output format. DENSE, MESSAGE, CSV or ANSWERS
            schema:
                type: string
            default: MESSAGE
          - in: query
            name: max_connectivity
            description: Max connectivity of nodes considered in the answers, Use 0 for no restriction
            schema:
                type: integer
            default: -1
          - in: query
            name: use_novelty
            description: Novelty weighting helps prevent over-weighting answers with popular nodes
            schema:
                type: boolean
            default: false
        responses:
            200:
                description: Successfull queued a task
                content:
                    text/plain:
                        schema:
                            type: string
        """
        max_results = int(request.args.get('max_results', -1))
        output_format = request.args.get('output_format', 'MESSAGE')
        max_connectivity = int(request.args.get('max_connectivity', -1))
        use_novelty = request.args.get('use_novelty', 'false').lower() == 'true'

        task = answer_question.apply_async(
            args=[request.json],
            kwargs={
                'max_results': max_results,
                'output_format': output_format,
                'max_connectivity': max_connectivity,
                'use_novelty': use_novelty,
            }
        )
        return {'task_id': task.id}, 202

api.add_resource(AnswerQuestion, '/')


class QuestionSubgraph(Resource):
    def post(self):
        """
        Get knowledge graph for message.
        ---
        tags: [knowledgeGraph]
        requestBody:
            description: A message with question graph and/or answers.
            content:
                application/json:
                    schema:
                        $ref: '#/definitions/Message'
            required: true
        responses:
            200:
                description: A knowledge graph.
                content:
                    application/json:
                        schema:
                            $ref: '#/definitions/KGraph'
        """
        message = request.json
        if not message.get('answers', []):
            message_obj = Message(request.json)

            message_obj.fetch_knowledge_graph()
            kg = message_obj.knowledge_graph

            return kg, 200

        # get nodes and edge ids from message answers
        node_ids = [knode_id for answer in message['answers'] for knode_id in answer['node_bindings'].values()]
        edge_ids = [kedge_id for answer in message['answers'] for kedge_id in answer['edge_bindings'].values()]
        node_ids = flatten_semilist(node_ids)
        edge_ids = flatten_semilist(edge_ids)

        # unique over node and edge ids
        node_ids = list(set(node_ids))
        edge_ids = list(set(edge_ids))

        nodes = get_node_properties(node_ids)
        edges = get_edge_properties(edge_ids)

        kg = {
            'nodes': nodes,
            'edges': edges
        }
        return kg, 200

api.add_resource(QuestionSubgraph, '/knowledge_graph/')


class MultiNodeLookup(Resource):
    """Multi-node lookup endpoints."""

    def post(self):
        """
        Get properties of nodes by id.
        Ignores nodes that are not found.
        If 'fields' is provided, returns only the requested fields.
        Returns null for any unknown fields.

        RESULTS MAY NOT BE SORTED!
        ---
        tags: [knowledgeGraph]
        requestBody:
            name: request
            description: The node ids for lookup.
            content:
                application/json:
                    schema:
                        required:
                          - node_ids
                        properties:
                            node_ids:
                                type: array
                                items:
                                    type: string
                            fields:
                                type: array
                                items:
                                    type: string
                        example:
                            node_ids:
                              - "MONDO:0005737"
                              - "HGNC:16361"
            required: true
        responses:
            200:
                description: Node
                content:
                    application/json:
                        schema:
                            $ref: '#/definitions/KNode'
        """
        node_ids = request.json['node_ids']
        fields = request.json.get('fields', None)

        return get_node_properties(node_ids, fields), 200

api.add_resource(MultiNodeLookup, '/multinode_lookup/')


def batches(arr, n):
    """Iterator separating arr into batches of size n."""
    for i in range(0, len(arr), n):
        yield arr[i:i + n]


def get_node_properties(node_ids, fields=None):
    functions = {
        'type': 'labels(n)'
    }

    if fields is not None:
        functions.update({
            'degree': 'size((n)-[]-())'
        })
        prop_string = ', '.join([f'{key}:{functions[key]}' if key in functions else f'{key}:n.{key}' for key in fields])
    else:
        prop_string = ', '.join([f'{key}:{functions[key]}' for key in functions] + ['.*'])

    output = []
    n = 10000
    for batch in batches(node_ids, n):
        where_string = 'n.id IN [' + ', '.join([f'"{node_id}"' for node_id in batch]) + ']'
        query_string = f'MATCH (n) WHERE {where_string} RETURN n{{{prop_string}}}'

        with KnowledgeGraph() as database:
            with database.driver.session() as session:
                result = session.run(query_string)

        for record in result:
            r = record['n']
            if 'type' in r and 'named_thing' in r['type']:
                r['type'].remove('named_thing')

            output.append(r)

    return output


class MultiEdgeLookup(Resource):
    """Multi-edge lookup endpoints."""

    def post(self):
        """
        Get properties of edges by id.
        Ignores edges that are not found.
        If 'fields' is provided, returns only the requested fields.
        Returns null for any unknown fields.

        RESULTS MAY NOT BE SORTED!
        ---
        tags: [knowledgeGraph]
        requestBody:
            name: request
            description: The edge id for lookup
            content:
                application/json:
                    schema:
                        required:
                          - edge_ids
                        properties:
                            edge_ids:
                                type: array
                                items:
                                    type: string
                            fields:
                                type: array
                                items:
                                    type: string
                        example:
                            edge_ids:
                              - "1e3b89d466ad612b6cf93c846d6f3323"
                              - "ab556faa169c3ecf9f4beaec395958d4"
            required: true
        responses:
            200:
                description: Edge
                content:
                    application/json:
                        schema:
                            $ref: '#/definitions/KEdge'
        """
        edge_ids = request.json['edge_ids']
        fields = request.json.get('fields', None)

        return get_edge_properties(edge_ids, fields), 200

api.add_resource(MultiEdgeLookup, '/multiedge_lookup/')


def get_edge_properties(edge_ids, fields=None):
    functions = {
        'source_id': 'startNode(e).id',
        'target_id': 'endNode(e).id',
        'type': 'type(e)'
    }

    if fields is not None:
        prop_string = ', '.join([f'{key}:{functions[key]}' if key in functions else f'{key}:e.{key}' for key in fields])
    else:
        prop_string = ', '.join([f'{key}:{functions[key]}' for key in functions] + ['.*'])

    output = []
    statement = f"CALL db.index.fulltext.queryRelationships('edge_id_index', {{edge_ids}}) YIELD relationship WITH relationship as e RETURN e{{{prop_string}}}"
    with KnowledgeGraph() as database:
        with database.driver.session() as session:
            tx = session.begin_transaction()
            for batch in batches(edge_ids, 1024):
                result = tx.run(statement, {'edge_ids': ' '.join(batch)})

                for record in result:
                    output.append(record['e'])
            tx.commit()

    return output

class Results(Resource):
    def get(self, task_id):
        """
        Fetch results from task
        ---
        tags: [tasks]
        parameters:
          - in: path
            name: task_id
            description: ID of task
            schema:
                type: string
            required: true
        responses:
            200:
                description: result
                content:
                    application/json:
                        schema:
                            $ref: "#/definitions/Graph"
        """
        r = redis.Redis(
            host=os.environ['RESULTS_HOST'],
            port=os.environ['RESULTS_PORT'],
            db=os.environ['RANKER_RESULTS_DB'],
            password=os.environ['RESULTS_PASSWORD'])

        task_id = 'celery-task-meta-'+task_id
        task_string = r.get(task_id)
        if task_string is None:
            return 'No such task', 404
        info = json.loads(task_string)
        if info['status'] != 'SUCCESS':
            return 'This task is incomplete or failed', 404

        filename = info['result']
        if filename:
            result_path = os.path.join(os.environ['ROBOKOP_HOME'], 'robokop-rank', 'answers', filename)

            with open(result_path, 'r') as f:
                file_contents = json.load(f)
            os.remove(result_path)

            return file_contents, 200
        else:
            return 'No results found', 200

api.add_resource(Results, '/task/<task_id>/result/')

class Omnicorp(Resource):
    def get(self, id1, id2):
        """
        Get publications for a pair of identifiers
        ---
        tags: [util]
        parameters:
          - in: path
            name: id1
            description: "curie of first term"
            schema:
                type: string
            required: true
            default: "MONDO:0005737"
          - in: path
            name: id2
            description: "curie of second term"
            schema:
                type: string
            required: true
            default: "HGNC:7897"
        responses:
            200:
                description: publications
                content:
                    application/json:
                        schema:
                            type: array
                            items:
                                type: string
        """
        with OmnicorpSupport() as supporter:
            # Ids would be cached in sorted order
            ids = [id1,id2]
            ids.sort()
            
            publications = supporter.term_to_term(ids[0], ids[1])

        return publications, 200

api.add_resource(Omnicorp, '/omnicorp/<id1>/<id2>/')

class Omnicorp1(Resource):
    def get(self, id1):
        """
        Get publications for a pair of identifiers
        ---
        tags: [util]
        parameters:
          - in: path
            name: id1
            description: "curie of first term"
            schema:
                type: string
            required: true
            default: "MONDO:0005737"
        responses:
            200:
                description: publications
                content:
                    application/json:
                        schema:
                            type: array
                            items:
                                type: string
        """
        with OmnicorpSupport() as supporter:
            publications = supporter.get_node_publications(id1)

        return publications, 200

api.add_resource(Omnicorp1, '/omnicorp/<id1>/')

class TaskStatus(Resource):
    def get(self, task_id):
        """
        Get status of task
        ---
        tags: [tasks]
        parameters:
          - in: path
            name: task_id
            description: ID of task
            schema:
                type: string
            required: true
        responses:
            200:
                description: result
                content:
                    application/json:
                        schema:
                            $ref: "#/definitions/Graph"
        """

        task_id = 'celery-task-meta-'+task_id
        task_string = redis_client.get(task_id)
        if task_string is None:
            return 'No such task', 404
        info = json.loads(task_string)
        
        return info, 200

    def delete(self, task_id):
        """Revoke task
        ---
        tags: [tasks]
        parameters:
          - in: path
            name: task_id
            description: "task id"
            schema:
                type: string
            required: true
        responses:
            204:
                description: task revoked
                content:
                    text/plain:
                        schema:
                            type: string
        """
        
        task_redis_id = 'celery-task-meta-'+task_id
        task_string = redis_client.get(task_redis_id)
        if task_string is None:
            return 'No such task', 404

        try:
            celery.control.revoke(task_id, terminate=True)
        except Exception as err:
            return 'We failed to revoke the task', 500

        return '', 204

api.add_resource(TaskStatus, '/task/<task_id>/')

class TaskLog(Resource):
    def get(self, task_id):
        """
        Get activity log for a task
        ---
        tags: [tasks]
        parameters:
          - in: path
            name: task_id
            description: ID of task
            schema:
                type: string
            required: true
        responses:
            200:
                description: text
        """

        task_log_file = os.path.join(os.environ['ROBOKOP_HOME'], 'logs', 'ranker_task_logs', f'{task_id}.log')
        if os.path.isfile(task_log_file):
            with open(task_log_file, 'r') as log_file:
                log_contents = log_file.read()
            return log_contents, 200
        else:
            return 'Task ID not found', 404


api.add_resource(TaskLog, '/task/<task_id>/log/')

class EnrichedExpansion(Resource):
    def post(self, type1, type2):
        """
        Enriched search in the local knowledge graph
        ---
        tags: [simple]
        parameters:
          - in: path
            name: type1
            description: "type of query node"
            schema:
                type: string
            required: true
            default: "disease"
          - in: path
            name: type2
            description: "type of return nodes"
            schema:
                type: string
            required: true
            default: "phenotypic_feature"
        requestBody:
            content:
                application/json:
                    schema:
                        type: object
                        properties:
                            identifiers:
                                description: "identifiers of query nodes"
                                schema:
                                    type: list
                                default:
                                  - MONDO:0005737
                            threshhold:
                                description: "Number between 0 and 1 indicating the maximum p-value to return"
                                schema:
                                    type: float
                                default: 0.05
                            max_results:
                                description: "The maximum number of results to return. Set to 0 to return all results."
                                schema:
                                    type: integer
                                default: 100
                            num_type1:
                                description: "The total number of type1 entities that can exist.  If not specified, this is estimated from the cache"
                                schema:
                                    type: integer
                        required:
                          - identifiers
                        example:
                            identifiers:
                              - MONDO:0005737
                            threshhold: 0.05
                            max_results: 100
        responses:
            200:
                description: result
                content:
                    application/json:
                        schema:
                            $ref: "#/definitions/SimilarityResult"
        """
        parameters = request.json
        identifiers = parameters['identifiers']
        if 'threshhold' in parameters:
            threshhold = parameters['threshhold']
        else:
            threshhold = 0.05
        if ('max_results' in parameters) and parameters['max_results']:
            maxresults = parameters['max_results']
            maxresults = int(maxresults) if isinstance(maxresults, str) else maxresults
            maxresults = maxresults if maxresults is not None else 0
        else:
            maxresults = 250
        if 'num_type1' in parameters:
            num_type1 = parameters['num_type1']
        else:
            num_type1 = None

        with KnowledgeGraph() as database:
            enr_results = database.enrichment_search(identifiers, type1, type2, threshhold, maxresults, num_type1)
        
        return enr_results, 200

api.add_resource(EnrichedExpansion, '/enrichment/<type1>/<type2>/')


class SimilaritySearch(Resource):
    def get(self, type1, id1, type2, by_type):
        """
        Similarity search in the local knowledge graph
        ---
        tags: [simple]
        parameters:
          - in: path
            name: type1
            description: "type of query node"
            schema:
                type: string
            required: true
            default: "disease"
          - in: path
            name: id1
            description: "curie of query node"
            schema:
                type: string
            required: true
            default: "MONDO:0005737"
          - in: path
            name: type2
            description: "type of return nodes"
            schema:
                type: string
            required: true
            default: "disease"
          - in: path
            name: by_type
            description: "type used to evaluate similarity"
            schema:
                type: string
            required: true
            default: "phenotypic_feature"
          - in: query
            name: threshold
            description: "Number between 0 and 1 indicating the minimum similarity to return"
            schema:
                type: number
            default: 0.4
          - in: query
            name: max_results
            description: "The maximum number of results to return. Set to 0 to return all results."
            schema:
                type: integer
            default: 100
        responses:
            200:
                description: result
                content:
                    application/json:
                        schema:
                            $ref: "#/definitions/SimilarityResult"
        """
        threshold = request.args.get('threshold', default=0.4)
        
        max_results = parse_args_max_results(request.args)

        with KnowledgeGraph() as database:
            sim_results = database.similarity_search(type1, id1, type2, by_type, threshold, max_results)

        return sim_results, 200

api.add_resource(SimilaritySearch, '/similarity/<type1>/<id1>/<type2>/<by_type>/')


class CypherKnowledgeGraph(Resource):
    def post(self):
        """
        Transpile a question into a cypher query to retrieve a knowledge graph
        ---
        tags: [cypher]
        requestBody:
            description: A message with a machine-readable question graph.
            content:
                application/json:
                    schema:
                        $ref: '#/definitions/Message'
            required: true
        parameters:
          - in: query
            name: max_connectivity
            description: Max connectivity of nodes considered in the answers, Use 0 for no restriction
            schema:
                type: integer
            default: 0
        responses:
            200:
                description: A cypher query to retrieve a knowledge graph
                content:
                    application/txt:
        """

        max_connectivity = parse_args_max_connectivity(request.args)

        try:
            message_obj = Message(request.json)
            c = message_obj.cypher_query_knowledge_graph({'max_connectivity': max_connectivity})
        except:
            logger.debug(f"Unexpected error: {sys.exc_info()}")
            return "Unable to transpile question to cypher query.", 404

        return c, 200

api.add_resource(CypherKnowledgeGraph, '/cypher/knowledge_graph/')


class CypherAnswers(Resource):
    def post(self):
        """
        Transpile question into a cypher query to retrieve a list of potential answer maps
        ---
        tags: [cypher]
        requestBody:
            description: A message with a machine-readable question graph.
            content:
                application/json:
                    schema:
                        $ref: '#/definitions/Message'
            required: true
        parameters:
          - in: query
            name: max_connectivity
            description: Max connectivity of nodes considered in the answers, Use 0 for no restriction
            schema:
                type: integer
            default: 0
        responses:
            200:
                description: A cypher query to retrieve a list of potential answer maps
                content:
                    application/txt:
        """
        
        max_connectivity = parse_args_max_connectivity(request.args)

        message_obj = Message(request.json)
        c = message_obj.cypher_query_answer_map({'max_connectivity': max_connectivity})
        try:
            pass
        except:
            logger.debug(f"Unexpected error: {sys.exc_info()}")
            return "Unable to transpile question to cypher query.", 404

        return c, 200

api.add_resource(CypherAnswers, '/cypher/answers/')



class Normalize(Resource):
    def post(self):
        """
        Transform and normalize an answerset into a standardized MESSAGE format
        ---
        tags: [normalize]
        requestBody:
            description: A 0.8 answerset or 0.9 answerset or a question graph.
            content:
                application/json:
                    schema:
                        $ref: '#/definitions/Message'
            required: true
        responses:
            200:
                description: A normalized message
                schema:
                    $ref: '#/definitions/Message'
        """
        
        logger.info('Normalizing answerset')
        logger.info(request)
        provided = request.json
        if 'question_graph' not in provided:
            if 'machine_question' not in provided:
                return 'input must contain the field question_graph', 400
            else:
                question_graph = provided['machine_question']
        else:
            question_graph = provided['question_graph']

        # Detect if we have a dense or a message
        if 'result_list' in provided:
            logger.info('  Detected dense answerset, converting')
            # Assume dense
            # If dense, we need to translate
            message = Message(question_graph = question_graph)
            message.from_dense(provided)
            logger.info('  Conversion complete')

        elif 'knowledge_graph' in provided and 'answers' in provided:
            # Assume message
            logger.info('  Detected message')
            message = Message(question_graph=question_graph, knowledge_graph=provided['knowledge_graph'], answers=provided['answers'])
        else:
            return 'could not determine the format of the input answerset', 400

        # Call the builder normalize service to fix various identifiers
        response = requests.post( f'http://{os.environ["BUILDER_HOST"]}:{os.environ["BUILDER_PORT"]}/api/normalize/', json=message.dump())
        if not response.ok:
            logger.info(response.status_code)
            logger.info(response.text)
            return 'There was a problem connecting to the ROBOKOP builder', 500

        return response.json(), 200

api.add_resource(Normalize, '/normalize/')

if __name__ == '__main__':

    # Get host and port from environmental variables
    server_host = '0.0.0.0'
    server_port = int(os.environ['RANKER_PORT'])

    app.run(host=server_host,\
        port=server_port,\
        debug=False,\
        use_reloader=True)
