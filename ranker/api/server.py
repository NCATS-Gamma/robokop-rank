#!/usr/bin/env python

"""Flask REST API server for ranker."""

import os
import logging
import json

import redis
from flask_restful import Resource
from flask import request, send_from_directory

from ranker.api.setup import app, api
from ranker.message import Message, output_formats
from ranker.tasks import answer_question
# import ranker.api.definitions
import ranker.definitions
from ranker.knowledgegraph import KnowledgeGraph
from ranker.support.omnicorp import OmnicorpSupport
from ranker.cache import Cache

# get supporter_cache - connect to redis
support_cache = Cache(
    redis_host=os.environ['CACHE_HOST'],
    redis_port=os.environ['CACHE_PORT'],
    redis_db=os.environ['CACHE_DB'],
    redis_password=os.environ['CACHE_PASSWORD'])
cached_prefixes = support_cache.get('OmnicorpPrefixes')

logger = logging.getLogger("ranker")

class AnswerQuestionNow(Resource):
    def post(self):
        """
        Get answers to a question
        ---
        tags: [answer]
        requestBody:
            description: A message with a machine-readable question graph.
            content:
                application/json:
                    schema:
                        $ref: '#/definitions/Message'
            required: true
        parameters:
          - in: query
            name: max_results
            description: Maximum number of results to return. Provide -1 to indicate no maximum.
            schema:
                type: integer
            default: 250
          - in: query
            name: output_format
            description: Requested output format. APIStandard or Message
            schema:
                type: string
            default: APIStandard
        responses:
            200:
                description: Answer
                content:
                    application/json:
                        schema:
                            $ref: '#/definitions/Response'
        """
        max_results = request.args.get('max_results', default=250)
        # logger.debug("max_results: %s", str(max_results))
        try:
            max_results = int(max_results)
        except ValueError:
            return 'max_results should be an integer', 400
        if max_results < 0:
            max_results = None

        output_format = request.args.get('output_format', default=output_formats[0])
        if output_format not in output_formats:
            return f'output_format must be one of [{" ".join(output_formats)}]', 400

        try:
            result = answer_question.apply(
                args=[request.json],
                kwargs={'max_results': max_results, 'output_format': output_format}
            )
            result = result.get()
        except:
            # Celery tasks log errors internally. Just return.
            return "Internal server error. See the logs for details.", 500
        if result is None:
            return None, 200

        logger.debug(f'Fetching answerset file: {result}')
        filename = os.path.join(os.environ['ROBOKOP_HOME'], 'robokop-rank', 'answers', result)
        with open(filename, 'r') as f:
            output = json.load(f)
        os.remove(filename)
        return output, 200

api.add_resource(AnswerQuestionNow, '/now')

class AnswerQuestion(Resource):
    def post(self):
        """
        Get answers to a question
        ---
        tags: [answer]
        requestBody:
            description: A message with a machine-readable question graph.
            content:
                application/json:
                    schema:
                        $ref: '#/definitions/Message'
            required: true
        parameters:
          - in: query
            name: max_results
            description: Maximum number of results to return. Provide -1 to indicate no maximum.
            schema:
                type: integer
            default: 250
          - in: query
            name: output_format
            description: Requested output format. APIStandard, Message, Answers
            schema:
                type: string
            default: APIStandard
        responses:
            200:
                description: Successfull queued a task
                content:
                    application/json:
        """

        max_results = request.args.get('max_results', default=250)
        try:
            max_results = int(max_results)
        except ValueError:
            return 'max_results should be an integer', 400
        except:
            raise
        if max_results < 0:
            max_results = None

        output_format = request.args.get('output_format', default=output_formats[0])
        if output_format not in output_formats:
            return f'output_format must be one of [{" ".join(output_formats)}]', 400

        task = answer_question.apply_async(
            args=[request.json],
            kwargs={'max_results': max_results, 'output_format': output_format}
        )
        return {'task_id':task.id}, 202

api.add_resource(AnswerQuestion, '/')


class QuestionSubgraph(Resource):
    def post(self):
        """
        Get knowledge graph for message.
        ---
        tags: [knowledgeGraph]
        requestBody:
            description: A message with a machine-readable question graph.
            content:
                application/json:
                    schema:
                        $ref: '#/definitions/Message'
            required: true
        responses:
            200:
                description: Answer
                content:
                    application/json:
                        schema:
                            $ref: '#/definitions/Response'
        """
        message = request.json
        if not message.get('answers', []):
            message_obj = Message(request.json)

            try:
                kg = message_obj.knowledge_graph
            except:
                return "Unable to retrieve knowledge graph.", 404

            return kg, 200

        def flatten_semilist(x):
            lists = [n if isinstance(n, list) else [n] for n in x]
            return [e for el in lists for e in el]

        # get nodes and edge ids from message answers
        node_ids = [knode_id for answer in message['answers'] for knode_id in answer['node_bindings'].values()]
        edge_ids = [kedge_id for answer in message['answers'] for kedge_id in answer['edge_bindings'].values()]
        node_ids = flatten_semilist(node_ids)
        edge_ids = flatten_semilist(edge_ids)

        nodes = get_node_properties(node_ids)
        edges = get_edge_properties(edge_ids)

        kg = {
            'nodes': nodes,
            'edges': edges
        }
        return kg, 200

api.add_resource(QuestionSubgraph, '/knowledge_graph')


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

api.add_resource(MultiNodeLookup, '/multinode_lookup')


def get_node_properties(node_ids, fields=None):
    functions = {
        'labels': 'labels(n)',
    }

    if fields is not None:
        prop_string = ', '.join([f'{key}:{functions[key]}' if key in functions else f'{key}:n.{key}' for key in fields])
    else:
        prop_string = ', '.join([f'{key}:{functions[key]}' for key in functions] + ['.*'])


    where_string = ' OR '.join([f'n.id="{node_id}"' for node_id in node_ids])
    query_string = f'MATCH (n) WHERE {where_string} RETURN n{{{prop_string}}}'

    with KnowledgeGraph() as database:
        with database.driver.session() as session:
            result = session.run(query_string)

    output = []
    for record in result:
        r = record['n']
        if 'labels' in r and 'named_thing' in r['labels']:
            r['labels'].remove('named_thing')
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
                              - "636"
                              - "634"
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

api.add_resource(MultiEdgeLookup, '/multiedge_lookup')


def get_edge_properties(edge_ids, fields=None):
    functions = {
        'source_id': 'startNode(e).id',
        'target_id': 'endNode(e).id',
        'type': 'type(e)',
        'id': 'toString(id(e))'
    }

    if fields is not None:
        prop_string = ', '.join([f'{key}:{functions[key]}' if key in functions else f'{key}:e.{key}' for key in fields])
    else:
        prop_string = ', '.join([f'{key}:{functions[key]}' for key in functions] + ['.*'])

    where_string = ' OR '.join([f'id(e)={edge_id}' for edge_id in edge_ids])
    query_string = f'MATCH ()-[e]->() WHERE {where_string} RETURN e{{{prop_string}}}'
    # logger.debug(query_string)

    with KnowledgeGraph() as database:
        with database.driver.session() as session:
            result = session.run(query_string)

    return [record['e'] for record in result]


class Tasks(Resource):
    def get(self):
        """
        Fetch queued/active task list
        ---
        tags: [tasks]
        responses:
            200:
                description: tasks
                content:
                    application/json:
                        schema:
                            type: string
        """
        r = redis.Redis(
            host=os.environ['RESULTS_HOST'],
            port=os.environ['RESULTS_PORT'],
            db=os.environ['RANKER_RESULTS_DB'],
            password=os.environ['RESULT_PASSWORD'])

        tasks = []
        for name in r.scan_iter('*'):
            name = name.decode() # convert bytes to str
            tasks.append(json.loads(r.get(name)))

        return tasks

api.add_resource(Tasks, '/tasks/')

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
        
        # These are defined above
        # support_cache
        # cached_prefixes

        with OmnicorpSupport() as supporter:
            # Ids would be cached in sorted order
            ids = [id1,id2]
            ids.sort()
            key = f"{supporter.__class__.__name__}({ids[0]},{ids[1]})"
            support_edge = support_cache.get(key)
            if support_edge is None:
                # Not found in cache
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
                        support_edge = supporter.term_to_term(ids[0], ids[1])
                        # logger.info(f'Support {support_edge}')
                        support_cache.set(key, support_edge)
                    except Exception as e:
                        # raise e
                        logger.debug('Support error, not caching')
            # else: # Found in cache
            
            # support_edge now is either None or the object
            if support_edge is None:
                publications = []
            else:
                publications = support_edge

        return publications, 200

api.add_resource(Omnicorp, '/omnicorp/<id1>/<id2>')


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
        
        return info, 200

api.add_resource(TaskStatus, '/task/<task_id>')

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

        task_log_file = os.path.join(os.environ['ROBOKOP_HOME'], 'task_logs', f'{task_id}.log')
        if os.path.isfile(task_log_file):
            with open(task_log_file, 'r') as log_file:
                log_contents = log_file.read()
            return log_contents, 200
        else:
            return 'Task ID not found', 404


        # task_log_dir = os.path.join(os.environ['ROBOKOP_HOME'], 'task_logs')
        # task_log_file = f'{task_id}.log'

        # if os.path.isfile(os.path.join(task_log_dir, task_log_file)):
        #     return send_from_directory(task_log_dir, task_log_file,mimetype='text/plain',as_attachment=False)
        # else:
        #     return '', 404

api.add_resource(TaskLog, '/task/<task_id>/log')

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
            default: "disease"
          - in: query
            name: identifiers
            description: "identifiers of query nodes"
            schema:
                type: list
            required: true
          - in: query
            name: threshhold
            description: "Number between 0 and 1 indicating the maximum p-value to return"
            schema:
                type: float
            default: 0.05
          - in: query
            name: maxresults
            description: "The maximum number of results to return. Set to 0 to return all results."
            schema:
                type: integer
            default: 100
          - in: query
            name: num_type1
            description: "The total number of type1 entities that can exist.  If not specified, this is estimated from the cache"
            schema:
                type: integer
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
        if 'maxresults' in parameters:
            maxresults = parameters['maxresults']
        else:
            maxresults = 100
        if 'num_type1' in parameters:
            num_type1 = parameters['num_type1']
        else:
            num_type1 = None
        with KnowledgeGraph() as database:
            sim_results = database.enrichment_search(identifiers, type1, type2, threshhold, maxresults,num_type1)

        return sim_results, 200

api.add_resource(EnrichedExpansion, '/enrichment/<type1>/<type2>')


class SimilaritySearch(Resource):
    def get(self, type1, identifier, type2, by_type):
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
            name: threshhold
            description: "Number between 0 and 1 indicating the minimum similarity to return"
            schema:
                type: float
            default: 0.4
          - in: query
            name: maxresults
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
        threshhold = request.args.get('threshhold', default = 0.4)
        maxresults = int(request.args.get('maxresults', default = 100))
        with KnowledgeGraph() as database:
            sim_results = database.similarity_search(type1, identifier, type2, by_type, threshhold, maxresults)

        return sim_results, 200

api.add_resource(SimilaritySearch, '/similarity/<type1>/<identifier>/<type2>/<by_type>')


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
        responses:
            200:
                description: A cypher query to retrieve a knowledge graph
                content:
                    application/txt:
        """
        try:
            message_obj = Message(request.json)
            c = message_obj.cypher_query_knowledge_graph()
        except:
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
        responses:
            200:
                description: A cypher query to retrieve a list of potential answer maps
                content:
                    application/txt:
        """
        
        try:
            message_obj = Message(request.json)
            c = message_obj.cypher_query_answer_map()
        except:
            return "Unable to transpile question to cypher query.", 404

        return c, 200

api.add_resource(CypherAnswers, '/cypher/answers/')


if __name__ == '__main__':

    # Get host and port from environmental variables
    server_host = '0.0.0.0' #os.environ['ROBOKOP_HOST']
    server_port = int(os.environ['RANKER_PORT'])

    app.run(host=server_host,\
        port=server_port,\
        debug=False,\
        use_reloader=True)
