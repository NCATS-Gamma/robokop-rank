#!/usr/bin/env python

"""Flask REST API server for ranker"""

import os
import logging
from datetime import datetime
import json

import redis
from flask_restful import Resource
from flask import request

from ranker.api.setup import app, api
from ranker.api.logging_config import logger
from ranker.question import Question, NoAnswersException
from ranker.tasks import answer_question
import ranker.api.definitions
import ranker.api.logging_config
from ranker.knowledgegraph import KnowledgeGraph

logger = logging.getLogger("ranker")

class QueryTemplate(Resource):
    def post(self):
        """
        Query
        ---
        tags: [query]
        summary: "Query RTX using a predefined question type"
        description: ""
        operationId: "query"
        consumes:
          - "application/json"
        produces:
          - "application/json"
        parameters:
          - in: "body"
            name: "body"
            description: "Query information to be submitted"
            required: true
            schema:
                $ref: "#/definitions/Query"
        responses:
            200:
                description: "successful operation"
                schema:
                    $ref: "#/definitions/Response"
            400:
                description: "Invalid status value"
        """
        logger.debug(f"Received request {request.json}.")
        if not request.json['query_type_id'] == 'Q3':
            return f"I don't know what a '{request.json['query_type_id']} is.", 200
        drug_id = request.json['terms']['chemical_substance']
        q_json = {
            "edges": [
                {
                    "end": 1,
                    "start": 0
                }
            ],
            "nodes": [
                {
                    "id": 0,
                    "identifiers": [
                        drug_id
                    ],
                    "type": "chemical_substance"
                },
                {
                    "id": 1,
                    "type": "gene"
                }
            ]
        }
        question = Question(q_json)
        answerset = answer_question(question)
        if isinstance(answerset, BaseException):
            response = {
                'datetime': datetime.now().isoformat(),
                'id': '',
                'message': f"Found 0 answers.",
                'response_code': 'OK'
            }
        else:
            response = answerset.toStandard()
            response.update(request.json)
        logger.debug(f"Prepared response {response}.")
        return response, 200

api.add_resource(QueryTemplate, '/query')

class AnswerQuestionNow(Resource):
    def post(self):
        """
        Get answers to a question
        ---
        tags: [answer]
        parameters:
          - in: body
            name: question
            description: The machine-readable question graph.
            schema:
                $ref: '#/definitions/Question'
            required: true
        responses:
            200:
                description: Answer
                schema:
                    type: object
                    required:
                      - thingsandstuff
                    properties:
                        thingsandstuff:
                            type: string
                            description: all the things and stuff
        """
        # replace `parameters` with this when OAS 3.0 is fully supported by Swagger UI
        # https://github.com/swagger-api/swagger-ui/issues/3641
        """
        requestBody:
            description: The machine-readable question graph.
            required: true
            content:
                application/json:
                    schema:
                        $ref: '#/definitions/Question'
        """
        result = answer_question.apply(args=[request.json])
        with open(os.path.join(os.environ['ROBOKOP_HOME'], 'robokop-rank', 'answers', result.get()), 'r') as f:
            answers = json.load(f)
        return answers, 202

api.add_resource(AnswerQuestionNow, '/now')

class AnswerQuestion(Resource):
    def post(self):
        """
        Get answers to a question
        ---
        tags: [answer]
        parameters:
          - in: body
            name: question
            description: The machine-readable question graph.
            schema:
                $ref: '#/definitions/Question'
            required: true
        responses:
            200:
                description: Answer
                schema:
                    type: object
                    required:
                      - thingsandstuff
                    properties:
                        thingsandstuff:
                            type: string
                            description: all the things and stuff
        """
        # replace `parameters` with this when OAS 3.0 is fully supported by Swagger UI
        # https://github.com/swagger-api/swagger-ui/issues/3641
        """
        requestBody:
            description: The machine-readable question graph.
            required: true
            content:
                application/json:
                    schema:
                        $ref: '#/definitions/Question'
        """
        task = answer_question.apply_async(args=[request.json])
        return {'task_id':task.id}, 202

api.add_resource(AnswerQuestion, '/')

class AnswerQuestionStandard(Resource):
    def post(self):
        """
        Get answers to a question in standard format
        ---
        tags: [answer]
        parameters:
          - in: body
            name: question
            description: The machine-readable question graph.
            schema:
                $ref: '#/definitions/Question'
            required: true
        responses:
            200:
                description: Answer
                schema:
                    type: object
                    required:
                      - thingsandstuff
                    properties:
                        thingsandstuff:
                            type: string
                            description: all the things and stuff
        """
        # replace `parameters` with this when OAS 3.0 is fully supported by Swagger UI
        # https://github.com/swagger-api/swagger-ui/issues/3641
        """
        requestBody:
            description: The machine-readable question graph.
            required: true
            content:
                application/json:
                    schema:
                        $ref: '#/definitions/Question'
        """
        question = Question(request.json)
        logger.debug(question.cypher_match_string())
        answer = answer_question(question)
        if isinstance(answer, BaseException):
            return "No answers", 204
        return answer.toStandard(), 200

api.add_resource(AnswerQuestionStandard, '/standard/')

class QuestionSubgraph(Resource):
    def post(self):
        """
        Get question subgraph
        ---
        tags: [util]
        parameters:
          - in: body
            name: question
            description: The machine-readable question graph.
            schema:
                $ref: '#/definitions/Question'
            required: true
        responses:
            200:
                description: Knowledge subgraph
                schema:
                    $ref: '#/definitions/Question'
        """
        # https://github.com/swagger-api/swagger-ui/issues/3641
        """
        requestBody:
            description: The machine-readable question graph.
            required: true
            content:
                application/json:
                    schema:
                        $ref: '#/definitions/Question'
        """

        question = Question(request.json)
        
        try:
            subgraph = question.relevant_subgraph()
        except NoAnswersException:
            return "Question not found in neo4j cache.", 404

        return subgraph, 200

api.add_resource(QuestionSubgraph, '/subgraph')

class IDMap(Resource):
    def post(self, concept):
        """
        Get id map
        ---
        tags: [util]
        parameters:
          - in: path
            name: concept
            description: Biolink concept
            type: string
            required: true
        responses:
            200:
                description: Knowledge subgraph
                schema:
                    type: object
        """

        database = KnowledgeGraph()
        id_map = database.get_map_for_type(concept)

        return id_map, 200

api.add_resource(IDMap, '/id_map/<concept>')

class MapID(Resource):
    def post(self, concept, identifier):
        """
        Get canonical id, IF AVAILABLE IN NEO4J CACHE
        ---
        tags: [util]
        parameters:
          - in: path
            name: concept
            description: Biolink concept
            type: string
            required: true
          - in: path
            name: identifier
            description: curie
            type: string
            required: true
        responses:
            200:
                description: Knowledge subgraph
                schema:
                    type: string
        """

        database = KnowledgeGraph()
        id_map = database.get_map_for_type(concept)

        return id_map[identifier], 200

api.add_resource(MapID, '/canonicalize/<concept>/<identifier>')

class Tasks(Resource):
    def get(self):
        """
        Fetch queued/active task list
        ---
        tags: [util]
        responses:
            200:
                description: tasks
                schema:
                    type: string
        """
        r = redis.Redis(
            host=os.environ['RESULTS_HOST'],
            port=os.environ['RESULTS_PORT'],
            db=os.environ['RANKER_RESULTS_DB'])

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
        tags: [util]
        parameters:
          - in: path
            name: task_id
            description: ID of task
            type: string
            required: true
        responses:
            200:
                description: result
                schema:
                    $ref: "#/definitions/Graph"
        """
        r = redis.Redis(
            host=os.environ['RESULTS_HOST'],
            port=os.environ['RESULTS_PORT'],
            db=os.environ['RANKER_RESULTS_DB'])

        task_id = 'celery-task-meta-'+task_id
        task_string = r.get(task_id)
        if task_string is None:
            return 'No such task', 404
        info = json.loads(task_string)
        if info['status'] != 'SUCCESS':
            return 'This task is incomplete or failed', 404

        filename = info['result']
        result_path = os.path.join(os.environ['ROBOKOP_HOME'], 'robokop-rank', 'answers', filename)
        with open(result_path, 'r') as f:
            return json.load(f)

api.add_resource(Results, '/result/<task_id>')

class TaskStatus(Resource):
    def get(self, task_id):
        """
        Get status of task
        ---
        tags: [util]
        parameters:
          - in: path
            name: task_id
            description: ID of task
            type: string
            required: true
        responses:
            200:
                description: result
                schema:
                    $ref: "#/definitions/Graph"
        """
        r = redis.Redis(
            host=os.environ['RESULTS_HOST'],
            port=os.environ['RESULTS_PORT'],
            db=os.environ['RANKER_RESULTS_DB'])

        task_id = 'celery-task-meta-'+task_id
        task_string = r.get(task_id)
        if task_string is None:
            return 'No such task', 404
        info = json.loads(task_string)
        
        return info, 200

api.add_resource(TaskStatus, '/task/<task_id>')

class SimilaritySearch(Resource):
    def get(self, type1, identifier, type2, by_type):
        """
        Similarity search in the local knowledge graph
        ---
          - in: path
            name: type1
            description: "type of query node"
            type: string
            required: true
            default: "disease"
          - in: path
            name: id1
            description: "curie of query node"
            type: string
            required: true
            default: "MONDO:0005737"
          - in: path
            name: type2
            description: "type of return nodes"
            type: string
            required: true
            default: "disease"
          - in: path
            name: by_type
            description: "type used to evaluate similarity"
            type: string
            required: true
            default: "phenotypic_feature"
          - in: query
            name: threshhold
            description: "Number between 0 and 1 indicating the minimum similarity to return"
            type: float
            default: 0.4
          - in: query
            name: maxresults
            description: "The maximum number of results to return. Set to 0 to return all results."
            type: integer
            default: 100
        responses:
            200:
                description: result
                schema:
                    $ref: "#/definitions/SimilarityResult"
        """
        database = KnowledgeGraph()
        threshhold = request.args.get('threshhold', default = 0.4)
        maxresults = request.args.get('maxresults', default = 100)
        sim_results = database.similarity_search(type1, identifier, type2, by_type, threshhold, maxresults)
        del database

        return sim_results, 200

api.add_resource(SimilaritySearch, '/similarity/<type1>/<identifier>/<type2>/<by_type>')


if __name__ == '__main__':

    # Get host and port from environmental variables
    server_host = '0.0.0.0' #os.environ['ROBOKOP_HOST']
    server_port = int(os.environ['RANKER_PORT'])

    app.run(host=server_host,\
        port=server_port,\
        debug=False,\
        use_reloader=True)
