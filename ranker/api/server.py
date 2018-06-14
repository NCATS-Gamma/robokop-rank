#!/usr/bin/env python

"""Flask REST API server for ranker"""

import os
import logging
from datetime import datetime
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
        question = Question(request.json)
        answer = answer_question(question)
        if isinstance(answer, BaseException):
            return "No answers", 204
        return answer.toJSON(), 200

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
        del database

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
        del database

        return id_map[identifier], 200

api.add_resource(MapID, '/canonicalize/<concept>/<identifier>')

if __name__ == '__main__':

    # Get host and port from environmental variables
    server_host = '0.0.0.0' #os.environ['ROBOKOP_HOST']
    server_port = int(os.environ['RANKER_PORT'])

    app.run(host=server_host,\
        port=server_port,\
        debug=True,\
        use_reloader=True)
