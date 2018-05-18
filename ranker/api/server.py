#!/usr/bin/env python

"""Flask REST API server for ranker"""

import os
import logging
from flask_restful import Resource
from flask import request
from ranker.api.setup import app, api
from ranker.api.logging_config import logger
from ranker.question import Question
from ranker.tasks import answer_question
import ranker.api.definitions
import ranker.api.logging_config

logger = logging.getLogger("ranker")

class AnswerQuestion(Resource):
    def post(self):
        """
        Get answers to a question
        ---
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
        answer = answer_question.apply(args=[question]).result
        if isinstance(answer, BaseException):
            return "No answers", 204
        return answer.toJSON(), 200

api.add_resource(AnswerQuestion, '/')

class AnswerQuestionStandard(Resource):
    def post(self):
        """
        Get answers to a question in standard format
        ---
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
        answer = answer_question.apply(args=[question]).result
        if isinstance(answer, BaseException):
            return "No answers", 204
        return answer.toStandard(), 200

api.add_resource(AnswerQuestionStandard, '/standard/')

class QuestionSubgraph(Resource):
    def post(self):
        """
        Get question subgraph
        ---
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
            
        subgraph = question.relevant_subgraph()

        return subgraph, 200

api.add_resource(QuestionSubgraph, '/subgraph')

if __name__ == '__main__':

    # Get host and port from environmental variables
    server_host = '0.0.0.0' #os.environ['ROBOKOP_HOST']
    server_port = int(os.environ['RANKER_PORT'])

    app.run(host=server_host,\
        port=server_port,\
        debug=True,\
        use_reloader=True)
