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

class AnswerQuestionNow(Resource):
    def post(self):
        """
        Get answers to a question
        ---
        tags: [answer]
        requestBody:
            description: The machine-readable question graph.
            content:
                application/json:
                    schema:
                        $ref: '#/definitions/Question'
            required: true
        parameters:
          - in: query
            name: max_results
            description: Maximum number of results to return. Provide -1 to indicate no maximum.
            schema:
                type: integer
            default: 250
        responses:
            200:
                description: Answer
                content:
                    application/json:
                        schema:
                            $ref: '#/definitions/Response'
        """
        max_results = request.args.get('max_results', default=250)
        logger.debug("max_results: %s", str(max_results))
        try:
            max_results = int(max_results)
        except ValueError:
            return 'max_results should be an integer', 400
        except:
            raise
        if max_results < 0:
            max_results = None

        result = answer_question.apply(
            args=[request.json],
            kwargs={'max_results': max_results}
        )
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
        requestBody:
            description: The machine-readable question graph.
            content:
                application/json:
                    schema:
                        $ref: '#/definitions/Question'
            required: true
        parameters:
          - in: query
            name: max_results
            description: Maximum number of results to return. Provide -1 to indicate no maximum.
            schema:
                type: integer
            default: 250
        responses:
            200:
                description: Answer
                content:
                    application/json:
                        schema:
                            $ref: '#/definitions/Response'
        """
        max_results = request.args.get('max_results', default=250)
        logger.debug("max_results: %s", str(max_results))
        try:
            max_results = int(max_results)
        except ValueError:
            return 'max_results should be an integer', 400
        except:
            raise
        if max_results < 0:
            max_results = None

        task = answer_question.apply_async(
            args=[request.json],
            kwargs={'max_results': max_results}
        )
        return {'task_id':task.id}, 202

api.add_resource(AnswerQuestion, '/')

class QuestionSubgraph(Resource):
    def post(self):
        """
        Get question subgraph
        ---
        tags: [util]
        requestBody:
            name: question
            description: The machine-readable question graph.
            content:
                application/json:
                    schema:
                        $ref: '#/definitions/Question'
            required: true
        responses:
            200:
                description: Knowledge subgraph
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

class Tasks(Resource):
    def get(self):
        """
        Fetch queued/active task list
        ---
        tags: [util]
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
            db=os.environ['RANKER_RESULTS_DB'])

        task_id = 'celery-task-meta-'+task_id
        task_string = r.get(task_id)
        if task_string is None:
            return 'No such task', 404
        info = json.loads(task_string)
        
        return info, 200

api.add_resource(TaskStatus, '/task/<task_id>')


class EnrichedExpansion(Resource):
    def post(self, type1, type2):
        """
        Enriched search in the local knowledge graph
        ---
        tags: [util]
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
        database = KnowledgeGraph()
        sim_results = database.enrichment_search(identifiers, type1, type2, threshhold, maxresults,num_type1)
        del database

        return sim_results, 200

api.add_resource(EnrichedExpansion, '/enrichment/<type1>/<type2>')


class SimilaritySearch(Resource):
    def get(self, type1, identifier, type2, by_type):
        """
        Similarity search in the local knowledge graph
        ---
        tags: [util]
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
        database = KnowledgeGraph()
        threshhold = request.args.get('threshhold', default = 0.4)
        maxresults = int(request.args.get('maxresulta', default = 100))
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
        debug=True,\
        use_reloader=True)
