#!/usr/bin/env python

"""Flask REST API server for ranker"""

import os
from flask_restplus import Resource
from flask import request
from setup import app, api
from question import Question
from ranker_tasks import answer_question

@api.route('/')
@api.doc(params={'question': 'A question specification'})
class AnswerQuestion(Resource):
    @api.response(200, 'Success')
    @api.response(204, 'No answers')
    def post(self):
        """Get answer for question"""
        question = Question(request.json)
        answer = answer_question.apply(args=[question]).result
        if isinstance(answer, BaseException):
            return "No answers", 204
        return answer.toJSON(), 200

if __name__ == '__main__':

    # Get host and port from environmental variables
    server_host = os.environ['ROBOKOP_HOST']
    server_port = int(os.environ['RANKER_PORT'])

    app.run(host=server_host,\
        port=server_port,\
        debug=True,\
        use_reloader=True)
