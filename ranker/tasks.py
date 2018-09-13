'''
Tasks for Celery workers
'''

import os
import sys
import logging
import json
import uuid

from celery import Celery, signals
from kombu import Queue

from ranker.api.setup import app
from ranker.question import Question, NoAnswersException
import ranker.api.logging_config

logger = logging.getLogger(__name__)

# set up Celery
celery = Celery(app.name)
celery.conf.update(
    broker_url=os.environ["CELERY_BROKER_URL"],
    result_backend=os.environ["CELERY_RESULT_BACKEND"],
)
celery.conf.task_queues = (
    Queue('ranker', routing_key='ranker'),
)
# Tell celery not to mess with logging at all
@signals.setup_logging.connect
def setup_celery_logging(**kwargs):
    pass
celery.log.setup()

@celery.task(bind=True, queue='ranker')
def answer_question(self, question_json, max_results=250):
    '''
    Generate answerset for a question
    '''

    question = Question(question_json)

    self.update_state(state='ANSWERING')
    logger.info("Answering your question...")

    try:
        answerset = question.answer(max_results=max_results)
    except NoAnswersException as err:
        logger.debug(err)
        raise err
    except Exception as err:
        logger.exception(f"Something went wrong with question answering: {err}")
        raise err
    logger.info("%d answers found.", len(answerset.answers))

    self.update_state(state='SAVING')

    filename = f"{uuid.uuid4()}.json"
    answers_dir = os.path.join(os.environ['ROBOKOP_HOME'], 'robokop-rank', 'answers')
    if not os.path.exists(answers_dir):
        os.makedirs(answers_dir)
    result_path = os.path.join(answers_dir, filename)
    try:
        with open(result_path, 'w') as f:
            json.dump(answerset.toJSON(), f)
    except Exception as err:
        logger.exception(err)
        raise err

    logger.info("Answerset saved.")
    
    return filename
