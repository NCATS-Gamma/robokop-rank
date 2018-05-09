'''
Tasks for Celery workers
'''

import os
import sys
import logging
from celery import Celery, signals
from kombu import Queue

from ranker.api.setup import app

# set up Celery
app.config['broker_url'] = f'redis://{os.environ["REDIS_HOST"]}:{os.environ["REDIS_PORT"]}/{os.environ["MANAGER_REDIS_DB"]}'
app.config['result_backend'] = f'redis://{os.environ["REDIS_HOST"]}:{os.environ["REDIS_PORT"]}/{os.environ["MANAGER_REDIS_DB"]}'
celery = Celery(app.name, broker=app.config['broker_url'])
celery.conf.update(app.config)
celery.conf.task_queues = (
    Queue('answer', routing_key='answer'),
)
# Tell celery not to mess with logging at all
@signals.setup_logging.connect
def setup_celery_logging(**kwargs):
    pass
celery.log.setup()

logger = logging.getLogger(__name__)

@celery.task(bind=True, queue='answer')
def answer_question(self, question):
    '''
    Generate answerset for a question
    '''

    self.update_state(state='ANSWERING')
    logger.info("Answering your question...")

    try:
        answerset = question.answer()
    except Exception as err:
        logger.exception(f"Something went wrong with question answering: {err}")
        raise err
    if answerset.answers:
        self.update_state(state='ANSWERS FOUND')
        logger.info("Answers found.")
    else:
        logger.exception("Question answering completed: no answers found.")
        raise ValueError("Question answering completed: no answers found.")

    logger.info("Done answering.")
    return answerset
