"""Tasks for Celery workers."""

import os
import logging
import json
import uuid

from celery import Celery, signals
from kombu import Queue
from ranker.question import Question, NoAnswersException
from ranker.api.logging_config import setup_main_logger, add_task_id_based_handler, clear_log_handlers

# set up Celery
celery = Celery('ranker.api.setup')
celery.conf.update(
    broker_url=os.environ["CELERY_BROKER_URL"],
    result_backend=os.environ["CELERY_RESULT_BACKEND"],
)
celery.conf.task_queues = (
    Queue('ranker', routing_key='ranker'),
)

logger = logging.getLogger('ranker')

@signals.task_prerun.connect()
def setup_logging(signal=None, sender=None, task_id=None, task=None, *args, **kwargs):
    """
    Changes the main logger's handlers so they could log to a task specific log file.    
    """
    logger = logging.getLogger('ranker')
    clear_log_handlers(logger)
    add_task_id_based_handler(logger, task_id)

@signals.task_postrun.connect()
def tear_down_task_logging(**kwargs):
    """
    Reverts back logging to main configuration once task is finished.
    """
    logger = logging.getLogger('ranker')
    clear_log_handlers(logger)
    # change logging config back to the way it was
    set_up_main_logger()
    #finally log task has finished to main file
    logger = logging.getLogger('ranker')
    logger.info(f"task {kwargs.get('task_id')} finished ...")


@celery.task(bind=True, queue='ranker', task_acks_late=True, track_started=True, worker_prefetch_multiplier=1)
def answer_question(self, question_json, max_results=250):
    """Generate answerset for a question."""
    question = Question(question_json)
    self.update_state(state='ANSWERING')
    logger.info("Answering your question...")

    try:
        answerset = question.answer(max_results=max_results)
    except Exception as err:
        logger.exception(f"Something went wrong with question answering: {err}")
        raise err
    if answerset is None:
        logger.info("0 answers found. Returning None.")
        return None
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
