"""Tasks for Celery workers."""

import os
import logging
import json
import uuid

from celery import Celery, signals
from kombu import Queue
from ranker.message import Message, output_formats
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
    logger.info(f'Starting task specific log for task {task_id}')
    clear_log_handlers(logger)
    add_task_id_based_handler(logger, task_id)
    logger.info(f'This is a task specific log for task {task_id}')

@signals.task_postrun.connect()
def tear_down_task_logging(**kwargs):
    """
    Reverts back logging to main configuration once task is finished.
    """
    logger = logging.getLogger('ranker')
    logger.info('Task is complete. Ending task specific log.')
    clear_log_handlers(logger)
    # change logging config back to the way it was
    setup_main_logger()
    #finally log task has finished to main file
    logger = logging.getLogger('ranker')
    logger.info(f"Task {kwargs.get('task_id')} is complete")


@celery.task(bind=True, queue='ranker', task_acks_late=True, track_started=True, worker_prefetch_multiplier=1)
def answer_question(self, message_json, max_results=250, output_format=output_formats[1], max_connectivity=0):
    """Generate a message from the input json for a question."""

    self.update_state(state='ANSWERING')
    logger.info("Answering Question")
    logger.info(message_json)

    try:
        message = Message(message_json)
        
        try:
            message.fill(max_connectivity=max_connectivity)
            message.rank(max_results)
        except Exception as err:
            logger.exception(f"Something went wrong with question answering: {err}")
            raise err
        
        if message.answers is None:
            logger.info("0 answers found. Returning None.")
            return None

        logger.info(f'{len(message.answers)} answers found.')

        self.update_state(state='SAVING')

        filename = f"{uuid.uuid4()}.json"
        answers_dir = os.path.join(os.environ['ROBOKOP_HOME'], 'robokop-rank', 'answers')
        if not os.path.exists(answers_dir):
            os.makedirs(answers_dir)
        result_path = os.path.join(answers_dir, filename)

        if output_format.upper() == output_formats[0]:
            message_dump = message.dump_dense()
        elif output_format.upper() == output_formats[1]:
            message_dump = message.dump()
        elif output_format.upper() == output_formats[2]:
            message_dump = message.dump_csv()
        elif output_format.upper() == output_formats[3]:
            message_dump = message.dump_answers()
        else:
            logger.info('Problem encountered during exporting.')
            raise Exception('Problem encountered during exporting.') 

        try:
            with open(result_path, 'w') as f:
                json.dump(message_dump, f)
        except Exception as err:
            logger.exception(err)
            raise err

        logger.info("Answers saved.")
    
        return filename

    except Exception as err:
        logger.exception(err)
        raise err
