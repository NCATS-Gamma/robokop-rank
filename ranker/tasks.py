"""Tasks for Celery workers."""

import os
import logging
import json
import uuid
import redis
import requests
from celery import Celery, signals
from kombu import Queue
from ranker.message import Message, output_formats
from ranker.api.logging_config import setup_main_logger, add_task_id_based_handler, clear_log_handlers
from ranker.core import run, dense, to_robokop, strip_kg, csv

# set up Celery
celery = Celery('ranker.api.setup')
celery.conf.update(
    broker_url=os.environ["CELERY_BROKER_URL"],
    result_backend=os.environ["CELERY_RESULT_BACKEND"],
)
celery.conf.task_queues = (
    Queue('ranker', routing_key='ranker'),
)

redis_client = redis.Redis(
    host=os.environ['RESULTS_HOST'],
    port=os.environ['RESULTS_PORT'],
    db=os.environ['RANKER_RESULTS_DB'],
    password=os.environ['RESULTS_PASSWORD'])

logger = logging.getLogger('ranker')

@signals.after_task_publish.connect()
def initialize_queued_task_results(**kwargs):
    # headers=None, body=None, exchange=None, routing_key=None
    task_id = kwargs['headers']['id']
    logger.info(f'Queuing task: {task_id}')

    redis_key = 'celery-task-meta-'+task_id
    initial_status = {"status": "QUEUED",
        "result": None,
        "traceback": None,
        "children": [],
        "task_id": task_id
    }
    redis_client.set(redis_key, json.dumps(initial_status))

    # initial_status_again = redis_client.get(redis_key)
    # logger.info(f'Got initial status {initial_status_again}')


@signals.task_prerun.connect()
def setup_logging(signal=None, sender=None, task_id=None, task=None, *args, **kwargs):
    """
    Changes the main logger's handlers so they could log to a task specific log file.    
    """
    logger = logging.getLogger('ranker')
    logger.info(f'Starting to work on task {task_id}. Starting task specific log.')
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
def answer_question(self, request_json, max_results=-1, output_format=output_formats[1], max_connectivity=-1, use_novelty=False):
    """Generate a message from the input json for a question."""

    self.update_state(state='ANSWERING')
    logger.info("Answering Question")

    message_json = {
        'knowledge_graph': {
            'url': f'bolt://{os.environ["NEO4J_HOST"]}:{os.environ["NEO4J_BOLT_PORT"]}',
            'credentials': {
                'username': 'neo4j',
                'password': os.environ["NEO4J_PASSWORD"],
            },
        },
        'query_graph': request_json['question_graph'],
        'results': [],
    }
    logger.info(message_json)

    try:
        message_json = run(
            message_json,
            max_results=max_results,
            max_connectivity=max_connectivity,
            use_novelty=use_novelty,
        )
    except Exception as err:
        logger.exception(err)
        raise err

    logger.info('%d answers found.', len(message_json["results"]))

    # convert output format
    if output_format.upper() == 'DENSE':
        output = dense(message_json)
    elif output_format.upper() == 'ANSWERS':
        output = to_robokop(strip_kg(message_json))
    elif output_format.upper() == 'MESSAGE':
        output = to_robokop(message_json)
    elif output_format.upper() == 'CSV':
        output = csv(message_json)
    else:
        raise ValueError(f'Unrecognized output format "{output_format}"')

    # save output
    self.update_state(state='SAVING')
    filename = f"{uuid.uuid4()}.json"
    answers_dir = os.path.join(os.environ['ROBOKOP_HOME'], 'robokop-rank', 'answers')
    if not os.path.exists(answers_dir):
        os.makedirs(answers_dir)
    result_path = os.path.join(answers_dir, filename)
    try:
        with open(result_path, 'w') as f:
            json.dump(output, f)
    except Exception as err:
        logger.exception(err)
        raise err

    logger.info("Answers saved.")

    return filename
