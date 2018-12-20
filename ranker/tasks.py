"""Tasks for Celery workers."""

import os
import logging
import json
import uuid

from celery import Celery, signals
from kombu import Queue
from ranker.question import Question
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
    Queue('ranker-ti', routing_key='ranker-ti')
)

logger = logging.getLogger('ranker')

@signals.task_prerun.connect()
def setup_logging(signal=None, sender=None, task_id=None, task=None, *args, **kwargs):
    """
    Changes the main logger's handlers so they could log to a task specific log file.    
    """
    logger = logging.getLogger('ranker')
    logger.info(f"Task {kwargs.get('task_id')} is starting.")
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
    setup_main_logger()
    #finally log task has finished to main file
    logger = logging.getLogger('ranker')
    logger.info(f"Task {kwargs.get('task_id')} completed.")


@celery.task(bind=True, queue='ranker', task_acks_late=True, track_started=True, worker_prefetch_multiplier=1)
def answer_question(self, message_json, max_results=250, output_format=output_formats[0]):
    """Generate a message from the input json for a question."""

    message = Message(message_json)

    self.update_state(state='ANSWERING')
    logger.info("Answering your question...")
    try:
        message.rank_answers(max_results=250)
    except Exception as err:
        logger.exception(f"Something went wrong with question answering: {err}")
        raise err
    
    if message.answer_maps is None:
        logger.info("0 answers found. Returning None.")
        return None

    logger.info("%d answers found.", len(message.answer_maps))

    self.update_state(state='SAVING')

    filename = f"{uuid.uuid4()}.json"
    answers_dir = os.path.join(os.environ['ROBOKOP_HOME'], 'robokop-rank', 'answers')
    if not os.path.exists(answers_dir):
        os.makedirs(answers_dir)
    result_path = os.path.join(answers_dir, filename)

    if output_format == output_formats[0]:
        message_dump = message.dump_standard()
    elif output_format == output_formats[1]:
        message_dump = message.dump()
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
    self.update_state(state='COMPLETE')

    return filename

@celery.task(bind=True, queue='ranker-ti', task_acks_late=True, track_started=True, worker_prefetch_multiplier=1)
def answer_question_ti(self, message_json):
    """Generate answerset for a question."""

    self.update_state(state='ANSWERING')
    logger.info("Completing message...")
   
    message = Message(message_json)
    if not message.answers:
        message.answers.append({
            "node_bindings": {},
            "edge_bindings": {}
        })

    logger.info(f"Answering {len(message.answers)} questions.")
    big_answerset = None
    for i, kmap in enumerate(message.answers):
        logger.info(f"Answering question {i}...")
        question_json = message.question_graph.apply(kmap)
        question = Question(question_json)

        answerset = question.fetch_answers()
        if answerset is None:
            continue
        logger.debug(answerset)
        answerset = Message(answerset)
        logger.info("%d answers found.", len(answerset.answers))
        answerset.answers = [{
            "node_bindings": {**kmap['node_bindings'], **km['node_bindings']},
            "edge_bindings": {**kmap['edge_bindings'], **km['edge_bindings']}
        } for km in answerset.answers]
        if big_answerset is None:
            big_answerset = answerset
            big_answerset.knowledge_graph.merge(message.knowledge_graph)
        else:
            big_answerset.knowledge_graph.merge(answerset.knowledge_graph)
            big_answerset.answers = big_answerset.answers + answerset.answers

    self.update_state(state='SAVING')

    if big_answerset is None:
        logger.info("0 answers found. Returning None.")
        return None

    big_answerset.question_graph = message.question_graph
    logger.info("All questions answered.")

    self.update_state(state='SAVING')

    filename = f"{uuid.uuid4()}.json"
    answers_dir = os.path.join(os.environ['ROBOKOP_HOME'], 'robokop-rank', 'answers')
    if not os.path.exists(answers_dir):
        os.makedirs(answers_dir)
    result_path = os.path.join(answers_dir, filename)
    
    try:
        with open(result_path, 'w') as f:
            json.dump(big_answerset.dump(), f)
    except Exception as err:
        logger.exception(err)
        raise err

    logger.info("Completed Message saved.")

    return filename
