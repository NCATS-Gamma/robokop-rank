'''
Tasks for Celery workers
'''

import os
import sys
import logging

from ranker.api.setup import app
from ranker.question import NoAnswersException

logger = logging.getLogger(__name__)

def answer_question(question):
    '''
    Generate answerset for a question
    '''

    logger.info("Answering your question...")

    try:
        answerset = question.answer()
    except NoAnswersException as err:
        logger.debug(err)
        raise err
    except Exception as err:
        logger.exception(f"Something went wrong with question answering: {err}")
        raise err
    if answerset.answers:
        logger.info("Answers found.")
    else:
        raise NoAnswersException("Question answering complete, found 0 answers.")

    logger.info("Done answering.")
    return answerset
