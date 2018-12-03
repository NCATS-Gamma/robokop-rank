"""Logging configuration."""

import logging.config
import os
from celery._state import get_current_task


def setup_logger():
    """Set up logger."""

    dict_config = {
        'version': 1,
        'formatters': {
            'default': {
                'format': "[%(asctime)s: %(levelname)s/%(name)s(%(processName)s)]: %(message)s"
            }
        },
        'filters': {},
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'default'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'default',
                'filename': f"{os.environ['ROBOKOP_HOME']}/logs/ranker.log",
                'mode': 'a',
                'encoding': 'utf-8',
                'maxBytes': 1e6,
                'backupCount': 9
            },
            'smtp': {
                'class': 'logging.handlers.SMTPHandler',
                'level': 'ERROR',
                'formatter': 'default',
                'mailhost': (os.environ["ROBOKOP_MAIL_SERVER"], 587),
                'fromaddr': os.environ["ROBOKOP_DEFAULT_MAIL_SENDER"],
                'toaddrs': os.environ['ADMIN_EMAIL'],
                'subject': 'ROBOKOP ranker error log',
                'credentials': [os.environ["ROBOKOP_MAIL_USERNAME"], os.environ["ROBOKOP_MAIL_PASSWORD"]]
            }
        },
        'loggers': {
            'ranker': {
                'level': 'DEBUG',
                'handlers': [
                    'console',
                    'file',
                    'smtp'
                ]
            },
            'ranker.task': {
                'level': 'DEBUG',
                'handlers': [
                    'smtp'
                ],
                'propagate': False
            }
        },
        'root': {},
        'incremental': False,
        'disable_existing_loggers': True
    }
    logging.config.dictConfig(dict_config)

def get_task_logger(module_name=None):
    """Sets up and returns a new task-id based logger if a task exist else a normal logger."""
    setup_logger()
    
    current_task = get_current_task()
    if current_task == None :
        return logging.getLogger(module_name or __name__)
    task_id = current_task.request.id
    # create a handler with <this-module>.<task-id>
    logger = logging.getLogger('ranker.task' + f'.{task_id}')
    # prevent log getting to higher loggers
    # logger.propagate = False
    file_path = os.path.join(os.environ.get('ROBOKOP_HOME'),'robokop-rank', 'task_logs', f'{task_id}.log')
    fileHandler = logging.handlers.RotatingFileHandler(filename=file_path)
    formatter = logging.Formatter("[%(asctime)s: %(levelname)s/%(name)s(%(processName)s)]: %(message)s")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    return logger