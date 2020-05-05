"""Logging configuration."""

import logging.config
import os


def setup_main_logger():
    """Set up default logger."""

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
                    # 'smtp'
                ]
            }
        },
        'root': {},
        'incremental': False,
        'disable_existing_loggers': True
    }
    logging.config.dictConfig(dict_config)
setup_main_logger()

def clear_log_handlers(logger):
    """ Clears any handlers from the logger."""
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    logger.handlers = []
def add_task_id_based_handler(logger, task_id):
    """Adds a file handler with task_id as file name to the logger."""
    formatter = logging.Formatter("[%(asctime)s: %(levelname)s/%(name)s(%(processName)s)]: %(message)s")
    # create file handler and set level to debug
    file_handler = logging.handlers.RotatingFileHandler(f"{os.environ['ROBOKOP_HOME']}/logs/ranker_task_logs/{task_id}.log",
        mode="a",
        encoding="utf-8",
        maxBytes=1e6,
        backupCount=9)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)