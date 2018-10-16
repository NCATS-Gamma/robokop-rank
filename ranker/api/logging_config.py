"""Logging configuration."""

import logging.config
import os


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
            }
        },
        'root': {},
        'incremental': False,
        'disable_existing_loggers': True
    }
    logging.config.dictConfig(dict_config)
