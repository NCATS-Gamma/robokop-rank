#!/bin/bash

cd $ROBOKOP_HOME/robokop-rank

celery -A ranker.tasks.celery worker --loglevel=info -c $RANKER_NUM_WORKERS -n ranker@robokop -Q ranker -Ofair
