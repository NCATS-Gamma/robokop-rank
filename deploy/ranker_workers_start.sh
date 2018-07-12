#!/bin/bash

cd $ROBOKOP_HOME/robokop-rank

celery -A ranker.tasks.celery worker --loglevel=info -c $NUM_RANKERS -n ranker@robokop -Q ranker
