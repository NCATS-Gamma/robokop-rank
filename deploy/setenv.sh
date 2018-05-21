DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export ROBOKOP_HOME="$DIR/../.."
if [ "$DEPLOY" != "docker" ]; then
    export $(cat $ROBOKOP_HOME/shared/robokop.env | grep -v ^# | xargs)
fi

export CELERY_BROKER_URL="redis://$BROKER_HOST:$BROKER_PORT/$RANKER_REDIS_DB"
export CELERY_RESULT_BACKEND="redis://$BROKER_HOST:$BROKER_PORT/$RANKER_REDIS_DB"
export FLOWER_BROKER_API="redis://$BROKER_HOST:$BROKER_PORT/$RANKER_REDIS_DB"
export FLOWER_PORT="$RANKER_FLOWER_PORT"
export FLOWER_BASIC_AUTH=${FLOWER_USER}:${FLOWER_PASSWORD}
export SUPERVISOR_PORT="$RANKER_SUPERVISOR_PORT"
export PYTHONPATH=$ROBOKOP_HOME/robokop-rank