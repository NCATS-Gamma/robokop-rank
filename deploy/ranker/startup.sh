#!/bin/bash
### every exit != 0 fails the script
set -e

cd $ROBOKOP_HOME/robokop-rank

echo "Creating task logs dir"
mkdir -p $ROBOKOP_HOME/task_logs
chmod 777 $ROBOKOP_HOME/task_logs

if [ -n "$(find . -user "murphy" -prune)" ]; then
    echo "Files are owned by murphy."
else
    echo "Changing file ownership..."
    chown -R murphy:murphy $ROBOKOP_HOME
fi

echo "Setting up environment..."
source ./deploy/ranker/setenv.sh

echo "Finding and removing stray pid files..."
find . -prune -name "*.pid" -exec rm -rf {} \;

cd - > /dev/null
echo "Running supervisord..."
exec "$@"