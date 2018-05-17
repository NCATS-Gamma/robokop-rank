#!/bin/bash
### every exit != 0 fails the script
set -e

chown -R murphy:murphy $ROBOKOP_HOME

cd $ROBOKOP_HOME/robokop-rank
source ./deploy/setenv.sh

cd - > /dev/null
exec "$@"