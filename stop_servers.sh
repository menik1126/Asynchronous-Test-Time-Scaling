#!/bin/bash

# ==============================================================================
# SGLang Server Stopper
# This script stops the SGLang servers started by start_servers.sh
# ==============================================================================

echo "Stopping SGLang servers..."

if [ -f .small_model_server.pid ]; then
    SMALL_PID=$(cat .small_model_server.pid)
    if ps -p $SMALL_PID > /dev/null 2>&1; then
        kill $SMALL_PID
        echo "Small model server (PID: $SMALL_PID) stopped"
    else
        echo "Small model server (PID: $SMALL_PID) not running"
    fi
    rm .small_model_server.pid
else
    echo "No small model server PID file found"
fi

if [ -f .eval_model_server.pid ]; then
    EVAL_PID=$(cat .eval_model_server.pid)
    if ps -p $EVAL_PID > /dev/null 2>&1; then
        kill $EVAL_PID
        echo "Eval model server (PID: $EVAL_PID) stopped"
    else
        echo "Eval model server (PID: $EVAL_PID) not running"
    fi
    rm .eval_model_server.pid
else
    echo "No eval model server PID file found"
fi

echo "All servers stopped"
