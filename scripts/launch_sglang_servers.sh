#!/bin/bash

# ==============================================================================
# SGLang Server Launcher
# This script launches two SGLang servers: a small model and an evaluation model
# ==============================================================================

# --- Configuration ---
SGLANG_HOST="0.0.0.0"
SMALL_MODEL_PORT=40000
EVAL_MODEL_PORT=40001

# Default models (can be overridden by command line arguments)
SMALL_MODEL="${1:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
EVAL_MODEL="${2:-Qwen/QwQ-32B}"

# GPU devices
SMALL_MODEL_DEVICE="${3:-0}"
EVAL_MODEL_DEVICES="${4:-1,2}"

# Memory fraction
MEM_FRACTION="${5:-0.9}"

# Tensor parallel size for eval model
EVAL_TP="${6:-2}"

echo "===================================================================================="
echo "Launching SGLang Servers"
echo "  - Small Model:     $SMALL_MODEL (GPU: $SMALL_MODEL_DEVICE, Port: $SMALL_MODEL_PORT)"
echo "  - Eval Model:      $EVAL_MODEL (GPU: $EVAL_MODEL_DEVICES, Port: $EVAL_MODEL_PORT, TP: $EVAL_TP)"
echo "  - Memory Fraction: $MEM_FRACTION"
echo "===================================================================================="

# --- Function to check if a server is ready ---
wait_for_server() {
    local port=$1
    local server_name=$2
    echo "Waiting for $server_name on port $port to start..."
    while true; do
        # Try to connect to the server health endpoint
        if curl -s http://localhost:$port/health > /dev/null 2>&1 || \
           curl -s http://localhost:$port/v1/models > /dev/null 2>&1 || \
           curl -s http://localhost:$port/get_model_info > /dev/null 2>&1; then
            echo "✓ $server_name on port $port is ready!"
            break
        else
            echo "  Waiting for $server_name on port $port..."
            sleep 10  # Wait 10 seconds before retrying
        fi
    done
}

# --- Launch Small Model Server ---
echo ""
echo "Starting SGLang server for small model ($SMALL_MODEL)..."
CUDA_VISIBLE_DEVICES=$SMALL_MODEL_DEVICE python3 -m sglang.launch_server \
    --model-path "$SMALL_MODEL" \
    --tp 1 \
    --mem-fraction-static $MEM_FRACTION \
    --host "$SGLANG_HOST" \
    --port "$SMALL_MODEL_PORT" > small_model_server.log 2>&1 &
SMALL_MODEL_PID=$!
echo "Small model server started with PID: $SMALL_MODEL_PID"
echo "Logs: small_model_server.log"

# --- Launch Evaluation Model Server ---
echo ""
echo "Starting SGLang server for evaluation model ($EVAL_MODEL)..."
CUDA_VISIBLE_DEVICES=$EVAL_MODEL_DEVICES python3 -m sglang.launch_server \
    --model-path "$EVAL_MODEL" \
    --tp $EVAL_TP \
    --mem-fraction-static $MEM_FRACTION \
    --host "$SGLANG_HOST" \
    --port "$EVAL_MODEL_PORT" > eval_model_server.log 2>&1 &
EVAL_MODEL_PID=$!
echo "Evaluation model server started with PID: $EVAL_MODEL_PID"
echo "Logs: eval_model_server.log"

# --- Wait for servers to be ready ---
echo ""
echo "===================================================================================="
wait_for_server "$SMALL_MODEL_PORT" "Small model server"
wait_for_server "$EVAL_MODEL_PORT" "Evaluation model server"

echo ""
echo "===================================================================================="
echo "✓ All servers are running!"
echo ""
echo "Server Information:"
echo "  - Small Model:  http://$SGLANG_HOST:$SMALL_MODEL_PORT (PID: $SMALL_MODEL_PID)"
echo "  - Eval Model:   http://$SGLANG_HOST:$EVAL_MODEL_PORT (PID: $EVAL_MODEL_PID)"
echo ""
echo "To stop the servers, run:"
echo "  kill $SMALL_MODEL_PID $EVAL_MODEL_PID"
echo ""
echo "Or save PIDs to file for later cleanup:"
echo "  echo $SMALL_MODEL_PID > small_model.pid"
echo "  echo $EVAL_MODEL_PID > eval_model.pid"
echo "===================================================================================="

# Save PIDs to files
echo $SMALL_MODEL_PID > small_model.pid
echo $EVAL_MODEL_PID > eval_model.pid
echo ""
echo "PIDs saved to small_model.pid and eval_model.pid"
