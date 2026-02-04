#!/bin/bash

# ==============================================================================
# SGLang Server Launcher
# This script starts two SGLang servers for ATTS evaluation
# ==============================================================================

# --- Configuration ---
SGLANG_HOST="0.0.0.0"
SMALL_MODEL_PORT=40000
EVAL_MODEL_PORT=40001

# Default models (can be overridden by command line arguments)
SMALL_MODEL="${1:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
EVAL_MODEL="${2:-Qwen/QwQ-32B}"

# GPU allocation
SMALL_MODEL_DEVICE="${SMALL_GPU:-2}"
EVAL_MODEL_DEVICES="${EVAL_GPU:-3,4}"

echo "===================================================================================="
echo "Starting SGLang Servers"
echo "  Small Model: $SMALL_MODEL (Port: $SMALL_MODEL_PORT, GPU: $SMALL_MODEL_DEVICE)"
echo "  Eval Model:  $EVAL_MODEL (Port: $EVAL_MODEL_PORT, GPU: $EVAL_MODEL_DEVICES)"
echo "===================================================================================="

# --- Start Small Model Server ---
echo "Starting SGLang server for small model..."
CUDA_VISIBLE_DEVICES=$SMALL_MODEL_DEVICE python3 -m sglang.launch_server \
    --model-path "$SMALL_MODEL" \
    --tp 1 \
    --mem-fraction-static 0.9 \
    --host "$SGLANG_HOST" \
    --port "$SMALL_MODEL_PORT" > "small_model_server.log" 2>&1 &
SMALL_MODEL_PID=$!
echo "Small model server started with PID: $SMALL_MODEL_PID"

# --- Start Evaluation Model Server ---
echo "Starting SGLang server for evaluation model..."
CUDA_VISIBLE_DEVICES=$EVAL_MODEL_DEVICES python3 -m sglang.launch_server \
    --model-path "$EVAL_MODEL" \
    --tp 2 \
    --mem-fraction-static 0.9 \
    --host "$SGLANG_HOST" \
    --port "$EVAL_MODEL_PORT" > "eval_model_server.log" 2>&1 &
EVAL_MODEL_PID=$!
echo "Evaluation model server started with PID: $EVAL_MODEL_PID"

# --- Save PIDs ---
echo "$SMALL_MODEL_PID" > .small_model_server.pid
echo "$EVAL_MODEL_PID" > .eval_model_server.pid

echo ""
echo "===================================================================================="
echo "Servers started successfully!"
echo "  Small Model PID: $SMALL_MODEL_PID (Port: $SMALL_MODEL_PORT)"
echo "  Eval Model PID:  $EVAL_MODEL_PID (Port: $EVAL_MODEL_PORT)"
echo ""
echo "Logs:"
echo "  Small Model: small_model_server.log"
echo "  Eval Model:  eval_model_server.log"
echo ""
echo "To stop servers, run: bash stop_servers.sh"
echo "===================================================================================="
