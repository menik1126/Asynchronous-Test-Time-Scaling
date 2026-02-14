#!/bin/bash

# ==============================================================================
# SGLang Server Launcher
# Launches small model and eval model in parallel so both log files are created
# from the start; then waits for each server to be ready.
# ==============================================================================

# --- Paths: logs and PIDs in project root (parent of scripts/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SMALL_LOG="${PROJECT_DIR}/small_model_server.log"
EVAL_LOG="${PROJECT_DIR}/eval_model_server.log"
SMALL_PID_FILE="${PROJECT_DIR}/small_model.pid"
EVAL_PID_FILE="${PROJECT_DIR}/eval_model.pid"

# --- Configuration ---
SGLANG_HOST="${SGLANG_HOST:-0.0.0.0}"
SMALL_MODEL_PORT=40000
EVAL_MODEL_PORT=40001

SMALL_MODEL="${1:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
EVAL_MODEL="${2:-Qwen/QwQ-32B}"
SMALL_MODEL_DEVICE="${3:-0}"
EVAL_MODEL_DEVICES="${4:-1,2}"
MEM_FRACTION="${5:-0.9}"
EVAL_TP="${6:-2}"

# Flashinfer sm90 optimization (set to 1 to disable sm90 kernels if compilation fails)
# This uses generic kernels instead - slightly slower but more stable
FLASHINFER_DISABLE_SM90="${FLASHINFER_DISABLE_SM90:-0}"

echo "===================================================================================="
echo "Launching SGLang Servers"
echo "  - Small Model:     $SMALL_MODEL (GPU: $SMALL_MODEL_DEVICE, Port: $SMALL_MODEL_PORT)"
echo "  - Eval Model:      $EVAL_MODEL (GPU: $EVAL_MODEL_DEVICES, Port: $EVAL_MODEL_PORT, TP: $EVAL_TP)"
echo "  - Memory Fraction: $MEM_FRACTION"
echo "  - Logs:            $SMALL_LOG , $EVAL_LOG"
if [ "$FLASHINFER_DISABLE_SM90" = "1" ]; then
    echo "  - flashinfer:      sm90 optimizations DISABLED (using generic kernels)"
else
    echo "  - flashinfer:      sm90 optimizations ENABLED"
fi
echo "===================================================================================="

# --- Cleanup handler for Ctrl+C ---
cleanup() {
    echo ""
    echo "===================================================================================="
    echo "Caught interrupt signal. Cleaning up..."
    if [ -n "$SMALL_MODEL_PID" ]; then
        echo "Killing small model server (PID: $SMALL_MODEL_PID)..."
        kill $SMALL_MODEL_PID 2>/dev/null || true
    fi
    if [ -n "$EVAL_MODEL_PID" ]; then
        echo "Killing eval model server (PID: $EVAL_MODEL_PID)..."
        kill $EVAL_MODEL_PID 2>/dev/null || true
    fi
    echo "Cleanup complete."
    echo "===================================================================================="
    exit 130
}
trap cleanup INT TERM

# --- Wait until GET /get_model_info returns HTTP 200 ---
# Use env -u to clear proxy so curl hits 127.0.0.1 directly; otherwise proxy may return 503 for localhost
# Initial delay avoids treating leftover processes (from previous run) as "ready"
WAIT_POLL_INTERVAL=5
WAIT_INITIAL_DELAY=8
WAIT_TIMEOUT=600  # 10 minutes timeout

wait_for_server() {
    local port=$1
    local server_name=$2
    local timeout=${3:-$WAIT_TIMEOUT}
    
    echo "Waiting for $server_name on port $port to start..."
    if [ "$timeout" -gt 0 ]; then
        echo "  (giving process ${WAIT_INITIAL_DELAY}s to bind; then polling every ${WAIT_POLL_INTERVAL}s; timeout: ${timeout}s)"
    else
        echo "  (giving process ${WAIT_INITIAL_DELAY}s to bind; then polling every ${WAIT_POLL_INTERVAL}s; no timeout)"
    fi
    
    sleep "$WAIT_INITIAL_DELAY"
    
    local start_time=$(date +%s)
    local attempt=1
    
    while true; do
        # Check timeout
        if [ "$timeout" -gt 0 ]; then
            local current_time=$(date +%s)
            local elapsed=$((current_time - start_time))
            if [ "$elapsed" -ge "$timeout" ]; then
                echo "  ✗ Timeout: $server_name on port $port did not become ready after ${elapsed}s"
                return 1
            fi
        fi
        
        # Check server status
        local code
        code=$(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u no_proxy -u NO_PROXY \
            curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "http://127.0.0.1:$port/get_model_info" 2>/dev/null || echo "000")
        
        if [ "$code" = "200" ]; then
            local end_time=$(date +%s)
            local total_elapsed=$((end_time - start_time))
            echo "  ✓ $server_name on port $port is ready after ${total_elapsed}s (${attempt} check(s))"
            return 0
        else
            echo "  Attempt ${attempt}: get_model_info returned $code, retrying in ${WAIT_POLL_INTERVAL}s..."
            sleep "$WAIT_POLL_INTERVAL"
            attempt=$((attempt + 1))
        fi
    done
}

# --- Launch both servers in parallel so both log files exist from the start ---
echo ""
echo "Starting both servers (logs will be written to $SMALL_LOG and $EVAL_LOG)..."
echo "Note: Proxy environment variables will be unset for server processes to allow local connections."

# Use 'env -u' to unset proxy variables for the server processes
# This prevents proxy interference with localhost connections during warmup
# Set HF_HUB_OFFLINE=1 to prevent HuggingFace from trying to connect to huggingface.co
# when loading models from local cache
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u no_proxy -u NO_PROXY \
    HF_HUB_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=$SMALL_MODEL_DEVICE python3 -m sglang.launch_server \
    --model-path "$SMALL_MODEL" \
    --tp 1 \
    --mem-fraction-static $MEM_FRACTION \
    --host "$SGLANG_HOST" \
    --port "$SMALL_MODEL_PORT" \
    --watchdog-timeout 600 > "$SMALL_LOG" 2>&1 &
SMALL_MODEL_PID=$!

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u no_proxy -u NO_PROXY \
    HF_HUB_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=$EVAL_MODEL_DEVICES python3 -m sglang.launch_server \
    --model-path "$EVAL_MODEL" \
    --tp $EVAL_TP \
    --mem-fraction-static $MEM_FRACTION \
    --host "$SGLANG_HOST" \
    --port "$EVAL_MODEL_PORT" \
    --watchdog-timeout 600 > "$EVAL_LOG" 2>&1 &
EVAL_MODEL_PID=$!

echo "  Small model server PID: $SMALL_MODEL_PID  ->  $SMALL_LOG"
echo "  Eval model server PID:  $EVAL_MODEL_PID  ->  $EVAL_LOG"

# --- Wait for servers to be ready (order: small first, then eval) ---
echo ""
echo "===================================================================================="

if ! wait_for_server "$SMALL_MODEL_PORT" "Small model server"; then
    echo ""
    echo "ERROR: Small model server failed to start. Check log: $SMALL_LOG"
    echo "Killing server processes..."
    kill $SMALL_MODEL_PID $EVAL_MODEL_PID 2>/dev/null || true
    exit 1
fi

if ! wait_for_server "$EVAL_MODEL_PORT" "Evaluation model server"; then
    echo ""
    echo "ERROR: Evaluation model server failed to start. Check log: $EVAL_LOG"
    echo "Killing server processes..."
    kill $SMALL_MODEL_PID $EVAL_MODEL_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "===================================================================================="
echo "✓ All servers are running!"
echo ""
echo "Server Information:"
echo "  - Small Model:  http://$SGLANG_HOST:$SMALL_MODEL_PORT (PID: $SMALL_MODEL_PID)"
echo "  - Eval Model:   http://$SGLANG_HOST:$EVAL_MODEL_PORT (PID: $EVAL_MODEL_PID)"
echo ""
echo "Logs:"
echo "  - $SMALL_LOG"
echo "  - $EVAL_LOG"
echo ""
echo "To stop:  kill $SMALL_MODEL_PID $EVAL_MODEL_PID"
echo "===================================================================================="

echo "$SMALL_MODEL_PID" > "$SMALL_PID_FILE"
echo "$EVAL_MODEL_PID"  > "$EVAL_PID_FILE"
echo "PIDs saved to $SMALL_PID_FILE and $EVAL_PID_FILE"
