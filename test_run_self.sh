#!/bin/bash
set -eo pipefail

# ==============================================================================
# Single-Model ATTS Test: one model for both draft generation and PPL evaluation
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPL_OUTPUT_DIR="${PROJECT_DIR}/evaluation"
mkdir -p "$PPL_OUTPUT_DIR"

PYTHON="${PROJECT_DIR}/.sglang/bin/python"

SGLANG_HOST="0.0.0.0"
MODEL_PORT=40000

MODEL="${1:-Qwen/Qwen2.5-Math-7B}"
DATASET_NAME="${2:-math500}"
SAMPLE_SIZE="${3:-4}"
GPU_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

MAX_TOKENS=500
CONTINUE_MAX_TOKENS=500
TEMPERATURE=0.8
TURNS=10
CONCURRENCY=8
USE_CHAT_TEMPLATE=1

model_short=$(echo "$MODEL" | tr '/' '_')
PPL_ARRAY_PATH="${PPL_OUTPUT_DIR}/ppls_self_${DATASET_NAME}_${model_short}_s${SAMPLE_SIZE}_t${MAX_TOKENS}_temp${TEMPERATURE}.npy"
OUTPUT_DIR="${PROJECT_DIR}/results/self_${model_short}_${DATASET_NAME}_s${SAMPLE_SIZE}_t${MAX_TOKENS}"
mkdir -p "$OUTPUT_DIR"

echo "===================================================================================="
echo "ATTS Single-Model Test"
echo "  Model:      $MODEL"
echo "  GPU:        $GPU_DEVICE"
echo "  Port:       $MODEL_PORT"
echo "  Dataset:    $DATASET_NAME (sample_size=$SAMPLE_SIZE)"
echo "  PPL Path:   $PPL_ARRAY_PATH"
echo "  Output Dir: $OUTPUT_DIR"
echo "===================================================================================="

cleanup() {
    echo ""
    echo "Cleaning up server..."
    [ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null || true
    sleep 2
    echo "Done."
}
trap cleanup EXIT INT TERM

wait_for_server() {
    local port=$1 timeout=300 elapsed=0
    while [ $elapsed -lt $timeout ]; do
        code=$(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
            curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 \
            "http://127.0.0.1:$port/get_model_info" 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            echo "  Port $port ready (${elapsed}s)"
            return 0
        fi
        sleep 5; elapsed=$((elapsed + 5))
    done
    echo "ERROR: Server on port $port not ready after ${timeout}s"
    return 1
}

echo ""
echo "[Step 1/3] Starting SGLang server..."
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u no_proxy -u NO_PROXY \
    HF_HUB_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE $PYTHON -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp 1 \
    --mem-fraction-static 0.85 \
    --host "$SGLANG_HOST" \
    --port "$MODEL_PORT" \
    --watchdog-timeout 600 > "$OUTPUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "  PID: $SERVER_PID"

if ! wait_for_server "$MODEL_PORT"; then
    echo "Server log:"; tail -20 "$OUTPUT_DIR/server.log"
    exit 1
fi
sleep 5
echo "  Server ready!"

CHAT_TEMPLATE_FLAG=""
[ "$USE_CHAT_TEMPLATE" = "1" ] && CHAT_TEMPLATE_FLAG="--use_chat_template"

echo ""
echo "[Step 2/3] Running single-model conformal calibration..."
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u no_proxy -u NO_PROXY \
    $PYTHON -m ATTS.ref_conformal_self \
    --model_name "$MODEL" \
    --dataset_name "$DATASET_NAME" \
    --ppl_array_path "$PPL_ARRAY_PATH" \
    --model_port "$MODEL_PORT" \
    --max_tokens "$MAX_TOKENS" \
    --sample_size "$SAMPLE_SIZE" \
    --max_concurrent "$CONCURRENCY" \
    --temperature "$TEMPERATURE" \
    $CHAT_TEMPLATE_FLAG 2>&1 | tee "$OUTPUT_DIR/conformal_self.log"

echo "  PPL array saved."

echo ""
echo "[Step 3/3] Running single-model async evaluation..."
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u no_proxy -u NO_PROXY \
    $PYTHON -m ATTS.ref_async_self \
    --model_name "$MODEL" \
    --dataset_name "$DATASET_NAME" \
    --ppl_array_path "$PPL_ARRAY_PATH" \
    --model_port "$MODEL_PORT" \
    --max_tokens "$MAX_TOKENS" \
    --continue_max_tokens "$CONTINUE_MAX_TOKENS" \
    --turns "$TURNS" \
    --repeats "$SAMPLE_SIZE" \
    --temperature "$TEMPERATURE" \
    --concurrency "$CONCURRENCY" \
    --max_retries 3 \
    --extract_mode "regex" \
    --output_dir "$OUTPUT_DIR" \
    $CHAT_TEMPLATE_FLAG 2>&1 | tee "$OUTPUT_DIR/async_self.log"

echo ""
echo "===================================================================================="
echo "Single-model test completed!"
echo "  Results: $OUTPUT_DIR"
echo "  PPL:     $PPL_ARRAY_PATH"
echo "===================================================================================="
