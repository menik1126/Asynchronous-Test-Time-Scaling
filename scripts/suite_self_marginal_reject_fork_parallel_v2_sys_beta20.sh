#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PPL_OUTPUT_DIR="${PROJECT_DIR}/evaluation"
mkdir -p "$PPL_OUTPUT_DIR"

PYTHON="${PROJECT_DIR}/.sglang-kv-compress/bin/python"

SGLANG_HOST="0.0.0.0"
MODEL_PORT=40031

MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DATASET_NAME="math500"
SAMPLE_SIZE=16
MAX_TOKENS=500
TEMPERATURE=0.8
CONCURRENCY=16
TURNS=15
EXTRACT_MODE="regex"
MAX_RETRIES=3

MODEL_DEVICE="1"

PPL_THRESHOLD=0.6
MAX_REJECT_ATTEMPTS=3
ALPHA=0.01
FORK_TEMPERATURE=1.0
FORK_GAP=0.02
BETA=0.2

export HF_HOME=/tmp/hf_cache
export OPENAI_API_KEY="sk-f7Oh115pfz6REQeFKesLFOhrk85Yd8ySvnqmRDZ08oDT8nyr"
export OPENAI_BASE_URL="https://chatapi.littlewheat.com/v1"
export OPENAI_MODEL="gpt-4o"

model_short=$(echo "$MODEL" | tr '/' '_')

PPL_ARRAY_PATH="${PPL_OUTPUT_DIR}/ppls_self_prefill_v2sys_${DATASET_NAME}_${model_short}_s${SAMPLE_SIZE}_t${MAX_TOKENS}_temp${TEMPERATURE}.npy"
OUTPUT_DIR="${PROJECT_DIR}/results/self_reject_fork_v2sys_${model_short}_${DATASET_NAME}_sm${MAX_TOKENS}_t${TURNS}_temp${TEMPERATURE}_thr${PPL_THRESHOLD}_att${MAX_REJECT_ATTEMPTS}_a${ALPHA}_b${BETA}"
mkdir -p "$OUTPUT_DIR"

echo "===================================================================================="
echo "ATTS Self-Evaluation (Rejection + Fork + Length Reward, ALL PARALLEL)"
echo "  Model:              $MODEL (GPU $MODEL_DEVICE, port $MODEL_PORT)"
echo "  Dataset:            $DATASET_NAME (sample_size=$SAMPLE_SIZE)"
echo "  PPL Path:           $PPL_ARRAY_PATH"
echo "  Output Dir:         $OUTPUT_DIR"
echo "  PPL Threshold:      $PPL_THRESHOLD"
echo "  Max Reject Attempts: $MAX_REJECT_ATTEMPTS"
echo "  Alpha (progress):   $ALPHA"
echo "  Beta (length):      $BETA"
echo "  Fork Temperature:   $FORK_TEMPERATURE"
echo "  Fork Gap (score):   $FORK_GAP"
echo "===================================================================================="

cleanup() {
    echo ""
    echo "Cleaning up..."
    [ -n "$MODEL_PID" ] && kill $MODEL_PID 2>/dev/null || true
    sleep 2
    echo "Done."
}
trap 'cleanup; exit 130' INT TERM
trap cleanup EXIT

wait_for_server() {
    local port=$1 timeout=600 elapsed=0
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
echo "[Step 1/3] Starting model server..."
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u no_proxy -u NO_PROXY \
    HF_HOME=/tmp/hf_cache \
    HF_HUB_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=$MODEL_DEVICE $PYTHON -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp 1 \
    --mem-fraction-static 0.85 \
    --host "$SGLANG_HOST" \
    --port "$MODEL_PORT" \
    --watchdog-timeout 600 > "$OUTPUT_DIR/server.log" 2>&1 &
MODEL_PID=$!
echo "  PID: $MODEL_PID"

if ! wait_for_server "$MODEL_PORT"; then
    echo "Server log:"; tail -20 "$OUTPUT_DIR/server.log"
    exit 1
fi
sleep 5
echo "  Server ready!"

if [ -f "$PPL_ARRAY_PATH" ]; then
    echo ""
    echo "[Step 2/3] PPL array already exists, skipping calibration."
    echo "  Using: $PPL_ARRAY_PATH"
else
    echo ""
    echo "[Step 2/3] Running self-conformal calibration (prefill PPL, prefix-cache friendly)..."
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u no_proxy -u NO_PROXY \
        $PYTHON -m ATTS.ref_conformal_self \
        --model_name "$MODEL" \
        --dataset_name "$DATASET_NAME" \
        --ppl_array_path "$PPL_ARRAY_PATH" \
        --model_port "$MODEL_PORT" \
        --max_tokens "$MAX_TOKENS" \
        --sample_size "$SAMPLE_SIZE" \
        --max_concurrent "$CONCURRENCY" \
        --temperature "$TEMPERATURE" 2>&1 | tee "$OUTPUT_DIR/conformal_self.log"

    if [ $? -ne 0 ]; then
        echo "ERROR: Self-conformal calibration failed."
        exit 1
    fi
    echo "  PPL array saved to $PPL_ARRAY_PATH"
fi

echo ""
echo "[Step 3/3] Running self-async evaluation (rejection + fork + length reward)..."
no_proxy="localhost,127.0.0.1,0.0.0.0" NO_PROXY="localhost,127.0.0.1,0.0.0.0" \
    $PYTHON -m ATTS.ref_async_self_reject_fork_parallel \
    --model_name "$MODEL" \
    --dataset_name "$DATASET_NAME" \
    --ppl_array_path "$PPL_ARRAY_PATH" \
    --model_port "$MODEL_PORT" \
    --max_tokens "$MAX_TOKENS" \
    --turns "$TURNS" \
    --repeats "$SAMPLE_SIZE" \
    --temperature "$TEMPERATURE" \
    --concurrency "$CONCURRENCY" \
    --max_retries "$MAX_RETRIES" \
    --extract_mode "$EXTRACT_MODE" \
    --output_dir "$OUTPUT_DIR" \
    --ppl_threshold "$PPL_THRESHOLD" \
    --max_reject_attempts "$MAX_REJECT_ATTEMPTS" \
    --alpha "$ALPHA" \
    --fork_temperature "$FORK_TEMPERATURE" \
    --fork_gap "$FORK_GAP" \
    --beta "$BETA" 2>&1 | tee "$OUTPUT_DIR/async_self_reject_fork.log"

if [ $? -ne 0 ]; then
    echo "ERROR: Rejection + fork + length reward evaluation failed."
    exit 1
fi

echo ""
echo "===================================================================================="
echo "Rejection + fork + length reward test completed!"
echo "  Results: $OUTPUT_DIR"
echo "  PPL:     $PPL_ARRAY_PATH"
echo "===================================================================================="
