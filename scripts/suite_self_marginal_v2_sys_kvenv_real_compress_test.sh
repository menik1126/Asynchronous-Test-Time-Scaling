#!/bin/bash
set -eo pipefail

# =============================================================================
# ATTS Self-Evaluation — Real KV Compression (free unselected + tree insert)
#
# This script tests the new real-compression implementation against the
# no-compression baseline. Runs a small subset (10 problems) for a quick
# smoke test, then compares accuracy.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PPL_OUTPUT_DIR="${PROJECT_DIR}/evaluation"
mkdir -p "$PPL_OUTPUT_DIR"

PYTHON="${PROJECT_DIR}/.sglang-kv-compress/bin/python"

SGLANG_HOST="0.0.0.0"
MODEL_PORT=40070

MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DATASET_NAME="math500"
SAMPLE_SIZE=16
MAX_TOKENS=500
TEMPERATURE=0.8
CONCURRENCY=16
TURNS=15
CONTINUE_MAX_TOKENS=500
EXTRACT_MODE="regex"
MAX_RETRIES=3

MODEL_DEVICE="5"

# Use 10 problems for a quick regression test; set to 0 for full eval
MAX_PROBLEMS=10

# SnapKV compression params (same as previous runs for fair comparison)
KV_COMPRESS_BUDGET=256
KV_COMPRESS_RATIO=0.5
KV_COMPRESS_RECENT=64
KV_COMPRESS_OBS=32
KV_COMPRESS_MIN_SEQ=512
KV_COMPRESS_PROTECT_PREFIX=128

export HF_HOME=/tmp/hf_cache
export OPENAI_API_KEY="sk-f7Oh115pfz6REQeFKesLFOhrk85Yd8ySvnqmRDZ08oDT8nyr"
export OPENAI_BASE_URL="https://chatapi.littlewheat.com/v1"
export OPENAI_MODEL="gpt-4o"

model_short=$(echo "$MODEL" | tr '/' '_')

PPL_ARRAY_PATH="${PPL_OUTPUT_DIR}/ppls_self_prefill_v2sys_kvenv_nocompress_${DATASET_NAME}_${model_short}_s${SAMPLE_SIZE}_t${MAX_TOKENS}_temp${TEMPERATURE}.npy"
OUTPUT_DIR="${PROJECT_DIR}/results/self_prefill_v2sys_kvenv_real_compress_test_${model_short}_${DATASET_NAME}_sm${MAX_TOKENS}_t${TURNS}_temp${TEMPERATURE}_b${KV_COMPRESS_BUDGET}_r${KV_COMPRESS_RATIO}"
mkdir -p "$OUTPUT_DIR"

echo "===================================================================================="
echo "ATTS Self-Evaluation — Real KV Compression Regression Test"
echo "  Model:        $MODEL (GPU $MODEL_DEVICE, port $MODEL_PORT)"
echo "  Dataset:      $DATASET_NAME (sample_size=$SAMPLE_SIZE)"
echo "  Max Problems: $MAX_PROBLEMS (0=all)"
echo "  PPL Path:     $PPL_ARRAY_PATH"
echo "  Output Dir:   $OUTPUT_DIR"
echo "  KV Compress:  budget=$KV_COMPRESS_BUDGET, ratio=$KV_COMPRESS_RATIO,"
echo "                recent=$KV_COMPRESS_RECENT, obs=$KV_COMPRESS_OBS,"
echo "                protect=$KV_COMPRESS_PROTECT_PREFIX, min_seq=$KV_COMPRESS_MIN_SEQ"
echo "===================================================================================="

cleanup() {
    echo ""
    echo "Cleaning up..."
    [ -n "$MODEL_PID" ] && kill $MODEL_PID 2>/dev/null || true
    sleep 2
    echo "Done."
}
trap cleanup EXIT INT TERM

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

# Check PPL file exists
if [ ! -f "$PPL_ARRAY_PATH" ]; then
    echo "ERROR: PPL array not found at $PPL_ARRAY_PATH"
    echo "  Run the nocompress pipeline first to generate it."
    exit 1
fi
echo "  PPL array found: $PPL_ARRAY_PATH"

echo ""
echo "[Step 1/2] Starting model server with REAL KV compression (free + tree insert)..."
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u no_proxy -u NO_PROXY \
    HF_HOME=/tmp/hf_cache \
    HF_HUB_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=$MODEL_DEVICE $PYTHON -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp 1 \
    --mem-fraction-static 0.50 \
    --host "$SGLANG_HOST" \
    --port "$MODEL_PORT" \
    --watchdog-timeout 600 \
    --enable-kv-compress \
    --kv-compress-budget "$KV_COMPRESS_BUDGET" \
    --kv-compress-ratio "$KV_COMPRESS_RATIO" \
    --kv-compress-recent-window "$KV_COMPRESS_RECENT" \
    --kv-compress-obs-window "$KV_COMPRESS_OBS" \
    --kv-compress-protect-prefix "$KV_COMPRESS_PROTECT_PREFIX" \
    --kv-compress-min-seq-len "$KV_COMPRESS_MIN_SEQ" \
    --disable-cuda-graph > "$OUTPUT_DIR/server.log" 2>&1 &
MODEL_PID=$!
echo "  PID: $MODEL_PID"

if ! wait_for_server "$MODEL_PORT"; then
    echo "Server log:"; tail -30 "$OUTPUT_DIR/server.log"
    exit 1
fi
sleep 5
echo "  Server ready!"

echo ""
echo "[Step 2/2] Running self-async evaluation (real compress, max_problems=$MAX_PROBLEMS)..."
START_TIME=$(date +%s)

no_proxy="localhost,127.0.0.1,0.0.0.0" NO_PROXY="localhost,127.0.0.1,0.0.0.0" \
    HF_HOME=/tmp/hf_cache HF_HUB_OFFLINE=1 \
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
    --max_retries "$MAX_RETRIES" \
    --extract_mode "$EXTRACT_MODE" \
    --output_dir "$OUTPUT_DIR" \
    --max_problems "$MAX_PROBLEMS" 2>&1 | tee "$OUTPUT_DIR/async_self.log"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $? -ne 0 ]; then
    echo "ERROR: Self-async evaluation failed."
    exit 1
fi

echo ""
echo "===================================================================================="
echo "Real KV Compression Regression Test COMPLETED"
echo "  Results:  $OUTPUT_DIR"
echo "  Duration: ${ELAPSED}s"
echo "  Problems: $MAX_PROBLEMS"
echo ""
echo "Check server.log for 'SnapKV compress' lines to verify real compression:"
echo "  grep 'SnapKV compress' $OUTPUT_DIR/server.log | head -5"
echo ""
echo "Compare with no-compress baseline accuracy in:"
echo "  ${PROJECT_DIR}/results/self_prefill_v2sys_kvenv_nocompress_*/"
echo "===================================================================================="
