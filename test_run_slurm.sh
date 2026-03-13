#!/bin/bash
#SBATCH --job-name=atts_test
#SBATCH --partition=LocalQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=test_run_slurm_%j.log
#SBATCH --error=test_run_slurm_%j.log

set -e

PROJECT_DIR="/home/xiongjing/Asynchronous-Test-Time-Scaling"
PPL_OUTPUT_DIR="${PROJECT_DIR}/evaluation"
mkdir -p "$PPL_OUTPUT_DIR"

PYTHON="${PROJECT_DIR}/.sglang/bin/python"

SGLANG_HOST="0.0.0.0"
SMALL_MODEL_PORT=40000
EVAL_MODEL_PORT=40001

SMALL_MODEL="Qwen/Qwen2.5-Math-1.5B"
EVAL_MODEL="Qwen/Qwen2.5-Math-7B"
DATASET_NAME="math500"
SAMPLE_SIZE=4
SMALL_MODEL_MAX_TOKENS=500
SMALL_MODEL_TEMPERATURE=0.8
MAX_CONCURRENT=4
USE_EVAL_CHAT_TEMPLATE=1

PPL_ARRAY_PATH="${PPL_OUTPUT_DIR}/ppls_${DATASET_NAME}_test_s${SAMPLE_SIZE}_t${SMALL_MODEL_MAX_TOKENS}_temp${SMALL_MODEL_TEMPERATURE}_ct${USE_EVAL_CHAT_TEMPLATE}.npy"

OUTPUT_DIR="${PROJECT_DIR}/results/test_qwen1.5B_qwen7B_${DATASET_NAME}"
mkdir -p "$OUTPUT_DIR"

echo "===================================================================================="
echo "ATTS Test Run (SLURM Job: $SLURM_JOB_ID)"
echo "  Node:        $(hostname)"
echo "  GPU:         $CUDA_VISIBLE_DEVICES"
echo "  Small Model: $SMALL_MODEL (port $SMALL_MODEL_PORT)"
echo "  Eval Model:  $EVAL_MODEL (port $EVAL_MODEL_PORT)"
echo "  Dataset:     $DATASET_NAME (sample_size=$SAMPLE_SIZE)"
echo "  PPL Path:    $PPL_ARRAY_PATH"
echo "  Output Dir:  $OUTPUT_DIR"
echo "===================================================================================="

cleanup() {
    echo ""
    echo "Cleaning up servers..."
    [ -n "$SMALL_PID" ] && kill $SMALL_PID 2>/dev/null || true
    [ -n "$EVAL_PID" ] && kill $EVAL_PID 2>/dev/null || true
    sleep 2
    echo "Done."
}
trap cleanup EXIT INT TERM

echo ""
echo "[Step 1/4] Starting SGLang servers on GPU $CUDA_VISIBLE_DEVICES..."

HF_HUB_OFFLINE=1 $PYTHON -m sglang.launch_server \
    --model-path "$SMALL_MODEL" \
    --tp 1 \
    --mem-fraction-static 0.15 \
    --host "$SGLANG_HOST" \
    --port "$SMALL_MODEL_PORT" \
    --watchdog-timeout 600 > "$OUTPUT_DIR/small_server.log" 2>&1 &
SMALL_PID=$!
echo "  Small model PID: $SMALL_PID"

HF_HUB_OFFLINE=1 $PYTHON -m sglang.launch_server \
    --model-path "$EVAL_MODEL" \
    --tp 1 \
    --mem-fraction-static 0.35 \
    --host "$SGLANG_HOST" \
    --port "$EVAL_MODEL_PORT" \
    --watchdog-timeout 600 > "$OUTPUT_DIR/eval_server.log" 2>&1 &
EVAL_PID=$!
echo "  Eval model PID: $EVAL_PID"

echo ""
echo "[Step 2/4] Waiting for servers to be ready..."
for port in $SMALL_MODEL_PORT $EVAL_MODEL_PORT; do
    elapsed=0; timeout=300
    while [ $elapsed -lt $timeout ]; do
        code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 \
            "http://127.0.0.1:$port/get_model_info" 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            echo "  Port $port ready (${elapsed}s)"
            break
        fi
        sleep 5; elapsed=$((elapsed + 5))
    done
    if [ "$code" != "200" ]; then
        echo "ERROR: Server on port $port not ready after ${timeout}s"
        echo "Small model log:"; tail -20 "$OUTPUT_DIR/small_server.log"
        echo "Eval model log:"; tail -20 "$OUTPUT_DIR/eval_server.log"
        exit 1
    fi
done
sleep 5
echo "  Both servers ready!"

echo ""
echo "[Step 3/4] Running conformal calibration (PPL generation)..."
CHAT_TEMPLATE_FLAG=""
[ "$USE_EVAL_CHAT_TEMPLATE" = "1" ] && CHAT_TEMPLATE_FLAG="--use_chat_template"

$PYTHON -m ATTS.ref_conformal \
    --small_model_name "$SMALL_MODEL" \
    --eval_model_name "$EVAL_MODEL" \
    --dataset_name "$DATASET_NAME" \
    --ppl_array_path "$PPL_ARRAY_PATH" \
    --small_model_port "$SMALL_MODEL_PORT" \
    --eval_model_port "$EVAL_MODEL_PORT" \
    --sample_size "$SAMPLE_SIZE" \
    --max_concurrent "$MAX_CONCURRENT" \
    --small_model_max_tokens "$SMALL_MODEL_MAX_TOKENS" \
    --small_model_temperature "$SMALL_MODEL_TEMPERATURE" \
    $CHAT_TEMPLATE_FLAG

echo "  PPL array saved to $PPL_ARRAY_PATH"

echo ""
echo "[Step 4/4] Running async evaluation..."
$PYTHON -m ATTS.ref_async \
    --small_model_name "$SMALL_MODEL" \
    --eval_model_name "$EVAL_MODEL" \
    --dataset_name "$DATASET_NAME" \
    --ppl_array_path "$PPL_ARRAY_PATH" \
    --small_model_max_tokens "$SMALL_MODEL_MAX_TOKENS" \
    --evalator_max_tokens 500 \
    --turns 10 \
    --small_model_port "$SMALL_MODEL_PORT" \
    --eval_model_port "$EVAL_MODEL_PORT" \
    --output_dir "$OUTPUT_DIR" \
    --small_model_temperature "$SMALL_MODEL_TEMPERATURE" \
    --eval_model_temperature 0.8 \
    --small_model_concurrency 4 \
    --eval_model_concurrency 2 \
    --max_retries 3 \
    --extract_mode "regex" \
    --repeats "$SAMPLE_SIZE" \
    $CHAT_TEMPLATE_FLAG

echo ""
echo "===================================================================================="
echo "Test run completed successfully!"
echo "  Results: $OUTPUT_DIR"
echo "  PPL:     $PPL_ARRAY_PATH"
echo "===================================================================================="
