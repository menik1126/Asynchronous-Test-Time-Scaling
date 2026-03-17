#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PROJECT_DIR}/.sglang-kv-compress/bin/python"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_DEVICE="6"
MODEL_PORT=40098
OUTPUT_DIR="/tmp/debug_cg_compress_$$"
PPL_ARRAY_PATH="${PROJECT_DIR}/evaluation/ppls_self_prefill_v2sys_kvenv_nocompress_math500_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_s16_t500_temp0.8.npy"
mkdir -p "$OUTPUT_DIR"

export HF_HOME=/tmp/hf_cache
export SGLANG_DEBUG_CG_COMPRESS=1

cleanup() {
    echo "Cleaning up..."
    [ -n "$MODEL_PID" ] && kill $MODEL_PID 2>/dev/null || true
    sleep 2; echo "Done."
}
trap cleanup EXIT INT TERM

echo "=== Debug: CG + KV-Compress positions & kv_indices ==="
echo "  GPU: $MODEL_DEVICE, Port: $MODEL_PORT"
echo "  Output: $OUTPUT_DIR"

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u no_proxy -u NO_PROXY \
    HF_HOME=/tmp/hf_cache HF_HUB_OFFLINE=1 \
    SGLANG_DEBUG_CG_COMPRESS=1 \
    CUDA_VISIBLE_DEVICES=$MODEL_DEVICE $PYTHON -m sglang.launch_server \
    --model-path "$MODEL" --tp 1 --mem-fraction-static 0.80 \
    --host 0.0.0.0 --port "$MODEL_PORT" --watchdog-timeout 300 \
    --enable-kv-compress --kv-compress-budget 256 --kv-compress-ratio 0.5 \
    --kv-compress-recent-window 64 --kv-compress-obs-window 32 \
    --kv-compress-protect-prefix 128 --kv-compress-min-seq-len 512 \
    > "$OUTPUT_DIR/server.log" 2>&1 &
MODEL_PID=$!
echo "  Server PID: $MODEL_PID"

timeout=300; elapsed=0
while [ $elapsed -lt $timeout ]; do
    code=$(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
        curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 \
        "http://127.0.0.1:$MODEL_PORT/get_model_info" 2>/dev/null || echo "000")
    if [ "$code" = "200" ]; then echo "Server ready (${elapsed}s)"; break; fi
    sleep 5; elapsed=$((elapsed + 5))
done
sleep 3

echo ""
echo "=== Running 2 problems (32 samples) ==="
no_proxy="localhost,127.0.0.1,0.0.0.0" NO_PROXY="localhost,127.0.0.1,0.0.0.0" \
    HF_HOME=/tmp/hf_cache HF_HUB_OFFLINE=1 \
    $PYTHON -m ATTS.ref_async_self \
    --model_name "$MODEL" --dataset_name math500 \
    --ppl_array_path "$PPL_ARRAY_PATH" \
    --model_port "$MODEL_PORT" --max_tokens 500 --continue_max_tokens 500 \
    --turns 15 --repeats 16 --temperature 0.8 --concurrency 16 \
    --max_retries 3 --extract_mode regex \
    --output_dir "$OUTPUT_DIR" --max_problems 2 2>&1 | tee "$OUTPUT_DIR/eval.log"

echo ""
echo "=== Debug output from server log ==="
grep "\[DEBUG-CG\]" "$OUTPUT_DIR/server.log" | head -30
echo ""
echo "=== Compression stats ==="
grep "SnapKV attention-compress" "$OUTPUT_DIR/server.log" | head -10
echo ""
echo "=== Results ==="
ls "$OUTPUT_DIR"/problem_*.json 2>/dev/null | wc -l
echo "files generated"
