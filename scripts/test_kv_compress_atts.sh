#!/bin/bash
# Quick ATTS test with .sglang-kv-compress env.
# Only 3 problems × 2 repeats × 3 turns for fast validation.
# Must run inside ATTS docker container.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PPL_OUTPUT_DIR="${PROJECT_DIR}/evaluation"

PYTHON="${PROJECT_DIR}/.sglang-kv-compress/bin/python"

SGLANG_HOST="0.0.0.0"
MODEL_PORT=40098
MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_DEVICE="5"

# Small-scale test params
SAMPLE_SIZE=16
REPEATS=2
MAX_TOKENS=300
TEMPERATURE=0.8
CONCURRENCY=4
TURNS=3
MAX_RETRIES=2
PPL_THRESHOLD=0.6
MAX_REJECT_ATTEMPTS=2
ALPHA=0.01
FORK_TEMPERATURE=1.0
FORK_GAP=0.02

model_short=$(echo "$MODEL" | tr '/' '_')
PPL_ARRAY_PATH="${PPL_OUTPUT_DIR}/ppls_self_prefill_math500_${model_short}_s${SAMPLE_SIZE}_t500_temp${TEMPERATURE}.npy"
OUTPUT_DIR="/tmp/atts_kv_compress_test_$$"
mkdir -p "$OUTPUT_DIR"

echo "===================================================================================="
echo "ATTS KV-Compress Environment Test (small scale)"
echo "  Python:  $PYTHON"
echo "  Model:   $MODEL (GPU $MODEL_DEVICE, port $MODEL_PORT)"
echo "  Params:  repeats=$REPEATS, turns=$TURNS, max_tokens=$MAX_TOKENS"
echo "  PPL:     $PPL_ARRAY_PATH"
echo "  Output:  $OUTPUT_DIR"
echo "===================================================================================="

cleanup() {
    echo ""
    echo "Cleaning up..."
    [ -n "$MODEL_PID" ] && kill $MODEL_PID 2>/dev/null || true
    sleep 2
    echo "Server stopped. Results in: $OUTPUT_DIR"
}
trap 'cleanup; exit 130' INT TERM
trap cleanup EXIT

wait_for_server() {
    local port=$1 timeout=300 elapsed=0
    while [ $elapsed -lt $timeout ]; do
        code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 \
            "http://127.0.0.1:$port/get_model_info" 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            echo "  Server ready (${elapsed}s)"
            return 0
        fi
        sleep 5; elapsed=$((elapsed + 5))
    done
    echo "ERROR: Server not ready after ${timeout}s"
    return 1
}

# Check PPL file exists
if [ ! -f "$PPL_ARRAY_PATH" ]; then
    echo "ERROR: PPL array not found at $PPL_ARRAY_PATH"
    echo "  Run the full pipeline first to generate it."
    exit 1
fi
echo "  PPL array found."

echo ""
echo "[Step 1/2] Starting sglang server with .sglang-kv-compress env..."
HF_HUB_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=$MODEL_DEVICE $PYTHON -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp 1 \
    --mem-fraction-static 0.9 \
    --host "$SGLANG_HOST" \
    --port "$MODEL_PORT" \
    --watchdog-timeout 300 > "$OUTPUT_DIR/server.log" 2>&1 &
MODEL_PID=$!
echo "  PID: $MODEL_PID"

if ! wait_for_server "$MODEL_PORT"; then
    echo "Server log:"; tail -30 "$OUTPUT_DIR/server.log"
    exit 1
fi
sleep 3

echo ""
echo "[Step 2/2] Running ATTS ($REPEATS repeats × $TURNS turns, limited to 3 problems)..."
# We limit by only processing 3 problems worth of data.
# The script processes ALL 500 problems but we'll stop early via timeout.
# Better approach: just run and let it go, the small repeats/turns make it fast.

timeout 300 bash -c "
no_proxy='localhost,127.0.0.1,0.0.0.0' NO_PROXY='localhost,127.0.0.1,0.0.0.0' \
    $PYTHON -m ATTS.ref_async_self_reject_fork_parallel \
    --model_name '$MODEL' \
    --dataset_name 'math500' \
    --ppl_array_path '$PPL_ARRAY_PATH' \
    --model_port $MODEL_PORT \
    --max_tokens $MAX_TOKENS \
    --turns $TURNS \
    --repeats $REPEATS \
    --temperature $TEMPERATURE \
    --concurrency $CONCURRENCY \
    --max_retries $MAX_RETRIES \
    --extract_mode 'regex' \
    --output_dir '$OUTPUT_DIR' \
    --ppl_threshold $PPL_THRESHOLD \
    --max_reject_attempts $MAX_REJECT_ATTEMPTS \
    --alpha $ALPHA \
    --fork_temperature $FORK_TEMPERATURE \
    --fork_gap $FORK_GAP
" 2>&1 || true

echo ""
echo "===================================================================================="
echo "  Results"
echo "===================================================================================="

TOTAL=$(ls "$OUTPUT_DIR"/problem_*.json 2>/dev/null | wc -l)
echo "  Total completed: $TOTAL problems"

if [ "$TOTAL" -gt 0 ]; then
    echo ""
    echo "  Sample outputs (CoT history):"
    echo "  ---"
    for f in $(ls "$OUTPUT_DIR"/problem_*.json 2>/dev/null | head -4); do
        IDX=$(basename "$f" .json | sed 's/problem_//')
        ANS=$($PYTHON -c "import json; d=json.load(open('$f')); print(d['final_answer'])")
        TURNS_DONE=$($PYTHON -c "import json; d=json.load(open('$f')); print(len(d.get('ppl_history',[])))")
        AVG_PPL=$($PYTHON -c "import json; d=json.load(open('$f')); p=d.get('avg_ppl'); print(f'{p:.2f}' if p else 'N/A')")
        DUR=$($PYTHON -c "import json; d=json.load(open('$f')); print(f\"{d['duration_seconds']:.1f}\")")
        FORKS=$($PYTHON -c "import json; d=json.load(open('$f')); print(sum(1 for h in d.get('full_history',[]) if h.get('event')=='fork'))")
        QUESTION=$($PYTHON -c "import json; d=json.load(open('$f')); q=d.get('question',''); print(q[:80] if isinstance(q,str) else str(q)[:80])")

        echo "  Problem $IDX: answer=$ANS, turns=$TURNS_DONE, avg_ppl=$AVG_PPL, forks=$FORKS, time=${DUR}s"
        echo "    Q: $QUESTION"

        # Print first turn's CoT (truncated)
        $PYTHON -c "
import json
d = json.load(open('$f'))
hist = d.get('full_history', [])
for h in hist[:2]:
    if 'output' in h:
        text = h['output'][:200].replace(chr(10), ' ')
        status = 'accepted' if h.get('accepted') else 'rejected'
        ppl = h.get('ppl', 0)
        print(f'    Turn {h[\"turn\"]} att {h[\"attempt\"]} [{status}, ppl={ppl:.2f}]: {text}...')
    elif 'event' in h:
        print(f'    Turn {h[\"turn\"]} [FORK from idx={h.get(\"forked_from\")}]')
"
        echo "  ---"
    done
    echo ""
    echo "  ✓ ATTS pipeline works with .sglang-kv-compress environment!"
else
    echo "  ✗ No results generated. Check $OUTPUT_DIR/server.log"
fi
echo "===================================================================================="
