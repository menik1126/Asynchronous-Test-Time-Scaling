#!/bin/bash

# ==============================================================================
# SGLang 服务器和评估脚本运行器
# 此脚本将循环启动两个 SGLang 服务器，然后执行 ref_conformal.py 脚本进行评估
# 并在每次循环结束后关闭服务器。
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PPL_OUTPUT_DIR="${PROJECT_DIR}/evaluation"
mkdir -p "$PPL_OUTPUT_DIR"
# export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=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
# export NCCL_P2P_DISABLE=1

# --- 配置变量 ---
# 你可以根据需要修改这些参数。
SGLANG_HOST="0.0.0.0"
SMALL_MODEL_PORT=51106
EVAL_MODEL_PORT=51105

# Server CUDA devices
SMALL_MODEL_DEVICE="2"
# 注意：评估模型使用了两个CUDA设备
EVAL_MODEL_DEVICES="3,4"

# 评估轮次，如果配置数组中未指定，将使用此默认值
DEFAULT_TURNS=20
SMALL_MODEL_TEMPERATURE=0.8
SMALL_MODEL_MAX_TOKENS=500
EVAL_MODEL_MAX_TOKENS=500
EVAL_MODEL_TEMPERATURE=0.8

SMALL_MODEL_CONCURRENCY=16
EVAL_MODEL_CONCURRENCY=4
SAMPLE_SIZE=16

# --- 定义所有要运行的配置数组 ---
# 每个元素包含七个参数: SMALL_MODEL, EVAL_MODEL, DATASET_NAME, PPL_ARRAY_PATH, SMALL_MODEL_MAX_TOKENS, EVAL_MODEL_MAX_TOKENS, TURNS
CONFIGS=(
    "simplescaling/s1.1-7B simplescaling/s1.1-32B math500 ${PPL_OUTPUT_DIR}/ppls_math500_s1.1_32B_s1.1_7B_s${SAMPLE_SIZE}_t${SMALL_MODEL_MAX_TOKENS}_temp${SMALL_MODEL_TEMPERATURE}_per_question.npy $SMALL_MODEL_MAX_TOKENS $EVAL_MODEL_MAX_TOKENS $DEFAULT_TURNS"
    "simplescaling/s1.1-7B simplescaling/s1.1-32B amc ${PPL_OUTPUT_DIR}/ppls_amc_s1.1_32B_s1.1_7B_s${SAMPLE_SIZE}_t${SMALL_MODEL_MAX_TOKENS}_temp${SMALL_MODEL_TEMPERATURE}_per_question.npy $SMALL_MODEL_MAX_TOKENS $EVAL_MODEL_MAX_TOKENS $DEFAULT_TURNS"
    "simplescaling/s1.1-7B simplescaling/s1.1-32B aime25 ${PPL_OUTPUT_DIR}/ppls_aime25_s1.1_32B_s1.1_7B_s${SAMPLE_SIZE}_t${SMALL_MODEL_MAX_TOKENS}_temp${SMALL_MODEL_TEMPERATURE}_per_question.npy $SMALL_MODEL_MAX_TOKENS $EVAL_MODEL_MAX_TOKENS $DEFAULT_TURNS"
    "simplescaling/s1.1-7B simplescaling/s1.1-32B aime24 ${PPL_OUTPUT_DIR}/ppls_aime24_s1.1_32B_s1.1_7B_s${SAMPLE_SIZE}_t${SMALL_MODEL_MAX_TOKENS}_temp${SMALL_MODEL_TEMPERATURE}_per_question.npy $SMALL_MODEL_MAX_TOKENS $EVAL_MODEL_MAX_TOKENS $DEFAULT_TURNS"
)

# --- 循环遍历所有配置 ---
for config in "${CONFIGS[@]}"; do
    # 解析配置字符串
    read SMALL_MODEL EVAL_MODEL DATASET_NAME PPL_ARRAY_PATH SMALL_MODEL_MAX_TOKENS EVALATOR_MAX_TOKENS EVAL_TURNS <<< "$config"

    # 根据配置生成动态日志和输出目录名
    small_model_name_base=$(echo "$SMALL_MODEL" | tr '/' '_')
    eval_model_name_base=$(echo "$EVAL_MODEL" | tr '/' '_')
    run_identifier="${small_model_name_base}_${eval_model_name_base}_${DATASET_NAME}"
    hyperparams_suffix="sm${SMALL_MODEL_MAX_TOKENS}_em${EVALATOR_MAX_TOKENS}_t${EVAL_TURNS}_smt${SMALL_MODEL_TEMPERATURE}_emt${EVAL_MODEL_TEMPERATURE}_smc${SMALL_MODEL_CONCURRENCY}_emc${EVAL_MODEL_CONCURRENCY}_per_question"
    
    # 定义输出目录路径
    OUTPUT_DIR="./results/${run_identifier}_${hyperparams_suffix}"
    
    SMALL_MODEL_LOG="$OUTPUT_DIR/server_${run_identifier}_${hyperparams_suffix}.log"
    EVAL_MODEL_LOG="$OUTPUT_DIR/eval_${run_identifier}_${hyperparams_suffix}.log"
    EVAL_SCRIPT_LOG="$OUTPUT_DIR/async_eval_${run_identifier}_${hyperparams_suffix}.log"

    # 确保输出目录存在
    mkdir -p "$OUTPUT_DIR"

    echo "===================================================================================="
    echo "Starting a new run with:"
    echo "  - Small Model:             $SMALL_MODEL"
    echo "  - Eval Model:              $EVAL_MODEL"
    echo "  - Dataset:                 $DATASET_NAME"
    echo "  - PPL Array Path:          $PPL_ARRAY_PATH"
    echo "  - Small Model Max Tokens:  $SMALL_MODEL_MAX_TOKENS"
    echo "  - Evalator Max Tokens:     $EVALATOR_MAX_TOKENS"
    echo "  - Turns:                   $EVAL_TURNS"
    echo "  - Output Directory:        $OUTPUT_DIR"
    echo "===================================================================================="

    # --- 启动 SGLang 服务器 ---
    echo "Starting SGLang server for small model ($SMALL_MODEL)..."
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u no_proxy -u NO_PROXY \
        HF_HUB_OFFLINE=1 \
        CUDA_VISIBLE_DEVICES=$SMALL_MODEL_DEVICE python3 -m sglang.launch_server \
        --model-path "$SMALL_MODEL" \
        --tp 1 \
        --mem-fraction-static 0.9 \
        --host "$SGLANG_HOST" \
        --port "$SMALL_MODEL_PORT" > "$SMALL_MODEL_LOG" 2>&1 &
    SMALL_MODEL_PID=$!
    echo "Small model server started with PID: $SMALL_MODEL_PID"

    echo "Starting SGLang server for evaluation model ($EVAL_MODEL)..."
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u no_proxy -u NO_PROXY \
        HF_HUB_OFFLINE=1 \
        CUDA_VISIBLE_DEVICES=$EVAL_MODEL_DEVICES python3 -m sglang.launch_server \
        --model-path "$EVAL_MODEL" \
        --tp 2 \
        --mem-fraction-static 0.7 \
        --host "$SGLANG_HOST" \
        --port "$EVAL_MODEL_PORT" > "$EVAL_MODEL_LOG" 2>&1 &
    EVAL_MODEL_PID=$!
    echo "Evaluation model server started with PID: $EVAL_MODEL_PID"

    # --- Wait for servers to be fully ready (HTTP 200, not just TCP) ---
    echo "Waiting for SGLang servers to be ready..."
    for port in $SMALL_MODEL_PORT $EVAL_MODEL_PORT; do
        elapsed=0; timeout=600
        while [ $elapsed -lt $timeout ]; do
            code=$(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
                curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 \
                "http://127.0.0.1:$port/get_model_info" 2>/dev/null || echo "000")
            if [ "$code" = "200" ]; then
                echo "  Server on port $port is ready (${elapsed}s)"
                break
            fi
            sleep 5; elapsed=$((elapsed + 5))
        done
        if [ "$code" != "200" ]; then
            echo "ERROR: Server on port $port not ready after ${timeout}s. Aborting."
            kill $SMALL_MODEL_PID $EVAL_MODEL_PID 2>/dev/null
            exit 1
        fi
    done
    # Extra wait for warmup request to finish
    sleep 10

    # --- 运行 Python 评估脚本，并将输出重定向到日志文件 ---
    echo "Starting evaluation script in the background, output will be logged to $EVAL_SCRIPT_LOG..."
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u no_proxy -u NO_PROXY \
        python3 -m ATTS.ref_async_per_question \
        --small_model_name "$SMALL_MODEL" \
        --eval_model_name "$EVAL_MODEL" \
        --dataset_name "$DATASET_NAME" \
        --ppl_array_path "$PPL_ARRAY_PATH" \
        --small_model_max_tokens "$SMALL_MODEL_MAX_TOKENS" \
        --evalator_max_tokens "$EVALATOR_MAX_TOKENS" \
        --turns "$EVAL_TURNS" \
        --small_model_port "$SMALL_MODEL_PORT" \
        --eval_model_port "$EVAL_MODEL_PORT" \
        --output_dir "$OUTPUT_DIR" \
        --repeats "$SAMPLE_SIZE" \
        --small_model_temperature "$SMALL_MODEL_TEMPERATURE" \
        --eval_model_temperature "$EVAL_MODEL_TEMPERATURE" \
        --small_model_concurrency "$SMALL_MODEL_CONCURRENCY" \
        --eval_model_concurrency "$EVAL_MODEL_CONCURRENCY" > "$EVAL_SCRIPT_LOG" 2>&1 &
    EVAL_SCRIPT_PID=$!
    echo "Evaluation script started with PID: $EVAL_SCRIPT_PID"

    # 等待评估脚本完成
    echo "Waiting for the evaluation script (PID: $EVAL_SCRIPT_PID) to finish..."
    wait "$EVAL_SCRIPT_PID"
    
    # 检查评估脚本是否成功运行
    if [ $? -eq 0 ]; then
        echo "Evaluation script for this run finished successfully."
    else
        echo "Evaluation script for this run encountered an error."
    fi

    # --- 清理: 关闭所有服务器 ---
    echo "Killing SGLang servers with PIDs: $SMALL_MODEL_PID and $EVAL_MODEL_PID..."
    kill $SMALL_MODEL_PID $EVAL_MODEL_PID
    sleep 5 # 等待5秒，确保端口完全释放，避免下一轮启动失败

done

echo "All evaluation runs have been completed."
