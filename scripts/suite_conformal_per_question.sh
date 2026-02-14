#!/bin/bash

# ==============================================================================
# SGLang 服务器和评估脚本运行器
# 此脚本将循环启动两个 SGLang 服务器，然后执行 evaluation/baseline.py 脚本
# 来计算 PPL，并在每次循环结束后关闭服务器。
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PPL_OUTPUT_DIR="${PROJECT_DIR}/evaluation"
mkdir -p "$PPL_OUTPUT_DIR"
#export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=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

# --- 检查 wait-for-it.sh 脚本是否存在 ---
if [ ! -f "$SCRIPT_DIR/wait-for-it.sh" ]; then
    echo "Error: wait-for-it.sh not found. Please download it first."
    echo "Run: wget https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh -O $SCRIPT_DIR/wait-for-it.sh && chmod +x $SCRIPT_DIR/wait-for-it.sh"
    wget https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh -O "$SCRIPT_DIR/wait-for-it.sh" && chmod +x "$SCRIPT_DIR/wait-for-it.sh"
    #exit 1
fi

# --- 配置变量 ---
# 你可以根据需要修改这些参数。
SGLANG_HOST="0.0.0.0"
SMALL_MODEL_PORT=40000
EVAL_MODEL_PORT=40001

# 采样数量配置
SAMPLE_SIZE=16

# 小模型最大token数量配置
SMALL_MODEL_MAX_TOKENS=500

# 小模型采样温度配置
SMALL_MODEL_TEMPERATURE=0.8

# 最大并发请求数
MAX_CONCURRENT=16

# Server CUDA devices
SMALL_MODEL_DEVICE="2"
# 注意：评估模型使用了两个CUDA设备
EVAL_MODEL_DEVICES="3,4"

# --- 定义所有要运行的配置数组 ---
# 每个元素包含五个参数: SMALL_MODEL, EVAL_MODEL, DATASET_NAME, PPL_ARRAY_PATH, CONFIG_SMALL_MODEL_MAX_TOKENS
# PPL文件名格式: ppls_{dataset}_{eval_model}_{small_model}_s{SAMPLE_SIZE}_t{SMALL_MODEL_MAX_TOKENS}_temp{SMALL_MODEL_TEMPERATURE}_per_question.npy
# PPL filename helper: auto-generates path from DATASET, EVAL_SHORT, SMALL_SHORT
# Format: ${PPL_OUTPUT_DIR}/ppls_${DATASET}_${EVAL_SHORT}_${SMALL_SHORT}_s${SAMPLE_SIZE}_t${SMALL_MODEL_MAX_TOKENS}_temp${SMALL_MODEL_TEMPERATURE}_per_question.npy
# Each config: SMALL_MODEL EVAL_MODEL DATASET_NAME EVAL_SHORT SMALL_SHORT [SMALL_MODEL_MAX_TOKENS_OVERRIDE]
# Each config: SMALL_MODEL EVAL_MODEL DATASET EVAL_SHORT SMALL_SHORT SM_MAX_TOKENS
CONFIGS=(
    # --- Qwen/QwQ-32B as eval ---
    # "Qwen/Qwen2.5-7B-Instruct Qwen/QwQ-32B aime25 qwq_32B qwen_7B 500"
    # "meta-llama/Llama-3.1-8B-Instruct Qwen/QwQ-32B aime25 qwq_32B llama_8B 500"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B Qwen/QwQ-32B aime25 qwq_32B r1_1B 500"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B Qwen/QwQ-32B aime24 qwq_32B rllama_8B 500"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B Qwen/QwQ-32B aime25 qwq_32B rllama_8B 500"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B Qwen/QwQ-32B math500 qwq_32B rllama_8B 500"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B Qwen/QwQ-32B gpqa qwq_32B rllama_8B 500"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B Qwen/QwQ-32B olympiad qwq_32B rllama_8B 500"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B Qwen/QwQ-32B olympiad qwq_32B r1_1B 500"
    # "Qwen/Qwen2.5-7B-Instruct Qwen/QwQ-32B olympiad qwq_32B qwen_7B 500"
    # "meta-llama/Llama-3.1-8B-Instruct Qwen/QwQ-32B olympiad qwq_32B llama_8B 500"
    # --- simplescaling/s1.1-32B as eval ---
    # "Qwen/Qwen2.5-7B-Instruct simplescaling/s1.1-32B aime25 s1_32B qwen_7B 500"
    # "meta-llama/Llama-3.1-8B-Instruct simplescaling/s1.1-32B aime25 s1_32B llama_8B 500"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B simplescaling/s1.1-32B aime25 s1_32B r1_1B 500"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B simplescaling/s1.1-32B aime24 s1_32B rllama_8B 500"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B simplescaling/s1.1-32B aime25 s1_32B rllama_8B 500"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B simplescaling/s1.1-32B math500 s1_32B rllama_8B 500"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B simplescaling/s1.1-32B gpqa s1_32B rllama_8B 500"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B simplescaling/s1.1-32B olympiad s1_32B rllama_8B 500"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B simplescaling/s1.1-32B olympiad s1_32B r1_1B 500"
    # "Qwen/Qwen2.5-7B-Instruct simplescaling/s1.1-32B olympiad s1_32B qwen_7B 500"
    # "meta-llama/Llama-3.1-8B-Instruct simplescaling/s1.1-32B olympiad s1_32B llama_8B 500"
    # --- Qwen/Qwen2.5-32B-Instruct as eval ---
    # "Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-32B-Instruct aime24 qwen_32B qwen_7B 500"
    # "Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-32B-Instruct aime25 qwen_32B qwen_7B 500"
    # "Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-32B-Instruct math500 qwen_32B qwen_7B 500"
    # "Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-32B-Instruct gpqa qwen_32B qwen_7B 500"
    # --- Active config ---
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B Qwen/QwQ-32B math500 qwq_32B rllama_8B 500"
)


# --- 循环遍历所有配置 ---
for config in "${CONFIGS[@]}"; do
    # 解析配置字符串: SMALL_MODEL EVAL_MODEL DATASET EVAL_SHORT SMALL_SHORT [MAX_TOKENS_OVERRIDE]
    read SMALL_MODEL EVAL_MODEL DATASET_NAME EVAL_SHORT SMALL_SHORT CONFIG_SMALL_MODEL_MAX_TOKENS <<< "$config"

    # Auto-generate PPL output path from dataset + model short names (with _per_question suffix)
    PPL_ARRAY_PATH="${PPL_OUTPUT_DIR}/ppls_${DATASET_NAME}_${EVAL_SHORT}_${SMALL_SHORT}_s${SAMPLE_SIZE}_t${CONFIG_SMALL_MODEL_MAX_TOKENS:-$SMALL_MODEL_MAX_TOKENS}_temp${SMALL_MODEL_TEMPERATURE}_per_question.npy"

    # 根据配置生成动态日志文件名，并添加 "conformal" 前缀
    small_model_name_base=$(echo "$SMALL_MODEL" | tr '/' '_')
    eval_model_name_base=$(echo "$EVAL_MODEL" | tr '/' '_')
    log_file_base="${small_model_name_base}_${eval_model_name_base}_${DATASET_NAME}"
    SMALL_MODEL_LOG="conformal_small_${log_file_base}.log"
    EVAL_MODEL_LOG="conformal_eval_${log_file_base}.log"

    echo "===================================================================================="
    echo "Starting a new run with:"
    echo "  - Small Model:     $SMALL_MODEL"
    echo "  - Eval Model:      $EVAL_MODEL"
    echo "  - Dataset:         $DATASET_NAME"
    echo "  - PPL Array Path:  $PPL_ARRAY_PATH"
    echo "  - Small Model Max Tokens: ${CONFIG_SMALL_MODEL_MAX_TOKENS:-$SMALL_MODEL_MAX_TOKENS}"
    echo "  - Small Model Temperature: $SMALL_MODEL_TEMPERATURE"
    echo "  - Sample Size: $SAMPLE_SIZE"
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
        --mem-fraction-static 0.9 \
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

    # --- 运行 Python 评估脚本 ---
    echo "Starting evaluation script..."
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u no_proxy -u NO_PROXY \
        python3 -m ATTS.ref_conformal_per_question \
        --small_model_name "$SMALL_MODEL" \
        --eval_model_name "$EVAL_MODEL" \
        --dataset_name "$DATASET_NAME" \
        --ppl_array_path "$PPL_ARRAY_PATH" \
        --small_model_port "$SMALL_MODEL_PORT" \
        --eval_model_port "$EVAL_MODEL_PORT" \
        --sample_size "$SAMPLE_SIZE" \
        --max_concurrent "$MAX_CONCURRENT" \
        --small_model_max_tokens "${CONFIG_SMALL_MODEL_MAX_TOKENS:-$SMALL_MODEL_MAX_TOKENS}" \
        --small_model_temperature "$SMALL_MODEL_TEMPERATURE"
    
    # 检查评估脚本是否成功运行
    if [ $? -eq 0 ]; then
        echo "Evaluation script for this run finished successfully."
    else
        echo "Evaluation script for this run encountered an error."
    fi

    # --- 清理: 关闭服务器 ---
    echo "Killing SGLang servers with PIDs: $SMALL_MODEL_PID and $EVAL_MODEL_PID..."
    kill $SMALL_MODEL_PID $EVAL_MODEL_PID
    sleep 5 # 等待5秒，确保端口完全释放，避免下一轮启动失败

done

echo "All evaluation runs have been completed."
