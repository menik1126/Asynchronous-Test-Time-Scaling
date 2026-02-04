#!/bin/bash
#export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=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

#!/bin/bash

# --- 检查 wait-for-it.sh 脚本是否存在 ---
if [ ! -f "./wait-for-it.sh" ]; then
    echo "Error: wait-for-it.sh not found. Please download it first."
    echo "Run: wget https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh && chmod +x wait-for-it.sh"
    exit 1
fi

# --- 定义服务器的固定参数 ---
SGLANG_HOST="0.0.0.0"
SGLANG_PORT="60000"


CUDA_DEVICE=0 # 显卡设备号

# --- 定义评估脚本的固定参数 ---
EVAL_SCRIPT="evaluation/baseline.py"

# --- 定义所有要运行的配置数组 ---
# 每个元素包含四个参数: MODEL_NAME, TURNS, MAX_TOKENS, DATASET
CONFIGS=(
    # "nvidia/Llama-3.1-Nemotron-Nano-8B-v1 15 500 aime24"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 15 500 aime25"
    # "meta-llama/Llama-3.1-8B-Instruct 15 200 aime25"
    # "Qwen/Qwen2.5-7B-Instruct 15 100 math500"
    # "meta-llama/Llama-3.1-8B-Instruct 15 100 gpqa"
    # "meta-llama/Llama-3.1-8B-Instruct 15 100 math500"
    # "nvidia/Llama-3.1-Nemotron-Nano-8B-v1 15 200 math500"
    # "nvidia/Llama-3.1-Nemotron-Nano-8B-v1 15 200 gpqa"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B 15 200 math500"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B 15 500 gpqa"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B 15 500 aime24"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B 15 500 aime25"
    # "simplescaling/s1.1-32B 20 1000 math500"
    # "simplescaling/s1.1-32B 20 1000 aime25"
    # "simplescaling/s1.1-32B 20 1000 amc"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 15 500 olympiad"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B 15 500 olympiad"
    # "Qwen/Qwen2.5-7B-Instruct 15 500 olympiad"
    # "meta-llama/Llama-3.1-8B-Instruct 15 500 olympiad"
    # "Qwen/QwQ-32B 15 500 olympiad"
    # "simplescaling/s1.1-32B 15 500 olympiad"
)

# --- 循环遍历所有配置 ---
for config in "${CONFIGS[@]}"; do
    # 解析配置字符串
    read MODEL_NAME EVAL_TURNS EVAL_MAX_TOKENS EVAL_DATASET <<< "$config"

    # 根据配置生成动态日志文件名
    log_file_base=$(echo "$MODEL_NAME" | tr '/' '_')
    log_file_base="${log_file_base}_turns${EVAL_TURNS}_tokens${EVAL_MAX_TOKENS}_${EVAL_DATASET}"
    SGLANG_LOG_FILE="sglang_server_${log_file_base}1.log"
    EVAL_LOG_FILE="my_baseline_${log_file_base}1.log"

    echo "===================================================================================="
    echo "Starting a new run with:"
    echo "  - Model:           $MODEL_NAME"
    echo "  - Dataset:         $EVAL_DATASET"
    echo "  - Turns:           $EVAL_TURNS"
    echo "  - Max Tokens:      $EVAL_MAX_TOKENS"
    echo "===================================================================================="

    # --- 启动 SGLang 服务器 (使用数组传递参数) ---
    echo "Starting SGLang server with model: $MODEL_NAME on device $CUDA_DEVICE..."
    SGLANG_ARGS=(
        "--model-path" "$MODEL_NAME"
        "--tp" "1"
        "--mem-fraction-static" "0.8"
        "--host" "$SGLANG_HOST"
        "--port" "$SGLANG_PORT"
    )
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 -m sglang.launch_server "${SGLANG_ARGS[@]}" > "$SGLANG_LOG_FILE" 2>&1 &
    SGLANG_PID=$!
    echo "SGLang server started with PID: $SGLANG_PID"

    # --- 等待服务器启动并监听端口 ---
    echo "Waiting for SGLANG server to be ready on $SGLANG_HOST:$SGLANG_PORT..."
    ./wait-for-it.sh "$SGLANG_HOST:$SGLANG_PORT" --timeout=300 -- echo "SGLang server is up! Proceeding with evaluation..."

    # 检查等待是否成功
    if [ $? -ne 0 ]; then
        echo "SGLang server for $MODEL_NAME did not start within the timeout. Killing PID $SGLANG_PID. Exiting."
        kill $SGLANG_PID
        exit 1
    fi

    # --- 运行评估脚本 ---
    echo "Starting evaluation script..."
    python "$EVAL_SCRIPT" \
        --dataset_name "$EVAL_DATASET" \
        --turns "$EVAL_TURNS" \
        --small_model_max_tokens "$EVAL_MAX_TOKENS" \
        --small_model_name "$MODEL_NAME" \
        --sglang_port "$SGLANG_PORT" \
        > "$EVAL_LOG_FILE" 2>&1 &
    EVAL_PID=$!
    echo "Evaluation script started with PID: $EVAL_PID"

    # 等待评估脚本完成
    wait $EVAL_PID
    echo "Evaluation script for $MODEL_NAME on $EVAL_DATASET finished."

    # 在下一轮循环前杀死 SGLang 服务
    echo "Killing SGLANG server with PID $SGLANG_PID..."
    kill $SGLANG_PID
    sleep 5 # 等待5秒，确保端口完全释放，避免下一轮启动失败

done

echo "All evaluation runs have been completed."