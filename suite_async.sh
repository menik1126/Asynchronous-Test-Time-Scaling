#!/bin/bash

# ==============================================================================
# SGLang 服务器和评估脚本运行器
# 此脚本将循环启动两个 SGLang 服务器，然后执行 ref_conformal.py 脚本进行评估
# 并在每次循环结束后关闭服务器。
# ==============================================================================
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=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

# --- 检查 wait-for-it.sh 脚本是否存在 ---
if [ ! -f "./wait-for-it.sh" ]; then
    echo "Error: wait-for-it.sh not found. Please download it first."
    echo "Run: wget https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh && chmod +x wait-for-it.sh"
    exit 1
fi

# --- 配置变量 ---
# 你可以根据需要修改这些参数。
SGLANG_HOST="0.0.0.0"
SMALL_MODEL_PORT=51101
EVAL_MODEL_PORT=51100

# Server CUDA devices
SMALL_MODEL_DEVICE="2,3"
# 注意：评估模型使用了两个CUDA设备
EVAL_MODEL_DEVICES="2,3"

# 评估轮次，如果配置数组中未指定，将使用此默认值
DEFAULT_TURNS=15

# --- 定义所有要运行的配置数组 ---
# 每个元素包含七个参数: SMALL_MODEL, EVAL_MODEL, DATASET_NAME, PPL_ARRAY_PATH, SMALL_MODEL_MAX_TOKENS, EVAL_MODEL_MAX_TOKENS, TURNS
CONFIGS=(
    # "nvidia/Llama-3.1-Nemotron-Nano-8B-v1 Qwen/QwQ-32B aime24 /zju_0038/xj/sglang-parallel-test-time-scaling/ppls_aime24_64_qwq_32_nn_8.npy 500 500 15"
    # "nvidia/Llama-3.1-Nemotron-Nano-8B-v1 Qwen/QwQ-32B aime25 /zju_0038/xj/sglang-parallel-test-time-scaling/ppls_aime25_64_qwq_32_nn_8.npy 500 500 15"
    # "nvidia/Llama-3.1-Nemotron-Nano-8B-v1 Qwen/QwQ-32B math500 /zju_0038/xj/sglang-parallel-test-time-scaling/ppls_math500_64_qwq_32_nn_8.npy 200 200 15"
    # "nvidia/Llama-3.1-Nemotron-Nano-8B-v1 Qwen/QwQ-32B gpqa /zju_0038/xj/sglang-parallel-test-time-scaling/ppls_gpqa_64_qwq_32_nn_8.npy 200 200 15"
    # "Qwen/Qwen2.5-7B-Instruct Qwen/QwQ-32B aime25 /zju_0038/xj/sglang-parallel-test-time-scaling/ppls_aime25_64_qwq_32_qwen_7.npy 200 200 15"
    # "meta-llama/Llama-3.1-8B-Instruct Qwen/QwQ-32B aime25 /zju_0038/xj/sglang-parallel-test-time-scaling/ppls_aime25_64_qwq_32_llama_8.npy 200 200 15"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B Qwen/QwQ-32B aime25 /zju_0038/xj/sglang-parallel-test-time-scaling/ppls_aime25_64_qwq_32_r1_1.npy 500 500 15"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B Qwen/QwQ-32B gpqa /zju_0038/xj/sglang-parallel-test-time-scaling/ppls_gpqa_64_qwq_32_r1_1.npy 200 200 15"
    # "nvidia/Llama-3.1-Nemotron-Nano-8B-v1 simplescaling/s1.1-32B aime24 /zju_0038/xj/sglang-parallel-test-time-scaling/ppls_aime24_64_s1_32_nn_8.npy 500 500 15"
    # "nvidia/Llama-3.1-Nemotron-Nano-8B-v1 simplescaling/s1.1-32B aime25 /zju_0038/xj/sglang-parallel-test-time-scaling/ppls_aime25_64_s1_32_nn_8.npy 500 500 15"
    # "nvidia/Llama-3.1-Nemotron-Nano-8B-v1 simplescaling/s1.1-32B math500 /zju_0038/xj/sglang-parallel-test-time-scaling/ppls_math500_64_s1_32_nn_8.npy 200 200 15"
    # "nvidia/Llama-3.1-Nemotron-Nano-8B-v1 simplescaling/s1.1-32B gpqa /zju_0038/xj/sglang-parallel-test-time-scaling/ppls_gpqa_64_s1_32_nn_8.npy 200 200 15"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B simplescaling/s1.1-32B aime25 /zju_0038/xj/sglang-parallel-test-time-scaling/ppls_aime25_64_s1_32_r1_1.npy 500 500 15"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B simplescaling/s1.1-32B gpqa /zju_0038/xj/sglang-parallel-test-time-scaling/ppls_gpqa_64_s1_32_r1_1.npy 300 300 40"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B simplescaling/s1.1-32B gpqa /zju_0038/xj/sglang-parallel-test-time-scaling/ppls_gpqa_64_s1_32_rllama_8.npy 300 300 40"
    # "Qwen/Qwen2.5-7B-Instruct simplescaling/s1.1-32B aime25 /zju_0038/xj/sglang-parallel-test-time-scaling/ppls_aime25_64_s1_32_qwen_7.npy 100 100 15"
    # "meta-llama/Llama-3.1-8B-Instruct simplescaling/s1.1-32B aime25 /zju_0038/xj/sglang-parallel-test-time-scaling/ppls_aime25_64_s1_32_llama_8.npy 100 100 15"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B simplescaling/s1.1-32B math500 /zju_0038/xj/sglang-parallel-test-time-scaling/ppls_math500_64_s1_32_rllama_8.npy 200 200 15"
)

# --- 循环遍历所有配置 ---
for config in "${CONFIGS[@]}"; do
    # 解析配置字符串
    read SMALL_MODEL EVAL_MODEL DATASET_NAME PPL_ARRAY_PATH SMALL_MODEL_MAX_TOKENS EVALATOR_MAX_TOKENS EVAL_TURNS <<< "$config"

    # 根据配置生成动态日志和输出目录名
    small_model_name_base=$(echo "$SMALL_MODEL" | tr '/' '_')
    eval_model_name_base=$(echo "$EVAL_MODEL" | tr '/' '_')
    run_identifier="${small_model_name_base}_${eval_model_name_base}_${DATASET_NAME}"
    
    # 定义输出目录路径
    OUTPUT_DIR="./results/$run_identifier"
    
    SMALL_MODEL_LOG="$OUTPUT_DIR/server_${run_identifier}.log"
    EVAL_MODEL_LOG="$OUTPUT_DIR/eval_${run_identifier}.log"
    EVAL_SCRIPT_LOG="$OUTPUT_DIR/async_eval_${run_identifier}.log"

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
    CUDA_VISIBLE_DEVICES=$SMALL_MODEL_DEVICE python3 -m sglang.launch_server \
        --model-path "$SMALL_MODEL" \
        --tp 2 \
        --mem-fraction-static 0.6 \
        --host "$SGLANG_HOST" \
        --port "$SMALL_MODEL_PORT" > "$SMALL_MODEL_LOG" 2>&1 &
    SMALL_MODEL_PID=$!
    echo "Small model server started with PID: $SMALL_MODEL_PID"

    echo "Starting SGLang server for evaluation model ($EVAL_MODEL)..."
    CUDA_VISIBLE_DEVICES=$EVAL_MODEL_DEVICES python3 -m sglang.launch_server \
        --model-path "$EVAL_MODEL" \
        --tp 2 \
        --mem-fraction-static 0.7 \
        --host "$SGLANG_HOST" \
        --port "$EVAL_MODEL_PORT" > "$EVAL_MODEL_LOG" 2>&1 &
    EVAL_MODEL_PID=$!
    echo "Evaluation model server started with PID: $EVAL_MODEL_PID"

    # --- 等待服务器启动并监听端口 ---
    echo "Waiting for SGLang servers to be ready..."
    ./wait-for-it.sh "$SGLANG_HOST:$SMALL_MODEL_PORT" --timeout=300 -- echo "Small model server is up."
    if [ $? -ne 0 ]; then
        echo "Small model server did not start. Killing PIDs $SMALL_MODEL_PID and $EVAL_MODEL_PID. Exiting."
        kill $SMALL_MODEL_PID $EVAL_MODEL_PID
        exit 1
    fi
    ./wait-for-it.sh "$SGLANG_HOST:$EVAL_MODEL_PORT" --timeout=300 -- echo "Evaluation model server is up."
    if [ $? -ne 0 ]; then
        echo "Evaluation model server did not start. Killing PIDs $SMALL_MODEL_PID and $EVAL_MODEL_PID. Exiting."
        kill $SMALL_MODEL_PID $EVAL_MODEL_PID
        exit 1
    fi

    # --- 运行 Python 评估脚本，并将输出重定向到日志文件 ---
    echo "Starting evaluation script in the background, output will be logged to $EVAL_SCRIPT_LOG..."
    python3 evaluation/ref_async.py \
        --small_model_name "$SMALL_MODEL" \
        --eval_model_name "$EVAL_MODEL" \
        --dataset_name "$DATASET_NAME" \
        --ppl_array_path "$PPL_ARRAY_PATH" \
        --small_model_max_tokens "$SMALL_MODEL_MAX_TOKENS" \
        --evalator_max_tokens "$EVALATOR_MAX_TOKENS" \
        --turns "$EVAL_TURNS" \
        --small_model_port "$SMALL_MODEL_PORT" \
        --eval_model_port "$EVAL_MODEL_PORT" \
        --output_dir "$OUTPUT_DIR" > "$EVAL_SCRIPT_LOG" 2>&1 &
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
