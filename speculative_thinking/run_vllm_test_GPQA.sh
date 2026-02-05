#!/bin/bash
# 激活虚拟环境


# 设置临时目录


export CUDA_VISIBLE_DEVICES=1,2 #6,7
export NCCL_P2P_DISABLE=1
# export RAY_BACKEND_LOG_LEVEL=40
# export RAY_DISABLE_MEMORY_MONITOR=1

# 定义模型名字用于文件命名
MODEL_NAME="Qwen2.5-32B-Instruct-7B-Instruct" #"DeepSeek-R1-Distill-Qwen-32B_1.5B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python /home/xiongjing/qj/sglang-parallel-test-time-scaling/speculative_thinking/skythought_evals/eval.py \
    --evals gpqa_diamond --n 16 --result-dir ./eval_vllm/gpqa \
    --tp 2  --output-file ./eval_vllm/gpqa/${MODEL_NAME}_${TIMESTAMP}.txt --spe_config ./speculative/config/1b_32b_Qwen2.5-32B-Instruct-7B-Instruct.yml \
    --start 0 --end 60 2>&1  | tee ./eval_vllm/gpqa/vllm_test_${MODEL_NAME}_${TIMESTAMP}.log