#!/bin/bash
# 激活虚拟环境
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=4,5
# export RAY_BACKEND_LOG_LEVEL=40
# export RAY_DISABLE_MEMORY_MONITOR=1

# 定义模型名字用于文件命名
MODEL_NAME="simplescaling-s1.1-32B-1.5B" #"DeepSeek-R1-Distill-Qwen-32B_1.5B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python /root/xj/sglang-parallel-test-time-scaling/speculative_thinking/skythought_evals/eval.py \
    --evals aime24 --n 16 --result-dir ./eval_vllm/aime24 \
    --tp 2  --output-file ./eval_vllm/aime24/${MODEL_NAME}_${TIMESTAMP}.txt --spe_config ./speculative/config/1b_32b_simplescaling-s1.1-32B-1.5B.yml \
    --start 0 --end 15 2>&1  | tee ./eval_vllm/aime24/vllm_test_${MODEL_NAME}_${TIMESTAMP}.log