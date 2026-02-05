export CUDA_VISIBLE_DEVICES=1,2
# export RAY_BACKEND_LOG_LEVEL=40
# export RAY_DISABLE_MEMORY_MONITOR=1

# 定义模型名字用于文件命名
MODEL_NAME="Qwen2.5-32B-Instruct-7B-Instruct" #"Skywork/Skywork-OR1-32B" #"simplescaling-s1.1-32B-1.5B" #"DeepSeek-R1-Distill-Qwen-32B_1.5B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python /home/xiongjing/qj/sglang-parallel-test-time-scaling/speculative_thinking/skythought_evals/eval.py \
    --evals aime25 --n 16 --result-dir ./eval_vllm/aime25 \
    --tp 2  --output-file ./eval_vllm/aime25/${MODEL_NAME}_${TIMESTAMP}.txt --spe_config ./speculative/config/1b_32b_Qwen2.5-32B-Instruct-7B-Instruct.yml \
    --start 0 --end 15 2>&1  | tee ./eval_vllm/aime25/vllm_test_${MODEL_NAME}_${TIMESTAMP}.log