# GPU配置示例：
# 方式1：使用前3个GPU (0,1,2)
# export CUDA_VISIBLE_DEVICES=1,2
# 方式2：使用指定的GPU
# export CUDA_VISIBLE_DEVICES=1,2,3,4
# 方式3：使用单GPU
# export CUDA_VISIBLE_DEVICES=0

# 可选：禁用P2P通信（在某些多GPU配置下可能需要）
export NCCL_P2P_DISABLE=1

python /home/xiongjing/speculative_thinking/skythought_evals/eval.py \
    --evals aime25 --n 16 --result-dir ./eval_vllm/aime25 \
    --tp 3 --output-file ./eval_vllm/aime25/1b_32b.txt --spe_config ./speculative/config/1b_32b.yml --temperatures 0.6

# 说明：
# --tp 3 表示使用3个GPU进行张量并行
# 确保CUDA_VISIBLE_DEVICES中的GPU数量 >= --tp的值