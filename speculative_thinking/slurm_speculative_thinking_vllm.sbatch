#!/bin/bash


#SBATCH  -J r1
#SBATCH  -N 1
#SBATCH  --gres=gpu:2
#SBATCH  --ntasks=1
#SBATCH  --qos=high_priority
#SBATCH  --output=/home/xiongjing/qj/sglang-parallel-test-time-scaling/speculative_thinking/sbatch/GPQA_Qwen2.5-32B-Instruct-7B-Instruct_GPU1,2_%j.out    
#SBATCH  --error=/home/xiongjing/qj/sglang-parallel-test-time-scaling/speculative_thinking/sbatch/GPQA_Qwen2.5-32B-Instruct-7B-Instruct_GPU1,2_%j.out
#SBATCH  --time=7-00:00:00 
#SBATCH  --partition=LocalQ

# export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
# echo 1
source ./miniconda3/etc/profile.d/conda.sh

nvidia-smi
cd /home/xiongjing/qj/sglang-parallel-test-time-scaling/speculative_thinking #/home/xiongjing/speculative_thinking #/home/xiongjing/speculative_thinking/


# sh run_vllm_pass@k_aime24.sh
#sh run_vllm_test_aime25.sh
#sh run_vllm_test.sh
#sh run_vllm_test_math500.sh
sh run_vllm_test_GPQA.sh
