#!/bin/bash
#SBATCH --job-name=specr_aime2024_32b_greedy9          # Job name
#SBATCH --output="/home/rp2773/slurm_logs/%A.out"       # Standard output log
#SBATCH --error="/home/rp2773/slurm_logs/%A.err"         # Standard error log
#SBATCH --ntasks=1                            # Number of tasks (1 process)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --gres=gpu:4                        # Number of GPUs to allocate
##SBATCH --constraint="gpu80"
#SBATCH --time=7:00:00                        # Time limit (24 hours max)
#SBATCH --mem=100G                            # Memory allocation (adjust as needed)
#SBATCH --mail-user=ruipan@princeton.edu  # Your email
#SBATCH --mail-type=ALL  # Options: BEGIN, END, FAIL, REQUEUE, TIME_LIMIT, etc.
#SBATCH --partition=pli
#SBATCH --account=specreason

# CLUSTER="ravi"
CLUSTER="della"
DATA_DIR="/home/xiongjing/qj"
# initialization: set environment variables based on the cluster
# if [ "$CLUSTER" = "ravi" ]; then
#     DATA_DIR="/home/ruipan/data2"
# elif [ "$CLUSTER" = "della" ]; then
#     DATA_DIR="/scratch/gpfs/rp2773/data"
#     export HF_HOME="/scratch/gpfs/rp2773/hf_cache"
#     export HF_HUB_OFFLINE=1
#     export HF_DATASETS_OFFLINE=1
#     export TRANSFORMERS_OFFLINE=1
#     source /scratch/gpfs/rp2773/miniconda3/etc/profile.d/conda.sh
#     nvidia-smi
# else
#     echo "Error: CLUSTER must be either 'ravi' or 'della'"
#     exit 1
# fi
# conda activate specreason

# SpecR experiment configuration
DATASET_NAME="aime"
AIME_YEAR="2024"
JUDGE_SCHEME="greedy"
THRESHOLD=3 #9
NUM_REPEATS=16 #1 #16
BASE_MODEL_NAME="Qwen/QwQ-32B-renew" #"Qwen/Qwen2.5-32B-Instruct" #"Skywork/Skywork-OR1-32B" #"simplescaling/s1.1-32B"  # "Qwen/QwQ-32B" or "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" or "NovaSky-AI/Sky-T1-32B-Preview"
SMALL_MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-renew" #"Qwen/Qwen2.5-7B-Instruct" #"Skywork/Skywork-OR1-7B" #"simplescaling/s1.1-7B"   #"deepseek-ai/DeepSeek-R1-Distill-Llama-8B" #"meta-llama/Llama-3.1-8B-Instruct" #"Qwen/Qwen2.5-7B-Instruct" #"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" or "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
#BASE_MODEL_ABBRV="Qwen-32B"  # "Qwen-32B" or "deepseek-70B" or "NovaSky-Preview"
#SMALL_MODEL_ABBRV="deepseek-1.5B"  # "deepseek-1.5B" or "deepseek-7B"
# ENGINE can be set from environment variable, otherwise default to "sglang"
export SGLANG_PORT_32B=40000
export SGLANG_PORT_1_5B=40001
ENGINE="${ENGINE:-sglang}" # "vllm" or "sglang"
OUTPUT_DIR="${DATA_DIR}/specreason/results/${ENGINE}/${JUDGE_SCHEME}_${THRESHOLD}_sample_${NUM_REPEATS}/${DATASET_NAME}${AIME_YEAR}/${BASE_MODEL_NAME}_${SMALL_MODEL_NAME}"
LOGFILE_DIR="${DATA_DIR}/specreason/logs"
# TP_SIZE=4  # applies for both models
TOKEN_BUDGET=8192
# Define the list of problem IDs
# ids1=($(seq 0 99))  # MATH and GPQA

ids1=($(seq 60 74))  # AIME

# Create engine-specific log directory
ENGINE_LOG_DIR="${LOGFILE_DIR}/${ENGINE}/${JUDGE_SCHEME}_${THRESHOLD}_sample_${NUM_REPEATS}/${DATASET_NAME}${AIME_YEAR}"
if [ ! -d "$ENGINE_LOG_DIR" ]; then
    mkdir -p "$ENGINE_LOG_DIR"
    echo "Created engine log directory: $ENGINE_LOG_DIR"
fi

# for dir in "$OUTPUT_DIR" "$LOGFILE_DIR"; do
#     if [ ! -d "$dir" ]; then
#         mkdir -p "$dir"
#         echo "Created directory: $dir"
#     fi
# done

# # Function to check if a server is ready
# wait_for_server() {
#     local port=$1
#     while true; do
#         # Try to connect to the server
#         curl -s http://localhost:$port/get_model_info > /dev/null
#         if [ $? -eq 0 ]; then
#             echo "Server on port $port is ready!"
#             break
#         else
#             echo "Waiting for server on port $port to start..."
#             sleep 10  # Wait 10 seconds before retrying
#         fi
#     done
# }

# # launch 32b model and 1.5b model one by one
# vllm serve "$BASE_MODEL_NAME" --dtype auto -tp "$TP_SIZE" --max_model_len 16384 --gpu-memory-utilization 0.75 --enable-prefix-caching --port 30000 &
# VLLM_BASE_PID=$!
# wait_for_server 30000
# nvidia-smi
# vllm serve "$SMALL_MODEL_NAME" --dtype auto -tp "$TP_SIZE" --max_model_len 16384 --gpu-memory-utilization 0.15 --enable-prefix-caching --port 30001 &
# VLLM_SMALL_PID=$!
# wait_for_server 30001


# Run for the first set of problem IDs
for id in "${ids1[@]}"; do
    timestamp=$(date +"%Y%m%d_%H%M%S")
    logfile="${LOGFILE_DIR}/${ENGINE}/${JUDGE_SCHEME}_${THRESHOLD}_sample_${NUM_REPEATS}/${DATASET_NAME}${AIME_YEAR}/${timestamp}.log"
    echo "Running problem $id, parallelizing across the $NUM_REPEATS runs of a job, logfile $logfile"

    # Initialize an empty array to store process IDs
    pid_list=()

    for repeat_id in $(seq 0 $((NUM_REPEATS - 1))); do
        python spec_reason.py --dataset_name "$DATASET_NAME" --aime_year "$AIME_YEAR" --problem_id "$id" --repeat_id "$repeat_id" --score_threshold "${THRESHOLD}" --score_method "${JUDGE_SCHEME}" --token_budget "$TOKEN_BUDGET" --output_dir "$OUTPUT_DIR" >> "$logfile" 2>&1 &

        # Capture the PID of the last background process and store it in the list
        pid_list+=($!)
    done

    # Explicitly wait for only the processes in pid_list
    for pid in "${pid_list[@]}"; do
        wait "$pid"
    done

    # Print confirmation message
    echo "Problem $id completed"

    # Sleep for 2s
    sleep 2
done

kill $VLLM_BASE_PID
kill $VLLM_SMALL_PID
