#!/bin/bash
#SBATCH --job-name=specr_amc23_32b_greedy9          # Job name
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
DATA_DIR="/home/xiongjing/qj/"

# Create main log file for all output
timestamp=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG_FILE="${DATA_DIR}/specreason/logs/spec_reason_amc23_${timestamp}.log"

# Ensure log directory exists
mkdir -p "$(dirname "$MAIN_LOG_FILE")"

# Redirect all output to the main log file
exec 1> >(tee -a "$MAIN_LOG_FILE")
exec 2>&1

echo "=== SpecR Experiment Started at $(date) ==="
echo "Main log file: $MAIN_LOG_FILE"

# SpecR experiment configuration
DATASET_NAME="amc23"
JUDGE_SCHEME="greedy"
THRESHOLD=3 #9
NUM_REPEATS=16 #1 #16
BASE_MODEL_NAME="Qwen/QwQ-32B"  # "Qwen/QwQ-32B" or "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" or "NovaSky-AI/Sky-T1-32B-Preview"
SMALL_MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" or "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# ENGINE can be set from environment variable, otherwise default to "sglang"
ENGINE="${ENGINE:-sglang}" # "vllm" or "sglang"
OUTPUT_DIR="${DATA_DIR}/specreason/results/${ENGINE}/${JUDGE_SCHEME}_${THRESHOLD}_sample_${NUM_REPEATS}/${DATASET_NAME}/${BASE_MODEL_NAME}_${SMALL_MODEL_NAME}"
LOGFILE_DIR="${DATA_DIR}/specreason/logs"
TOKEN_BUDGET=8192
# Define the list of problem IDs - AMC23 has 83 problems (0-82)
ids1=($(seq 12 59))  # AMC23 all 83 problems

# Create engine-specific log directory
ENGINE_LOG_DIR="${LOGFILE_DIR}/${ENGINE}/${JUDGE_SCHEME}_${THRESHOLD}_sample_${NUM_REPEATS}/${DATASET_NAME}"
if [ ! -d "$ENGINE_LOG_DIR" ]; then
    mkdir -p "$ENGINE_LOG_DIR"
    echo "Created engine log directory: $ENGINE_LOG_DIR"
fi

# Run for the first set of problem IDs
for id in "${ids1[@]}"; do
    timestamp=$(date +"%Y%m%d_%H%M%S")
    logfile="${ENGINE_LOG_DIR}/problem_${id}_${timestamp}.log"
    echo "Running problem $id, parallelizing across the $NUM_REPEATS runs of a job, logfile $logfile"

    # Initialize an empty array to store process IDs
    pid_list=()

    for repeat_id in $(seq 0 $((NUM_REPEATS - 1))); do
        python spec_reason.py --dataset_name "$DATASET_NAME" --problem_id "$id" --repeat_id "$repeat_id" --score_threshold "${THRESHOLD}" --score_method "${JUDGE_SCHEME}" --token_budget "$TOKEN_BUDGET" --output_dir "$OUTPUT_DIR" >> "$logfile" 2>&1 &

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

echo "=== SpecR Experiment Completed at $(date) ==="
