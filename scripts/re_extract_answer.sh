#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export OPENAI_API_KEY="sk-f7Oh115pfz6REQeFKesLFOhrk85Yd8ySvnqmRDZ08oDT8nyr"
export OPENAI_BASE_URL="https://chatapi.littlewheat.com/v1"
export OPENAI_MODEL="gpt-5.2"

INPUT_DIR="$PROJECT_DIR/results/deepseek-ai_DeepSeek-R1-Distill-Llama-8B_Qwen_QwQ-32B_math500_sm500_em500_t15_smt0.8_sct0.8_emt0.8_smc16_emc4_ct1"

DATASET_NAME="math500"
REPEATS=16

# 加 --dry_run 只打印不修改文件，去掉则正式写入
cd "$PROJECT_DIR"
python -m ATTS.re_extract \
    --input_dir "$INPUT_DIR" \
    --dataset_name "$DATASET_NAME" \
    --repeats "$REPEATS" \
    --concurrency 32 \
