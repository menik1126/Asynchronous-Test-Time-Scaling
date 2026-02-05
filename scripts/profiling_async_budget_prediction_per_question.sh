#!/bin/bash

# PPL计算脚本

# ==============================================================================
# 超参数配置
# ==============================================================================

# 模型配置
SMALL_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" #"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
EVAL_MODEL="Qwen/QwQ-32B"
DATASET="aime25"

# 采样配置
REPEATS=16      # 重复次数

# 端口配置
SMALL_PORT=40001  # 小模型端口
EVAL_PORT=40000   # 大模型端口

# PPL文件路径
PPL_ARRAY_PATH="/home/xiongjing/qj/sglang-parallel-test-time-scaling/ppls_aime25_64_qwq_32_r1_1_500_per_question.npy"

# 预测任务超参数
TURNS=1                   # 最大轮次
SMALL_MODEL_MAX_TOKENS=700 # 小模型最大token数
EVAL_MODEL_MAX_TOKENS=150  # 大模型最大token数
TAKEOVER_BUDGET=4          # 接管预算

# 自动计算百分位数阈值
PERCENTILE_THRESHOLD=$(python3 -c "print(1 - ($TAKEOVER_BUDGET / $REPEATS))")

# ==============================================================================
# 输出配置
# ==============================================================================

# 输出目录
OUTPUT_DIR="./results/prediction_$(date +%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# 日志文件
LOG_FILE="$OUTPUT_DIR/prediction.log"

echo "启动预测任务: $DATASET (采样数量: $REPEATS)" | tee -a "$LOG_FILE"
echo "使用已启动的服务器: 小模型端口$SMALL_PORT, 大模型端口$EVAL_PORT" | tee -a "$LOG_FILE"
echo "PPL文件路径: $PPL_ARRAY_PATH" | tee -a "$LOG_FILE"
echo "预测超参数: 轮次=$TURNS, 小模型token=$SMALL_MODEL_MAX_TOKENS, 大模型token=$EVAL_MODEL_MAX_TOKENS" | tee -a "$LOG_FILE"
echo "接管配置: 预算=$TAKEOVER_BUDGET, 自动计算百分位数阈值=$PERCENTILE_THRESHOLD (${TAKEOVER_BUDGET}/${REPEATS})" | tee -a "$LOG_FILE"

# 检查PPL文件是否存在
if [ ! -f "$PPL_ARRAY_PATH" ]; then
    echo "❌ PPL文件不存在: $PPL_ARRAY_PATH" | tee -a "$LOG_FILE"
    echo "请先运行PPL计算生成PPL文件" | tee -a "$LOG_FILE"
    exit 1
fi

echo "✅ PPL文件存在，开始预测任务..." | tee -a "$LOG_FILE"

# 运行预测任务
echo "开始预测任务..." | tee -a "$LOG_FILE"
python3 -m ATTS.ref_async_budget_prediction_per_question \
    --small_model_name "$SMALL_MODEL" \
    --eval_model_name "$EVAL_MODEL" \
    --dataset_name "$DATASET" \
    --turns "$TURNS" \
    --small_model_max_tokens "$SMALL_MODEL_MAX_TOKENS" \
    --evalator_max_tokens "$EVAL_MODEL_MAX_TOKENS" \
    --repeats "$REPEATS" \
    --small_model_port "$SMALL_PORT" \
    --eval_model_port "$EVAL_PORT" \
    --output_dir "$OUTPUT_DIR" \
    --takeover_budget "$TAKEOVER_BUDGET" \
    --ppl_array_path "$PPL_ARRAY_PATH" \
    --percentile_threshold "$PERCENTILE_THRESHOLD" 2>&1 | tee -a "$LOG_FILE"

echo "完成! 结果在: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "预测结果目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "完整日志: $LOG_FILE" | tee -a "$LOG_FILE"
