#!/bin/bash

# 简化版Profiling测试脚本

# 配置
SMALL_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" #"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
EVAL_MODEL="Qwen/QwQ-32B"
DATASET="aime25"

# 端口和设备
SMALL_PORT=40001  # 小模型端口
EVAL_PORT=40000   # 大模型端口
SMALL_GPU="1"
EVAL_GPU="2,3"

# 测试参数
TURNS=3
TOKENS=150
REPEATS=4
TAKEOVER_BUDGET=5  # 每个turn的接管预算 (减少到5，便于测试)
DEBUG_MODE= true   # 设置为true启用debug模式

# 输出目录
OUTPUT_DIR="./results/profiling_$(date +%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# 日志文件
LOG_FILE="$OUTPUT_DIR/profiling.log"

echo "启动profiling测试: $DATASET (turns=$TURNS, repeats=$REPEATS)" | tee -a "$LOG_FILE"
echo "使用已启动的服务器: 小模型端口$SMALL_PORT, 大模型端口$EVAL_PORT" | tee -a "$LOG_FILE"

# 运行profiling
echo "开始性能分析..." | tee -a "$LOG_FILE"
# 构建debug参数
DEBUG_FLAG=""
if [ "$DEBUG_MODE" = "true" ]; then
    DEBUG_FLAG="--debug_mode"
fi

python3 -m cProfile -o "$OUTPUT_DIR/profile.prof" ref_async_fix_budget.py \
    --small_model_name "$SMALL_MODEL" \
    --eval_model_name "$EVAL_MODEL" \
    --dataset_name "$DATASET" \
    --turns "$TURNS" \
    --small_model_max_tokens "$TOKENS" \
    --evalator_max_tokens "$TOKENS" \
    --repeats "$REPEATS" \
    --small_model_port "$SMALL_PORT" \
    --eval_model_port "$EVAL_PORT" \
    --output_dir "$OUTPUT_DIR" \
    --takeover_budget "$TAKEOVER_BUDGET" \
    $DEBUG_FLAG 2>&1 | tee -a "$LOG_FILE"

# 生成简单报告
echo "生成性能报告..." | tee -a "$LOG_FILE"
python3 -c "
import pstats
stats = pstats.Stats('$OUTPUT_DIR/profile.prof')
stats.sort_stats('cumulative').print_stats(10)
" > "$OUTPUT_DIR/top10_functions.txt" 2>&1 | tee -a "$LOG_FILE"

# 服务器保持运行，无需清理

echo "完成! 结果在: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "性能报告: $OUTPUT_DIR/top10_functions.txt" | tee -a "$LOG_FILE"
echo "完整日志: $LOG_FILE" | tee -a "$LOG_FILE"
