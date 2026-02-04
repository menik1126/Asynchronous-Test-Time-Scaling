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

# 使用 Nsight Systems 进行系统级性能分析
echo "使用 Nsight Systems 进行系统级性能分析..." | tee -a "$LOG_FILE"
nsys profile --stats=true --force-overwrite=true \
    --output="$OUTPUT_DIR/nsight_profile" \
    --duration=600 \
    --gpu-metrics-device=all \
    --trace=cuda,nvtx \
    python3 ref_async_fix_budget.py \
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

# 同时运行 cProfile 进行 Python 函数级分析
echo "同时运行 cProfile 进行 Python 函数级分析..." | tee -a "$LOG_FILE"
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

# 生成 cProfile 报告
python3 -c "
import pstats
stats = pstats.Stats('$OUTPUT_DIR/profile.prof')
stats.sort_stats('cumulative').print_stats(10)
" > "$OUTPUT_DIR/top10_functions.txt" 2>&1 | tee -a "$LOG_FILE"

# 生成 Nsight Systems 报告
echo "生成 Nsight Systems 报告..." | tee -a "$LOG_FILE"

# 检查不同格式的 Nsight Systems 文件
NSIGHT_FILE=""
if [ -f "$OUTPUT_DIR/nsight_profile.qdrep" ]; then
    NSIGHT_FILE="$OUTPUT_DIR/nsight_profile.qdrep"
elif [ -f "$OUTPUT_DIR/nsight_profile.nsys-rep" ]; then
    NSIGHT_FILE="$OUTPUT_DIR/nsight_profile.nsys-rep"
elif [ -f "$OUTPUT_DIR/nsight_profile.sqlite" ]; then
    NSIGHT_FILE="$OUTPUT_DIR/nsight_profile.sqlite"
fi

if [ -n "$NSIGHT_FILE" ]; then
    echo "找到 Nsight Systems 文件: $NSIGHT_FILE" | tee -a "$LOG_FILE"
    nsys stats "$NSIGHT_FILE" > "$OUTPUT_DIR/nsight_stats.txt" 2>&1 | tee -a "$LOG_FILE"
    
    # 生成详细的 GPU 指标报告
    echo "生成详细 GPU 指标报告..." | tee -a "$LOG_FILE"
    nsys stats --report gputrace "$NSIGHT_FILE" > "$OUTPUT_DIR/gpu_trace.txt" 2>&1 | tee -a "$LOG_FILE"
    
    # 如果安装了 nsys-ui，可以生成 SQLite 导出
    if command -v nsys-ui &> /dev/null && [ "$NSIGHT_FILE" != "*.sqlite" ]; then
        echo "生成 Nsight Systems SQLite 导出..." | tee -a "$LOG_FILE"
        nsys export --type sqlite --output "$OUTPUT_DIR/nsight_profile.sqlite" "$NSIGHT_FILE" 2>&1 | tee -a "$LOG_FILE"
    fi
else
    echo "警告: 没有找到 Nsight Systems 数据文件，跳过报告生成" | tee -a "$LOG_FILE"
    echo "Nsight Systems 数据文件不存在" > "$OUTPUT_DIR/nsight_stats.txt"
    echo "Nsight Systems 数据文件不存在" > "$OUTPUT_DIR/gpu_trace.txt"
fi

# 服务器保持运行，无需清理

echo "完成! 结果在: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Python 性能报告: $OUTPUT_DIR/top10_functions.txt" | tee -a "$LOG_FILE"
echo "Nsight Systems 统计报告: $OUTPUT_DIR/nsight_stats.txt" | tee -a "$LOG_FILE"
echo "GPU 追踪报告: $OUTPUT_DIR/gpu_trace.txt" | tee -a "$LOG_FILE"
echo "Nsight Systems 原始数据: $OUTPUT_DIR/nsight_profile.*" | tee -a "$LOG_FILE"
echo "完整日志: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "查看 Nsight Systems 报告的方法:" | tee -a "$LOG_FILE"
echo "1. 系统统计: cat $OUTPUT_DIR/nsight_stats.txt" | tee -a "$LOG_FILE"
echo "2. GPU 追踪: cat $OUTPUT_DIR/gpu_trace.txt" | tee -a "$LOG_FILE"
echo "3. 图形界面: nsys-ui \$OUTPUT_DIR/nsight_profile.*" | tee -a "$LOG_FILE"
