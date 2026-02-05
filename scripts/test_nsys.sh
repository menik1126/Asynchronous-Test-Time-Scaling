#!/bin/bash

# 简单的 nsys 测试脚本
echo "测试 Nsight Systems 功能..."

# 创建测试输出目录
TEST_DIR="./test_nsys_$(date +%H%M%S)"
mkdir -p "$TEST_DIR"

echo "使用 nsys profile 运行简单测试..."
nsys profile --stats=true --force-overwrite=true \
    --output="$TEST_DIR/test_profile" \
    --duration=10 \
    sleep 5

echo "检查生成的文件..."
ls -la "$TEST_DIR"

if [ -f "$TEST_DIR/test_profile.qdrep" ]; then
    echo "✅ nsys profile 成功生成数据文件"
    echo "生成统计报告..."
    nsys stats "$TEST_DIR/test_profile.qdrep" > "$TEST_DIR/test_stats.txt"
    echo "统计报告内容:"
    cat "$TEST_DIR/test_stats.txt"
else
    echo "❌ nsys profile 没有生成数据文件"
    echo "检查可能的错误..."
    ls -la "$TEST_DIR"
fi

echo "测试完成，清理测试文件..."
rm -rf "$TEST_DIR"
