#!/bin/bash
# 超快速HDF5文件收集脚本 - 使用移动操作
# 用法: ./fast_collect.sh [输入目录] [输出目录] [前缀]

INPUT_DIR="${1:-output}"
OUTPUT_DIR="${2:-output/collected}"
PREFIX="${3:-leapdata}"

echo "⚡ 快速移动HDF5文件: $INPUT_DIR -> $OUTPUT_DIR"

# 检查输入目录
[ ! -d "$INPUT_DIR" ] && { echo "❌ 目录不存在: $INPUT_DIR"; exit 1; }

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 一次性查找所有文件并按目录+ID排序
echo "🔍 扫描文件..."
files=($(find "$INPUT_DIR" -name "leap_action_*.hdf5" -type f | sort -V))

echo "📊 找到 ${#files[@]} 个文件"

# 快速移动
echo "⚡ 快速移动..."
counter=0
for file in "${files[@]}"; do
    new_name="${PREFIX}_${counter}.hdf5"
    if mv "$file" "$OUTPUT_DIR/$new_name"; then
        echo "✅ $counter: $new_name"
        ((counter++))
    else
        echo "❌ 移动失败: $file"
    fi
done

echo ""
echo "🎉 完成! 移动了 $counter 个文件到 $OUTPUT_DIR"
echo "📋 文件: ${PREFIX}_0.hdf5 到 ${PREFIX}_$((counter-1)).hdf5"

# 显示结果
ls -1 "$OUTPUT_DIR"/*.hdf5 2>/dev/null | wc -l | xargs echo "📄 总计:"
