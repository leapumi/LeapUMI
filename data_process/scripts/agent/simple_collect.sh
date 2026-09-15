#!/bin/bash
# 简单的HDF5文件收集脚本
# 用法: ./simple_collect.sh [输入目录] [输出目录] [前缀]

# 默认参数
INPUT_DIR="${1:-output}"
OUTPUT_DIR="${2:-output/collected}"
PREFIX="${3:-leapdata}"

echo "📦 收集HDF5文件: $INPUT_DIR -> $OUTPUT_DIR (前缀: $PREFIX)"

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 目录不存在: $INPUT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 查找并排序所有HDF5文件
echo "🔍 查找文件..."
temp_list=$(mktemp)

# 查找所有leap_action_*.hdf5文件，提取信息并排序
find "$INPUT_DIR" -name "leap_action_*.hdf5" -type f | while read -r file; do
    dir_name=$(basename "$(dirname "$file")")
    filename=$(basename "$file")
    if [[ $filename =~ leap_action_([0-9]+)\.hdf5 ]]; then
        original_id="${BASH_REMATCH[1]}"
        echo "$dir_name|$original_id|$file"
    fi
done | sort -t'|' -k1,1 -k2,2n > "$temp_list"

# 检查是否找到文件
file_count=$(wc -l < "$temp_list")
if [ "$file_count" -eq 0 ]; then
    echo "❌ 没有找到HDF5文件"
    rm -f "$temp_list"
    exit 1
fi

echo "📊 找到 $file_count 个文件"

# 移动和重命名文件
echo "🔄 移动文件..."
new_id=0
success=0

# 创建索引文件
index_file="$OUTPUT_DIR/file_index.txt"
echo "# HDF5文件重命名索引" > "$index_file"
echo "# 生成时间: $(date)" >> "$index_file"
echo "# 格式: 新文件名 <- 原目录/原文件名" >> "$index_file"
echo "" >> "$index_file"

while IFS='|' read -r dir_name original_id file_path; do
    new_filename="${PREFIX}_${new_id}.hdf5"
    if mv "$file_path" "$OUTPUT_DIR/$new_filename"; then
        echo "✅ $new_id: $new_filename <- $dir_name/leap_action_$original_id.hdf5"
        echo "$new_filename <- $dir_name/leap_action_$original_id.hdf5" >> "$index_file"
        ((success++))
    else
        echo "❌ $new_id: 移动失败"
    fi
    ((new_id++))
done < "$temp_list"

# 清理
rm -f "$temp_list"

# 结果统计
echo ""
echo "🎉 完成! 成功移动: $success/$file_count"
echo "📁 文件保存在: $OUTPUT_DIR"
echo "📋 文件范围: ${PREFIX}_0.hdf5 到 ${PREFIX}_$((success-1)).hdf5"
echo "📄 索引文件: $index_file"

# 显示文件列表
echo ""
echo "📂 输出文件:"
ls -lh "$OUTPUT_DIR"/*.hdf5 2>/dev/null | head -10
