#!/bin/bash
# 超快速HDF5文件收集脚本
# 直接移动文件，不复制，速度最快

INPUT_DIR="${1:-output}"
OUTPUT_DIR="${2:-output/collected}"
PREFIX="${3:-leapdata}"

echo "⚡ 超快速收集: $INPUT_DIR -> $OUTPUT_DIR"

# 检查和创建目录
[ ! -d "$INPUT_DIR" ] && { echo "❌ 目录不存在: $INPUT_DIR"; exit 1; }
mkdir -p "$OUTPUT_DIR"

# 生成文件列表并移动
echo "🔍 扫描和移动文件..."

# 创建索引文件
index_file="$OUTPUT_DIR/file_index.txt"
{
    echo "# HDF5文件重命名索引"
    echo "# 生成时间: $(date)"
    echo "# 格式: 新文件名 <- 原目录/原文件名"
    echo ""
} > "$index_file"

# 一次性处理：查找、排序、移动
new_id=0
success=0

find "$INPUT_DIR" -name "leap_action_*.hdf5" -type f -printf "%h|%f|%p\n" | \
sed 's|.*/\([^/]*\)|\1|' | \
sed 's/leap_action_\([0-9]*\)\.hdf5/\1/' | \
awk -F'|' '{print $1"|"$2"|"$3}' | \
sort -t'|' -k1,1 -k2,2n | \
while IFS='|' read -r dir_name original_id file_path; do
    new_filename="${PREFIX}_${new_id}.hdf5"
    
    if mv "$file_path" "$OUTPUT_DIR/$new_filename" 2>/dev/null; then
        echo "✅ $new_id: $new_filename <- $dir_name/leap_action_$original_id.hdf5"
        echo "$new_filename <- $dir_name/leap_action_$original_id.hdf5" >> "$index_file"
        ((success++))
    else
        echo "❌ $new_id: 移动失败"
    fi
    ((new_id++))
done

echo ""
echo "⚡ 完成! 移动了 $success 个文件"
echo "📁 位置: $OUTPUT_DIR"
echo "📄 索引: $index_file"

# 简单统计
total_files=$(find "$OUTPUT_DIR" -name "${PREFIX}_*.hdf5" 2>/dev/null | wc -l)
echo "📊 最终文件数: $total_files"
