#!/bin/bash
"""
HDF5文件收集和重命名脚本
扫描output目录下的所有HDF5文件，重新命名为连续ID并输出到collected目录
"""

# 默认参数
INPUT_DIR="output"
OUTPUT_DIR="output/collected"
PREFIX="leapdata"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -p|--prefix)
            PREFIX="$2"
            shift 2
            ;;
        -h|--help)
            echo "HDF5文件收集和重命名工具"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -i, --input DIR     输入目录 (默认: output)"
            echo "  -o, --output DIR    输出目录 (默认: output/collected)"
            echo "  -p, --prefix NAME   文件前缀 (默认: leapdata)"
            echo "  -h, --help          显示帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                                    # 使用默认参数"
            echo "  $0 -i output -o final_data -p robot  # 自定义参数"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "📦 HDF5文件收集和重命名"
echo "📁 输入目录: $INPUT_DIR"
echo "📁 输出目录: $OUTPUT_DIR"
echo "🏷️  文件前缀: $PREFIX"
echo "================================================"

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 查找所有HDF5文件
echo "🔍 扫描HDF5文件..."
hdf5_files=()
while IFS= read -r -d '' file; do
    hdf5_files+=("$file")
done < <(find "$INPUT_DIR" -name "leap_action_*.hdf5" -type f -print0)

# 检查是否找到文件
if [ ${#hdf5_files[@]} -eq 0 ]; then
    echo "❌ 没有找到HDF5文件"
    exit 1
fi

echo "📊 找到 ${#hdf5_files[@]} 个HDF5文件"

# 创建临时文件用于排序
temp_file=$(mktemp)
index_file="$OUTPUT_DIR/file_index.txt"

# 提取文件信息并排序
echo "📋 分析文件信息..."
for file in "${hdf5_files[@]}"; do
    # 提取目录名和文件名
    dir_name=$(basename "$(dirname "$file")")
    filename=$(basename "$file")
    
    # 提取原始ID
    if [[ $filename =~ leap_action_([0-9]+)\.hdf5 ]]; then
        original_id="${BASH_REMATCH[1]}"
        # 获取文件大小
        size_bytes=$(stat -c%s "$file")
        size_mb=$((size_bytes / 1024 / 1024))
        
        # 写入临时文件 (格式: 目录名|原始ID|文件路径|大小MB)
        echo "${dir_name}|${original_id}|${file}|${size_mb}" >> "$temp_file"
    fi
done

# 按目录名和ID排序
sort -t'|' -k1,1 -k2,2n "$temp_file" > "${temp_file}.sorted"

# 显示文件列表
echo ""
echo "📋 文件列表 (按目录和ID排序):"
id_counter=0
while IFS='|' read -r dir_name original_id file_path size_mb; do
    printf "   %2d. %s/leap_action_%s.hdf5 (%s MB)\n" "$id_counter" "$dir_name" "$original_id" "$size_mb"
    ((id_counter++))
done < "${temp_file}.sorted"

# 开始复制和重命名
echo ""
echo "🔄 开始收集和重命名..."

# 创建索引文件
echo "# HDF5文件重命名索引" > "$index_file"
echo "# 生成时间: $(date)" >> "$index_file"
echo "# 总文件数: $(wc -l < "${temp_file}.sorted")" >> "$index_file"
echo "" >> "$index_file"

new_id=0
success_count=0
failed_count=0

while IFS='|' read -r dir_name original_id file_path size_mb; do
    new_filename="${PREFIX}_${new_id}.hdf5"
    output_path="$OUTPUT_DIR/$new_filename"
    
    # 复制文件
    if cp "$file_path" "$output_path" 2>/dev/null; then
        # 验证文件大小
        original_size=$(stat -c%s "$file_path")
        copied_size=$(stat -c%s "$output_path")
        
        if [ "$original_size" -eq "$copied_size" ]; then
            printf "   ✅ %2d. %s <- %s/leap_action_%s.hdf5\n" "$new_id" "$new_filename" "$dir_name" "$original_id"
            echo "$new_filename <- $dir_name/leap_action_$original_id.hdf5" >> "$index_file"
            ((success_count++))
        else
            echo "   ❌ $new_id. 文件大小不匹配"
            rm -f "$output_path"  # 删除损坏的文件
            ((failed_count++))
        fi
    else
        echo "   ❌ $new_id. 复制失败: $file_path"
        ((failed_count++))
    fi
    
    ((new_id++))
done < "${temp_file}.sorted"

# 清理临时文件
rm -f "$temp_file" "${temp_file}.sorted"

# 统计结果
total_files=$((success_count + failed_count))
echo ""
echo "🎉 收集完成!"
echo "   ✅ 成功: $success_count/$total_files 个文件"
echo "   ❌ 失败: $failed_count 个文件"
echo "   📁 保存位置: $OUTPUT_DIR"
if [ $success_count -gt 0 ]; then
    echo "   📋 文件命名: ${PREFIX}_0.hdf5 到 ${PREFIX}_$((success_count-1)).hdf5"
fi
echo "   📄 索引文件: $index_file"

# 显示输出目录内容
if [ $success_count -gt 0 ]; then
    echo ""
    echo "📂 输出目录内容:"
    file_count=0
    for file in "$OUTPUT_DIR"/*.hdf5; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            size_bytes=$(stat -c%s "$file")
            size_mb=$((size_bytes / 1024 / 1024))
            printf "   📄 %s (%d MB)\n" "$filename" "$size_mb"
            ((file_count++))
            
            # 只显示前10个文件
            if [ $file_count -ge 10 ]; then
                remaining=$((success_count - 10))
                if [ $remaining -gt 0 ]; then
                    echo "   ... 还有 $remaining 个文件"
                fi
                break
            fi
        fi
    done
fi

echo ""
echo "✨ 完成! 所有文件已收集到 $OUTPUT_DIR"
