#!/usr/bin/env python3
"""
快速文件收集脚本
简单快速地收集和重命名HDF5文件
"""

import os
import shutil
import glob
from pathlib import Path
import re


def quick_collect(input_dir="output", output_dir="output/collected", prefix="leapdata"):
    """
    快速收集和重命名HDF5文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录  
        prefix: 文件名前缀
    """
    print(f"📦 快速收集HDF5文件")
    print(f"📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🏷️  文件前缀: {prefix}")
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 扫描所有HDF5文件
    pattern = os.path.join(input_dir, "*", "leap_action_*.hdf5")
    hdf5_files = glob.glob(pattern)
    
    if not hdf5_files:
        print(f"❌ 没有找到HDF5文件")
        return
    
    print(f"🔍 找到 {len(hdf5_files)} 个文件")
    
    # 提取文件信息并排序
    file_info = []
    for filepath in hdf5_files:
        dir_name = os.path.basename(os.path.dirname(filepath))
        filename = os.path.basename(filepath)
        
        # 提取原始ID
        match = re.search(r'leap_action_(\d+)\.hdf5', filename)
        if match:
            original_id = int(match.group(1))
            size_mb = os.path.getsize(filepath) / (1024*1024)
            file_info.append((filepath, dir_name, original_id, size_mb))
    
    # 按目录名和ID排序
    file_info.sort(key=lambda x: (x[1], x[2]))
    
    print(f"\n📋 文件列表:")
    for i, (filepath, dir_name, original_id, size_mb) in enumerate(file_info):
        print(f"   {i:2d}. {dir_name}/leap_action_{original_id}.hdf5 ({size_mb:.1f} MB)")
    
    # 生成索引记录
    index_records = []
    
    # 复制和重命名文件
    print(f"\n🔄 开始收集...")
    success_count = 0
    
    for new_id, (src_filepath, dir_name, original_id, size_mb) in enumerate(file_info):
        new_filename = f"{prefix}_{new_id}.hdf5"
        dst_filepath = os.path.join(output_dir, new_filename)
        
        try:
            # 复制文件
            shutil.copy2(src_filepath, dst_filepath)
            
            # 验证复制
            if os.path.getsize(dst_filepath) == os.path.getsize(src_filepath):
                print(f"   ✅ {new_id:2d}. {new_filename} <- {dir_name}/leap_action_{original_id}.hdf5")
                success_count += 1
                index_records.append(f"{new_filename} <- {dir_name}/leap_action_{original_id}.hdf5")
            else:
                print(f"   ❌ {new_id:2d}. 文件大小不匹配")
                
        except Exception as e:
            print(f"   ❌ {new_id:2d}. 复制失败: {e}")
    
    # 生成索引文件
    index_file = os.path.join(output_dir, "file_index.txt")
    try:
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write("# HDF5文件重命名索引\n")
            f.write(f"# 总共 {success_count} 个文件\n\n")
            for record in index_records:
                f.write(record + '\n')
        print(f"\n📋 索引文件已生成: {index_file}")
    except Exception as e:
        print(f"\n❌ 生成索引文件失败: {e}")
    
    # 统计结果
    print(f"\n🎉 收集完成!")
    print(f"   ✅ 成功: {success_count}/{len(file_info)} 个文件")
    print(f"   📁 保存位置: {output_dir}")
    print(f"   📋 文件命名: {prefix}_0.hdf5 到 {prefix}_{success_count-1}.hdf5")
    
    # 显示输出目录内容
    if success_count > 0:
        print(f"\n📂 输出目录内容:")
        output_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.hdf5')])
        for i, filename in enumerate(output_files[:10]):  # 只显示前10个
            size_mb = os.path.getsize(os.path.join(output_dir, filename)) / (1024*1024)
            print(f"   📄 {filename} ({size_mb:.1f} MB)")
        
        if len(output_files) > 10:
            print(f"   ... 还有 {len(output_files) - 10} 个文件")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # 默认参数
        quick_collect()
    elif len(sys.argv) == 4:
        # 自定义参数
        input_dir, output_dir, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
        quick_collect(input_dir, output_dir, prefix)
    else:
        print("用法:")
        print("  python quick_collect.py                                    # 使用默认参数")
        print("  python quick_collect.py <输入目录> <输出目录> <文件前缀>      # 自定义参数")
        print()
        print("示例:")
        print("  python quick_collect.py                                    # output -> output/collected, 前缀 leapdata")
        print("  python quick_collect.py output final_data robotdata       # output -> final_data, 前缀 robotdata")
        sys.exit(1)
