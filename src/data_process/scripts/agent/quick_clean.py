#!/usr/bin/env python3
"""
快速HDF5数据清理脚本
用于快速清理和还原滚动存储的机器人数据
"""

import h5py
import os
import glob
import shutil
from pathlib import Path


def quick_clean(input_pattern, output_dir):
    """
    快速清理HDF5数据
    
    Args:
        input_pattern: 输入目录模式，如 'leap_action_0626_*/'
        output_dir: 输出目录
    """
    print(f"🧹 快速清理模式")
    print(f"📁 输入模式: {input_pattern}")
    print(f"📁 输出目录: {output_dir}")
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 查找所有匹配的目录
    input_dirs = glob.glob(input_pattern)
    if not input_dirs:
        print(f"❌ 没有找到匹配的目录")
        return
    
    print(f"🎯 找到 {len(input_dirs)} 个目录需要处理")
    
    total_processed = 0
    
    for input_dir in sorted(input_dirs):
        dir_name = os.path.basename(input_dir.rstrip('/'))
        print(f"\n📂 处理: {dir_name}")
        
        # 查找需要保留的文件（不在trash中的）
        hdf5_files = glob.glob(os.path.join(input_dir, "leap_action_*.hdf5"))
        trash_files = set(glob.glob(os.path.join(input_dir, "trash", "leap_action_*.hdf5")))
        
        # 筛选出需要保留的文件
        keep_files = []
        for filepath in hdf5_files:
            if filepath not in trash_files:
                # 提取文件ID
                filename = os.path.basename(filepath)
                file_id = int(filename.replace('leap_action_', '').replace('.hdf5', ''))
                keep_files.append((file_id, filepath))
        
        keep_files.sort()  # 按ID排序
        
        if not keep_files:
            print(f"   ⚠️  没有找到需要保留的文件")
            continue
        
        print(f"   📋 需要保留 {len(keep_files)} 个文件:")
        for file_id, _ in keep_files:
            print(f"      📄 leap_action_{file_id}.hdf5")
        
        # 获取文件长度并计算切分点
        file_lengths = {}
        for file_id, filepath in keep_files:
            try:
                with h5py.File(filepath, 'r') as f:
                    length = f['data/demo/actions'].shape[0]
                    file_lengths[file_id] = length
                    print(f"      📊 ID {file_id}: {length} 步")
            except Exception as e:
                print(f"      ❌ 读取失败 ID {file_id}: {e}")
                continue
        
        # 计算切分索引
        sorted_ids = sorted(file_lengths.keys())
        split_points = {}
        
        for i, file_id in enumerate(sorted_ids):
            if i == 0:
                start_idx = 0
            else:
                start_idx = file_lengths[sorted_ids[i-1]]
            end_idx = file_lengths[file_id]
            split_points[file_id] = (start_idx, end_idx)
            print(f"      ✂️  ID {file_id}: 切分 [{start_idx}:{end_idx}] ({end_idx-start_idx} 步)")
        
        # 创建输出子目录
        output_subdir = os.path.join(output_dir, dir_name)
        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        
        # 处理每个文件
        for file_id in sorted_ids:
            if file_id not in split_points:
                continue
            
            start_idx, end_idx = split_points[file_id]
            
            # 找到源文件（包含足够数据的最新文件）
            source_file = None
            for src_id, src_path in reversed(keep_files):  # 从最新的开始找
                if file_lengths[src_id] >= end_idx:
                    source_file = src_path
                    break
            
            if source_file is None:
                print(f"      ⚠️  找不到包含足够数据的源文件 for ID {file_id}")
                continue
            
            # 输出文件路径
            output_file = os.path.join(output_subdir, f"leap_action_{file_id}.hdf5")
            
            try:
                print(f"      🔄 切分 ID {file_id}...")
                
                with h5py.File(source_file, 'r') as src:
                    with h5py.File(output_file, 'w') as dst:
                        # 递归复制并切分数据
                        def copy_item(name, obj):
                            if isinstance(obj, h5py.Dataset):
                                # 切分数据集
                                data = obj[start_idx:end_idx]
                                dst.create_dataset(name, data=data)
                                # 复制属性
                                for attr_name, attr_value in obj.attrs.items():
                                    dst[name].attrs[attr_name] = attr_value
                            elif isinstance(obj, h5py.Group):
                                # 创建群组
                                grp = dst.create_group(name)
                                # 复制属性
                                for attr_name, attr_value in obj.attrs.items():
                                    grp.attrs[attr_name] = attr_value
                        
                        # 复制根属性
                        for attr_name, attr_value in src.attrs.items():
                            dst.attrs[attr_name] = attr_value
                        
                        # 遍历并复制所有内容
                        src.visititems(copy_item)
                
                print(f"      ✅ 完成 leap_action_{file_id}.hdf5")
                total_processed += 1
                
            except Exception as e:
                print(f"      ❌ 处理失败 ID {file_id}: {e}")
    
    print(f"\n🎉 清理完成！总共处理了 {total_processed} 个文件")
    print(f"📁 输出目录: {output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python quick_clean.py <输入目录模式> <输出目录>")
        print("示例: python quick_clean.py 'leap_action_0626_*/' output/")
        sys.exit(1)
    
    input_pattern = sys.argv[1]
    output_dir = sys.argv[2]
    
    quick_clean(input_pattern, output_dir)
