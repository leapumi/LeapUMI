#!/usr/bin/env python3
"""
HDF5数据清理和还原脚本
用于处理滚动存储的机器人遥操作数据

功能：
1. 扫描指定目录下的leap_action_*.hdf5文件
2. 识别需要保留的文件（非trash目录中的文件）
3. 根据数据长度还原每个文件的正确数据
4. 输出到指定的output目录
"""

import h5py
import numpy as np
import os
import sys
import glob
import shutil
from pathlib import Path
import argparse
from typing import List, Tuple, Dict


class DataCleaner:
    def __init__(self, input_dir: str, output_dir: str, dry_run: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.valid_files = []  # 需要保留的文件
        self.file_lengths = {}  # 文件ID到数据长度的映射
        
        # 创建输出目录
        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def scan_directory(self) -> List[Tuple[int, str]]:
        """扫描目录，找到所有需要保留的文件（不在trash中的文件）"""
        print(f"🔍 扫描目录: {self.input_dir}")
        
        # 查找所有leap_action_*.hdf5文件
        all_files = glob.glob(str(self.input_dir / "leap_action_*.hdf5"))
        trash_files = glob.glob(str(self.input_dir / "trash" / "leap_action_*.hdf5"))
        
        # 获取文件ID
        def get_file_id(filepath):
            filename = os.path.basename(filepath)
            return int(filename.replace('leap_action_', '').replace('.hdf5', ''))
        
        # 需要保留的文件（不在trash中）
        valid_files = []
        for filepath in all_files:
            if filepath not in trash_files:
                file_id = get_file_id(filepath)
                valid_files.append((file_id, filepath))
        
        # 按ID排序
        valid_files.sort(key=lambda x: x[0])
        self.valid_files = valid_files
        
        print(f"📋 找到 {len(valid_files)} 个需要保留的文件:")
        for file_id, filepath in valid_files:
            print(f"   📄 leap_action_{file_id}.hdf5")
        
        return valid_files
    
    def get_data_lengths(self) -> Dict[int, int]:
        """获取每个文件的数据长度"""
        print(f"\n📏 分析文件数据长度...")
        
        file_lengths = {}
        for file_id, filepath in self.valid_files:
            try:
                with h5py.File(filepath, 'r') as f:
                    # 假设数据在 /data/demo/actions 路径下
                    actions = f['data/demo/actions']
                    length = actions.shape[0]
                    file_lengths[file_id] = length
                    print(f"   📊 leap_action_{file_id}.hdf5: {length} 步")
            except Exception as e:
                print(f"   ❌ 读取 leap_action_{file_id}.hdf5 失败: {e}")
                continue
        
        self.file_lengths = file_lengths
        return file_lengths
    
    def calculate_split_indices(self) -> Dict[int, Tuple[int, int]]:
        """计算每个文件应该切分的索引范围"""
        print(f"\n✂️  计算切分索引...")
        
        split_indices = {}
        sorted_ids = sorted(self.file_lengths.keys())
        
        for i, file_id in enumerate(sorted_ids):
            if i == 0:
                # 第一个文件从0开始
                start_idx = 0
                end_idx = self.file_lengths[file_id]
            else:
                # 后续文件从前一个文件的长度开始
                prev_id = sorted_ids[i-1]
                start_idx = self.file_lengths[prev_id]
                end_idx = self.file_lengths[file_id]
            
            split_indices[file_id] = (start_idx, end_idx)
            print(f"   📄 leap_action_{file_id}.hdf5: 切分范围 [{start_idx}:{end_idx}] ({end_idx-start_idx} 步)")
        
        return split_indices
    
    def copy_and_slice_data(self, source_file: str, target_file: str, 
                           start_idx: int, end_idx: int, file_id: int):
        """复制并切分HDF5文件数据"""
        print(f"   🔄 处理 leap_action_{file_id}.hdf5 [{start_idx}:{end_idx}]...")
        
        if self.dry_run:
            print(f"      [DRY RUN] 将从 {source_file} 切分数据到 {target_file}")
            return
        
        try:
            with h5py.File(source_file, 'r') as source:
                with h5py.File(target_file, 'w') as target:
                    # 递归复制数据结构，但对数组数据进行切分
                    def copy_group(src_group, dst_group, name=""):
                        for key in src_group.keys():
                            item = src_group[key]
                            if isinstance(item, h5py.Dataset):
                                # 数据集 - 需要切分
                                data = item[start_idx:end_idx]
                                dst_group.create_dataset(key, data=data, 
                                                       compression=item.compression,
                                                       compression_opts=item.compression_opts)
                                # 复制属性
                                for attr_name, attr_value in item.attrs.items():
                                    dst_group[key].attrs[attr_name] = attr_value
                            elif isinstance(item, h5py.Group):
                                # 群组 - 递归处理
                                new_group = dst_group.create_group(key)
                                # 复制群组属性
                                for attr_name, attr_value in item.attrs.items():
                                    new_group.attrs[attr_name] = attr_value
                                copy_group(item, new_group, name + "/" + key)
                    
                    # 复制根级属性
                    for attr_name, attr_value in source.attrs.items():
                        target.attrs[attr_name] = attr_value
                    
                    # 开始复制
                    copy_group(source, target)
            
            print(f"      ✅ 成功生成 {os.path.basename(target_file)}")
            
        except Exception as e:
            print(f"      ❌ 处理失败: {e}")
    
    def process_all_files(self):
        """处理所有文件"""
        print(f"\n🚀 开始数据清理和还原...")
        
        if not self.valid_files:
            print("❌ 没有找到需要处理的文件")
            return
        
        # 计算切分索引
        split_indices = self.calculate_split_indices()
        
        print(f"\n📤 输出到: {self.output_dir}")
        
        # 处理每个文件
        for file_id, source_filepath in self.valid_files:
            if file_id not in split_indices:
                print(f"⚠️  跳过 leap_action_{file_id}.hdf5 (无法计算切分索引)")
                continue
            
            start_idx, end_idx = split_indices[file_id]
            target_filepath = self.output_dir / f"leap_action_{file_id}.hdf5"
            
            # 找到对应的滚动数据文件（最长的那个）
            max_length = max(self.file_lengths.values())
            source_file_for_slicing = None
            
            # 找到包含当前数据范围的文件
            for src_id, src_path in self.valid_files:
                if self.file_lengths[src_id] >= end_idx:
                    source_file_for_slicing = src_path
                    break
            
            if source_file_for_slicing is None:
                print(f"⚠️  找不到包含索引范围 [{start_idx}:{end_idx}] 的源文件")
                continue
            
            self.copy_and_slice_data(source_file_for_slicing, str(target_filepath), 
                                   start_idx, end_idx, file_id)
    
    def generate_report(self):
        """生成处理报告"""
        print(f"\n📊 处理报告")
        print("=" * 60)
        print(f"📁 输入目录: {self.input_dir}")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"📄 处理文件数: {len(self.valid_files)}")
        
        if self.file_lengths:
            total_original_steps = max(self.file_lengths.values()) if self.file_lengths else 0
            total_restored_steps = sum(
                end - start for start, end in self.calculate_split_indices().values()
            )
            print(f"📈 原始总步数: {total_original_steps}")
            print(f"📈 还原总步数: {total_restored_steps}")
        
        print(f"🏃 运行模式: {'预演模式 (DRY RUN)' if self.dry_run else '实际执行'}")


def main():
    parser = argparse.ArgumentParser(
        description="HDF5数据清理和还原工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 预演模式（不实际执行）
  python data_cleaner.py leap_action_0626_15_22/ output/ --dry-run
  
  # 实际执行
  python data_cleaner.py leap_action_0626_15_22/ output/
  
  # 批量处理多个目录
  python data_cleaner.py leap_action_0626_*/ output/ --batch
        """
    )
    
    parser.add_argument('input_dir', help='输入目录路径')
    parser.add_argument('output_dir', help='输出目录路径')
    parser.add_argument('--dry-run', action='store_true', 
                       help='预演模式，只显示操作但不实际执行')
    parser.add_argument('--batch', action='store_true',
                       help='批量处理模式（输入路径支持通配符）')
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量处理模式
        input_dirs = glob.glob(args.input_dir)
        if not input_dirs:
            print(f"❌ 没有找到匹配的目录: {args.input_dir}")
            return
        
        print(f"🎯 批量处理模式，找到 {len(input_dirs)} 个目录")
        
        for i, input_dir in enumerate(input_dirs, 1):
            print(f"\n{'='*60}")
            print(f"📁 处理目录 {i}/{len(input_dirs)}: {input_dir}")
            print(f"{'='*60}")
            
            # 为每个输入目录创建对应的输出子目录
            dir_name = os.path.basename(input_dir.rstrip('/'))
            output_subdir = os.path.join(args.output_dir, dir_name)
            
            cleaner = DataCleaner(input_dir, output_subdir, args.dry_run)
            cleaner.scan_directory()
            cleaner.get_data_lengths()
            cleaner.process_all_files()
            cleaner.generate_report()
    else:
        # 单目录处理模式
        if not os.path.exists(args.input_dir):
            print(f"❌ 输入目录不存在: {args.input_dir}")
            return
        
        cleaner = DataCleaner(args.input_dir, args.output_dir, args.dry_run)
        cleaner.scan_directory()
        cleaner.get_data_lengths()
        cleaner.process_all_files()
        cleaner.generate_report()


if __name__ == "__main__":
    main()
