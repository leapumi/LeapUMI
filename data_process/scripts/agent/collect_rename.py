#!/usr/bin/env python3
"""
HDF5文件收集和重命名脚本
扫描output目录下的所有HDF5文件，重新命名为连续ID并输出到collected目录
"""

import os
import shutil
import glob
from pathlib import Path
import argparse
import sys
from typing import List, Tuple
import re


class FileCollector:
    def __init__(self, input_dir: str = "output", output_dir: str = "output/collected", 
                 prefix: str = "leapdata", dry_run: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.dry_run = dry_run
        self.collected_files = []
        
        # 创建输出目录
        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def scan_hdf5_files(self) -> List[Tuple[str, str, int]]:
        """
        扫描input_dir下的所有HDF5文件
        返回: [(文件路径, 目录名, 原始ID), ...]
        """
        print(f"🔍 扫描目录: {self.input_dir}")
        
        if not self.input_dir.exists():
            print(f"❌ 输入目录不存在: {self.input_dir}")
            return []
        
        # 查找所有HDF5文件
        pattern = str(self.input_dir / "*" / "leap_action_*.hdf5")
        hdf5_files = glob.glob(pattern)
        
        if not hdf5_files:
            print(f"❌ 在 {self.input_dir} 下没有找到HDF5文件")
            return []
        
        print(f"📁 找到 {len(hdf5_files)} 个HDF5文件")
        
        # 提取文件信息
        file_info = []
        for filepath in hdf5_files:
            # 提取目录名
            dir_name = os.path.basename(os.path.dirname(filepath))
            
            # 提取原始ID
            filename = os.path.basename(filepath)
            match = re.search(r'leap_action_(\d+)\.hdf5', filename)
            if match:
                original_id = int(match.group(1))
                file_info.append((filepath, dir_name, original_id))
            else:
                print(f"⚠️  无法解析文件名: {filename}")
        
        # 按目录名和原始ID排序
        file_info.sort(key=lambda x: (x[1], x[2]))
        
        print(f"📋 按目录和ID排序后的文件列表:")
        current_dir = None
        for filepath, dir_name, original_id in file_info:
            if current_dir != dir_name:
                current_dir = dir_name
                print(f"   📁 {dir_name}/")
            filename = os.path.basename(filepath)
            size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"      📄 {filename} ({size_mb:.1f} MB)")
        
        return file_info
    
    def collect_and_rename(self, file_info: List[Tuple[str, str, int]]):
        """收集并重命名文件"""
        print(f"\n📦 开始收集和重命名文件...")
        print(f"📤 输出目录: {self.output_dir}")
        
        if self.dry_run:
            print(f"🔍 [DRY RUN] 预览模式，不实际执行文件操作")
        
        success_count = 0
        failed_files = []
        
        for new_id, (src_filepath, dir_name, original_id) in enumerate(file_info):
            # 生成新文件名
            new_filename = f"{self.prefix}_{new_id}.hdf5"
            dst_filepath = self.output_dir / new_filename
            
            print(f"\n   🔄 处理 {new_id + 1}/{len(file_info)}")
            print(f"      📂 源目录: {dir_name}")
            print(f"      📄 源文件: leap_action_{original_id}.hdf5")
            print(f"      ➡️  新文件: {new_filename}")
            
            if self.dry_run:
                print(f"      [DRY RUN] 将复制: {src_filepath} -> {dst_filepath}")
                success_count += 1
                continue
            
            try:
                # 复制文件
                shutil.copy2(src_filepath, dst_filepath)
                
                # 验证文件大小
                src_size = os.path.getsize(src_filepath)
                dst_size = os.path.getsize(dst_filepath)
                
                if src_size == dst_size:
                    print(f"      ✅ 复制成功 ({dst_size / (1024*1024):.1f} MB)")
                    success_count += 1
                    self.collected_files.append((new_filename, dir_name, original_id))
                else:
                    print(f"      ❌ 文件大小不匹配: {src_size} != {dst_size}")
                    failed_files.append((src_filepath, "文件大小不匹配"))
                    
            except Exception as e:
                print(f"      ❌ 复制失败: {e}")
                failed_files.append((src_filepath, str(e)))
        
        # 统计结果
        print(f"\n📊 收集完成统计:")
        print(f"   ✅ 成功: {success_count}/{len(file_info)} 个文件")
        print(f"   ❌ 失败: {len(failed_files)} 个文件")
        
        if failed_files:
            print(f"\n⚠️  失败文件列表:")
            for filepath, error in failed_files:
                print(f"   📄 {os.path.basename(filepath)}: {error}")
        
        return success_count, failed_files
    
    def generate_index_file(self):
        """生成索引文件，记录新旧文件名的对应关系"""
        if self.dry_run:
            print(f"\n📋 [DRY RUN] 将生成索引文件: {self.output_dir / 'file_index.txt'}")
            return
        
        index_file = self.output_dir / "file_index.txt"
        
        try:
            with open(index_file, 'w', encoding='utf-8') as f:
                f.write("# HDF5文件重命名索引\n")
                f.write("# 格式: 新文件名 <- 原目录/原文件名\n")
                f.write(f"# 生成时间: {__import__('datetime').datetime.now()}\n\n")
                
                for new_filename, dir_name, original_id in self.collected_files:
                    f.write(f"{new_filename} <- {dir_name}/leap_action_{original_id}.hdf5\n")
            
            print(f"📋 索引文件已生成: {index_file}")
            
        except Exception as e:
            print(f"❌ 生成索引文件失败: {e}")
    
    def generate_summary(self, file_info: List[Tuple[str, str, int]], 
                        success_count: int, failed_count: int):
        """生成处理摘要"""
        print(f"\n{'='*60}")
        print(f"📊 处理摘要")
        print(f"{'='*60}")
        print(f"📁 输入目录: {self.input_dir}")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"📄 扫描到的文件: {len(file_info)} 个")
        print(f"✅ 成功处理: {success_count} 个")
        print(f"❌ 处理失败: {failed_count} 个")
        print(f"🏃 运行模式: {'预览模式 (DRY RUN)' if self.dry_run else '实际执行'}")
        
        if not self.dry_run and success_count > 0:
            print(f"📋 生成的文件命名规则: {self.prefix}_0.hdf5, {self.prefix}_1.hdf5, ...")
            print(f"📂 文件保存位置: {self.output_dir}")
        
        # 按目录统计
        if file_info:
            print(f"\n📁 按源目录统计:")
            dir_stats = {}
            for _, dir_name, _ in file_info:
                dir_stats[dir_name] = dir_stats.get(dir_name, 0) + 1
            
            for dir_name, count in sorted(dir_stats.items()):
                print(f"   📂 {dir_name}: {count} 个文件")


def main():
    parser = argparse.ArgumentParser(
        description="HDF5文件收集和重命名工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 预览模式（推荐先使用）
  python collect_rename.py --dry-run
  
  # 实际执行，使用默认设置
  python collect_rename.py
  
  # 指定输入输出目录
  python collect_rename.py --input output --output collected_data
  
  # 自定义文件前缀
  python collect_rename.py --prefix robotdata
  
  # 完整示例
  python collect_rename.py --input output --output final_collection --prefix dataset --dry-run
        """
    )
    
    parser.add_argument('--input', '-i', default='output',
                       help='输入目录路径 (默认: output)')
    parser.add_argument('--output', '-o', default='output/collected',
                       help='输出目录路径 (默认: output/collected)')
    parser.add_argument('--prefix', '-p', default='leapdata',
                       help='输出文件名前缀 (默认: leapdata)')
    parser.add_argument('--dry-run', action='store_true',
                       help='预览模式，只显示操作但不实际执行')
    
    args = parser.parse_args()
    
    # 创建收集器
    collector = FileCollector(
        input_dir=args.input,
        output_dir=args.output,
        prefix=args.prefix,
        dry_run=args.dry_run
    )
    
    # 扫描文件
    file_info = collector.scan_hdf5_files()
    
    if not file_info:
        print("❌ 没有找到可处理的文件")
        return 1
    
    # 收集和重命名
    success_count, failed_files = collector.collect_and_rename(file_info)
    
    # 生成索引文件
    if success_count > 0:
        collector.generate_index_file()
    
    # 生成摘要
    collector.generate_summary(file_info, success_count, len(failed_files))
    
    return 0 if len(failed_files) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
