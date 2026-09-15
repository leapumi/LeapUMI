#!/usr/bin/env python3
"""
批量HDF5数据清理脚本
自动处理所有leap_action_0626_*目录
"""

import os
import glob
import subprocess
import sys
from pathlib import Path


def batch_clean_all(output_dir="output"):
    """批量清理所有leap_action_0626_*目录"""
    
    print("🔍 扫描所有需要处理的目录...")
    
    # 查找所有匹配的目录
    pattern = "leap_action_0626_*/"
    directories = glob.glob(pattern)
    
    if not directories:
        print(f"❌ 没有找到匹配模式 '{pattern}' 的目录")
        return
    
    # 过滤掉空目录或无效目录
    valid_dirs = []
    for dir_path in directories:
        if os.path.isdir(dir_path):
            # 检查是否包含hdf5文件
            hdf5_files = glob.glob(os.path.join(dir_path, "leap_action_*.hdf5"))
            if hdf5_files:
                valid_dirs.append(dir_path)
                print(f"   📁 {dir_path} (包含 {len(hdf5_files)} 个HDF5文件)")
            else:
                print(f"   ⚠️  {dir_path} (无HDF5文件，跳过)")
    
    if not valid_dirs:
        print("❌ 没有找到包含HDF5文件的有效目录")
        return
    
    print(f"\n🎯 将处理 {len(valid_dirs)} 个目录")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {output_path.absolute()}")
    
    # 统计信息
    total_processed = 0
    failed_dirs = []
    
    # 处理每个目录
    for i, dir_path in enumerate(sorted(valid_dirs), 1):
        dir_name = os.path.basename(dir_path.rstrip('/'))
        print(f"\n{'='*60}")
        print(f"📂 处理 {i}/{len(valid_dirs)}: {dir_name}")
        print(f"{'='*60}")
        
        try:
            # 调用quick_clean脚本
            cmd = [sys.executable, "quick_clean.py", dir_path, output_dir]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {dir_name} 处理完成")
                # 统计处理的文件数
                lines = result.stdout.split('\n')
                for line in lines:
                    if "总共处理了" in line and "个文件" in line:
                        try:
                            count = int(line.split("总共处理了")[1].split("个文件")[0].strip())
                            total_processed += count
                        except:
                            pass
            else:
                print(f"❌ {dir_name} 处理失败:")
                print(result.stderr)
                failed_dirs.append(dir_name)
                
        except Exception as e:
            print(f"❌ {dir_name} 处理异常: {e}")
            failed_dirs.append(dir_name)
    
    # 最终统计
    print(f"\n{'='*60}")
    print(f"🎉 批量处理完成！")
    print(f"{'='*60}")
    print(f"📊 处理统计:")
    print(f"   📁 处理目录数: {len(valid_dirs)}")
    print(f"   📄 成功处理文件数: {total_processed}")
    print(f"   ✅ 成功目录数: {len(valid_dirs) - len(failed_dirs)}")
    print(f"   ❌ 失败目录数: {len(failed_dirs)}")
    
    if failed_dirs:
        print(f"\n⚠️  失败的目录:")
        for dir_name in failed_dirs:
            print(f"   📁 {dir_name}")
    
    print(f"\n📁 所有清理后的数据保存在: {output_path.absolute()}")
    
    # 显示输出目录结构
    print(f"\n📋 输出目录结构:")
    try:
        for item in sorted(os.listdir(output_dir)):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                hdf5_count = len(glob.glob(os.path.join(item_path, "*.hdf5")))
                print(f"   📁 {item}/ ({hdf5_count} 个文件)")
    except:
        print("   (无法读取输出目录结构)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="批量清理所有leap_action_0626_*目录的HDF5数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 默认输出到output目录
  python batch_clean_all.py
  
  # 指定输出目录
  python batch_clean_all.py --output cleaned_data
  
  # 列出将要处理的目录但不执行
  python batch_clean_all.py --list-only
        """
    )
    
    parser.add_argument('--output', '-o', default='output',
                       help='输出目录路径 (默认: output)')
    parser.add_argument('--list-only', action='store_true',
                       help='只列出要处理的目录，不实际执行')
    
    args = parser.parse_args()
    
    if args.list_only:
        print("🔍 扫描所有需要处理的目录...")
        pattern = "leap_action_0626_*/"
        directories = glob.glob(pattern)
        
        if not directories:
            print(f"❌ 没有找到匹配模式 '{pattern}' 的目录")
            return
        
        print(f"📋 找到 {len(directories)} 个目录:")
        for dir_path in sorted(directories):
            if os.path.isdir(dir_path):
                hdf5_files = glob.glob(os.path.join(dir_path, "leap_action_*.hdf5"))
                trash_files = set(glob.glob(os.path.join(dir_path, "trash", "leap_action_*.hdf5")))
                
                # 计算需要保留的文件（不在trash中的）
                keep_files = []
                for filepath in hdf5_files:
                    if filepath not in trash_files:
                        keep_files.append(filepath)
                
                print(f"   📁 {dir_path} (包含 {len(hdf5_files)} 个文件, 保留 {len(keep_files)} 个)")
    else:
        batch_clean_all(args.output)


if __name__ == "__main__":
    main()
