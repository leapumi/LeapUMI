#!/usr/bin/env python3
"""
HDF5数据分析工具集
提供多种分析HDF5文件的功能
"""

import argparse
import sys
import os

# 导入其他分析模块的函数
def print_help():
    """显示帮助信息"""
    print("""
🔧 HDF5数据分析工具集

📋 可用命令:

1. 快速查看 (quick)
   python hdf5_tools.py quick <文件路径>
   - 快速显示HDF5文件的结构和基本信息

2. 详细分析 (analyze)
   python hdf5_tools.py analyze <文件路径> [--plots]
   - 详细分析单个HDF5文件
   - --plots: 生成分析图表

3. 批量信息 (batch)
   python hdf5_tools.py batch <目录路径>
   - 批量分析目录中的所有HDF5文件
   - 生成统计报告

4. 机器人数据 (robot)
   python hdf5_tools.py robot <文件路径或目录> [--plots] [--batch]
   - 专门分析机器人遥操作数据
   - --plots: 生成轨迹图表
   - --batch: 批量模式

5. 交互模式 (interactive)
   python hdf5_tools.py interactive <目录路径>
   - 交互式浏览和分析HDF5文件

示例:
   python hdf5_tools.py quick data_wo_images/leap_action_0.hdf5
   python hdf5_tools.py batch data_wo_images/
   python hdf5_tools.py robot data_wo_images/leap_action_0.hdf5 --plots
   python hdf5_tools.py interactive data_wo_images/
""")


def interactive_mode(directory):
    """交互式模式"""
    print("🔍 交互式HDF5文件浏览器")
    print("=" * 50)
    
    # 收集HDF5文件
    hdf5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.hdf5', '.h5')):
                hdf5_files.append(os.path.join(root, file))
    
    if not hdf5_files:
        print("❌ 没有找到HDF5文件")
        return
    
    hdf5_files.sort()
    
    while True:
        print(f"\n📁 找到 {len(hdf5_files)} 个HDF5文件:")
        for i, filepath in enumerate(hdf5_files, 1):
            rel_path = os.path.relpath(filepath, directory)
            size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"  {i:2d}. {rel_path} ({size_mb:.2f} MB)")
        
        print("\n🔧 操作选项:")
        print("  q) 快速查看文件结构")
        print("  a) 详细分析文件")
        print("  r) 机器人数据分析")
        print("  b) 批量统计分析")
        print("  x) 退出")
        
        try:
            choice = input("\n请选择操作 (q/a/r/b/x): ").strip().lower()
            
            if choice == 'x':
                print("👋 再见!")
                break
            elif choice == 'b':
                os.system(f"python batch_hdf5_info.py '{directory}'")
            elif choice in ['q', 'a', 'r']:
                file_num = input("请输入文件编号: ").strip()
                try:
                    file_index = int(file_num) - 1
                    if 0 <= file_index < len(hdf5_files):
                        selected_file = hdf5_files[file_index]
                        
                        if choice == 'q':
                            os.system(f"python hdf5_quick_view.py '{selected_file}'")
                        elif choice == 'a':
                            os.system(f"python read_hdf5.py '{selected_file}'")
                        elif choice == 'r':
                            os.system(f"python robot_data_analyzer.py '{selected_file}'")
                    else:
                        print("❌ 无效的文件编号")
                except ValueError:
                    print("❌ 请输入有效数字")
            else:
                print("❌ 无效选择")
                
        except KeyboardInterrupt:
            print("\n👋 退出")
            break
        except EOFError:
            break


def main():
    parser = argparse.ArgumentParser(description='HDF5数据分析工具集')
    parser.add_argument('command', nargs='?', choices=['quick', 'analyze', 'batch', 'robot', 'interactive', 'help'], 
                       help='要执行的命令')
    parser.add_argument('path', nargs='?', help='文件或目录路径')
    parser.add_argument('--plots', '-p', action='store_true', help='生成图表')
    parser.add_argument('--batch', '-b', action='store_true', help='批量模式')
    
    args = parser.parse_args()
    
    if not args.command or args.command == 'help':
        print_help()
        return 0
    
    if not args.path:
        print("❌ 请提供文件或目录路径")
        return 1
    
    if not os.path.exists(args.path):
        print(f"❌ 路径不存在: {args.path}")
        return 1
    
    try:
        if args.command == 'quick':
            os.system(f"python hdf5_quick_view.py '{args.path}'")
        
        elif args.command == 'analyze':
            plot_flag = "--plots" if args.plots else ""
            os.system(f"python read_hdf5.py '{args.path}' {plot_flag}")
        
        elif args.command == 'batch':
            os.system(f"python batch_hdf5_info.py '{args.path}'")
        
        elif args.command == 'robot':
            plot_flag = "--plots" if args.plots else ""
            batch_flag = "--batch" if args.batch else ""
            os.system(f"python robot_data_analyzer.py '{args.path}' {plot_flag} {batch_flag}")
        
        elif args.command == 'interactive':
            if not os.path.isdir(args.path):
                print("❌ 交互模式需要目录路径")
                return 1
            interactive_mode(args.path)
        
    except KeyboardInterrupt:
        print("\n👋 操作已取消")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
