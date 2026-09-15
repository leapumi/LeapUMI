#!/usr/bin/env python3
"""
HDF5数据清理演示脚本
展示完整的数据清理和验证流程
"""

import os
import sys
import subprocess
import time


def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"💻 执行命令: {' '.join(cmd)}")
    print("-" * 40)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"⚠️  错误输出:\n{result.stderr}")
        
        if result.returncode == 0:
            print(f"✅ 命令执行成功")
        else:
            print(f"❌ 命令执行失败 (退出码: {result.returncode})")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ 命令执行超时")
        return False
    except Exception as e:
        print(f"❌ 执行异常: {e}")
        return False


def demo_data_cleaning():
    """演示数据清理流程"""
    print("""
🎯 HDF5数据清理演示
===================

本演示将展示如何清理和还原滚动存储的机器人遥操作数据。

演示流程:
1. 📋 列出所有待处理目录
2. 🔍 分析示例目录结构
3. 🧪 干运行模式测试
4. ✂️  实际清理小规模数据
5. ✅ 验证清理结果
6. 📊 批量处理演示
    """)
    
    input("按回车键开始演示...")
    
    # 步骤1: 列出所有待处理目录
    success = run_command(
        [sys.executable, "batch_clean_all.py", "--list-only"],
        "列出所有待处理目录"
    )
    
    if not success:
        print("❌ 无法列出目录，演示终止")
        return
    
    input("\n按回车键继续...")
    
    # 步骤2: 分析示例目录结构
    test_dir = "leap_action_0626_15_17"
    if os.path.exists(test_dir):
        run_command(
            ["ls", "-la", test_dir],
            f"查看示例目录 {test_dir} 的结构"
        )
        
        if os.path.exists(os.path.join(test_dir, "trash")):
            run_command(
                ["ls", "-la", os.path.join(test_dir, "trash")],
                f"查看 {test_dir}/trash 目录内容"
            )
    
    input("\n按回车键继续...")
    
    # 步骤3: 干运行模式测试
    if os.path.exists(test_dir):
        run_command(
            [sys.executable, "data_cleaner.py", f"{test_dir}/", "demo_output/", "--dry-run"],
            f"干运行模式测试目录 {test_dir}"
        )
    
    input("\n按回车键继续...")
    
    # 步骤4: 实际清理小规模数据
    if os.path.exists(test_dir):
        run_command(
            [sys.executable, "quick_clean.py", f"{test_dir}/", "demo_output/"],
            f"实际清理目录 {test_dir}"
        )
    
    input("\n按回车键继续...")
    
    # 步骤5: 验证清理结果
    output_dir = f"demo_output/{test_dir}"
    if os.path.exists(output_dir):
        run_command(
            ["ls", "-lh", output_dir],
            "查看清理后的文件"
        )
        
        # 验证第一个文件
        hdf5_files = [f for f in os.listdir(output_dir) if f.endswith('.hdf5')]
        if hdf5_files:
            first_file = os.path.join(output_dir, sorted(hdf5_files)[0])
            run_command(
                [sys.executable, "hdf5_quick_view.py", first_file],
                f"验证清理后文件: {sorted(hdf5_files)[0]}"
            )
    
    input("\n按回车键继续...")
    
    # 步骤6: 批量处理演示（干运行）
    print(f"\n{'='*60}")
    print(f"📊 批量处理演示（这里只显示命令，不实际执行大规模处理）")
    print(f"{'='*60}")
    
    print("""
实际批量处理命令:

# 处理所有目录到 cleaned_data 目录
python3 batch_clean_all.py --output cleaned_data

# 或者使用完整清理器进行批量处理
python3 data_cleaner.py 'leap_action_0626_*/' cleaned_data/ --batch

# 或者使用快速清理器
python3 quick_clean.py 'leap_action_0626_*/' cleaned_data/
    """)
    
    print(f"\n🎉 演示完成！")
    print(f"📁 演示输出文件保存在: demo_output/")
    print(f"📖 详细使用说明请查看: CLEANING_GUIDE.md")


def main():
    # 检查必要文件是否存在
    required_files = [
        "data_cleaner.py",
        "quick_clean.py", 
        "batch_clean_all.py",
        "hdf5_quick_view.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ 缺少必要文件: {', '.join(missing_files)}")
        print(f"请确保所有脚本文件都在当前目录中")
        sys.exit(1)
    
    # 检查是否有测试数据
    test_dirs = [d for d in os.listdir('.') if d.startswith('leap_action_0626_')]
    if not test_dirs:
        print(f"❌ 没有找到测试数据目录 (leap_action_0626_*)")
        print(f"请确保当前目录包含待处理的数据目录")
        sys.exit(1)
    
    try:
        demo_data_cleaning()
    except KeyboardInterrupt:
        print(f"\n\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n\n❌ 演示过程中发生错误: {e}")


if __name__ == "__main__":
    main()
