#!/usr/bin/env python3
"""
批量HDF5文件信息查看器
快速查看多个HDF5文件的基本信息和对比
"""

import h5py
import numpy as np
import os
import sys
from collections import defaultdict
import pandas as pd


def get_file_info(file_path):
    """获取单个HDF5文件的基本信息"""
    try:
        info = {
            'file': os.path.basename(file_path),
            'size_mb': os.path.getsize(file_path) / (1024*1024),
            'readable': True
        }
        
        with h5py.File(file_path, 'r') as f:
            # 获取数据集信息
            if 'data' in f and 'demo' in f['data']:
                demo = f['data']['demo']
                
                # 基本信息
                actions = demo['actions']
                info['timesteps'] = actions.shape[0]
                info['action_dim'] = actions.shape[1]
                
                # 时间信息
                if 'obs' in demo and 'time' in demo['obs']:
                    time_data = demo['obs']['time'][()]
                    info['duration'] = time_data[-1] - time_data[0]
                    info['frequency'] = len(actions) / info['duration']
                
                # 奖励信息
                if 'rewards' in demo:
                    rewards = demo['rewards'][()]
                    info['final_reward'] = rewards[-1]
                    info['max_reward'] = np.max(rewards)
                    info['success'] = info['final_reward'] > 0.5
                
                # 完成状态
                if 'dones' in demo:
                    dones = demo['dones'][()]
                    info['terminated'] = np.any(dones)
                
                # 机械臂状态
                if 'obs' in demo:
                    obs = demo['obs']
                    if 'arm_qpos' in obs:
                        info['arm_dof'] = obs['arm_qpos'].shape[1]
                    if 'eef_pos' in obs:
                        eef_pos = obs['eef_pos'][()]
                        distances = np.sqrt(np.sum(np.diff(eef_pos, axis=0)**2, axis=1))
                        info['travel_distance'] = np.sum(distances)
                    if 'gripper_qpos' in obs:
                        info['gripper_dof'] = obs['gripper_qpos'].shape[1]
                        
    except Exception as e:
        info = {
            'file': os.path.basename(file_path),
            'size_mb': os.path.getsize(file_path) / (1024*1024),
            'readable': False,
            'error': str(e)
        }
    
    return info


def analyze_directory(directory):
    """分析目录中的所有HDF5文件"""
    print(f"🔍 扫描目录: {directory}")
    
    # 收集所有HDF5文件
    hdf5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.hdf5', '.h5')):
                hdf5_files.append(os.path.join(root, file))
    
    if not hdf5_files:
        print("❌ 未找到HDF5文件")
        return
    
    print(f"📁 找到 {len(hdf5_files)} 个HDF5文件")
    print("⏳ 正在分析...")
    
    # 分析每个文件
    file_infos = []
    for file_path in sorted(hdf5_files):
        info = get_file_info(file_path)
        file_infos.append(info)
        
        # 简单进度指示
        if len(file_infos) % 10 == 0:
            print(f"   已处理 {len(file_infos)}/{len(hdf5_files)} 个文件...")
    
    # 创建DataFrame用于分析
    readable_files = [info for info in file_infos if info['readable']]
    unreadable_files = [info for info in file_infos if not info['readable']]
    
    print(f"\n✅ 可读文件: {len(readable_files)}")
    print(f"❌ 不可读文件: {len(unreadable_files)}")
    
    if unreadable_files:
        print("\n⚠️  不可读文件列表:")
        for info in unreadable_files:
            print(f"   {info['file']}: {info.get('error', '未知错误')}")
    
    if not readable_files:
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(readable_files)
    
    # 基本统计
    print("\n📊 基本统计信息:")
    print(f"   文件数量: {len(readable_files)}")
    print(f"   总大小: {df['size_mb'].sum():.2f} MB")
    print(f"   平均大小: {df['size_mb'].mean():.2f} MB")
    
    if 'timesteps' in df.columns:
        print(f"   总时间步: {df['timesteps'].sum():,}")
        print(f"   平均时间步: {df['timesteps'].mean():.0f}")
    
    if 'duration' in df.columns:
        print(f"   总时长: {df['duration'].sum():.1f} 秒 ({df['duration'].sum()/60:.1f} 分钟)")
        print(f"   平均时长: {df['duration'].mean():.1f} 秒")
    
    if 'success' in df.columns:
        success_rate = df['success'].mean() * 100
        print(f"   成功率: {df['success'].sum()}/{len(df)} ({success_rate:.1f}%)")
    
    # 详细列表
    print("\n📋 详细文件列表:")
    print("-" * 100)
    
    # 构建表头
    headers = ['文件名', '大小(MB)', '时间步']
    if 'duration' in df.columns:
        headers.append('时长(s)')
    if 'final_reward' in df.columns:
        headers.append('最终奖励')
    if 'success' in df.columns:
        headers.append('成功')
    if 'travel_distance' in df.columns:
        headers.append('移动距离')
    
    # 打印表头
    print(f"{headers[0]:<30} {headers[1]:<10} {headers[2]:<8}", end="")
    for header in headers[3:]:
        print(f" {header:<12}", end="")
    print()
    print("-" * 100)
    
    # 打印数据行
    for _, row in df.iterrows():
        filename = row['file'][:28] + '..' if len(row['file']) > 30 else row['file']
        print(f"{filename:<30} {row['size_mb']:<10.2f} {row.get('timesteps', 'N/A'):<8}", end="")
        
        if 'duration' in row:
            print(f" {row['duration']:<12.1f}", end="")
        if 'final_reward' in row:
            print(f" {row['final_reward']:<12.3f}", end="")
        if 'success' in row:
            status = "✅" if row['success'] else "❌"
            print(f" {status:<12}", end="")
        if 'travel_distance' in row:
            print(f" {row['travel_distance']:<12.3f}", end="")
        print()
    
    # 按子目录分组统计
    print("\n📁 按目录分组统计:")
    directory_stats = defaultdict(list)
    
    for file_path in hdf5_files:
        rel_path = os.path.relpath(file_path, directory)
        if '/' in rel_path:
            dir_name = rel_path.split('/')[0]
        else:
            dir_name = '根目录'
        directory_stats[dir_name].append(file_path)
    
    for dir_name, files in directory_stats.items():
        readable_count = sum(1 for f in files if any(info['file'] == os.path.basename(f) and info['readable'] for info in file_infos))
        total_size = sum(os.path.getsize(f) for f in files) / (1024*1024)
        print(f"   📂 {dir_name}: {readable_count}/{len(files)} 文件, {total_size:.2f} MB")


def main():
    if len(sys.argv) != 2:
        print("用法: python batch_hdf5_info.py <目录路径>")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"❌ 目录不存在: {directory}")
        sys.exit(1)
    
    analyze_directory(directory)


if __name__ == "__main__":
    main()
