#!/usr/bin/env python3
"""
机器人数据HDF5文件分析器
专门用于分析机器人遥操作数据的HDF5文件
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from typing import Dict, List, Any


def analyze_robot_data(file_path: str, save_plots: bool = False):
    """分析机器人数据HDF5文件"""
    print(f"🤖 分析机器人数据: {os.path.basename(file_path)}")
    print(f"📏 文件大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    print("=" * 60)
    
    try:
        with h5py.File(file_path, 'r') as f:
            # 分析数据结构
            demo_data = f['data']['demo']
            
            print("📊 数据集概览:")
            actions = demo_data['actions'][()]
            dones = demo_data['dones'][()]
            rewards = demo_data['rewards'][()]
            
            obs = demo_data['obs']
            arm_qpos = obs['arm_qpos'][()]
            eef_pos = obs['eef_pos'][()]
            eef_rot = obs['eef_rot'][()]
            gripper_qpos = obs['gripper_qpos'][()]
            time_data = obs['time'][()]
            
            print(f"   📈 时间步数: {len(actions)}")
            print(f"   ⏱️  持续时间: {time_data[-1] - time_data[0]:.2f} 秒")
            print(f"   🔄 采样频率: {len(actions) / (time_data[-1] - time_data[0]):.1f} Hz")
            print(f"   ✅ 完成状态: {'是' if np.any(dones) else '否'}")
            print(f"   🎯 最终奖励: {rewards[-1]:.3f}")
            
            print("\n🦾 动作数据分析:")
            print(f"   📐 动作维度: {actions.shape[1]}")
            print(f"   📊 动作范围: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
            print(f"   📈 动作标准差: {np.std(actions):.3f}")
            
            print("\n🔧 机械臂状态:")
            print(f"   📐 关节角度维度: {arm_qpos.shape[1]}")
            print(f"   📊 关节角度范围: [{np.min(arm_qpos):.3f}, {np.max(arm_qpos):.3f}]")
            
            print("\n📍 末端执行器位置:")
            print(f"   📊 位置范围 X: [{np.min(eef_pos[:, 0]):.3f}, {np.max(eef_pos[:, 0]):.3f}]")
            print(f"   📊 位置范围 Y: [{np.min(eef_pos[:, 1]):.3f}, {np.max(eef_pos[:, 1]):.3f}]")
            print(f"   📊 位置范围 Z: [{np.min(eef_pos[:, 2]):.3f}, {np.max(eef_pos[:, 2]):.3f}]")
            
            # 计算移动距离
            distances = np.sqrt(np.sum(np.diff(eef_pos, axis=0)**2, axis=1))
            total_distance = np.sum(distances)
            print(f"   📏 总移动距离: {total_distance:.3f} 米")
            print(f"   🏃 平均速度: {total_distance / (time_data[-1] - time_data[0]):.3f} m/s")
            
            print("\n🤏 夹爪状态:")
            print(f"   📐 夹爪维度: {gripper_qpos.shape[1]}")
            print(f"   📊 夹爪位置范围: [{np.min(gripper_qpos):.3f}, {np.max(gripper_qpos):.3f}]")
            
            # 分析动作变化
            action_changes = np.diff(actions, axis=0)
            action_smoothness = np.mean(np.std(action_changes, axis=0))
            print(f"\n📈 动作平滑度分析:")
            print(f"   📊 动作变化标准差: {action_smoothness:.6f}")
            print(f"   📈 最大单步变化: {np.max(np.abs(action_changes)):.6f}")
            
            # 如果需要保存图表
            if save_plots:
                plot_robot_data(actions, eef_pos, rewards, time_data, 
                              os.path.splitext(os.path.basename(file_path))[0])
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")


def plot_robot_data(actions, eef_pos, rewards, time_data, filename_prefix):
    """绘制机器人数据图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'机器人数据分析: {filename_prefix}', fontsize=16)
    
    # 时间轴（相对时间）
    time_rel = time_data - time_data[0]
    
    # 1. 动作数据
    axes[0, 0].plot(time_rel, actions[:, :6])  # 前6个动作维度
    axes[0, 0].set_title('机械臂动作 (前6维)')
    axes[0, 0].set_xlabel('时间 (秒)')
    axes[0, 0].set_ylabel('动作值')
    axes[0, 0].grid(True)
    axes[0, 0].legend([f'动作{i}' for i in range(6)], fontsize=8)
    
    # 2. 末端执行器位置
    axes[0, 1].plot(time_rel, eef_pos[:, 0], label='X')
    axes[0, 1].plot(time_rel, eef_pos[:, 1], label='Y')
    axes[0, 1].plot(time_rel, eef_pos[:, 2], label='Z')
    axes[0, 1].set_title('末端执行器位置')
    axes[0, 1].set_xlabel('时间 (秒)')
    axes[0, 1].set_ylabel('位置 (米)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. 3D轨迹
    ax_3d = fig.add_subplot(2, 2, 3, projection='3d')
    ax_3d.plot(eef_pos[:, 0], eef_pos[:, 1], eef_pos[:, 2])
    ax_3d.scatter(eef_pos[0, 0], eef_pos[0, 1], eef_pos[0, 2], 
                  color='green', s=100, label='起点')
    ax_3d.scatter(eef_pos[-1, 0], eef_pos[-1, 1], eef_pos[-1, 2], 
                  color='red', s=100, label='终点')
    ax_3d.set_title('3D轨迹')
    ax_3d.set_xlabel('X (米)')
    ax_3d.set_ylabel('Y (米)')
    ax_3d.set_zlabel('Z (米)')
    ax_3d.legend()
    
    # 4. 奖励
    axes[1, 1].plot(time_rel, rewards)
    axes[1, 1].set_title('奖励值')
    axes[1, 1].set_xlabel('时间 (秒)')
    axes[1, 1].set_ylabel('奖励')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📊 图表已保存: {filename_prefix}_analysis.png")
    plt.show()


def batch_analyze(directory: str, save_plots: bool = False):
    """批量分析目录中的所有HDF5文件"""
    hdf5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.hdf5', '.h5')):
                hdf5_files.append(os.path.join(root, file))
    
    if not hdf5_files:
        print("❌ 没有找到HDF5文件")
        return
    
    print(f"🔍 找到 {len(hdf5_files)} 个HDF5文件")
    
    # 统计信息
    total_episodes = len(hdf5_files)
    total_timesteps = 0
    total_duration = 0
    success_count = 0
    
    for i, file_path in enumerate(hdf5_files, 1):
        print(f"\n{'='*20} 文件 {i}/{total_episodes} {'='*20}")
        
        try:
            with h5py.File(file_path, 'r') as f:
                demo_data = f['data']['demo']
                actions = demo_data['actions'][()]
                rewards = demo_data['rewards'][()]
                time_data = demo_data['obs']['time'][()]
                
                timesteps = len(actions)
                duration = time_data[-1] - time_data[0]
                final_reward = rewards[-1]
                
                total_timesteps += timesteps
                total_duration += duration
                if final_reward > 0.5:  # 假设0.5以上为成功
                    success_count += 1
                
                print(f"📁 {os.path.basename(file_path)}")
                print(f"   ⏱️  时长: {duration:.1f}s, 步数: {timesteps}")
                print(f"   🎯 最终奖励: {final_reward:.3f}")
                
        except Exception as e:
            print(f"❌ 读取失败 {os.path.basename(file_path)}: {e}")
    
    # 总结统计
    print(f"\n{'='*60}")
    print("📈 批量分析总结:")
    print(f"   📊 总文件数: {total_episodes}")
    print(f"   ⏱️  总时长: {total_duration:.1f} 秒 ({total_duration/60:.1f} 分钟)")
    print(f"   📈 总时间步: {total_timesteps}")
    print(f"   🎯 成功率: {success_count}/{total_episodes} ({100*success_count/total_episodes:.1f}%)")
    print(f"   📊 平均时长: {total_duration/total_episodes:.1f} 秒/episode")
    print(f"   📈 平均步数: {total_timesteps/total_episodes:.0f} 步/episode")


def main():
    parser = argparse.ArgumentParser(description='机器人数据HDF5文件分析器')
    parser.add_argument('path', help='HDF5文件路径或目录路径')
    parser.add_argument('--plots', '-p', action='store_true', help='保存分析图表')
    parser.add_argument('--batch', '-b', action='store_true', help='批量分析模式')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print(f"❌ 路径不存在: {args.path}")
        return 1
    
    if args.batch or os.path.isdir(args.path):
        batch_analyze(args.path, args.plots)
    else:
        analyze_robot_data(args.path, args.plots)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
