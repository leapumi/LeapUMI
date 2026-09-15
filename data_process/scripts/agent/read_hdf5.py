#!/usr/bin/env python3
"""
HDF5文件读取脚本
用于读取和分析HDF5文件的内容、结构和数据格式
"""

import h5py
import numpy as np
import argparse
import os
from typing import Any, Dict, List
import sys


def print_separator(title: str, char: str = "=", width: int = 60):
    """打印分隔符"""
    print(f"\n{char * width}")
    print(f" {title} ".center(width, char))
    print(f"{char * width}")


def get_dataset_info(dataset: h5py.Dataset) -> Dict[str, Any]:
    """获取数据集的详细信息"""
    info = {
        'shape': dataset.shape,
        'dtype': str(dataset.dtype),
        'size': dataset.size,
        'chunks': dataset.chunks,
        'compression': dataset.compression,
        'compression_opts': dataset.compression_opts,
        'fillvalue': dataset.fillvalue,
        'maxshape': dataset.maxshape,
    }
    
    # 计算数据大小（字节）
    if dataset.dtype.kind in ['U', 'S']:  # 字符串类型
        item_size = dataset.dtype.itemsize
    else:
        item_size = dataset.dtype.itemsize
    
    info['size_bytes'] = dataset.size * item_size
    info['size_mb'] = info['size_bytes'] / (1024 * 1024)
    
    return info


def print_dataset_info(name: str, dataset: h5py.Dataset, indent: int = 0):
    """打印数据集信息"""
    prefix = "  " * indent
    info = get_dataset_info(dataset)
    
    print(f"{prefix}📊 Dataset: {name}")
    print(f"{prefix}   Shape: {info['shape']}")
    print(f"{prefix}   Type: {info['dtype']}")
    print(f"{prefix}   Size: {info['size']} elements ({info['size_mb']:.2f} MB)")
    
    if info['chunks']:
        print(f"{prefix}   Chunks: {info['chunks']}")
    if info['compression']:
        print(f"{prefix}   Compression: {info['compression']}")
    
    # 打印属性
    if dataset.attrs:
        print(f"{prefix}   Attributes:")
        for attr_name, attr_value in dataset.attrs.items():
            print(f"{prefix}     {attr_name}: {attr_value}")
    
    # 显示数据预览（如果数据不太大）
    if dataset.size <= 100:  # 只显示小数据集的完整内容
        print(f"{prefix}   Data:")
        try:
            data = dataset[()]
            if isinstance(data, np.ndarray):
                if data.ndim == 0:  # 标量
                    print(f"{prefix}     {data}")
                elif data.ndim == 1 and len(data) <= 10:  # 小的1D数组
                    print(f"{prefix}     {data}")
                else:  # 多维数组，显示形状和部分数据
                    print(f"{prefix}     Shape: {data.shape}")
                    print(f"{prefix}     Sample: {str(data).replace(chr(10), ' ')[:100]}...")
            else:
                print(f"{prefix}     {data}")
        except Exception as e:
            print(f"{prefix}     Error reading data: {e}")
    elif dataset.size <= 10000:  # 中等大小的数据集，显示统计信息
        try:
            if dataset.dtype.kind in ['f', 'i', 'u']:  # 数值类型
                print(f"{prefix}   Data preview (first 5 elements):")
                sample = dataset[:5] if dataset.ndim == 1 else dataset.flat[:5]
                print(f"{prefix}     {sample}")
                
                # 统计信息
                if dataset.size > 0:
                    data_sample = dataset[()]
                    if isinstance(data_sample, np.ndarray) and data_sample.size > 0:
                        print(f"{prefix}   Statistics:")
                        print(f"{prefix}     Min: {np.min(data_sample):.6f}")
                        print(f"{prefix}     Max: {np.max(data_sample):.6f}")
                        print(f"{prefix}     Mean: {np.mean(data_sample):.6f}")
                        print(f"{prefix}     Std: {np.std(data_sample):.6f}")
        except Exception as e:
            print(f"{prefix}   Error computing statistics: {e}")


def print_group_info(name: str, group: h5py.Group, indent: int = 0):
    """打印组信息"""
    prefix = "  " * indent
    print(f"{prefix}📁 Group: {name}")
    
    # 打印属性
    if group.attrs:
        print(f"{prefix}   Attributes:")
        for attr_name, attr_value in group.attrs.items():
            print(f"{prefix}     {attr_name}: {attr_value}")


def explore_hdf5_structure(item: h5py.Group, path: str = "/", indent: int = 0):
    """递归探索HDF5文件结构"""
    for key in item.keys():
        current_path = f"{path}{key}" if path.endswith("/") else f"{path}/{key}"
        obj = item[key]
        
        if isinstance(obj, h5py.Group):
            print_group_info(key, obj, indent)
            explore_hdf5_structure(obj, current_path, indent + 1)
        elif isinstance(obj, h5py.Dataset):
            print_dataset_info(key, obj, indent)


def analyze_hdf5_file(filepath: str, show_data: bool = True, max_preview_size: int = 100):
    """分析HDF5文件"""
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return
    
    print_separator(f"分析文件: {os.path.basename(filepath)}")
    print(f"📂 文件路径: {filepath}")
    print(f"📏 文件大小: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
    
    try:
        with h5py.File(filepath, 'r') as f:
            print_separator("文件结构")
            
            # 打印根级别属性
            if f.attrs:
                print("🏷️  根属性:")
                for attr_name, attr_value in f.attrs.items():
                    print(f"   {attr_name}: {attr_value}")
                print()
            
            # 探索文件结构
            explore_hdf5_structure(f)
            
            # 统计信息
            print_separator("统计信息")
            
            def count_items(group):
                groups = 0
                datasets = 0
                for item in group.values():
                    if isinstance(item, h5py.Group):
                        groups += 1
                        sub_g, sub_d = count_items(item)
                        groups += sub_g
                        datasets += sub_d
                    elif isinstance(item, h5py.Dataset):
                        datasets += 1
                return groups, datasets
            
            total_groups, total_datasets = count_items(f)
            print(f"📁 总组数: {total_groups}")
            print(f"📊 总数据集数: {total_datasets}")
            
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")


def list_hdf5_files(directory: str) -> List[str]:
    """列出目录中的所有HDF5文件"""
    hdf5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.hdf5', '.h5')):
                hdf5_files.append(os.path.join(root, file))
    return sorted(hdf5_files)


def main():
    parser = argparse.ArgumentParser(description='读取和分析HDF5文件的内容、结构和数据格式')
    parser.add_argument('path', help='HDF5文件路径或包含HDF5文件的目录路径')
    parser.add_argument('--list', '-l', action='store_true', help='列出目录中的所有HDF5文件')
    parser.add_argument('--no-data', action='store_true', help='不显示数据内容，只显示结构')
    parser.add_argument('--preview-size', type=int, default=100, help='数据预览的最大元素数量')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        hdf5_files = list_hdf5_files(args.path)
        
        if args.list:
            print_separator("HDF5文件列表")
            for i, filepath in enumerate(hdf5_files, 1):
                rel_path = os.path.relpath(filepath, args.path)
                size_mb = os.path.getsize(filepath) / (1024*1024)
                print(f"{i:3d}. {rel_path} ({size_mb:.2f} MB)")
            
            if hdf5_files:
                print(f"\n总共找到 {len(hdf5_files)} 个HDF5文件")
                
                # 询问是否要分析特定文件
                try:
                    choice = input("\n输入文件编号进行分析 (回车退出): ").strip()
                    if choice:
                        file_index = int(choice) - 1
                        if 0 <= file_index < len(hdf5_files):
                            analyze_hdf5_file(hdf5_files[file_index], not args.no_data, args.preview_size)
                        else:
                            print("❌ 无效的文件编号")
                except (ValueError, KeyboardInterrupt):
                    print("退出")
            else:
                print("❌ 在指定目录中没有找到HDF5文件")
        else:
            # 分析所有文件
            for filepath in hdf5_files:
                analyze_hdf5_file(filepath, not args.no_data, args.preview_size)
                
    elif os.path.isfile(args.path):
        analyze_hdf5_file(args.path, not args.no_data, args.preview_size)
    else:
        print(f"❌ 路径不存在: {args.path}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
