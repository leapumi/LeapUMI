#!/usr/bin/env python3
"""
简化版HDF5文件结构查看器
快速查看HDF5文件的结构和基本信息
"""

import h5py
import numpy as np
import sys
import os


def print_structure(name, obj, indent=0):
    """打印HDF5对象结构"""
    spaces = "  " * indent
    
    if isinstance(obj, h5py.Dataset):
        # 数据集
        print(f"{spaces}📊 {name}: {obj.shape} {obj.dtype}")
        
        # # 显示数据范围（如果是数值类型且不太大）
        # if obj.dtype.kind in ['f', 'i', 'u'] and obj.size < 1000000:
        #     try:
        #         data = obj[()]
        #         if isinstance(data, np.ndarray) and data.size > 0:
        #             print(f"{spaces}   ↳ 范围: [{np.min(data):.3f}, {np.max(data):.3f}]")
        #     except:
        #         pass
                
        # 显示属性
        if obj.attrs:
            for attr_name, attr_value in obj.attrs.items():
                print(f"{spaces}   • {attr_name}: {attr_value}")
                
    elif isinstance(obj, h5py.Group):
        # 组/群
        print(f"{spaces}📁 {name}/")
        if obj.attrs:
            for attr_name, attr_value in obj.attrs.items():
                print(f"{spaces}   • {attr_name}: {attr_value}")


def explore_hdf5(file_path):
    """探索HDF5文件"""
    print(f"📂 文件: {os.path.basename(file_path)}")
    print(f"📏 大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    print("-" * 50)
    
    try:
        with h5py.File(file_path, 'r') as f:
            def visit_func(name, obj):
                level = name.count('/')
                print_structure(name.split('/')[-1] if '/' in name else name, obj, level)
            
            # 处理根级别的属性
            if f.attrs:
                print("🏷️  文件属性:")
                for attr_name, attr_value in f.attrs.items():
                    print(f"   • {attr_name}: {attr_value}")
                print()
            
            # 遍历所有对象
            f.visititems(visit_func)
            
    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python hdf5_quick_view.py <文件路径>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        sys.exit(1)
    
    explore_hdf5(file_path)
