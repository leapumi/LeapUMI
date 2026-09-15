# 🧹 HDF5数据清理工具集

## 📋 快速开始

### 1. 查看所有待处理目录
```bash
python3 batch_clean_all.py --list-only
```

### 2. 测试单个目录（推荐）
```bash
# 干运行模式验证
python3 data_cleaner.py leap_action_0626_15_17/ test_output/ --dry-run

# 实际处理
python3 quick_clean.py leap_action_0626_15_17/ test_output/
```

### 3. 批量清理所有数据
```bash
python3 batch_clean_all.py --output cleaned_data
```

### 4. 验证结果
```bash
python3 hdf5_quick_view.py cleaned_data/leap_action_0626_15_17/leap_action_0.hdf5
```

## 🛠️ 工具列表

| 脚本名称 | 功能 | 推荐场景 |
|---------|------|----------|
| `data_cleaner.py` | 完整功能清理器 | 精确控制、详细报告 |
| `quick_clean.py` | 快速清理器 | 简单快速处理 |
| `batch_clean_all.py` | 自动批量处理 | 大规模自动化 |
| `demo_cleaning.py` | 演示脚本 | 学习和测试 |

## 📊 处理逻辑

1. **扫描目录**: 找到所有 `leap_action_*.hdf5` 文件
2. **过滤垃圾**: 跳过 `trash/` 目录中的文件  
3. **分析长度**: 读取每个文件的数据步数
4. **计算切分**: 根据滚动存储规律计算切分点
5. **还原数据**: 将滚动数据切分为独立的正确数据段
6. **验证输出**: 确保数据完整性和正确性

## 🎯 一键演示

```bash
python3 demo_cleaning.py
```

运行演示脚本，交互式体验完整的数据清理流程。
