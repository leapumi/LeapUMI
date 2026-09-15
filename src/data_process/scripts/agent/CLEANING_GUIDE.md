# HDF5数据清理工具使用指南

## 🎯 功能概述

这套工具专门用于清理和还原滚动存储的机器人遥操作HDF5数据文件。主要解决以下问题：

1. **滚动存储问题**: 数据文件以滚动方式存储，后续文件包含前面所有数据
2. **垃圾文件清理**: 自动识别和跳过trash目录中的无用文件
3. **数据还原**: 将滚动数据切分还原为独立的正确数据段
4. **批量处理**: 支持批量处理多个目录

## 📁 输入数据结构

期望的输入目录结构：
```
leap_action_0626_15_22/
├── leap_action_0.hdf5      # 保留：278步
├── leap_action_2.hdf5      # 保留：441步 (包含0-441步数据)
├── leap_action_3.hdf5      # 保留：638步 (包含0-638步数据)
├── leap_action_6.hdf5      # 保留：912步 (包含0-912步数据)
└── trash/                  # 垃圾目录
    ├── leap_action_1.hdf5  # 跳过
    ├── leap_action_4.hdf5  # 跳过
    └── leap_action_5.hdf5  # 跳过
```

## 📤 输出数据结构

清理后的输出结构：
```
output/
└── leap_action_0626_15_22/
    ├── leap_action_0.hdf5  # 还原：[0:278] → 278步
    ├── leap_action_2.hdf5  # 还原：[278:441] → 163步
    ├── leap_action_3.hdf5  # 还原：[441:638] → 197步
    └── leap_action_6.hdf5  # 还原：[638:912] → 274步
```

## 🛠️ 工具说明

### 1. `data_cleaner.py` - 完整功能脚本

最全面的数据清理工具，支持详细配置和干运行模式。

**基本用法:**
```bash
# 干运行模式（推荐先使用）
python3 data_cleaner.py leap_action_0626_15_22/ output/ --dry-run

# 实际执行
python3 data_cleaner.py leap_action_0626_15_22/ output/

# 批量处理
python3 data_cleaner.py 'leap_action_0626_*/' output/ --batch
```

**功能特点:**
- ✅ 支持干运行模式
- ✅ 详细的处理报告
- ✅ 错误处理和验证
- ✅ 批量处理支持

### 2. `quick_clean.py` - 快速清理脚本

轻量级的快速清理工具，适合简单场景。

**基本用法:**
```bash
# 单目录清理
python3 quick_clean.py leap_action_0626_15_22/ output/

# 批量清理（使用通配符）
python3 quick_clean.py 'leap_action_0626_*/' output/
```

**功能特点:**
- ✅ 操作简单快速
- ✅ 支持通配符批量处理
- ✅ 实时显示处理进度

### 3. `batch_clean_all.py` - 自动批量处理

全自动批量处理所有符合条件的目录。

**基本用法:**
```bash
# 列出要处理的目录（不执行）
python3 batch_clean_all.py --list-only

# 批量处理所有目录到默认output目录
python3 batch_clean_all.py

# 指定输出目录
python3 batch_clean_all.py --output cleaned_data
```

**功能特点:**
- ✅ 全自动扫描和处理
- ✅ 详细统计报告
- ✅ 失败目录追踪

## 🚀 推荐使用流程

### 步骤1: 预览和验证
```bash
# 1. 列出所有待处理目录
python3 batch_clean_all.py --list-only

# 2. 选择一个目录进行干运行测试
python3 data_cleaner.py leap_action_0626_15_22/ test_output/ --dry-run
```

### 步骤2: 小规模测试
```bash
# 3. 实际处理一个目录进行验证
python3 quick_clean.py leap_action_0626_15_22/ test_output/

# 4. 验证生成的文件
python3 hdf5_quick_view.py test_output/leap_action_0626_15_22/leap_action_0.hdf5
```

### 步骤3: 批量处理
```bash
# 5. 确认无误后，批量处理所有目录
python3 batch_clean_all.py --output final_output
```

## 📊 输出示例

### 干运行模式输出
```
🔍 扫描目录: leap_action_0626_15_22
📋 找到 4 个需要保留的文件:
   📄 leap_action_0.hdf5
   📄 leap_action_2.hdf5
   📄 leap_action_3.hdf5
   📄 leap_action_6.hdf5

📏 分析文件数据长度...
   📊 leap_action_0.hdf5: 278 步
   📊 leap_action_2.hdf5: 441 步
   📊 leap_action_3.hdf5: 638 步
   📊 leap_action_6.hdf5: 912 步

✂️  计算切分索引...
   📄 leap_action_0.hdf5: 切分范围 [0:278] (278 步)
   📄 leap_action_2.hdf5: 切分范围 [278:441] (163 步)
   📄 leap_action_3.hdf5: 切分范围 [441:638] (197 步)
   📄 leap_action_6.hdf5: 切分范围 [638:912] (274 步)
```

### 批量处理输出
```
🎉 批量处理完成！
============================================================
📊 处理统计:
   📁 处理目录数: 8
   📄 成功处理文件数: 45
   ✅ 成功目录数: 8
   ❌ 失败目录数: 0

📁 所有清理后的数据保存在: /path/to/output
```

## ⚠️ 注意事项

### 1. 磁盘空间
- 确保有足够的磁盘空间（通常需要1.5-2倍原始数据大小）
- 处理大文件时监控磁盘使用情况

### 2. 内存使用
- 大文件处理时需要足够内存
- 如遇内存不足，可分批处理小目录

### 3. 数据完整性
- 处理前建议备份原始数据
- 使用干运行模式验证处理逻辑
- 处理后验证关键文件的数据完整性

### 4. 文件权限
- 确保对源文件有读权限
- 确保对输出目录有写权限

## 🔧 故障排除

### 常见问题

**1. 找不到文件**
```bash
❌ 错误: [Errno 2] No such file or directory
```
- 检查文件路径是否正确
- 确保当前工作目录正确

**2. 权限错误**
```bash
❌ 错误: Permission denied
```
- 检查文件和目录权限
- 使用适当的用户权限运行

**3. 内存不足**
```bash
❌ 错误: MemoryError
```
- 关闭其他程序释放内存
- 分批处理较小的目录

**4. 磁盘空间不足**
```bash
❌ 错误: No space left on device
```
- 清理磁盘空间
- 选择有足够空间的输出目录

### 验证命令

```bash
# 验证输出文件结构
python3 hdf5_quick_view.py output/leap_action_0626_15_22/leap_action_0.hdf5

# 检查文件大小
ls -lh output/leap_action_0626_15_22/

# 统计总文件数
find output/ -name "*.hdf5" | wc -l
```

## 📈 性能优化建议

1. **SSD存储**: 使用SSD作为输出目录可大幅提升处理速度
2. **并行处理**: 大规模处理时可分组并行执行
3. **内存管理**: 处理超大文件时定期清理内存
4. **网络存储**: 避免在网络存储上直接处理大文件
