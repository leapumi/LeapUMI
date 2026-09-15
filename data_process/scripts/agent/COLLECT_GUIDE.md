# HDF5文件收集脚本使用说明

## 📦 脚本说明

### `simple_collect.sh` - 完整功能版本
功能完整的HDF5文件收集和重命名脚本。

### `ultrafast_collect.sh` - 超快速版本  
极简版本，专注于速度。

## 🚀 使用方法

### 基本用法
```bash
# 使用默认参数
./simple_collect.sh

# 自定义参数
./simple_collect.sh <输入目录> <输出目录> <文件前缀>
```

### 示例
```bash
# 默认：output -> output/collected，前缀 leapdata
./simple_collect.sh

# 自定义：从 output 目录收集到 final_data 目录，前缀 robotdata
./simple_collect.sh output final_data robotdata

# 超快速版本
./ultrafast_collect.sh output final_data robotdata
```

## 📊 输出示例

```bash
📦 收集HDF5文件: output -> output/collected (前缀: leapdata)
🔍 查找文件...
📊 找到 51 个文件
🔄 移动文件...
✅ 0: leapdata_0.hdf5 <- leap_action_0626_15_17/leap_action_0.hdf5
✅ 1: leapdata_1.hdf5 <- leap_action_0626_15_17/leap_action_1.hdf5
✅ 2: leapdata_2.hdf5 <- leap_action_0626_15_22/leap_action_0.hdf5
✅ 3: leapdata_3.hdf5 <- leap_action_0626_15_22/leap_action_2.hdf5
...

🎉 完成! 成功移动: 51/51
📁 文件保存在: output/collected
📋 文件范围: leapdata_0.hdf5 到 leapdata_50.hdf5
📄 索引文件: output/collected/file_index.txt
```

## 📄 索引文件

脚本会自动生成 `file_index.txt` 用于溯源：

```
# HDF5文件重命名索引
# 生成时间: 2025年 06月 30日 星期一 20:52:28 CST
# 格式: 新文件名 <- 原目录/原文件名

leapdata_0.hdf5 <- leap_action_0626_15_17/leap_action_0.hdf5
leapdata_1.hdf5 <- leap_action_0626_15_17/leap_action_1.hdf5
leapdata_2.hdf5 <- leap_action_0626_15_22/leap_action_0.hdf5
...
```

## 🔄 处理逻辑

1. **扫描**: 在输入目录下查找所有 `leap_action_*.hdf5` 文件
2. **排序**: 按目录名和原始ID排序
3. **移动**: 将文件移动（不是复制）到输出目录并重命名为连续ID
4. **索引**: 生成包含原始路径映射的索引文件

## ⚠️ 注意事项

- **文件会被移动**：原始文件将从源位置消失，移动到新位置
- **速度很快**：使用移动而非复制，速度极快
- **原子操作**：每个文件移动是原子操作，中途中断不会损坏文件
- **索引追踪**：通过索引文件可以追溯每个文件的原始位置

## 🛠️ 故障排除

### 权限问题
```bash
chmod +x simple_collect.sh
```

### 目录不存在
脚本会自动创建输出目录，但请确保输入目录存在。

### 文件冲突
如果输出目录已存在同名文件，移动操作会覆盖现有文件。
