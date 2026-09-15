


1. 查看hdf5 文件内容
```bash
python hdf5_quick_view.py /media/yaxun/B197/teleop_data/output/leap_action_0626_15_22/leap_action_16.hdf5

```
--------------------------------------------------
📁 data/
  📁 demo/
    📊 actions: (101, 23) float64
    📊 dones: (101,) bool
    📁 obs/
      📊 agentview_image: (101, 720, 1280, 3) uint8
      📊 arm_qpos: (101, 6) float64
      📊 depth_image: (101, 720, 1280) float32
      📊 eef_pos: (101, 3) float64
      📊 eef_rot: (101, 4) float64
      📊 gripper_qpos: (101, 16) float32
      📊 time: (101,) float64
    📊 rewards: (101,) float64

2. 依据trash对文件进行切分,保留



```bash
python3 batch_clean_all.py


python3 quick_clean.py 'leap_action_xxxx' output
```

3. 提取命名

python3 quick_collect.py output output/collected leapdata






`

```



```



# HDF5数据分析工具集

这是一套用于读取和分析HDF5文件的Python脚本工具集，特别适用于机器人遥操作数据的分析。

## 📋 脚本列表

### 1. `hdf5_quick_view.py` - 快速查看器
快速显示HDF5文件的结构和基本信息。

```bash
python hdf5_quick_view.py <文件路径>
```

**示例:**
```bash
python hdf5_quick_view.py data_wo_images/leap_action_0.hdf5
```

**输出:**
- 文件大小
- 数据集结构
- 数据类型和形状
- 数值数据的范围统计

### 2. `read_hdf5.py` - 详细分析器
提供HDF5文件的详细分析，包括完整的数据结构、属性和内容预览。

```bash
python read_hdf5.py <文件路径或目录> [选项]
```

**选项:**
- `--list, -l`: 列出目录中的所有HDF5文件
- `--no-data`: 只显示结构，不显示数据内容
- `--preview-size N`: 设置数据预览的最大元素数量

**示例:**
```bash
# 分析单个文件
python read_hdf5.py data_wo_images/leap_action_0.hdf5

# 列出目录中的文件并选择分析
python read_hdf5.py data_wo_images/ --list

# 只显示结构，不显示数据
python read_hdf5.py data_wo_images/leap_action_0.hdf5 --no-data
```

### 3. `batch_hdf5_info.py` - 批量信息统计
批量分析目录中的所有HDF5文件，生成统计报告。

```bash
python batch_hdf5_info.py <目录路径>
```

**示例:**
```bash
python batch_hdf5_info.py data_wo_images/
```

**输出:**
- 文件数量和总大小
- 成功率统计
- 时间步和时长统计
- 详细文件列表
- 按目录分组统计

### 4. `robot_data_analyzer.py` - 机器人数据专用分析器
专门用于分析机器人遥操作数据的HDF5文件。

```bash
python robot_data_analyzer.py <文件路径或目录> [选项]
```

**选项:**
- `--plots, -p`: 生成分析图表（需要matplotlib）
- `--batch, -b`: 批量分析模式

**示例:**
```bash
# 分析单个文件
python robot_data_analyzer.py data_wo_images/leap_action_0.hdf5

# 生成图表
python robot_data_analyzer.py data_wo_images/leap_action_0.hdf5 --plots

# 批量分析
python robot_data_analyzer.py data_wo_images/ --batch
```

**输出:**
- 时间步数和采样频率
- 动作数据分析
- 末端执行器轨迹分析
- 夹爪状态分析
- 动作平滑度分析
- 3D轨迹图（如果使用--plots）

### 5. `hdf5_tools.py` - 统一工具入口
提供统一的命令行界面来访问所有分析功能。

```bash
python hdf5_tools.py <命令> <路径> [选项]
```

**命令:**
- `quick`: 快速查看文件结构
- `analyze`: 详细分析文件
- `batch`: 批量信息统计
- `robot`: 机器人数据分析
- `interactive`: 交互式模式
- `help`: 显示帮助信息

**示例:**
```bash
# 快速查看
python hdf5_tools.py quick data_wo_images/leap_action_0.hdf5

# 批量统计
python hdf5_tools.py batch data_wo_images/

# 机器人数据分析（带图表）
python hdf5_tools.py robot data_wo_images/leap_action_0.hdf5 --plots

# 交互式模式
python hdf5_tools.py interactive data_wo_images/
```

## 🧹 数据清理工具

### 6. `data_cleaner.py` - 完整数据清理器
用于清理和还原滚动存储的HDF5数据文件。

```bash
python data_cleaner.py <输入目录> <输出目录> [选项]
```

**选项:**
- `--dry-run`: 预演模式，只显示操作但不执行
- `--batch`: 批量处理模式

**示例:**
```bash
# 预演模式
python data_cleaner.py leap_action_0626_15_22/ output/ --dry-run

# 实际清理
python data_cleaner.py leap_action_0626_15_22/ output/

# 批量清理
python data_cleaner.py 'leap_action_0626_*/' output/ --batch
```

### 7. `quick_clean.py` - 快速清理器
轻量级的快速数据清理工具。

```bash
python quick_clean.py <输入目录模式> <输出目录>
```

**示例:**
```bash/media/yaxun/B197/teleop_data/output/leap_action_0626_15_22/leap_action_16.hdf5
# 单目录清理
python quick_clean.py leap_action_0626_15_22/ output/

# 批量清理
python quick_clean.py 'leap_action_0626_*/' output/
```

### 8. `batch_clean_all.py` - 自动批量清理器
全自动批量处理所有符合条件的目录。

```bash
python batch_clean_all.py [选项]
```

**选项:**
- `--output, -o`: 指定输出目录
- `--list-only`: 只列出要处理的目录

**示例:**
```bash
# 列出待处理目录
python batch_clean_all.py --list-only

# 自动批量清理
python batch_clean_all.py --output cleaned_data
```

**功能说明:**
- 🔍 自动扫描所有 `leap_action_0626_*` 目录
- 🗑️ 识别并跳过 `trash/` 目录中的文件
- ✂️ 根据滚动数据结构自动切分还原数据
- 📊 生成详细的处理统计报告

## 🛠️ 依赖要求

### 基础依赖
```bash
pip install h5py numpy
```

### 可选依赖（用于图表生成）
```bash
pip install matplotlib pandas
```

### 完整安装
```bash
pip install h5py numpy matplotlib pandas
```

## 📊 数据格式支持

这些工具主要设计用于分析机器人遥操作数据，支持以下数据结构：

```
/data/demo/
├── actions          # 动作数据 (N, action_dim)
├── dones           # 完成标志 (N,)
├── rewards         # 奖励值 (N,)
└── obs/            # 观测数据
    ├── arm_qpos    # 机械臂关节角度 (N, 6)
    ├── eef_pos     # 末端执行器位置 (N, 3)
    ├── eef_rot     # 末端执行器旋转 (N, 4)
    ├── gripper_qpos # 夹爪位置 (N, gripper_dim)
    └── time        # 时间戳 (N,)
```

## 🚀 使用建议

1. **首次使用**: 使用`quick`命令快速了解数据结构
2. **详细分析**: 使用`robot`命令进行专业的机器人数据分析
3. **批量处理**: 使用`batch`命令统计多个文件的信息
4. **交互探索**: 使用`interactive`模式进行交互式分析

## 📈 输出示例

### 快速查看输出
```
📂 文件: leap_action_0.hdf5
📏 大小: 0.09 MB
--------------------------------------------------
📁 data/
  📁 demo/
    📊 actions: (223, 23) float64
       ↳ 范围: [-0.834, 1.518]
    📊 dones: (223,) bool
    ...
```

### 机器人数据分析输出
```
🤖 分析机器人数据: leap_action_0.hdf5
============================================================
📊 数据集概览:
   📈 时间步数: 223
   ⏱️  持续时间: 44.41 秒
   🔄 采样频率: 5.0 Hz
   ✅ 完成状态: 是
   🎯 最终奖励: 1.000
...
```

## ⚠️ 注意事项

1. 确保有足够的内存来加载大型HDF5文件
2. 图表生成功能需要图形界面支持
3. 某些统计功能可能对超大文件较慢
4. 建议在分析前使用快速查看模式了解文件结构

## 🔧 故障排除

### 常见问题

1. **ImportError: No module named 'h5py'**
   ```bash
   pip install h5py
   ```

2. **无法显示图表**
   - 确保安装了matplotlib
   - 确保有图形界面支持

3. **内存不足**
   - 使用`--no-data`选项跳过数据内容显示
   - 分批处理大型文件

4. **文件权限问题**
   - 确保对HDF5文件有读取权限
   - 检查文件路径是否正确
