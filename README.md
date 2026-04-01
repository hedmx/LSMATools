# LSMATOOLS - Lumbar Spine MRI Analysis Tools

**智能腰椎 MRI 分析工具** | Automated Lumbar Spine MRI Segmentation and Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 简介 (Introduction)

LSMATOOLS 是一个基于 Python 的腰椎 MRI 图像智能分析工具，专为医学影像研究人员设计。它能够自动识别椎体结构、检测终板位置、分析椎体前缘/后缘几何特征，并提供可视化的分析结果。

LSMATOOLS is an intelligent lumbar MRI analysis tool designed for medical imaging researchers. It automatically identifies vertebral structures, detects endplate positions, analyzes anterior/posterior vertebral geometry, and provides visualized analysis results.

### 🔑 核心功能 (Key Features)

- **自动椎管分割** - 基于 Otsu 阈值和形态学操作的椎管 ROI 提取
- **终板智能检测** - 状态机驱动的上升沿/下降沿终板识别算法（V15 全局回退机制）
- **椎体前缘分析** - 双模态（压脂/压水）融合的前缘线检测
- **椎体后缘分析** - 双皮质线体系的后缘角点定位
- **脊髓 ROI 分析** - 自动化的脊髓区域分割与测量
- **胸椎扩展识别** - 支持 L5/S1 → T12-T8 的自动命名
- **可视化输出** - 双图对比（左压脂/右压水），含完整标注
- **NIfTI 掩模导出** - 多标签椎体掩模文件（影像组学研究标准格式）
- **几何数据 CSV** - 椎体面积、角度、高度等定量参数

### 🎯 适用场景 (Use Cases)

- 腰椎 MRI 图像的自动化预处理
- 椎体几何参数的批量测量
- 终板退行性变的定量分析
- 术前规划与术后评估
- 医学影像 AI 算法的基准测试
- 影像组学特征提取

---

## 🚀 快速开始 (Quick Start)

### 安装依赖 (Install Dependencies)

```bash
pip install -r requirements.txt
```

### 基础用法 (Basic Usage)

LSMATOOLS V15 采用**命令行交互式**运行方式：

```bash
# 运行主程序（交互式菜单）
python main.py

# 选择运行模式：
#   1. 单张图像测试 - 处理单个 NIfTI 文件
#   2. 批量处理 - 批量处理多个病例
```

#### 模式 1：单张图像测试

运行后输入 `1`，按提示输入：
- NIfTI 文件路径
- Metadata JSON 路径（如有）
- 输出目录

#### 模式 2：批量处理

运行后输入 `2`，按提示输入：
- 输入目录（包含多个子目录，每个子目录含 scan.nii.gz）
- 输出目录

### 代码级使用 (Programmatic Usage)

如需在 Python 代码中调用：

```python
from LSMATOOLS_CP.main import test_single_image

# 单病例处理
test_single_image(
    nifti_path='path/to/scan.nii.gz',
    metadata_path='path/to/metadata.json',
    output_dir='./output'
)

# 或直接调用核心模块
from LSMATOOLS_CP.segmentation.anterior_edge import find_endplates_on_water_image
from LSMATOOLS_CP.postprocessing.visualization import visualize_results
```

> 注意：完整的高级封装类可在后续版本中添加，目前建议使用上述函数调用方式或交互式菜单。

### 典型目录结构

```
project/
├── main.py                     # 主程序
├── requirements.txt          # 依赖
├── LSMATOOLS_CP/
│   ├── config/              # 参数配置
│   ├── preprocessing/       # 预处理
│   ├── segmentation/        # 分割算法
│   └── postprocessing/      # 后处理
└── data/
    ├── case001/
    │   ├── scan.nii.gz      # T2 Dixon W 图像
    │   └── metadata.json    # 元数据（可选）
    └── case002/
        └── scan.nii.gz
```

---

## 📸 可视化预览 (Visualization Preview)

以下是 LSMATOOLS V15 的典型输出示例（点击放大查看细节）：

### Case 1: L5/S1 起始场景（高分辨率 HR，0.5mm）

![Case 1 - L5/S1 起始](examples/Clumbar_T2_TSE_DIXON_SAG_W_0005_TRACED.png)

**图像特征**:
- ✅ **序列类型**: T2 Dixon W（压脂图）
- ✅ **像素间距**: 0.5mm（高分辨率）
- ✅ **椎体链路**: S1 → L5 → L4 → L3 → L2 → L1（完整腰椎）
- ✅ **可视化内容**:
  - 白色实线：皮质线 1（椎管前壁）
  - 紫色虚线：皮质线 1 二次平滑（所有扫描检测基准 + 椎体弧度可视化）
  - 橙色实线：背部线（5mm 平滑增强）
  - 橙/绿实线：上/下终板线
  - 红色实线：双模态融合前缘线
  - 黄色轮廓：脊髓 ROI
  - 青色虚线框：椎体四角点

---

### Case 2: 胸腰段联合扫描（标准分辨率 STD，0.8mm，并行采集 P2）

![Case 2 - 胸腰段联合](examples/Clumbar_T2_TSE_DIXON_SAG_P2_320_W_0008_TRACED.png)

**图像特征**:
- ✅ **序列类型**: T2 Dixon W（压脂图，并行采集）
- ✅ **像素间距**: 0.8mm（标准分辨率）
- ✅ **场强修正**: 并行采集 × 0.92
- ✅ **椎体链路**: L5 → L4 → L3 → L2 → L1 → T12（包含胸椎）
- ✅ **V14 新特性验证**:
  - 胸椎命名扩展（T12 自动识别）
  - 几何中心标注优化（文字位于椎体中心高度）
  - 背部线平滑增强（与皮质线 1 参数一致）

---

### 可视化语义说明 (Visualization Semantics)

| 颜色 | 线条样式 | 解剖结构 | 备注 |
|------|---------|---------|------|
| ⚪ 白色 | 实线 | 皮质线 1（Cortical Line 1） | 椎管前壁高信号边界 |
| 🟣 紫色 | 虚线 | 皮质线 1 二次平滑（Smoothed C1） | 所有扫描检测基准 + 椎体弧度可视化 |
| 🟠 橙色 | 实线 | 背部线（Dorsal Line） | V14 增强：5mm 移动均值平滑 |
| 🔴 番茄红 | 实线 | 上终板（Superior Endplate） | `ep_type='superior'` |
| 🟢 草绿 | 实线 | 下终板（Inferior Endplate） | `ep_type='inferior'` |
| 🔴 红色 | 实线 | 前缘线（Anterior Edge） | 双模态融合输出（上升沿 + 下降沿） |
| 🟡 黄色 | 轮廓线 | 脊髓 ROI（Spinal Cord ROI） | 骨髓区域分割结果 |
| 🔵 深蓝 | 细实线 | 法线扫描线（V15） | 40 条，1mm 间距，右图调试显示 |
| 🔵 青色 | 虚线框 | 椎体四角点 | 掩模生成基准点 |

**左图（压脂图）**: 最终输出视图，仅显示关键结构  
**右图（压水图）**: 完整调试视图，包含所有中间态和检测点

---

## 📊 输出说明 (Output Description)

### 可视化文件 (Visualization Files)

| 文件名模式 | 内容描述 |
|-----------|---------|
| `*_TRACED.png` | 双图对比：左压脂 + 右压水，含所有标注 |

### 数据文件 (Data Files)

| 文件名模式 | 内容描述 |
|-----------|---------|
| `*_LOG.json` | 完整处理日志（含调试信息、参数配置、检测结果） |
| `*_masks.nii.gz` | 多标签椎体掩模（L5=2, L4=3, ..., S1=1） |
| `*_coords.csv` | 椎体几何参数（四角点坐标、面积、角度、高度） |

### 数据标注 (Data Annotations)

每个椎体输出以下参数（CSV 格式）：

```csv
level,BP_top_row,BP_top_col,AP_top_row,AP_top_col,...,area_mm2,angle_sup_deg,angle_inf_deg,angle_c1_deg,angle_ant_deg,ant_height_mm,pos_height_mm
L5,245.5,312.0,248.0,280.5,...,4523.67,12.3,8.7,45.2,38.9,42.5,38.2
```

- **level**: 椎体名称 (L5/L4/L3/L2/L1/T12/T11/...)
- **BP/AP_top/bot_row/col**: 后/前缘上/下角点行列坐标（像素单位）
- **area_mm²**: 椎体截面积（平方毫米）
- **Sup°**: 上终板角度
- **Inf°**: 下终板角度
- **C1°**: 皮质线1 角度
- **Ant°**: 前缘线角度
- **ant/pos_height_mm**: 前/后缘高度（毫米）

---

## 🧠 算法架构 (Algorithm Architecture)

LSMATOOLS V15 采用模块化设计，从输入到输出经历以下处理流程：

### 系统总体流程 (System Overview)

```
输入 (W 压脂图 NIfTI 3D)，推荐使用 https://github.com/hedmx/dicom_converter_enhanced 进行DICOM to NIfTI 转换。
    │
    ├─ [预处理] 序列筛选 + 最优切片选择
    │       ├─ 几何硬约束过滤（列中心/宽高比/最小高度）
    │       └─ 复合评分：垂直覆盖行数 × (1 + 0.2×形状分)
    │
    ├─ [SpinalCanalProcessor] 椎管分割 → 皮质线 1 + 皮质线1 二次平滑
    │       ├─ segment_initial(): 双区域 Otsu + 最大连通域 + 1px 膨胀
    │       ├─ extract_boundary(): 骨骼化 + 追踪 → c1/c2 原始点
    │       ├─ find_dorsal_edge(): 参考信号回退 5mm + 最大梯度点
    │       └─ smooth_boundary(): MAD 过滤 + 线性插值 + 移动均值平滑
    │
    ├─ [SpinalCordLocator] 脊髓 ROI 定位
    │
    ├─ [V15 法线扫描线生成] 40 条扫描线（1mm 间距）
    │       ├─ 逐点计算皮质线 1 局部切线（前后差分）
    │       ├─ 切线顺时针旋转 90° 得法线 (nx, ny)
    │       └─ 每条线沿法线方向偏移 k×step_mm（k=1..40）
    │
    └─ [压水图分析] F 序列
            │
            ├─ [仿射坐标对齐] 压脂 → 压水 扫描线映射
            │
            ├─ [终板检测] 
            │       ├─ 三区信号统计（Otsu）→ high/low mean 1/2/3
            │       ├─ 状态机下降沿/上升沿检测（含回退重扫）
            │       ├─ 弧长坐标系聚类 → consensus_endplates
            │       ├─ anatomical_gap_correction() 解剖间距复核
            │       ├─ 全局回退：终板线<5 条时降低阈值重扫
            │       └─ 点数回退：单条终板线点数<28 时降阈值重扫（最多 2 轮）
            │
            ├─ [前缘检测 - 上升沿主流程]
            │       ├─ find_arc_roi_min_points() 谷底查找
            │       └─ refine_arc_roi_to_anterior_edge() 上升沿精修
            │               → arc_refined（含 refined/kept/kept_low flag）
            │
            ├─ [前缘检测 - 下降沿并行方案]
            │       └─ find_anterior_edge_by_descent() 法线下降沿检测
            │               ├─ 逐行插值映射 row→(base_col, nx, ny)
            │               ├─ 终板段划分（lower_ep→上终板，upper_ep→下终板）
            │               ├─ 沿法线双线性插值采样（20mm 起点，最多 20mm）
            │               └─ 滑动最大值 + 绝对低信号双条件触发
            │
            ├─ [双模态融合 → 红色前缘线]
            │       ├─ 上升沿 refined 点 + 下降沿 confirmed 点 → 合并点集
            │       ├─ filter_arc_roi_by_dense_offset() 密集窗口过滤（expand_ratio=3.0）
            │       └─ MAD + 插值 + 8mm 平滑 → 红线
            │
            └─ [V14 椎体链路分析]
                    ├─ identify_vertebrae_chain() 识别完整椎体链路
                    │       ├─ 自动判定 S1/L5 起始位置
                    │       ├─ 命名扩展至 T12-T8（禁用 V6/V7 占位符）
                    │       └─ 向上追踪 L4/L3/L2/L1
                    ├─ compute_geometric_center() 计算几何中心
                    │       └─ 四角点平均坐标 → 文字标注位置
                    ├─ generate_vertebra_masks() 椎体掩模生成（四条真实曲线拼合）
                    └─ export_to_nifti_csv() 导出 NIfTI 掩模 + CSV 几何数据
```

### 核心模块详解 (Core Modules)

#### 1. 预处理模块 (Preprocessing)
- **序列筛选**: 验证 series_description 包含 t2 + dixon + W
- **最优切片选择**: 几何硬约束过滤 + 复合评分（行数 × 形状分）

#### 2. 椎管分割模块 (Spinal Canal Segmentation)
- **双区域 Otsu**: 行 5%-55% / 40%-90% 独立分割
- **最大连通域**: 保留主体结构 + 1px 膨胀
- **边界追踪**: 骨骼化处理后提取皮质线1/2

#### 3. V15 法线扫描线生成
- **40 条扫描线**: 1mm 间距，1-40mm 深度覆盖
- **法线计算**: 逐点切线 → 旋转 90° → 法向量 (nx, ny)
- **坐标系**: 以皮质线1 二次平滑为基准

#### 4. 终板检测模块 (Endplate Detection) - V15 增强
- **三区统计**: 后区/中区/前区独立 Otsu
- **状态机**: looking_for='drop' → 'rise' 交替检测
- **聚类**: 5mm 弧长滑动窗口 → 峰值合并
- **全局回退**: 终板线总数<5 条时，global_med×0.5 + drop/rise_ratio×0.7 重扫
- **点数回退**: 单条终板线点数<28 时，drop_ratio3×[0.80, 0.65] 重扫（最多 2 轮）

#### 5. 前缘线检测模块 (Anterior Edge Detection)
- **上升沿路径**: 谷底查找 → 脂肪过滤 → 高低高模式过滤 → 确认
- **下降沿路径**: 20mm 起点 → 水平扫描 → 双条件触发
- **双模态融合**: 合并两种路径 → 密集窗口过滤（梯形 expand_ratio=3.0）→ 红线输出

#### 6. 椎体链路识别模块 (V14 新增) ⭐
- **S1/L5 判定**: 基于前缘夹角（<45° → S1, ≥45° → L5）
- **胸椎扩展**: L5 → L4 → L3 → L2 → L1 → T12 → T11 → T10 → T9 → T8
- **几何中心**: 四角点平均坐标（top_c1 + top_front + bot_front + bot_c1）/ 4
- **掩模生成**: 四条真实曲线（上终板、前缘、下终板、皮质线 1）拼合闭合多边形

### 关键技术特点 (Key Technical Features)

| 特性 | 描述 |
|------|------|
| **双模态融合** | 上升沿 + 下降沿合并，增强鲁棒性 |
| **参数自适应** | 基于像素间距/场强/并行采集动态调整 |
| **MAD 平滑** | 鲁棒离群点过滤，避免噪声干扰 |
| **法线扫描** | V15 沿皮质线曲度追踪，符合解剖结构 |
| **胸椎扩展** | V14 支持 T12-T8 命名，禁用通用占位符 |
| **双重回退** | V15 全局回退 + 点数回退，保证检出率 |
| **梯形密集窗口** | expand_ratio=3.0，顶部 6mm 底部 18mm 动态扩展 |
| **影像组学输出** | NIfTI 多标签掩模 + CSV 几何参数，符合科研标准 |

---

## 📁 项目结构 (Project Structure)

```
LSMATOOLS_CP/
├── main.py                     # 主程序入口
├── ARCHITECTURE.md             # 架构设计文档
├── config/
│   ├── params.py              # V15 核心参数常量
│   └── param_route.py         # 参数路由（分辨率/场强/并行采集）
├── preprocessing/
│   ├── slice_selector.py      # 最优切片选择
│   └── spinal_canal_processor.py  # 椎管分割处理器
├── segmentation/
│   ├── anterior_edge.py       # 前缘检测（上升沿 + 下降沿）
│   ├── endplate_detector.py   # 终板检测（含回退机制）
│   └── scan_lines.py          # V15 法线扫描线生成
└── postprocessing/
    ├── visualization.py       # 可视化输出
    ├── export.py              # NIfTI 掩模 + CSV 导出
    └── geometric_modeling.py  # 椎体几何建模
```

---

## ⚙️ 配置参数 (Configuration)

### 像素间距自适应 (Pixel Spacing Adaptation)

LSMATOOLS 根据图像分辨率自动调整参数：

| 等级 | 像素间距 | tol_mm | min_pts |
|------|---------|--------|---------|
| HR   | ≤0.50mm | 2.0mm  | 7       |
| STD  | ≤0.75mm | 2.5mm  | 7       |
| LR   | >0.75mm | 3.0mm  | 6       |

### 场强修正 (Field Strength Correction)

- **3T 场强** (≥2.5T): depth_thresh × 1.10
- **并行采集**: depth_thresh × 0.92

---

## 🔬 技术细节 (Technical Details)

### 输入要求 (Input Requirements)

- **格式**: NIfTI (.nii.gz)
- **序列**: T2 Dixon W（压脂） + F（压水）配对
- **平面**: 矢状位 (Sagittal)
- **推荐分辨率**: ≤0.75mm (STD 等级)

### 坐标系统 (Coordinate System)

- **base_col**: 皮质线 2（c2）的列坐标基准
- **offset_mm**: `(base_col - col) × pixel_spacing`
  - 从皮质线2 向腹侧（椎体方向）的距离
  - offset 增大 → col 减小 → 图像中向左移动

### 关键算法参数 (Key Parameters)

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `left_off_mm` | 20.0mm | 谷底搜索 ROI 右边界 |
| `right_off_mm` | 40.0mm | 谷底搜索 ROI 左边界基准 |
| `scan_mm` | 40.0mm | 上升沿精修扫描距离 |
| `rise_ratio` | 0.50 | 上升沿触发比例 |
| `drop_ratio3` | 动态 [0.25-0.60] | 下降沿动态阈值 |
| `window_mm` | 6.0mm | 密集窗口宽度 |
| `expand_ratio` | 3.0 | 密集窗口梯形扩展倍数 |
| `smooth_k` | 8.0mm | 前缘线平滑窗口 |

---

## 📄 许可证 (License)

本项目采用 **MIT 许可证** - 详见 [LICENSE](LICENSE) 文件

Copyright (c) 2026 LSMATOOLS Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## 📜 学术引用与原创性声明 (Academic Citation & Originality)

### 核心算法存证 (Core Algorithm Certification)

本项目的核心算法及设计文档已于 **2026 年 3 月 24 日** 通过至信链进行区块链存证，存证 id：`02d0906fd2124bd4acc575629b2bbdf0`。

可通过"一点存"微信小程序核验存证信息。

### 开源协议 (Open Source License)

本项目基于 **MIT 许可证** 开源，欢迎自由使用、修改和分发。

### 学术引用规范 (Academic Citation Guidelines)

若在学术论文、研究报告或商业产品中使用本代码，请按学术规范注明原始来源，引用格式如下：

```
LSMATOOLS Contributors. LSMATOOLS: Lumbar Spine MRI Analysis Tools V15.0, 2026.
GitHub repository, https://github.com/hedmx/LSMATOOLS.
核心算法已进行区块链存证，ID: 02d0906fd2124bd4acc575629b2bbdf0
```

**BibTeX 格式**:
```
@software{lsma_tools_2026,
  author = {LSMATOOLS Contributors},
  title = {LSMATOOLS: Lumbar Spine MRI Analysis Tools},
  version = {15.0},
  year = {2026},
  url = {https://github.com/hedmx/LSMATOOLS},
  note = {Core algorithm certified with blockchain ID: 02d0906fd2124bd4acc575629b2bbdf0}
}
```

### 重要提示 (Important Notice)

本项目的核心算法及设计文档已在发表前完成原创性存证。任何未经授权的抢先发表行为，将可能构成学术不端。

---

## 🧪 贡献指南 (Contributing)

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范 (Code Style)

- 遵循 PEP 8 规范
- 函数添加类型提示 (Type Hints)
- 关键算法添加英文注释

---

## 📧 联系方式 (Contact)

- **问题反馈**: 请通过 GitHub Issues 提交
- **功能建议**: 欢迎发起 Discussion 讨论
- **邮箱联系**: xhfs_marvin@163.com

---

## 🙏 致谢 (Acknowledgements)

感谢所有为腰椎 MRI 分析算法研究做出贡献的研究团队！

---

**最后更新**: 2026-03-09  
**版本**: V15.0
