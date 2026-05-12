# LSMATools - Lumbar Spine MRI Analysis Tools

**腰椎矢状位 MRI 自动分割与几何量化工具** | Automated Lumbar Spine MRI Segmentation and Geometric Quantification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-V15.6-green.svg)](https://github.com/hedmx/LSMATools)

---

## 简介 (Introduction)

LSMATools 是一套基于 T2 Dixon/WFI Water/同相位 双序列的腰椎 MRI 自动分割与几何量化工具。它能够自动识别椎管结构、检测终板位置、量化椎体前缘/后缘几何特征，并输出多种格式的分析结果。

### 核心功能 (Key Features)

- **自动切片优选** — V15 五候选三步合并 + 四重形态校验 + 相对评分（45/45/10）+ col共识排除 + 宽度回退
- **椎管自动追踪** — SpinalCanalProcessor 完整分割椎管，输出皮质线 1
- **Mode4 同相位图膜态分割** — 加载同患者 IN/IP 序列，三pass自适应偏移量扫描终板汇合点
- **椎体几何量化** — 输出四交点（AP_top/AP_bot/BP_top/BP_bot）及形态角度
- **命名置信度评估** — T1 三条件评估椎体命名的可信度（presumed_ok / uncertain）
- **四种格式输出** — NIfTI 掩模（多标签）+ Fiji ROI ZIP + CSV 几何表 + PNG 可视化
- **批量处理** — 全量模式 / 快速轻量模式，自动匹配 T2 Dixon/WFI/Flex W 序列

### 适配设备 (Supported Devices)

- **西门子 Dixon**：T2 Dixon 序列，`_W`（水相）+ `_IN` / `_INPHASE`（同相位）
- **联影 WFI**：T2 WFI 序列，`_W`（水相）+ `_IP`（同相位）
- **GE Flex/IDEAL**：`WATER:` 前缀命名，无标准 IN 时自动 fallback 同位置 T2 FSE 作为 IN 替代

### 适用场景 (Use Cases)

- 腰椎 MRI 图像的自动化预处理
- 椎体几何参数的批量测量
- 终板退行性变的定量分析
- 术前规划与术后评估
- 医学影像 AI 算法的基准测试

---

## 快速开始 (Quick Start)

### 安装依赖

```bash
cd LSMATOOLS
pip install -r requirements.txt
```

### 命令行运行

```bash
python main.py
```

启动后出现交互菜单：

```
LSMATools – 腰椎 WATER/同相位 序列膜态分割工具

请选择处理模式:
  1. 单张图像处理
  2. 批量处理（全量分割模式：掩膜+CSV+日志+ROI+双图可视化）
  3. 批量处理（快速轻量模式：掩膜+CSV+W图可视化，无ROI/单例日志）
```

#### 模式 1：单张图像处理

输入 `1` 后按提示输入：
- NIfTI 文件路径 或 序列目录（自动追加 `scan.nii.gz`）
- metadata.json 路径（不存在时使用默认 pixel_spacing=0.9375mm）
- 输出目录

#### 模式 2：批量处理（全量）

输入 `2` 后按提示输入：
- 输入父目录路径

递归遍历输入目录，匹配 T2 Dixon/WFI W 序列，输出：四种文件（NIfTI掩模 + ROI ZIP + CSV + PNG）+ `batch_summary.json`

#### 模式 3：批量处理（快速）

输入 `3` 后按提示输入：
- 输入父目录路径

同模式 2，但跳过 ROI ZIP 和单例日志，仅输出掩模 + CSV + 左图可视化，适用于大批量快速筛查。

### Python 代码调用

```python
import sys
sys.path.insert(0, '/path/to/LSMATOOLS')

from main import process_single

result = process_single(
    nifti_path='/path/to/T2_DIXON_W/scan.nii.gz',
    metadata_path='/path/to/T2_DIXON_W/metadata.json',
    output_dir='/path/to/output',
    patient_dir='P12345',
    seq_dir='T2_TSE_DIXON_SAG_W_0005',
    fast_mode=False,
)
print(result)  # {'status': 'success', 'stem': '...', 'n_vertebrae': 5, ...}
```

### 批量调用

```python
from main import process_batch, process_batch_fast

# 全量模式
process_batch('/path/to/input', '/path/to/output')

# 快速模式
process_batch_fast('/path/to/input', '/path/to/output')
```

### 典型目录结构

```
患者目录/
├── T2_TSE_DIXON_SAG_W_0005/      # 压脂图（W 序列，西门子 Dixon / 联影 WFI）
│   ├── scan.nii.gz
│   └── metadata.json
└── T2_TSE_DIXON_SAG_IN_0006/     # 同相位图（西门子 _IN / 联影 _IP，自动配对）
    ├── scan.nii.gz
    └── metadata.json
```

`metadata.json` 至少包含：

```json
{
  "acquisition_params": {
    "pixel_spacing_mm": [0.9375, 0.9375]
  },
  "series_info": {
    "series_description": "T2_TSE_DIXON_SAG_W_0005"
  }
}
```

> 若没有 `metadata.json`，系统使用默认像素间距 0.9375mm。

---

## 输出说明 (Output Description)

每个病例在 `output_dir/{patient}_{seq}/` 下生成：

| 文件 | 格式 | 内容 |
|------|------|------|
| `{stem}_seg.nii.gz` | NIfTI | 多标签分割掩模（2D，S2 不纳入）|
| `{stem}_roi.zip` | ZIP | Fiji ImagejRoi 格式，每个标签一个 .roi 文件 |
| `{stem}_geom.csv` | CSV | 椎体几何形态表（四角点坐标/mm、角度、面积、椎管径）|
| `{stem}_log.txt` | TXT | 完整处理日志（头部含处理时间戳）|
| `{stem}_vis.png` | PNG | W图 + IN图双图可视化 |

批量输出：`output_dir/batch_summary.json`（含时间戳、成功失败统计、各病例处理详情）

### CSV 字段

| 字段组 | 字段名 |
|--------|--------|
| 定位 | `slice_index`, `level` |
| 四角点 | `BP_top/AP_top/AP_bot/BP_bot`（行/列/mm）|
| 几何 | `centroid_row/col`, `area_px_count`, `area_geo_mm2` |
| 角度 | `angle_sup/inf/c1/ant_deg` |
| 高度 | `ant_height_mm`, `pos_height_mm` |
| 椎管 | `canal_area_px/mm2`, `canal_ap_min/max/center_mm`, `canal_centroid_row/col` |

### 掩模标签映射

| 标签 | 值 |
|------|-----|
| S1 | 1 |
| L5 | 2 |
| L4 | 3 |
| L3 | 4 |
| L2 | 5 |
| L1 | 6 |
| T12 | 7 |
| T11 | 8 |
| ... | ... |
| T9 | 10 |
| CANAL | 15 |

> S2 不纳入掩模。

---

## 可视化预览 (Visualization Preview)

### 左图（W 图）

| 图层 | 颜色/样式 |
|------|-----------|
| 皮质线 1 | 白色实线 |
| 皮质线 2-2 | 青色实线 |
| 信号剖面采样线 | 白色虚线 |
| 椎管轮廓 | 青色虚线 |

### 右图（同相位图 IN/IP）

| 图层 | 颜色/样式 |
|------|-----------|
| 终板汇合点 | 黄色圆点 |
| 椎间盘中心 | 品红菱形 |
| 上终板线（实体段）| 草绿实线 |
| 下终板线（实体段）| 番茄红实线 |
| 终板延伸段 | 对应颜色虚线 |
| 前缘线 | 天蓝色实线 |
| 四边形轮廓 | 黄色实线 |
| 四角点 | 橙色 × |
| 名称标注 | 白色文字（黑底），含前缘角度/上终板角 |

### disc 模式可视化元素（同相位图额外）

| 元素 | 颜色/样式 |
|------|-----------|
| 掩模叠加 | 彩色半透明（按标签）|
| 起点连线（完整宽度）| 白色细虚线 |
| 扫描方向箭头 | 上=lime，下=salmon |
| 前缘扇形（disc 模式）| deepskyblue 虚线 |
| 候选点 | lime / salmon / deepskyblue 小圆点 |

---

## 算法架构 (Algorithm Architecture)

LSMATools V15.6 采用模块化 6 层架构：

### 系统总体流程

```
NIfTI (W图 3D)
      │
      ▼
[preprocessing]
  series_utils   → 序列类型识别
  slice_selector → 切片优选 → best_slice_idx, green_mask, canal_seed
      │
      ▼
[segmentation]
  SpinalCanalProcessor → traced(椎管掩模), v9_data{all_rows, smooth_cols}
  皮质线1 (c1_rows, c1_cols)
  皮质线延伸 → 皮质线2-2 (c2_rows_mode4, c2_cols_mode4)
      │
      ▼
[preprocessing]
  image_loader → 同相位图 IN/IP (in_img_2d, pixel_spacing)
      │
      ▼
[detection – Mode4 流水线]
  Step1: signal_ref       → low_mean, high_mean
  Step2: junction_detector→ junction_pts, anchor_pts_list
  Step2b: repair_junction → 去多/补少/末尾实扫补全
  Step3: disc_centers     → disc_centers, vert_centers
  Step3.5: verify_last    → 末尾两椎体汇合点校正
  Step4: fan_scanner/disc → scan_results[sup_pts, inf_pts, ant_pts, ...]
  Step4c: _verify_ant_pts → 前缘候选点二次校验（夹角<65°且角度过滤）
  Step5: anterior_edge    → cluster_results[sup, inf, ant]
      │
      ▼
[chain]
  build_vertebra_chain → vertebra_chain, ant_line
      │
      ▼
[output]
  export_masks    → {stem}_seg.nii.gz + {stem}_roi.zip
  export_csv      → {stem}_geom.csv
  export_log      → {stem}_log.txt
  visualize_wifs  → {stem}_vis.png
```

### 核心模块详解

#### 1. 预处理 (preprocessing)

| 模块 | 功能 |
|------|------|
| `series_utils.py` | 序列类型识别（W/IN/IP/F），支持西门子/联影命名 |
| `image_loader.py` | 同相位序列（IN/IP）自动查找与加载 |
| `slice_selector.py` | 切片优选：V14_2 三步掩模合并 + 四重形态校验 |

**切片优选三步算法**：

| 步骤 | 操作 |
|------|------|
| Step1 | 上区域 Otsu → 核心绿色掩模（锁定椎管身份）|
| Step2 | 下区域连通域与核心重叠判断 → 合并 |
| Step3 | 底行不足时向下搜索，四重形态校验（宽度比/左边界自洽/CV/细长比）|

#### 2. 椎管分割 (segmentation)

| 模块 | 功能 |
|------|------|
| `canal_processor.py` | SpinalCanalProcessor：双区域 Otsu → 边界追踪 → 皮质线 1 |
| `cortical_line.py` | 皮质线后处理：20mm 移动均值平滑 + 斜率修复 + 尾部延伸 |
| `endplate_detector.py` | V15 终板候选点检测（状态机扫描）|
| `endplate_clusterer.py` | 终板候选点聚类（cluster_endplates_v15）|
| `scan_lines_v15.py` | V15 扫描线构建（沿皮质线法线方向 40 条线）|

#### 3. 特征检测 (detection) — Mode4 核心流水线

| 模块 | 步骤 | 功能 |
|------|------|------|
| `signal_ref.py` | Step1 | 同相位图信号参考值（c2 前 70% 法线偏移采样）|
| `junction_detector.py` | Step2/2b | 终板汇合点扫描 + 修补（去多/补少/末尾 19 档实扫）|
| `disc_centers.py` | Step3/3.5 | 椎间盘/椎体中心 + 末尾两椎体独立校验 |
| `fan_scanner.py` | Step4/4c | disc 矩阵扫描终板 + 扇形扫描前缘 + 角度过滤 |
| `anterior_edge.py` | Step5 | 前缘聚类 + 终板聚类 + 平滑工具 |
| `_scan_utils.py` | 工具 | 皮质线投影 + 四函数低谷扫描（横向/对角/纵向/单像素）|

#### 4. 椎体链路 (chain)

| 模块 | 功能 |
|------|------|
| `vertebra_chain.py` | Step6：前缘拼接 + 终板延伸 + 符号变化法求交 + 四分支命名 |

**椎体命名规则**：

```
sup_angle ≥ 30°        → S 椎体
sup_angle ∈ [10°, 30°) → 灰色区间：width_ratio ≥ 1.2 OR hw_ratio ≥ 1.2 → S
sup_angle < 10°         → L 椎体

四分支：last=S+sec=S → S2,S1,L5...
       last=S+sec=L → S1,L5,L4...
       last=L+sec=S → S2,S1,L5... (sec 假阳性忽略)
       last=L+sec=L → L5,L4,L3...
```

#### 5. 结果输出 (output)

| 模块 | 功能 |
|------|------|
| `mask_export.py` | NIfTI 多标签掩模 + Fiji ROI ZIP 导出 |
| `csv_export.py` | CSV 几何形态导出 |
| `log_export.py` | 单例处理日志导出 |
| `visualization.py` | W 图 / 同相位图（IN/IP）双图可视化 PNG |

#### 6. 配置 (config)

| 模块 | 功能 |
|------|------|
| `params.py` | 全局参数常量 + 椎体标签映射 |
| `metadata_parser.py` | metadata.json 解析 |

#### 7. 通用工具 (utils)

| 模块 | 功能 |
|------|------|
| `geometry.py` | 通用几何函数（点距、向量夹角、线段交点等）|

---

## 项目结构 (Project Structure)

```
LSMATOOLS/
├── main.py                     # 主流程入口（单例/批量/CLI）
├── config/
│   ├── __init__.py
│   ├── params.py               # 全局参数常量 / LEVEL_LABEL
│   └── metadata_parser.py      # metadata.json 解析
├── preprocessing/
│   ├── __init__.py
│   ├── series_utils.py         # 序列类型识别（W/IN/IP/F）
│   ├── image_loader.py         # IN 序列自动查找与加载
│   └── slice_selector.py       # 切片优选（三步掩模合并 + 四重校验）
├── segmentation/
│   ├── __init__.py
│   ├── canal_processor.py      # 椎管追踪主处理器
│   ├── cortical_line.py        # 皮质线后处理工具
│   ├── endplate_detector.py    # 终板候选点检测（V15 状态机）
│   ├── endplate_clusterer.py   # 终板候选点聚类
│   └── scan_lines_v15.py       # V15 扫描线构建
├── detection/
│   ├── __init__.py
│   ├── _scan_utils.py          # 皮质线投影 + 低谷扫描工具
│   ├── signal_ref.py           # Step1 同相位图信号参考值
│   ├── junction_detector.py    # Step2/2b 汇合点扫描+修补
│   ├── disc_centers.py         # Step3/3.5 椎间盘中心+校验
│   ├── fan_scanner.py          # Step4 扇形/矩阵扫描 + 4c 校验
│   └── anterior_edge.py        # Step5 前缘聚类 + 平滑
├── chain/
│   ├── __init__.py
│   └── vertebra_chain.py       # Step6 椎体链路构建 + 命名
├── output/
│   ├── __init__.py
│   ├── mask_export.py          # NIfTI 掩模 + ROI ZIP 导出
│   ├── csv_export.py           # CSV 几何导出
│   ├── log_export.py           # 单例日志导出
│   └── visualization.py        # W 图 / 同相位图（IN/IP）双图可视化
├── utils/
│   ├── __init__.py
│   └── geometry.py             # 通用几何工具
├── requirements.txt            # 依赖列表
├── LICENSE                     # MIT 许可证
└── README.md                   # 本文件
```

---

## 配置参数 (Configuration)

### 关键参数速查

| 参数 | 值 | 说明 |
|------|-----|------|
| 切片候选数 | 5（中心 ±2 张）| 候选切片范围 |
| 椎管最小面积 | 300 mm² | 椎管连通域最小面积 |
| 椎管宽度上限 | 30 mm | 宽度异常检测上限 |
| 皮质线 2-2 平滑窗 | 20mm | 移动均值窗口 |
| 信号剖面采样范围 | c2 前 70% | 排除后 30% 骶椎高信号区 |
| Step1 信号偏移 | 20mm | 法线方向偏移距离 |
| 终板扫描模式 | disc | 矩阵扫描终板 + 扇形扫描前缘 |
| 前缘扇形半角 | 50° | 扇形扫描角度范围 |
| 前缘扫描距离 | 40mm | 扇形射线扫描长度 |
| 前缘 drop_ratio | 0.35（标准）/ 0.20（S1）| 下降沿触发阈值 |
| 终板 drop_ratio | 0.25（上/下终板）| 终板扫描阈值 |
| ref 锁定机制 | Step2 失败锁定 ref | 防止基准漂移 |
| 前缘聚类 offset_min | 15mm | 排除近端伪影 |
| 前缘聚类窗口宽 | 末尾椎体动态映射（40°→20mm，80°→5mm）| 动态窗口 |
| 终板聚类窗口宽 | 5mm | 滑动窗口 |
| 终板线延伸距离 | 25mm | 两端各延伸 |
| 上终板角命名阈值 | 30° | ≥30° 为 S 椎体 |
| 灰色区间 | 10°~30° | width_ratio ≥ 1.2 OR hw_ratio ≥ 1.2 → S |
| 切片评分权重 | 45% + 45% + 10% | 面积 + 底部 + 空洞 |

---

## 技术细节 (Technical Details)

### 输入要求

| 项目 | 要求 |
|------|------|
| 格式 | NIfTI (.nii.gz) |
| 序列 | 西门子 T2 Dixon（W 水相 + IN 同相位）或 联影 T2 WFI（W 水相 + IP 同相位）|
| 平面 | 矢状位 (Sagittal) |
| 推荐分辨率 | ≤0.75mm |

### 坐标系统

| 轴 | 方向 | 说明 |
|----|------|------|
| row | 向下为正 | 图像行索引，0 = 图像顶部 |
| col | 向右为正 | 图像列索引，0 = 图像左侧 |
| 腹侧 | col 减小（向左）| 矢状位图像中椎体前方 |
| 背侧 | col 增大（向右）| 矢状位图像中椎管侧 |
| 头颅侧 | row 减小（向上）| superior 方向 |
| 尾骨侧 | row 增大（向下）| inferior 方向 |

**扇形角度语义**：
```
angle = atan2(dr, dc)
dr = sin(angle)，dc = cos(angle)

angle 增大（+方向） → dr 变负 → 射线朝头颅侧（superior）
angle 减小（-方向） → dr 变正 → 射线朝尾骨侧（inferior）
```

### 法线方向

`_project_to_c2` 返回法线方向：
- 切线 `(t_dr, t_dc)` 顺时针旋转 90° → 法线 `(n_row, n_col) = (t_dc, -t_dr)`
- 强制 `n_col < 0`（朝腹侧）

---

## 版本历史 (Version History)

### V15.6（2026-05）
- **GE Flex/IDEAL 支持**: WATER:/FAT: 命名识别 + flex 关键词 + 无 IN 时自动 fallback 同位置 T2
- **三pass自适应偏移量**: 汇合点扫描 offset 1→3mm → 3→5mm → 5→7mm 逐步放大
- **col中心共识排除**: 掩模位置异常切片自动排除
- **宽度回退机制**: ≥3张切片异常时病灶性增宽自动识别
- **命名置信度 T1**: 三条件评估椎体命名可信度
- **pos_exact 三态语义**: 精确拒绝跨采集位置 IN 错配
- **日志必出 + None 防御**: crash 亦留 traceback

### V15.5（2026-03）

**1. Step2 信号确认：三点均值 → 两两组合任一满足**

四个 `_scan_normal_descent*` 函数的 Step2 确认逻辑从「三点均值」改为「两两组合任一满足」，可捕获单侧信号突变。

**2. S1 前缘信号确认：对角配对 → 单像素直判**

S1 骶椎前缘骨骼-软组织过渡区信号对比度低，单一像素更灵敏。

**3. S1 前缘下降沿阈值动态适配**

最后椎体满足角度校验时，前缘扫描 `drop_ratio` 从 0.35 降至 0.20。

**4. 切片优选 Step3 四重形态校验**

桥接候选区域校验从单一宽度比扩展为四重：宽度比 / 左边界自洽 / 宽度一致性 CV / 细长比。

**5. S1 局部信号日志输出**

最后两个椎体的局部 low_mean/high_mean 采样值打印到日志。

---

## 许可证 (License)

本项目采用 **MIT 许可证** - 详见 [LICENSE](LICENSE) 文件

Copyright (c) 2026 LSMATools Contributors

---

## 学术引用 (Academic Citation)

### 核心算法存证

本项目的核心算法及设计文档已于 **2026 年 3 月 24 日** 通过至信链进行区块链存证，存证 id：`02d0906fd2124bd4acc575629b2bbdf0`。

可通过"一点存"微信小程序核验存证信息。

### 引用格式

```
LSMATools Contributors. LSMATools: Lumbar Spine MRI Analysis Tools V15.5, 2026.
GitHub repository, https://github.com/hedmx/LSMATools.
核心算法已进行区块链存证，ID: 02d0906fd2124bd4acc575629b2bbdf0
```

**BibTeX**:
```bibtex
@software{lsma_tools_2026,
  author = {LSMATools Contributors},
  title = {LSMATools: Lumbar Spine MRI Analysis Tools},
  version = {15.5},
  year = {2026},
  url = {https://github.com/hedmx/LSMATools},
  note = {Core algorithm certified with blockchain ID: 02d0906fd2124bd4acc575629b2bbdf0}
}
```

---

## 贡献指南 (Contributing)

欢迎提交 Issue 和 Pull Request！

### 代码规范

- 遵循 PEP 8 规范
- 函数添加类型提示 (Type Hints)
- 关键算法添加英文注释

---

## 联系方式 (Contact)

- **问题反馈**: 请通过 GitHub Issues 提交
- **功能建议**: 欢迎发起 Discussion 讨论

---

**最后更新**: 2026-05-07  
**版本**: V15.5
