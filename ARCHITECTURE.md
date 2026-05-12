# LSMATOOLS 架构设计文档

> **版本**：V15.6  
> **更新日期**：2026-05-12  
> **输入序列**：西门子 T2 Dixon（W + IN）| 联影 T2 WFI（W + IP）| GE Flex/IDEAL（WATER + T2 fallback）

---

## 一、总体架构

```
LSMATOOLS/
├── main.py                     # 主流程入口（单例/批量/CLI）
├── config/                     # 参数配置层
├── preprocessing/              # 图像预处理层
├── segmentation/               # 椎管分割层
├── detection/                  # 特征检测层（Mode4 核心流水线）
├── chain/                      # 椎体链路构建层
├── output/                     # 结果输出层
└── utils/                      # 通用工具层
```

### 数据流向

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
  皮质线1 → 皮质线延伸 → 皮质线2-2
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
  Step4c: _verify_ant_pts → 前缘候选点二次校验
  Step5: anterior_edge    → cluster_results[sup, inf, ant]
      │
      ▼
[chain]
  build_vertebra_chain → vertebra_chain, ant_line
      │
      ▼
[output]
  export_masks    → NIfTI 掩模 + ROI ZIP
  export_csv      → CSV 几何表
  export_log      → 处理日志
  visualize_wifs  → PNG 可视化
```

---

## 二、模块详细说明

### 2.1 `main.py` — 主流程入口

**导出函数：**

| 函数 | 功能 |
|------|------|
| `process_single(nifti_path, metadata_path, output_dir, ...)` | 处理单例 W 图，返回处理结果字典 |
| `process_batch(input_dir, output_dir)` | 批量处理（全量模式：掩膜+CSV+日志+ROI+双图可视化）|
| `process_batch_fast(input_dir, output_dir)` | 批量快速模式（掩膜+CSV+W图可视化）|
| `main()` | CLI 交互入口（三级菜单：单张/批量全量/批量快速）|

**单例处理流程（`_run_single`）：**

1. 加载 NIfTI + 解析 metadata
2. `select_best_slice`：从中间 ±2 张候选切片中优选最佳矢状位
3. `SpinalCanalProcessor.process_with_mask`：椎管追踪 → 皮质线 1
4. 皮质线 1 尾部延伸 → 皮质线 2-2（20mm 平滑）
5. `find_in_image`：配对同相位序列（IN/IP）
6. **Mode4 流水线**：Step1 → Step2 → Step2b → Step3 → Step3.5 → Step4 → Step4c → Step5
7. `build_vertebra_chain`：椎体链路构建 + 四交点求解 + 命名
8. 输出：NIfTI 掩模 + ROI ZIP + CSV + 日志 + PNG

---

### 2.2 `config/` — 参数配置层

#### `config/params.py` — 全局参数常量

模块级常量，按功能分组：

| 参数组 | 关键常量 | 说明 |
|--------|----------|------|
| 切片优选 | `SLICE_CANDIDATES=5` | 候选切片数（中心 ±2）|
| 椎管检测 | `CANAL_MIN_AREA_MM2=300.0` | 椎管最小面积 |
| 椎管检测 | `MAX_CANAL_WIDTH_MM=30.0` | 椎管宽度上限 |
| 同相位图信号 | `SMOOTH_MM_C2=20.0` | 皮质线 2-2 平滑窗口 |
| 同相位图信号 | `SMOOTH_MM_C3=40.0` | 皮质线 3-3 平滑窗口 |
| 同相位图信号 | `OFFSET_MM_SIGNAL=20.0` | Step1 信号参考剖面偏移 |
| 椎体命名 | `SUP_ANGLE_THRESH=30.0` | S 椎体判定阈值（度）|
| 椎体命名 | `SUP_ANGLE_GRAY_LOW=10.0` | 灰色区间下界 |
| 椎体命名 | `WIDTH_RATIO_THRESH=1.2` | 宽度比/高宽比阈值 |
| 掩模标签 | `LEVEL_LABEL={'S1':1, ...}` | 椎体标签映射（S2 不纳入）|
| 掩模标签 | `CANAL_LABEL=15` | 椎管标签值 |

#### `config/metadata_parser.py` — 元数据解析

| 函数 | 功能 |
|------|------|
| `load_metadata(path)` | 加载 JSON metadata 文件 |
| `parse_pixel_spacing(meta)` | 从 metadata 提取像素间距，默认 0.9375mm |

---

### 2.3 `preprocessing/` — 图像预处理层

#### `preprocessing/series_utils.py` — 序列类型工具

| 函数 | 逻辑 |
|------|------|
| `_get_series_type(desc)` | 识别序列类型：`W`/`IN`/`F`/`OPP`/`None`。支持西门子/联影命名 |
| `_is_dixon_sequence(desc)` | 判断是否为 Dixon/WFI 序列（含 DIXON 或 WFI 关键词）|

#### `preprocessing/image_loader.py` — 同相位序列加载

`find_in_image(nifti_path, slice_idx)` — 在 W 图同目录或父目录中搜索同相位序列（IN/IP）。

匹配策略：
1. `_get_series_type(desc) == 'IN'`
2. 宽松匹配：末尾分段含 `'in'` 或 `'ip'`（兼容联影 `_IP` 命名）

#### `preprocessing/slice_selector.py` — 切片优选

`select_best_slice(data, pixel_spacing)` — V14_2 三步掩模合并 + 相对评分。

| 步骤 | 操作 |
|------|------|
| Step1 | 上区域（0%~65%行）Otsu → 核心绿色掩模 |
| Step2 | 下区域（30%~95%行）连通域与核心重叠判断 → 合并 |
| Step3 | 底行 < 60% 时向下搜索，四重形态校验全部通过才桥接 |

**Step3 四重形态校验**：

| 校验项 | 条件 |
|--------|------|
| 宽度比 | `0.5 ≤ w_region / w_green ≤ 1.5` |
| 左边界自洽 | `max_adjacent_jump ≤ max(3px, 0.20×med_w)` |
| 宽度一致性 CV | `CV ≤ 0.50` |
| 细长比 | `height / width ≥ 1.5` |

**评分**：`total_score = 0.45×面积 + 0.45×底部位置 + 0.10×空洞惩罚`

---

### 2.4 `segmentation/` — 椎管分割层

#### `segmentation/canal_processor.py` — SpinalCanalProcessor

主处理器，负责从 2D 切片追踪完整椎管并输出皮质线。

主要接口：
- `process_with_mask(slice_2d, green_mask, canal_seed, ...)` — 有初始掩模时调用
- `process(slice_2d)` — 无初始掩模时调用

输出 `v9_data`：
```python
{
  'all_rows':    [...],  # 皮质线1行坐标（未平滑）
  'smooth_cols': [...],  # 皮质线1列坐标（平滑后）
  ...
}
```

#### `segmentation/cortical_line.py` — 皮质线后处理

| 函数 | 功能 |
|------|------|
| `build_cortical2(cols, rows, ps, smooth_mm=20)` | 对皮质线1做 20mm 移动均值平滑 |
| `_repair_cortical2_slope(...)` | 修复皮质线两端斜率异常（MAD 检测 + 线性插值）|
| `extend_cortical2_tail(...)` | 向尾侧延伸皮质线 |

#### `segmentation/scan_lines_v15.py` — V15 扫描线生成

| 函数 | 功能 |
|------|------|
| `build_scan_lines_v15(c2_cols, c2_rows, img_shape, ps, ...)` | 沿皮质线2法线方向生成 40 条扫描线 |

#### `segmentation/endplate_detector.py` / `endplate_clusterer.py`

终板候选点检测（V15 状态机扫描）与聚类，在 `canal_processor.py` 内部使用。

---

### 2.5 `detection/` — 特征检测层（Mode4 核心流水线）

#### `detection/_scan_utils.py` — 低谷扫描工具

| 函数 | 功能 |
|------|------|
| `_project_to_c2(r0, c0, c2_rows, c2_cols)` | 将点投影到皮质线2，返回法线方向 |
| `_scan_normal_descent(...)` | 沿法线扫描找低谷（横向两两组合确认）|
| `_scan_normal_descent_diag(...)` | 对角确认版（`/` 方向，用于最后椎体后缘角 < 65°）|
| `_scan_normal_descent_ant(...)` | 前缘专用（纵向两两组合确认）|
| `_scan_normal_descent_ant_diag(...)` | 前缘单像素确认版（S1 前缘专用）|

**ref 锁定机制**：下降沿触发但 Step2 失败时锁定 ref，防止基准漂移。解锁条件：`cur ≥ ref_locked × (1 - drop_ratio / 2)`。

#### `detection/signal_ref.py` — Step1

`compute_signal_references(in_img_2d, c2_rows, c2_cols, pixel_spacing, offset_mm=20.0)`

沿皮质线 2 **前 70% 点**（排除末尾 30% 骶椎高信号区）法线方向偏移 20mm 采样同相位图信号，统计 `low_mean` / `high_mean`。

#### `detection/junction_detector.py` — Step2 / Step2b

**Step2 主扫描**：

```
沿皮质线 2-2 从顶向下滑动窗口（2mm×2mm）
offset 分段线性渐变：
  行程 0%~60%：offset = 1mm → 3mm（线性）
  行程 60%~100%：offset 固定 3mm
滑动均值平滑（smooth_win=3）
低于 low_mean×0.8 的区间取中点 → 终板汇合点
静默距离：8mm
```

**Step2b 修补**：

| 步骤 | 操作 |
|------|------|
| A. 去多 | 遍历相邻三点，两侧间距之和 < 1.5×median_gap → 删除假阳性 |
| B. 补少 | 间距 > 1.8×median_gap → 皮质线 2 弧长等分位置插入补全 |
| C. 末尾实扫 | 逐档测试 offset（-2mm ~ 16mm，共 19 档），找到低信号即停止 |

#### `detection/disc_centers.py` — Step3 / Step3.5

**Step3**：终板汇合点沿皮质线 2 法线延伸（渐变 15mm→20mm）→ 椎间盘中心。

**Step3.5**：末尾两椎体独立校验。
- 动态偏移量：`offset_mm = 8 + (clamp(angle, 40°, 80°) - 40) / 40 × 12`
- 40° 对应 8mm，80° 对应 20mm

#### `detection/fan_scanner.py` — Step4 / Step4c

**disc 模式（默认）**：

| 方向 | 参数 | 扫描函数 |
|------|------|---------|
| 上终板 | `scan_up_mm=30mm`, `drop_ratio=0.35` | `scan_disc_endplates`（矩阵扫描）|
| 下终板 | `scan_dn_mm=30mm`, `drop_ratio=0.35` | `scan_disc_endplates`（矩阵扫描）|
| 前缘 | `fan_half_deg=50°`, `scan_ant_mm=40mm` | `_scan_normal_descent_ant` / `_ant_diag` |

**Step4c 前缘二次校验**：
- 触发条件：末尾两椎体汇合点连线夹角 < 65°
- 过滤区间：`delta ∈ [+25°, +50°]`（直接删除）

#### `detection/anterior_edge.py` — Step5

**终板聚类（cluster_correction_pts）**：
1. 去倾斜变换 → 5mm 滑动窗口找最密区间
2. col 排序 MAD 过滤 → 移动均值平滑
3. 弧长超限回退：总弧长 > 13mm → 40% 窗口重聚类

**前缘聚类（cluster_anterior_edge）**：
1. 投影到法线轴，仅保留 offset ≥ 15mm 的点
2. 动态窗口宽度：最后椎体 40°→20mm / 80°→5mm
3. row 排序移动均值平滑（3mm）

---

### 2.6 `chain/` — 椎体链路构建层

#### `chain/vertebra_chain.py` — Step6：build_vertebra_chain

**A. 前缘线拼接 + 两端法线延伸**：
- 各椎体聚类前缘段 → MAD 平滑 → 插值 → 两端法线延伸
- 顶端终止：椎间盘中心 + 汇合点投影弧长 −5mm
- 底端终止：椎间盘中心 + 汇合点投影弧长 +5mm

**B. 终板线延长**：两端延伸 25mm，夹角约束 ±20°。

**C. 四交点求解（_find_crossing）**：

符号变化法（替代旧版线段求交）：
1. 参考线构建为 `row→col` 查找表（LUT）
2. 终板线各点计算 `diff = ep_col - lut[row]`
3. 相邻点 diff 符号反转 → 穿越 → 线性插值得交点

**D. 椎体命名（四分支）**：

```
sup_angle ≥ 30°        → S 椎体
sup_angle ∈ [10°, 30°) → width_ratio ≥ 1.2 OR hw_ratio ≥ 1.2 → S
sup_angle < 10°         → L 椎体

四分支命名（从下往上）：
  last=S, sec=S → S2, S1, L5, L4...
  last=S, sec=L → S1, L5, L4...
  last=L, sec=S → S2, S1, L5... (sec 假阳性忽略)
  last=L, sec=L → L5, L4, L3...
```

---

### 2.7 `output/` — 结果输出层

| 模块 | 功能 |
|------|------|
| `mask_export.py` | NIfTI 多标签掩模 + Fiji ROI ZIP。优先曲线轮廓，缺失降级四角点 |
| `csv_export.py` | CSV 几何形态表（四角点、角度、面积、椎管径）|
| `log_export.py` | 单例处理日志（头部含时间戳）|
| `visualization.py` | W 图 / 同相位图（IN/IP）双图可视化 |

---

### 2.8 `utils/` — 通用工具层

| 模块 | 功能 |
|------|------|
| `geometry.py` | 通用几何函数（点距、向量夹角、线段交点等）|

---

## 三、模块依赖关系

```
main.py
  ├── config.params
  ├── config.metadata_parser
  ├── preprocessing.series_utils
  ├── preprocessing.image_loader      → preprocessing.series_utils
  ├── preprocessing.slice_selector    → config.params
  ├── segmentation.canal_processor    → segmentation.cortical_line
  │                                   → segmentation.scan_lines_v15
  ├── segmentation.endplate_detector
  ├── segmentation.endplate_clusterer
  ├── detection.signal_ref            → detection._scan_utils
  ├── detection.junction_detector     → detection._scan_utils
  ├── detection.disc_centers          → detection._scan_utils
  ├── detection.fan_scanner           → detection._scan_utils
  │                                   → detection.anterior_edge
  ├── detection.anterior_edge         → config.params
  ├── chain.vertebra_chain            → detection.anterior_edge
  │                                   → utils.geometry
  ├── output.mask_export              → config.params
  ├── output.csv_export
  ├── output.log_export
  └── output.visualization            → config.params
```

---

## 四、关键设计决策

### 4.1 ref 锁定机制

下降沿触发但 Step2 确认失败时锁定 ref（`ref_locked = ref`），防止 ref 随低值漂移导致后续阈值下沉。解锁条件：信号回升至 `cur ≥ ref_locked × (1 - drop_ratio / 2)`。

### 4.2 Step2 offset 分段线性渐变

```
行程比例 t = (当前行 - 起始行) / 总行程  ∈ [0, 1]

if t ≤ 0.6:
    offset = 1.0 + (t/0.6) × 2.0   # 1mm → 3mm
else:
    offset = 3.0
```

设计意图：脊柱上段汇合点靠近皮质线，offset 从 1mm 起步；腰骶段椎间盘空间增大，offset 增至 3mm。

### 4.3 Step2b 末尾实扫 19 档策略

```python
for off_mm in range(-2, 17):   # -2mm ~ 16mm，步进 1mm，共 19 档
    ...
```

- 负值（-2mm~0mm）：向背侧探测，适应骶椎区域
- 正值（1mm~16mm）：向腹侧逐档搜索

### 4.4 Step4c 前缘二次校验角度过滤

过滤区间 `delta ∈ [+25°, +50°]` 对应椎体前上角区域，该区域最容易因腹侧软组织产生假阳性候选点，直接删除。

### 4.5 符号变化法求交（替代线段求交）

旧版线段求交需要行范围重叠才能相交，当终板延伸线已越过参考线（骶椎区域上终板倾角大时常见）会漏检。符号变化法只要检测到 diff 符号反转即可，对越界情况鲁棒。

### 4.6 S1 专属参数

最后一个椎体后缘角 < 65° 时：
- 前缘扫描函数切换为 `_scan_normal_descent_ant_diag`（单像素确认）
- `drop_ratio` 从 0.35 降至 0.20
- 终板扫描同步切换为 `_scan_normal_descent_diag`（`/` 对角确认）

---

## 五、运行使用方法

### 5.1 环境要求

```bash
Python >= 3.8
```

安装依赖：
```bash
pip install -r requirements.txt
```

### 5.2 运行方式

**必须在 `LSMATOOLS/` 目录内运行**（`main.py` 使用相对 import）：

```bash
cd /path/to/LSMATOOLS
python3 main.py
```

### 5.3 批量输出

`batch_summary.json` 字段：

```json
{
  "batch_start_time": "2026-03-09 10:00:00",
  "batch_end_time": "2026-03-09 12:30:00",
  "total_cases": 60,
  "success": 58,
  "failed": 2,
  "total_vertebrae": 310,
  "total_elapsed_s": 9000.0,
  "avg_elapsed_s": 150.0,
  "cases": [
    {
      "patient": "patient_id",
      "seq": "sequence_name",
      "status": "success",
      "n_vertebrae": 5,
      "elapsed_s": 148.3
    }
  ]
}
```

---

## 六、文件速查表

| 文件 | 核心函数/类 |
|------|------------|
| `main.py` | `process_single`, `process_batch`, `process_batch_fast`, `main` |
| `config/params.py` | 全局常量 + LEVEL_LABEL |
| `config/metadata_parser.py` | `load_metadata`, `parse_pixel_spacing` |
| `preprocessing/series_utils.py` | `_get_series_type`, `_is_dixon_sequence` |
| `preprocessing/image_loader.py` | `find_in_image` |
| `preprocessing/slice_selector.py` | `select_best_slice` |
| `segmentation/canal_processor.py` | `SpinalCanalProcessor` |
| `segmentation/cortical_line.py` | `build_cortical2`, `_repair_cortical2_slope`, `extend_cortical2_tail` |
| `segmentation/scan_lines_v15.py` | `build_scan_lines_v15` |
| `segmentation/endplate_detector.py` | 终板候选点检测 |
| `segmentation/endplate_clusterer.py` | `cluster_endplates_v15` |
| `detection/_scan_utils.py` | `_project_to_c2`, `_scan_normal_descent*`（四函数）|
| `detection/signal_ref.py` | `compute_signal_references` |
| `detection/junction_detector.py` | `scan_endplate_junction_points`, `repair_junction_pts` |
| `detection/disc_centers.py` | `compute_disc_and_vertebra_centers`, `verify_last_junction_point` |
| `detection/fan_scanner.py` | `scan_vertebra_by_disc`, `_verify_ant_pts_forward` |
| `detection/anterior_edge.py` | `cluster_correction_pts`, `cluster_anterior_edge`, `_smooth_ant_line` |
| `chain/vertebra_chain.py` | `build_vertebra_chain`, `_find_crossing`, `_stitch_ant_line` |
| `output/mask_export.py` | `export_masks`, `export_roi_zip` |
| `output/csv_export.py` | `export_csv` |
| `output/log_export.py` | `export_log` |
| `output/visualization.py` | `visualize_wifs` |
| `utils/geometry.py` | 通用几何工具 |
| **合计** | **31 个 .py 文件** |
