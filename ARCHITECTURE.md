# LSMATOOLS_CP 架构设计文档

> **版本**：V15.0.0  
> **基准源文件**：`batch_spinal_cord_roi_test_enhance14_2.py`  
> **重构目标**：将 7920 行单文件拆分为 4 层模块，提高可维护性与可测试性

---

## 一、总体架构

```
LSMATOOLS_CP/
├── main.py                     # 主流程编排器（入口）
├── config/                     # 参数配置层
├── preprocessing/              # 图像预处理层
├── segmentation/               # 分割与检测层（核心算法）
└── postprocessing/             # 后处理与可视化层
```

### 数据流向

```
NIfTI压脂图(W)
    │
    ▼
preprocessing: 加载 → 切片优选 → 椎管处理
    │
    ▼
segmentation: 皮质线 → V15扫描线 → 终板检测(全局回退/点数回退) → 前缘检测
    │
    ▼
postprocessing: 解剖校正 → 椎体编号 → 几何建模 → 可视化输出
```

---

## 二、模块详细说明

### 2.1 `main.py` — 主流程编排器

| 属性 | 说明 |
|------|------|
| 来源行 | 原文件 7008–7920 行 |
| 行数 | ~913 行 |

**导出函数：**

| 函数 | 功能 |
|------|------|
| `test_single_image(nifti_path, ...)` | 处理单张 NIfTI 图像，输出 PNG 结果和 JSON 日志 |
| `process_batch(parent_dir, output_dir)` | 遍历患者目录树，批量调用 `test_single_image` |
| `main()` | 命令行入口，解析 `--input / --output` 参数 |

**主流程步骤（`test_single_image`）：**
1. 加载 NIfTI + 解析 metadata（像素间距、场强、序列类型校验）
2. `select_best_slice`：从中间 ±2 张候选切片中优选最佳矢状位
3. `SpinalCanalProcessor.process_with_mask`：椎管追踪 → 皮质线 → V15 扫描线
4. `find_fat_water_image`：配对压水图（F 序列）
5. `align_scan_lines_to_f`：坐标对齐校验
6. `find_endplates_on_water_image`：V15 终板候选点扫描
7. `cluster_endplates_v15`：弧长坐标系聚类
8. **全局回退**：上/下终板线 < 5 条时降低阈值重扫
9. **点数回退**：点数不足 28（或动态值）的终板线降阈值重扫（最多 2 轮）
10. `scan_anterior_edge_v15`：前缘轮廓扫描
11. `compute_vertebra_geometry`：椎体四角点 + 几何中心建模
12. `visualize_results`：双图拼接输出

---

### 2.2 `config/` — 参数配置层

#### `config/params.py` — V15 核心参数常量

| 属性 | 说明 |
|------|------|
| 来源 | 从全文 `#` 注释与魔法数字中归纳提取 |
| 行数 | 49 行 |

提供 `V15Params` 类，集中管理所有 V15 算法常量：

| 参数组 | 关键常量 | 说明 |
|--------|----------|------|
| 扫描线 | `N_SCAN_LINES=40` | 法线方向扫描线数量 |
| 聚类 | `CLUSTER_MIN_LINES=15` | 默认最小支持线数 |
| 全局回退 | `MIN_EP_COUNT=5` | 触发全局回退的终板线门槛 |
| 全局回退因子 | `GLOBAL_MED_FACTOR=0.5` | global_med 降低倍数 |
| 全局回退因子 | `DROP/RISE_RATIO_FACTOR=0.7` | 状态机阈值降低倍数 |
| 点数回退 | `MIN_PTS_EP_BASE=28` | 点数阈值基准 |
| 点数回退 | `RETRY_FACTORS=[0.80, 0.65]` | 两轮回退的阈值比例 |

#### `config/param_route.py` — 参数路由

| 属性 | 说明 |
|------|------|
| 来源行 | 原文件 38–92 行（`SpinalCanalProcessor._build_param_route`） |
| 行数 | 68 行 |

`build_param_route(pixel_spacing, meta)` 函数：根据像素间距（HR/STD/LR 三档）、场强（3T 上调 10%）、并行采集（下调 8%）动态生成 `tol_mm / min_pts / depth_thresh` 参数字典，供 `SpinalCanalProcessor` 初始化使用。

**分辨率档位对应：**

| 档位 | 像素间距 | tol_mm | min_pts |
|------|---------|--------|---------|
| HR   | ≤0.50mm | 2.0mm  | 7       |
| STD  | ≤0.75mm | 2.5mm  | 7       |
| LR   | >0.75mm | 3.0mm  | 6       |

#### `config/metadata_parser.py` — 元数据解析

| 属性 | 说明 |
|------|------|
| 来源 | 新增工具函数，封装散落在各处的 JSON 字段读取逻辑 |
| 行数 | 44 行 |

提供 `load_metadata / parse_pixel_spacing / parse_patient_id / parse_series_desc / parse_image_origin` 五个工具函数，统一从 `metadata.json` 中提取字段，避免重复的 `.get()` 链式调用。

---

### 2.3 `preprocessing/` — 图像预处理层

#### `preprocessing/series_utils.py` — 序列类型工具

| 属性 | 说明 |
|------|------|
| 来源行 | 原文件 2215–2241 行 |
| 行数 | 33 行 |

三个轻量函数，专门解析 Dixon 序列 metadata：

| 函数 | 逻辑 |
|------|------|
| `_get_series_type(desc)` | 取末尾 `_X` 判断 `W`/`F`/`None` |
| `_get_series_prefix(desc)` | 去掉末尾 `_W/_F` 作为配对前缀 |
| `_get_series_number(folder)` | 从文件夹名末尾数字提取 series number |

#### `preprocessing/image_loader.py` — 图像加载与配对

| 属性 | 说明 |
|------|------|
| 来源行 | 原文件 3864–3953 行（`find_fat_water_image`） |
| 行数 | 130 行 |

- `load_nifti(path, slice_idx)`: 加载 NIfTI 文件，自动读取同目录 `metadata.json` 获取像素间距
- `find_fat_water_image(w_nii_path, slice_idx)`: 在同患者目录下搜索最佳配对压水图（F 序列），按"同前缀 → series number 差最小"双优先级排序

#### `preprocessing/slice_selector.py` — 切片优选

| 属性 | 说明 |
|------|------|
| 来源行 | 原文件 3509–3861 行 |
| 行数 | 349 行 |

实现 V14_2 三步掩模合并算法：

**`segment_initial_enhanced_v2(img, ps)`**：
- 上区域（0~65%行）+ 下区域（30~95%行）各做 Otsu 二值化
- 保留所有面积 ≥ 300mm² 的连通域（不止最大）
- 返回 `(canal_seed, upper_mask, lower_mask)` 三元组

**`select_best_slice(data, ps)`**：
- 候选范围：中间切片 ±2 共 5 张
- Step1：上区域最大连通域 → 核心绿色掩模（锁定椎管身份）
- Step2：下区域与核心有像素重叠的连通域合并
- Step3：底行 < 60% 高度时，向下搜索 20mm×10mm 窗口
- 宽度异常检测：≥5 行超过 60mm → 强制 0 分
- 评分：面积（50%）+ 底部位置（50%），全零时选中间切片

#### `preprocessing/coordinate_align.py` — 坐标对齐

| 属性 | 说明 |
|------|------|
| 来源行 | 原文件 3956–3997 行 |
| 行数 | 57 行 |

`align_scan_lines_to_f(scan_lines, w_meta, f_meta, f_img_2d)`: 比对压脂图和压水图的像素间距与坐标原点，计算缩放比例和行列偏移，若偏差 < 1px 则直接复用，否则线性映射所有扫描线坐标。

---

### 2.4 `segmentation/` — 分割与检测层（核心算法）

#### `segmentation/canal_processor.py` — 椎管处理器

| 属性 | 说明 |
|------|------|
| 来源行 | 原文件 28–2213 行 |
| 行数 | 2199 行（最大模块）|

包含 `SpinalCanalProcessor` 类和 `SpinalCordLocator` 类。

**`SpinalCanalProcessor` 核心方法：**

| 方法 | 功能 |
|------|------|
| `__init__(pixel_spacing, meta)` | 初始化，调用 `_build_param_route` 设置自适应参数 |
| `segment_initial(img)` | 双区域 Otsu 初始化椎管种子掩模 |
| `find_dorsal_edge(img, col, row)` | 背侧边缘扫描（斜率比较法） |
| `find_ventral_edge(img, row, col)` | 腹侧边缘扫描 |
| `trace_by_profile(seed, img)` | 逐行追踪椎管轮廓（主追踪函数） |
| `process(img)` | 完整流程：初始化 → 追踪 → ROI |
| `process_with_mask(img, green_mask, seed)` | 带优选掩模的追踪（V14_2 主入口） |
| `smooth_ventral_line / smooth_dorsal_line` | 腹/背侧线 MAD 过滤 + 大窗口平滑 |
| `build_scan_lines(smooth_cols, rows, img)` | 生成腹侧扫描线 |
| `find_endplates_unified(scan_lines, img)` | 统一终板检测（状态机，旧接口） |
| `validate_alternating_pattern(...)` | 交替校验（上/下终板必须交替） |
| `anatomical_distance_recheck(lines)` | 解剖间距复核（20~45mm 窗口） |
| `assign_endplate_colors(lines)` | 颜色分配（上终板=绿，下终板=红） |
| `fallback_endplate_detection(...)` | 回退终板检测（阈值降低重扫） |

**`SpinalCordLocator`：** 在椎管内用 Otsu 定位骨髓，返回质心坐标和掩模。

#### `segmentation/cortical_line.py` — 皮质线构建

| 属性 | 说明 |
|------|------|
| 来源行 | 原文件 2248–2433 行 |
| 行数 | 194 行 |

| 函数 | 功能 |
|------|------|
| `build_cortical2(cols, rows, ps, smooth_mm=50)` | 对皮质线1做 50mm 大窗口移动均值强平滑，生成皮质线2（法线基准） |
| `_repair_cortical2_slope(c2_cols, ps, ...)` | 修复皮质线2两端斜率异常（MAD 检测 + 线性外推） |
| `extend_cortical2_tail(c2_cols, c2_rows, ps, ...)` | 向尾侧延伸皮质线2（用于覆盖更多腰椎节段） |

#### `segmentation/scan_lines_v15.py` — V15 扫描线生成

| 属性 | 说明 |
|------|------|
| 来源行 | 原文件 2434–2521 行 |
| 行数 | 95 行 |

| 函数 | 功能 |
|------|------|
| `build_scan_lines_v15(c2_cols, c2_rows, img_shape, ps, n_lines=40, spacing_mm=2.0)` | 沿皮质线2法线方向生成 40 条扫描线，每条间距 2mm，返回 `(offset_mm, rows, cols, nx, ny)` |
| `convert_to_arc_coord(row, col, c2_cols, c2_rows, arc_len_mm)` | 将像素坐标转换为弧长坐标系（沿皮质线的弧长 mm 值），用于聚类时去除皮质线弯曲影响 |

**设计要点**：V15 采用法线方向扫描（而非早期的水平扫描），确保沿椎体终板真实走向检测，减少斜向椎体的漏检。

#### `segmentation/clustering.py` — 聚类与终板线构建

| 属性 | 说明 |
|------|------|
| 来源行 | 原文件 2539–2748、4158–4510 行 |
| 行数 | 574 行 |

| 函数 | 功能 |
|------|------|
| `cluster_endplates_v15(raw_cands, scan_lines, c2_cols, c2_rows, arc_len_mm, ps, win_mm=5.0, min_lines=15)` | **V15 核心聚类**：将候选点转为弧长坐标，5mm 滑动窗口聚类，≥ min_lines 条扫描线支持则确认为一条终板线 |
| `build_endplate_line_v15(ep, c2_cols, c2_rows, arc_len_mm, ps)` | 将聚类结果拟合为带斜率/截距/法线方向的终板线段对象 |
| `find_vertebra_right_edge_from_candidates(raw_cands, scan_lines_f, ps, endplates_f)` | V12.3 后缘双线共识：取 offset 最小的 3 条线，±4mm 容差内 ≥2 条线有同类型候选点则确认；含交替校验和锚点验证 |
| `sliding_window_cluster_endplates(raw_cands, scan_lines_f, ps, win_mm=5.0, min_lines=7)` | V12.4 滑动窗口聚类（旧版，保留兼容） |

**聚类原理**：将候选点投影到弧长坐标系，消除皮质线弯曲的影响，使同一终板的点在弧长轴上紧密分布，便于 5mm 窗口准确聚类。

#### `segmentation/endplate_detector.py` — 终板检测主函数

| 属性 | 说明 |
|------|------|
| 来源行 | 原文件 4511–5147 行 |
| 行数 | 645 行（含全局回退与点数回退逻辑） |

`find_endplates_on_water_image(scan_lines_f, f_img_2d, ps, ...)` 是 V15 终板检测的核心函数：

**内部实现流程：**
1. **三区信号统计**：将扫描线按 offset 分为后区/中区/前区，各区独立计算 high_mean/low_mean
2. **状态机扫描**：对每条扫描线用信号幅度状态机（global_med/drop_ratio/rise_ratio）检测下降沿（inferior）和上升沿（superior）候选点
3. **点数阈值过滤**：每条终板线的候选点数 ≥ min_pts_ep（默认 28）才保留
4. **解剖校正**：调用 `anatomical_gap_correction` 修正异常间距
5. **缺失插补**：调用 `fill_missing_endplates` 补全漏检终板

**关键参数（可由 main.py 覆盖）：**
- `use_half_global_med=True`：全局回退时将 global_med 降至 50%
- `drop_ratio_override / rise_ratio_override`：强制指定状态机阈值
- 返回值包含 `raw_candidates / endplates / all_sigs / drop_ratio / rise_ratio` 等供调用方使用

#### `segmentation/anterior_edge.py` — 前缘检测

| 属性 | 说明 |
|------|------|
| 来源行 | 原文件 2749–3507、5148–5707 行 |
| 行数 | 1334 行 |

包含 11 个函数，实现椎体前缘轮廓的完整检测流程：

| 函数 | 功能 |
|------|------|
| `build_anterior_scan_lines_v15(...)` | 沿皮质线前方生成前缘扫描线组（法线反方向，多层偏移） |
| `find_anterior_min_points_h(...)` | 水平方向扫描找前缘信号最低点 |
| `find_anterior_min_points_v15(...)` | V15 法线方向扫描找前缘信号最低点 |
| `refine_anterior_edge_v15(...)` | 精化前缘点（去噪 + 斜率修正） |
| `scan_anterior_edge_v15(...)` | 主扫描函数：整合多层扫描线结果，输出前缘轮廓点集 |
| `cluster_anterior_edge_v15(...)` | 前缘点聚类（去除离群点，保留主轮廓） |
| `smooth_anterior_edge_v15(...)` | 前缘轮廓平滑（5mm 移动均值） |
| `find_anterior_corner_v15(...)` | 前缘角定位：沿终板线延伸方向逐像素搜索，连续 2 点双侧探测低信号则确认角点 |
| `find_anterior_edge_by_descent(...)` | 下降法前缘检测（ROI 内逐列找最小值） |
| `find_arc_roi_min_points(...)` | 弧长 ROI 内最小信号点检测 |
| `refine_arc_roi_to_anterior_edge(...)` | ROI 最小值点精化为前缘 |
| `filter_arc_roi_by_dense_offset(...)` | 密度过滤：只保留 offset 密集区域的前缘点（白点筛选） |

---

### 2.5 `postprocessing/` — 后处理与可视化层

#### `postprocessing/anatomical_correction.py` — 解剖校正

| 属性 | 说明 |
|------|------|
| 来源行 | 原文件 4816–5117 行（从 `find_endplates_on_water_image` 内嵌函数提取，已去缩进） |
| 行数 | 312 行 |

| 函数 | 功能 |
|------|------|
| `enforce_alternating(eps)` | **交替校验**：上下终板必须严格交替（superior → inferior → ...），连续同类型时保留行号更小（更靠头侧）的 |
| `align_to_midline(endplates, scan_lines_f)` | 将所有终板点列坐标对齐到第 7 条（中心）扫描线，消除横向偏移误差 |
| `anatomical_gap_correction(endplates, all_sigs, scan_lines_f, ps)` | **间距复核**：MAD 鲁棒统计正常椎体/椎间盘间距，对异常点做前后双侧检查，预测修正位置并重新扫描信号 |
| `fill_missing_endplates(eps, scan_lines_f, ps)` | **缺失插补**：发现连续同类型终板（隐含缺失）时，按正常间距预测插入缺失点 |

**设计注意**：这 4 个函数原为 `find_endplates_on_water_image` 的局部函数，提取为模块级函数后，调用方需显式传入参数（不再依赖闭包变量）。

#### `postprocessing/vertebrae_chain.py` — 椎体编号分配

| 属性 | 说明 |
|------|------|
| 来源行 | 原文件 5763–5865 行 |
| 行数 | 112 行 |

`assign_vertebra_labels(endplates_f, ps)`: 基于已排序的终板对，从最尾侧起按 L5→L4→L3→L2→L1→T12→T11... 顺序分配编号。不足 5 节腰椎时，根据椎体间距外推推断编号（标记 `inferred=True`，可视化时用圆圈而非方框标注）。

#### `postprocessing/geometric_center.py` — 椎体几何建模

| 属性 | 说明 |
|------|------|
| 来源行 | 原文件 4000–4156 行 |
| 行数 | 168 行 |

`compute_vertebra_geometry(endplates_f, arc_final, smooth_cols, all_rows, ps, vert_edge_data)`:

对每对椎体（上终板 + 下终板）建模 8 个几何点：

| 点 | 含义 |
|-----|------|
| `AP_top / AP_bot` | 前缘上/下角点（arc_final 黄红/红黄切换行） |
| `BP_top / BP_bot` | 后缘上/下角点（双线共识结果，皮质线精确插值列） |
| `mid_top / mid_bot` | 上/下边中点 |
| `mid_ant / mid_pos` | 前/后边中点 |
| `centroid` | 椎体质心（四角平均） |

列坐标统一从皮质线/arc 插值，不使用两点列均值，保证精度。

#### `postprocessing/visualization.py` — 可视化渲染

| 属性 | 说明 |
|------|------|
| 来源行 | 原文件 5867–7007 行 |
| 行数 | 1155 行 |

`visualize_results(img_raw, traced, cord_mask, roi_points, ...)`: 生成双图拼接 PNG：

**左图（压脂图 W）：**
- 灰度图叠加椎管轮廓（绿色）
- 皮质线1（青色）、皮质线2（黄色）
- V15 扫描线（橙色，40 条）
- 骨髓质心（紫色圆点）

**右图（压水图 F）：**
- 灰度图叠加终板线（上终板=绿色，下终板=红色）
- 前缘轮廓（黄色/橙色）
- 前缘角点（青色三角）
- 椎体编号标注（L5/L4/L3... 方框或圆圈）
- 几何中心点（蓝色系列）

---

## 三、关键设计决策

### 3.1 V15 全局回退机制

**触发条件**：初始聚类后上终板线 < 5 条 OR 下终板线 < 5 条  
**操作**：
```python
global_med  × 0.5   # 降低状态机基准信号
drop_ratio  × 0.7   # 更容易检测下降沿
rise_ratio  × 0.7   # 更容易检测上升沿
min_lines   × 0.7   # 降低聚类最小支持线数（15 → 10）
```
重新扫描全 40 条线，去重合并候选点，重新聚类，**并补全交替校验**。

### 3.2 点数回退机制

触发条件：存在点数 < `_min_pts_ep`（默认 28，全局回退后按比例下调）的终板线  
操作：对弱线降阈值（×0.80 → ×0.65）重扫，最多 2 轮，**每轮聚类后补全交替校验**。

### 3.3 交替校验的两处修复

`cluster_endplates_v15` 本身只做弧长聚类，不含交替校验。V15 在以下两处手动补全：
1. 全局回退聚类后（`main.py` 第 7211–7217 行逻辑）
2. 每轮点数回退聚类后（`main.py` 第 7288–7294 行逻辑）

### 3.4 旧流程清理（方案 C）

已移除的内容：
- `vert_segments_all` 变量（V12 椎体段检测，已废弃）
- `find_endplates_on_water_image` 返回值中的 `'vert_segments'` 键
- 约 50 行 V12 旧流程日志（`聚类后终板线组`、`[V12.3]插补后`、`压水终板`列表等）

### 3.5 解剖校正函数的提取

`enforce_alternating / align_to_midline / anatomical_gap_correction / fill_missing_endplates` 原为 `find_endplates_on_water_image` 内部的局部函数（4 格缩进）。提取到 `postprocessing/anatomical_correction.py` 时需**去掉 4 格缩进**，并将闭包变量改为显式参数传入。

---

## 四、模块依赖关系（import 图）

```
main.py
  ├── config.metadata_parser
  ├── preprocessing.series_utils
  ├── preprocessing.image_loader        → preprocessing.series_utils
  ├── preprocessing.slice_selector
  ├── preprocessing.coordinate_align
  ├── segmentation.canal_processor      → segmentation.cortical_line
  │                                     → segmentation.scan_lines_v15
  ├── segmentation.clustering           → segmentation.scan_lines_v15
  ├── segmentation.endplate_detector    （内嵌 enforce_alternating 等4个解剖校正函数）
  ├── segmentation.anterior_edge
  ├── postprocessing.geometric_center
  └── postprocessing.visualization      → postprocessing.vertebrae_chain（内部调用）
```

`postprocessing.anatomical_correction` 中的 4 个函数（`enforce_alternating` / `align_to_midline` / `anatomical_gap_correction` / `fill_missing_endplates`）在 `segmentation.endplate_detector` 内部以**内嵌局部函数**方式重复定义并直接调用，不依赖外部 import。`postprocessing.anatomical_correction` 模块当前仅作为独立提取版本存档，供后续解耦重构使用。

---

## 五、运行使用方法

### 5.1 环境要求

```bash
Python >= 3.8
nibabel >= 3.0
numpy
scipy
scikit-image
matplotlib
```

安装依赖（在项目根目录执行）：

```bash
pip install nibabel numpy scipy scikit-image matplotlib
```

### 5.2 目录结构要求

输入数据需符合以下目录结构（与原始 DICOM 转换器输出一致）：

```
患者目录/
├── T2_TSE_DIXON_SAG_W_0005/      # 压脂图（W 序列）
│   ├── scan.nii.gz
│   └── metadata.json
└── T2_TSE_DIXON_SAG_F_0006/      # 压水图（F 序列，自动配对）
    ├── scan.nii.gz
    └── metadata.json
```

`metadata.json` 至少包含：
```json
{
  "acquisition_params": {
    "pixel_spacing_mm": [0.9375, 0.9375],
    "magnetic_field_strength": 1.5,
    "slice_thickness_mm": 4.0
  },
  "series_info": {
    "series_description": "T2_TSE_DIXON_SAG_W_0005"
  },
  "patient_info": {
    "patient_id": "P12345"
  }
}
```

> 若没有 `metadata.json`，系统使用默认像素间距 0.9375mm，可能影响参数路由精度。

### 5.3 运行方式

#### ⚠️ 重要：必须在 `LSMATOOLS_CP/` 目录内运行

`main.py` 使用相对 import，**必须切换到该目录后再执行**，否则会报 `ModuleNotFoundError`。

```bash
cd /Users/mac/mri_lumbarpv/lumbar_roitest/LSMATOOLS_CP
```

---

#### 方式 A：交互式菜单

```bash
python3 main.py
```

启动后出现交互菜单：

```
请选择模式 (1/2):
```

**选 `1`（单张图像）：**
```
请输入图像文件或目录路径: /path/to/T2_TSE_DIXON_SAG_W_0005
请输入输出目录 (直接回车默认为./test_output):
```
- 输入目录时，自动使用目录下的 `scan.nii.gz` 和 `metadata.json`
- 输入 `.nii.gz` 文件路径时，自动从同目录读取 `metadata.json`

**选 `2`（批量处理）：**
```
请输入父目录路径: /path/to/input
```
- 遍历父目录下所有患者子目录，自动识别 T2 Dixon W 序列并处理
- 输出目录固定为 `batch_output_v14_2/`（可在 `main.py` 第 938 行修改）

---

#### 方式 B：Python 脚本调用

```python
import sys
sys.path.insert(0, '/Users/mac/mri_lumbarpv/lumbar_roitest/LSMATOOLS_CP')

from main import test_single_image

test_single_image(
    nifti_path='/path/to/scan.nii.gz',
    metadata_path='/path/to/metadata.json',   # 可选，None 时用默认间距
    output_dir='/path/to/output',              # 可选，默认 ./test_output
    patient_dir='P12345',                      # 可选，用于输出文件命名
    seq_dir='T2_TSE_DIXON_SAG_W_0005',        # 可选，用于输出文件命名
)
```

---

#### 方式 C：批量调用（脚本）

```python
import sys
sys.path.insert(0, '/Users/mac/mri_lumbarpv/lumbar_roitest/LSMATOOLS_CP')

from main import process_batch

process_batch(
    parent_dir='/Users/mac/mri_lumbarpv/lumbar_roitest/input',
    output_dir='/Users/mac/mri_lumbarpv/lumbar_roitest/batch_output_v15',
)
```

### 5.4 输出文件说明

每张图像处理完成后，在 `output_dir` 生成两个文件：

| 文件 | 格式 | 内容 |
|------|------|------|
| `{患者ID}_{序列名}_TRACED.png` | PNG | 双图拼接可视化结果（左压脂 + 右压水） |
| `{患者ID}_{序列名}_log.json` | JSON | 完整处理日志（终板坐标、状态机参数、椎体编号等） |

**PNG 图像内容示例：**
- 左图：椎管轮廓（绿）+ 皮质线1（青）+ 皮质线2（黄）+ 40 条扫描线（橙）
- 右图：终板线（上终板=绿，下终板=红）+ 前缘轮廓（黄）+ 椎体编号（L5/L4/L3/L2/L1/T12）

### 5.5 常见问题

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| `ModuleNotFoundError: No module named 'config'` | 不在 `LSMATOOLS_CP/` 目录内运行 | `cd LSMATOOLS_CP` 后再运行 |
| `[WARN] 未找到压水图F序列` | 同患者目录下无 T2 Dixon F 序列 | 检查目录结构，确保 W/F 序列在同一患者目录下 |
| 终板线数量异常（< 5 条）| 信号对比度低，状态机触发门槛过高 | 系统会自动触发全局回退（`global_med × 0.5`） |
| 输出目录无文件 | 椎管追踪失败 | 查看控制台日志，检查图像信号质量 |

---

## 六、文件速查表

| 文件 | 行数 | 核心函数/类 |
|------|------|------------|
| `main.py` | 913 | `test_single_image`, `process_batch`, `main` |
| `config/params.py` | 49 | `V15Params` |
| `config/param_route.py` | 68 | `build_param_route` |
| `config/metadata_parser.py` | 44 | `load_metadata`, `parse_pixel_spacing` 等 |
| `preprocessing/series_utils.py` | 33 | `_get_series_type/prefix/number` |
| `preprocessing/image_loader.py` | 130 | `load_nifti`, `find_fat_water_image` |
| `preprocessing/slice_selector.py` | 349 | `segment_initial_enhanced_v2`, `select_best_slice` |
| `preprocessing/coordinate_align.py` | 57 | `align_scan_lines_to_f` |
| `segmentation/canal_processor.py` | 2199 | `SpinalCanalProcessor`, `SpinalCordLocator` |
| `segmentation/cortical_line.py` | 194 | `build_cortical2`, `_repair_cortical2_slope`, `extend_cortical2_tail` |
| `segmentation/scan_lines_v15.py` | 95 | `build_scan_lines_v15`, `convert_to_arc_coord` |
| `segmentation/clustering.py` | 574 | `cluster_endplates_v15`, `build_endplate_line_v15` 等 |
| `segmentation/endplate_detector.py` | 645 | `find_endplates_on_water_image` |
| `segmentation/anterior_edge.py` | 1334 | 11 个前缘检测函数 |
| `postprocessing/anatomical_correction.py` | 312 | `enforce_alternating`, `anatomical_gap_correction` 等 |
| `postprocessing/vertebrae_chain.py` | 112 | `assign_vertebra_labels` |
| `postprocessing/geometric_center.py` | 168 | `compute_vertebra_geometry` |
| `postprocessing/visualization.py` | 1155 | `visualize_results` |
| **合计** | **~7700** | **23 个 .py 文件** |
