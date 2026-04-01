# LSMATOOLS API Reference

**Version**: V15.0  
**Last Updated**: 2026-03-09

---

## Table of Contents

- [Main Entry Points](#main-entry-points)
- [Segmentation Module](#segmentation-module)
- [Preprocessing Module](#preprocessing-module)
- [Postprocessing Module](#postprocessing-module)
- [Configuration](#configuration)
- [Data Structures](#data-structures)

---

## Main Entry Points

### `main.py`

#### `test_single_image(nifti_path, metadata_path, output_dir)`

处理单个 NIfTI 文件。

**Parameters**:
- `nifti_path` (str): NIfTI 文件路径（.nii.gz）
- `metadata_path` (str): Metadata JSON 文件路径（可选）
- `output_dir` (str): 输出目录路径

**Returns**: None

**Example**:
```python
from LSMATOOLS_CP.main import test_single_image

test_single_image(
    nifti_path='patient/T2_Dixon_W.nii.gz',
    metadata_path='patient/metadata.json',
    output_dir='./results'
)
```

#### `batch_process(input_dir, output_dir)`

批量处理多个病例。

**Parameters**:
- `input_dir` (str): 输入目录（包含多个子目录，每个含 scan.nii.gz）
- `output_dir` (str): 输出目录

**Returns**: None

**Example**:
```python
from LSMATOOLS_CP.main import batch_process

batch_process(
    input_dir='./patients',
    output_dir='./batch_results'
)
```

---

## Segmentation Module

### `segmentation/spinal_canal.py`

#### `class SpinalCanalProcessor`

椎管分割处理器。

**Methods**:

##### `__init__(pixel_spacing, image_shape)`

初始化处理器。

**Parameters**:
- `pixel_spacing` (float): 像素间距（mm）
- `image_shape` (tuple): 图像形状 (rows, cols)

##### `process(img_w)`

执行椎管分割。

**Parameters**:
- `img_w` (np.ndarray): T2 Dixon W 图像（压脂图）

**Returns**:
- `cortical_line_1`: 皮质线 1（白色实线）
- `cortical_line_2`: 皮质线 2（紫色虚线）
- `dorsal_line`: 背部线（橙色实线）
- `marrow_mask`: 脊髓掩模

**Example**:
```python
from LSMATOOLS_CP.segmentation.spinal_canal import SpinalCanalProcessor

processor = SpinalCanalProcessor(pixel_spacing=0.5, image_shape=(512, 512))
c1, c2, dorsal, marrow = processor.process(img_w)
```

### `segmentation/anterior_edge.py`

#### `find_endplates_on_water_image(scan_lines, img_f, params)`

在压水图上检测终板候选点。

**Parameters**:
- `scan_lines` (list): 扫描线列表（来自 cortical line 2）
- `img_f` (np.ndarray): T2 Dixon F 图像（压水图）
- `params` (dict): 参数配置

**Returns**:
- `endplates`: List[(row, col, ep_type)] 终板点列表

**Example**:
```python
from LSMATOOLS_CP.segmentation.anterior_edge import find_endplates_on_water_image

endplates = find_endplates_on_water_image(
    scan_lines=scan_lines,
    img_f=img_f,
    params=params
)
```

#### `refine_arc_roi_to_anterior_edge(arc_initial, img_w, img_f, params)`

精修前缘 ROI 到真实前缘位置。

**Parameters**:
- `arc_initial` (list): 初始前缘点集
- `img_w` (np.ndarray): 压脂图
- `img_f` (np.ndarray): 压水图
- `params` (dict): 参数配置

**Returns**:
- `arc_refined`: List[(row, col, val, base_col, flag)] 精修点集

#### `find_anterior_edge_by_descent(cortical_line_2, img_w, params)`

通过下降沿检测前缘。

**Parameters**:
- `cortical_line_2` (list): 皮质线 2 点集
- `img_w` (np.ndarray): 压脂图
- `params` (dict): 参数配置

**Returns**:
- `descent_points`: List[(row, col, flag, src_tag, base_col)]

#### `filter_arc_roi_by_dense_offset(points, pixel_spacing, window_mm, step_mm, expand_ratio)`

密集窗口过滤，找到最佳密度区间。

**Parameters**:
- `points` (list): 前缘点集（refined + confirmed）
- `pixel_spacing` (float): 像素间距
- `window_mm` (float): 窗口宽度（默认 6.0mm）
- `step_mm` (float): 步长（默认 0.5mm）
- `expand_ratio` (float): 扩展比例（默认 3.0）

**Returns**:
- `filtered_points`: 过滤后的点集
- `best_range`: (best_lo, best_hi) 最佳区间

---

## Preprocessing Module

### `preprocessing/series_selection.py`

#### `select_best_t2_series(series_list)`

选择最佳 T2 序列。

**Parameters**:
- `series_list` (list): 序列元数据列表

**Returns**:
- `best_series`: 最佳序列索引

### `preprocessing/slice_selection.py`

#### `select_best_slice(image_stack)`

选择最佳中间层。

**Parameters**:
- `image_stack` (np.ndarray): 图像堆叠

**Returns**:
- `best_index`: 最佳层索引

---

## Postprocessing Module

### `postprocessing/visualization.py`

#### `visualize_results(img_w, img_f, vertebrae_chain, output_file)`

生成可视化结果图。

**Parameters**:
- `img_w` (np.ndarray): 压脂图
- `img_f` (np.ndarray): 压水图
- `vertebrae_chain` (list): 椎体链数据
- `output_file` (str): 输出文件路径

**Returns**: None

**Example**:
```python
from LSMATOOLS_CP.postprocessing.visualization import visualize_results

visualize_results(
    img_w=img_w,
    img_f=img_f,
    vertebrae_chain=vertebrae_chain,
    output_file='results/overlay.png'
)
```

#### `build_scan_lines_v15(cortical_line_2, pixel_spacing, num_lines=40)`

生成 V15 法线方向扫描线。

**Parameters**:
- `cortical_line_2` (list): 皮质线 2 点集
- `pixel_spacing` (float): 像素间距
- `num_lines` (int): 扫描线数量（默认 40）

**Returns**:
- `scan_lines`: List[(offset_mm, rows, cols, nx, ny)]

### `postprocessing/export.py`

#### `export_vertebra_masks(vertebrae_chain, reference_nifti, output_file)`

导出多标签椎体掩模 NIfTI 文件。

**Parameters**:
- `vertebrae_chain` (list): 椎体链数据
- `reference_nifti` (nib.Nifti1Image): 参考 NIfTI 图像
- `output_file` (str): 输出文件路径

**Returns**: None

**Example**:
```python
from LSMATOOLS_CP.postprocessing.export import export_vertebra_masks

export_vertebra_masks(
    vertebrae_chain=vertebrae_chain,
    reference_nifti=nifti_img,
    output_file='results/vertebra_masks.nii.gz'
)
```

#### `export_geometric_data(vertebra_analyses, output_file)`

导出椎体几何数据 CSV 文件。

**Parameters**:
- `vertebra_analyses` (list): 椎体分析结果列表
- `output_file` (str): 输出 CSV 文件路径

**Returns**: None

**CSV Columns**:
- `Vertebra`: 椎体名称（L5, L4, ...）
- `Area_mm2`: 椎体面积（mm²）
- `Top_Angle`: 上终板角度（°）
- `Bot_Angle`: 下终板角度（°）
- `C1_Angle`: 皮质线 1 角度（°）
- `Front_Angle`: 前缘线角度（°）
- `Height_Ant_mm`: 前缘高度（mm）
- `Height_Post_mm`: 后缘高度（mm）

---

## Configuration

### `config/parameters.py`

#### 默认参数表

```python
DEFAULT_PARAMS = {
    # 前缘检测
    'rise_ratio': 0.50,           # 上升沿触发比例
    'scan_mm': 40.0,              # 扫描距离
    'left_off_mm': 20.0,          # 右边界 offset
    'right_off_mm': 40.0,         # 左边界 offset
    
    # 密集窗口
    'window_mm': 6.0,             # 窗口宽度
    'step_mm': 0.5,               # 步长
    'expand_ratio': 3.0,          # 底行扩展倍数（统一参数）
    
    # 终板检测
    'probe_mm': 5.0,              # 验证探针长度
    'probe_ratio': 0.50,          # 验证比例阈值
    
    # 平滑参数
    'smooth_k': 8.0,              # 前缘线平滑窗口（mm）
    
    # 分辨率路由
    'hr_threshold': 0.50,         # HR 等级上限
    'std_threshold': 0.75,        # STD 等级上限
}
```

---

## Data Structures

### Vertebra Chain

```python
{
    'name': str,                  # 椎体名称（L5, L4, ...）
    'top_meta': dict,             # 上终板元数据
    'bot_meta': dict,             # 下终板元数据
    'r_top': float,               # 上终板 row 坐标
    'c_top': float,               # 上终板 col 坐标
    'r_bot': float,               # 下终板 row 坐标
    'c_bot': float,               # 下终板 col 坐标
    'row_top': int,               # 上终板切片索引
    'row_bot': int,               # 下终板切片索引
    'top_ix': int,                # 上终板索引
    'bot_ix': int,                # 下终板索引
    'top_c1': Tuple[float, float],   # 上终板 - 皮质线 1 交点
    'top_front': Tuple[float, float],# 上终板 - 前缘交点
    'bot_c1': Tuple[float, float],   # 下终板 - 皮质线 1 交点
    'bot_front': Tuple[float, float],# 下终板 - 前缘交点
}
```

### Analysis Result

```python
{
    'name': str,                  # 椎体名称
    'area_mm2': float,            # 椎体面积
    'angles': {
        'top': float,             # 上终板角度
        'bot': float,             # 下终板角度
        'c1': float,              # 皮质线 1 角度
        'fr': float,              # 前缘线角度
    }
}
```

### Point Formats

#### Refined Point (5-tuple)
```python
(row: int, col: float, val: float, base_col: float, flag: str)
# flag: 'refined' | 'kept' | 'kept_low'
```

#### Descent Detection Point (5-tuple)
```python
(row: int, col: float, flag: str, src_tag: str, base_col: float)
# flag: 'confirmed' | 'not_found'
# src_tag: 'main' | 'supp_upper' | 'supp_lower'
```

---

## Error Handling

### Common Exceptions

#### `NoEndplateFoundError`

当无法检测到足够的终板点时抛出。

```python
from LSMATOOLS_CP.segmentation.anterior_edge import NoEndplateFoundError

try:
    endplates = find_endplates_on_water_image(...)
except NoEndplateFoundError as e:
    print(f"Endplate detection failed: {e}")
```

#### `SegmentationFailedError`

椎管分割失败时抛出。

### Retry Strategy

V15 引入双重回退机制：

1. **全局回退**: 总点数 < 5 → 更换策略重新扫描
2. **点数回退**: 单条 < 28 点 → 最多 2 轮回退（阈值因子 [0.80, 0.65]）

---

## Examples

### Complete Workflow

```python
from LSMATOOLS_CP.main import test_single_image
from LSMATOOLS_CP.postprocessing.visualization import visualize_results
from LSMATOOLS_CP.postprocessing.export import export_vertebra_masks, export_geometric_data

# 单病例处理
test_single_image(
    nifti_path='patient/T2_Dixon_W.nii.gz',
    metadata_path='patient/metadata.json',
    output_dir='./results'
)

# 自定义可视化
visualize_results(
    img_w=img_w,
    img_f=img_f,
    vertebrae_chain=vertebrae_chain,
    output_file='custom_output.png'
)

# 导出掩模
export_vertebra_masks(
    vertebrae_chain=vertebrae_chain,
    reference_nifti=nifti_img,
    output_file='masks.nii.gz'
)

# 导出 CSV
export_geometric_data(
    vertebra_analyses=analyses,
    output_file='geometric_data.csv'
)
```

---

**API Version**: V15.0  
**Last Updated**: 2026-03-09  
**Maintained by**: LSMATOOLS Contributors
