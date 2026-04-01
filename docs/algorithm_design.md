# LSMATOOLS Algorithm Design Overview

**Version**: V15.0  
**Last Updated**: 2026-03-09

---

## 1. System Architecture

```
Input: T2 Dixon W (fat-suppressed) + F (water-only) NIfTI volumes
    │
    ├─ Preprocessing
    │   ├─ Series selection (T2 + Dixon + W-type validation)
    │   └─ Best slice selection (geometric constraints + scoring)
    │
    ├─ Spinal Canal Segmentation
    │   ├─ Dual-region Otsu thresholding
    │   ├─ Maximum connected component extraction
    │   └─ Boundary tracking → Cortical Line 1 & 2
    │
    ├─ Scan Line Generation (V15)
    │   ├─ 40 normal-direction lines (1mm spacing)
    │   └─ Following cortical line 2 curvature
    │
    ├─ Endplate Detection (on water image)
    │   ├─ Three-zone signal statistics (Otsu)
    │   ├─ State machine descent/ascent edge detection
    │   ├─ Arc-length clustering
    │   └─ V15 dual retry strategy
    │
    ├─ Anterior Edge Detection
    │   ├─ Main stream: Ascent edge refinement (yellow line)
    │   ├─ Parallel scheme: Descent edge detection (cyan line)
    │   └─ Dual-modal fusion → Red anterior line
    │
    └─ Vertebra Chain Recognition
        ├─ Automatic S1/L5 identification
        ├─ Thoracic extension naming (T12-T8)
        ├─ Geometric center annotation
        └─ Dorsal line smoothing enhancement
```

---

## 2. Core Modules

### 2.1 Spinal Canal Segmentation

**Algorithm**: `SpinalCanalProcessor`

**Key Steps**:

1. **Dual-Region Initialization**
   - Lower region: rows 40%-90% (covers lumbosacral segments)
   - Upper region: rows 5%-55% (covers thoracolumbar segments)
   - Independent Otsu + max connected component + 1px dilation

2. **Dorsal Edge Detection**
   - Reference signal: 5mm backward from canal boundary
   - Scan range: up to 30mm forward
   - Trigger: ≥40% signal drop from reference
   - Strategy: Find first drop, continue 5mm, select maximum gradient point

3. **Boundary Smoothing** (3-step robust smoothing)
   - MAD filtering: Remove outliers > 2.0σ in sliding window
   - Linear interpolation: Fill gaps using `np.interp`
   - Moving average: Window size based on pixel_spacing (~5mm)

### 2.2 Scan Line Generation (V15)

**Function**: `build_scan_lines_v15`

**Design**:

- **Old (V12.5)**: Horizontal translation (fixed 2mm spacing, 13 lines)
- **New (V15)**: Normal-direction translation (1mm spacing, 40 lines)

**Normal Vector Calculation**:

```python
# For each point i on cortical line 2:
tangent = (c2_rows[i+1] - c2_rows[i-1], 
           c2_cols[i+1] - c2_cols[i-1])  # Forward-backward difference
normal = (t_dc, -t_dr)  # Clockwise rotation 90°
# Ensure normal points ventrally (column decreasing direction)
```

**Output**: List of `(offset_mm, rows_arr, cols_arr, nx_arr, ny_arr)`

### 2.3 Endplate Detection

**Function**: `find_endplates_on_water_image`

**Three-Zone Statistics**:

| Zone | Representative Lines | Output Parameters |
|------|---------------------|-------------------|
| Posterior | index 3-5 | high_mean1, low_mean1, drop_ratio1 |
| Middle | index 13-15 | high_mean2, low_mean2, drop_ratio2 |
| Anterior | index 23-25 | high_mean3, low_mean3, drop_ratio3 |

**State Machine**:

```
looking_for = 'drop' (upper endplate)
  ↓ Trigger: cur < ref × (1 - drop_ratio)
  ↓ Verify: probe_mm low-signal ratio >= probe_ratio
  ↓ Confirm: label 'upper'
  ↓ Switch: looking_for = 'rise'

looking_for = 'rise' (lower endplate)
  ↓ Trigger: cur > ref × (1 + rise_ratio)
  ↓ Verify: probe_mm high-signal ratio >= probe_ratio
  ↓ Confirm: label 'lower'
  ↓ Switch: looking_for = 'drop'
```

**Clustering**: Arc-length coordinate system with 5mm sliding window

**V15 Retry Strategy**:
- Global retry: Total points < 5 → restart with different strategy
- Point count retry: Single line < 28 points → up to 2 retries with factors [0.80, 0.65]

### 2.4 Anterior Edge Detection

#### Main Stream: Ascent Edge Refinement

**Function**: `refine_arc_roi_to_anterior_edge`

**Search Space**:

- Right boundary: offset = 20mm (near cortical line)
- Left boundary: offset = 40mm (far from cortical line)
- Expand ratio: 3.0× (bottom row expansion, unified parameter)

**Refinement Process**:

```python
For each valley point (row, col):
  1. Extract scan segment: col_start = col + 2px (skip cortical bone)
                          col_end   = col + 40mm
  2. Smooth: 2px moving average
  3. Mode C check: if mean(seg < low_mean) > 50% → 'kept_low' (skip)
  4. Scan for ascent edge:
     if cur > ref × (1 + rise_ratio):  # Signal increase > 50%
       
       [Fat Filter]
       Check 12 neighborhood pixels at j+1
       if neighborhood_mean > high_mean × 1.3:
         continue  # Fat signal, skip
       
       [Mode B: High-Low-High Filter]
       if rest contains consecutive ≥2px signal < low_mean:
         continue  # Fat + cortical bone pattern, skip
       
       [Mode A: High Signal Confirmation]
       if mean(rest >= high_thr) >= probe_ratio AND mean(rest) >= high_thr:
         → Confirmed as anterior edge
         → Mark 'refined', new_col = col_start + j
```

#### Parallel Scheme: Descent Edge Detection

**Function**: `find_anterior_edge_by_descent`

**Starting Point**: offset = 20mm from cortical line 2 (inside vertebral marrow)

**Scan Direction**: Horizontal leftward (not normal direction)

**Trigger Conditions** (all must be satisfied):

```python
for t in 1..scan_px:
  cur = seg_sm[t]
  running_max = max(running_max, cur)
  
  if running_max > 1.0 and \
     cur < running_max × (1 - drop_ratio) and \
     cur < low_mean3 × 1.3:  # [V13_2] Relaxed for cortical bone
    → Record (row, col, 'confirmed', src_tag, base_col)
```

**Return Format**: `(row, col, flag, src_tag, base_col)` (5-tuple with base_col)

### 2.5 Dual-Modal Fusion

**Function**: `filter_arc_roi_by_dense_offset`

**Input**: Combined points from:
- Ascent refinement: `'refined'` points only (exclude `'kept'`, `'kept_low'`)
- Descent detection: `'confirmed'` points only

**Process**:

1. Re-run dense window filtering on combined point set
2. Window: 6.0mm width, 0.5mm step
3. Expand ratio: 3.0× (unified parameter for search/filter/visualization)
4. Find best density interval `[best_lo, best_hi]`

**Red Line Generation**:

```python
Data sources:
  1. Dense window points (arc_combined_v13)
     Condition: best_lo - ε ≤ offset ≤ best_lo + window_mm × (1+(expand-1)×t)
  
  2. Right-side supplementary points (ascent refined only)
     Condition:
       ① offset ∈ [best_lo - 5mm, best_lo)  ← [V13_2] 5mm constraint
       ② No dense window point in this row
       ③ Row ≤ top_point_row + half_cortical_height

Smoothing:
  Step A: MAD filtering (window=11, threshold=2.0σ)
  Step B: np.interp linear interpolation
  Step C: Moving average (k = max(3, round(8.0/pixel_spacing)))
```

---

## 3. Vertebra Chain Recognition

### 3.1 Naming Convention

**Naming Sequences** (from bottom to top):

| Starting Vertebra | Sequence |
|------------------|----------|
| **L5** | L5 → L4 → L3 → L2 → L1 → **T12 → T11 → T10 → T9 → T8** |
| **S1** | S1 → L5 → L4 → L3 → L2 → L1 → **T12 → T11 → T10 → T9 → T8** |

**Code Logic**:

```python
if _lowest_name == 'L5':
    _names_from_bottom = ['L5', 'L4', 'L3', 'L2', 'L1', 
                          'T12', 'T11', 'T10', 'T9', 'T8']
else:
    _names_from_bottom = ['S1', 'L5', 'L4', 'L3', 'L2', 'L1', 
                          'T12', 'T11', 'T10', 'T9', 'T8']

for idx, vertebra in enumerate(reversed(vertebrae_chain)):
    if idx < len(_names_from_bottom):
        vertebra['name'] = _names_from_bottom[idx]
    else:
        # Auto-generate T7/T6/... for >10 vertebrae
        vertebra['name'] = f'T{12-(idx-5)}' if idx >= 5 else f'V{idx+1}'
```

### 3.2 Geometric Center Annotation

**Calculation**:

```python
# Four corner points
pt_top_c1      # Superior endplate - cortical line 1 intersection
pt_top_front   # Superior endplate - anterior edge intersection
pt_bot_c1      # Inferior endplate - cortical line 1 intersection
pt_bot_front   # Inferior endplate - anterior edge intersection

# Geometric center
center_row = (pt_top_c1.row + pt_top_front.row + 
              pt_bot_front.row + pt_bot_c1.row) / 4.0
center_col = (pt_top_c1.col + pt_top_front.col + 
              pt_bot_front.col + pt_bot_c1.col) / 4.0
```

**Annotation Position**:

- X coordinate: Fixed left column (`x = 5` pixels)
- Y coordinate: Aligned with geometric center row
- Style: White text, black rounded box background

### 3.3 Dorsal Line Smoothing Enhancement

**Parameters** (consistent with cortical line 1):

- MAD filter: Window=21, threshold=3.5σ (wider window for higher noise)
- Interpolation: `np.interp` (same)
- Moving average: `k = max(3, int(5/pixel_spacing))` (~5mm window, same)

---

## 4. Parameter Routing System

### Resolution Grades

| Grade | Pixel Spacing | tol_mm | tol_mm_fallback | min_pts |
|-------|--------------|--------|-----------------|---------|
| HR    | ≤0.50mm      | 2.0mm  | 3.0mm           | 7       |
| STD   | ≤0.75mm      | 2.5mm  | 4.0mm           | 7       |
| LR    | >0.75mm      | 3.0mm  | 5.0mm           | 6       |

### Field Strength Correction

- **3T** (≥2.5T): `depth_thresh` × 1.10 (higher SNR, raise threshold)
- **Parallel Imaging**: `depth_thresh` × 0.92 (lower SNR, relax threshold)

### Depth Thresholds (STD grade, before correction)

| Level | Threshold | Meaning |
|-------|-----------|---------|
| 1     | 0.22      | First depth threshold |
| 2     | 0.13      | Second depth threshold |
| 3     | 0.07      | Third depth threshold |
| 4     | 0.09      | Fourth depth threshold |

---

## 5. Key Data Structures

### Vertebra Chain Structure

```python
vertebrae_chain = [
    {
        'name': 'L5',  # Auto-assigned name
        'top_meta': {...},      # Superior endplate metadata
        'bot_meta': {...},      # Inferior endplate metadata
        'r_top': float,         # Superior endplate row
        'c_top': float,         # Superior endplate column
        'r_bot': float,         # Inferior endplate row
        'c_bot': float,         # Inferior endplate column
        'row_top': int,         # Superior endplate slice index
        'row_bot': int,         # Inferior endplate slice index
        'top_ix': int,          # Superior endplate index
        'bot_ix': int,          # Inferior endplate index
        'top_c1': (row, col),      # Superior endplate - c1 intersection
        'top_front': (row, col),   # Superior endplate - anterior edge
        'bot_c1': (row, col),      # Inferior endplate - c1 intersection
        'bot_front': (row, col),   # Inferior endplate - anterior edge
    },
    ...
]
```

### Analysis Result Structure

```python
vertebra_analyses = [
    {
        'name': 'L5',
        'area_mm2': 1234.5,
        'angles': {
            'top': 12.3,  # Superior endplate angle
            'bot': 15.6,  # Inferior endplate angle
            'c1': 8.9,    # Cortical line 1 angle
            'fr': 5.4     # Anterior edge angle
        }
    },
    ...
]
```

### Refined Point Format (5-tuple)

```python
(row, col, val, base_col, flag)
# row: Image row coordinate
# col: Refined column coordinate
# val: Signal value at that point
# base_col: Cortical line 2 column (reference)
# flag: 'refined' / 'kept' / 'kept_low'
```

### Descent Detection Point Format (5-tuple)

```python
(row, col, flag, src_tag, base_col)
# row:      Image row coordinate
# col:      Detected anterior edge column (or start column if not found)
# flag:     'confirmed' / 'not_found'
# src_tag:  'main' / 'supp_upper' / 'supp_lower'
# base_col: Cortical line 2 column for this row (interpolated)
```

**Offset Calculation** (unified for both methods):

```python
offset_mm = (base_col - col) * pixel_spacing
```

---

## 6. Visualization Semantics

### Left Image (Fat-Suppressed) - Final Output

| Element | Style | Meaning |
|---------|-------|---------|
| Cortical Line 1 | White solid, lw=1.8 | Anterior vertebral canal wall |
| Dorsal Line | Orange solid, lw=1.8 | Posterior vertebral canal edge |
| Marrow Contour | Yellow contour | Spinal cord/marrow region |
| Cortical Line 2 | Purple dashed, lw=1.2 | Posterior wall (detection baseline) |
| Superior Endplate | Tomato solid, lw=1.8 | `ep_type='upper'` smoothed line |
| Inferior Endplate | Lawngreen solid, lw=1.8 | `ep_type='lower'` smoothed line |
| Anterior Edge | Red solid, lw=1.2 | Dual-modal fusion output |
| Vertebra Labels | White text, fs=7.0 | Area + angles per vertebra |

### Right Image (Water) - Debug View

Includes all elements from left image plus:

- Scan lines (steelblue thin lines)
- Raw endplate candidates (▽/△ markers)
- Endplate × annotations (color-coded by detection phase)
- Ascent refinement points (orange/gray/red scatter)
- Yellow line (original ascent refinement, debug only)
- Descent detection triangles (cyan/yellow markers)
- Start scan lines (deepskyblue, offset=20mm)
- Dense window box (yellow dashed, 4 edges)
- Offset search ROI (cyan dashed box)

---

## 7. Performance Expectations

### Recommended Validation Metrics

- **Canal Segmentation**: Dice coefficient > 0.85
- **Endplate Detection**: Accuracy > 90%
- **Anterior Edge Continuity**: Completeness > 85%

### Typical Processing Time

- Single case (5-6 vertebrae): ~2-5 seconds
- Batch processing (100 cases): ~5-10 minutes

(Depends on image size, number of vertebrae, and hardware)

---

## 8. References & Acknowledgements

This algorithm builds upon research in:

- Vertebral canal segmentation using Otsu thresholding
- Endplate detection via edge detection state machines
- Multi-modal fusion for anterior edge refinement
- Normal-direction scan line following anatomical curvature

We thank all research teams contributing to lumbar MRI analysis methodologies!

---

**Document Version**: V15.0  
**Last Updated**: 2026-03-09  
**Maintained by**: LSMATOOLS Contributors
