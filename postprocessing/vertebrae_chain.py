"""
椎体链路识别与编号分配模块
包含 assign_vertebra_labels（从下往上分配 L5/L4/.../T12 编号）
"""
import numpy as np


# ============ 椎体编号分配 ============
def assign_vertebra_labels(endplates_f, pixel_spacing):
    """
    根据已排好序的终板对（upper+lower）从下往上分配椎体编号。
    编号顺序：L5 → L4 → L3 → L2 → L1 → T12 → T11 ...
    不足5节腰椎时，根据椎体中心间距往上推断编号（圆圈标记）。
    推断点的列坐标通过最头侧真实椎体的两终板连线斜率外推得到。

    返回：
      list of dict {
        'label':      str,   e.g. 'L5'
        'center_row': int,   椎体中心行号
        'center_col': int,   椎体中心列号
        'inferred':   bool,  True 表示推断编号（圆圈标记）
      }
    """
    LUMBAR_LABELS   = ['L5', 'L4', 'L3', 'L2', 'L1']
    THORACIC_LABELS = ['T12', 'T11', 'T10', 'T9', 'T8']

    eps = sorted(endplates_f, key=lambda x: x[0])  # 按行号升序（从上到下）
    if len(eps) < 2:
        return []

    # 椎体 = superior(绿色上终板, 上方) → inferior(红色下终板, 下方) 之间
    # 同时保存两个终板点的完整坐标，用于计算斜率外推
    vertebra_info = []  # (center_row, center_col, superior_row, superior_col, inferior_row, inferior_col)
    for i in range(len(eps) - 1):
        a, b = eps[i], eps[i + 1]
        if a[3] == 'superior' and b[3] == 'inferior':
            vertebra_info.append((
                (a[0] + b[0]) // 2,   # center_row
                (a[1] + b[1]) // 2,   # center_col
                a[0], a[1],            # lower_row, lower_col
                b[0], b[1],            # upper_row, upper_col
            ))

    if not vertebra_info:
        return []

    n_real = len(vertebra_info)
    vertebra_centers = [(v[0], v[1]) for v in vertebra_info]

    # 相邻椎体中心行间距均値（用于推断）
    if n_real >= 2:
        avg_spacing_px = float(np.mean(
            [vertebra_centers[i+1][0] - vertebra_centers[i][0] for i in range(n_real - 1)]
        ))
    else:
        avg_spacing_px = max(1, int(round(30.0 / pixel_spacing)))

    # 最尾侧（行号最大）的椎体 = L5，往头侧依次 L4/L3/L2/L1/T12...
    label_pool = LUMBAR_LABELS + THORACIC_LABELS
    labels_real = [None] * n_real
    for i in range(n_real):
        idx_from_bottom = n_real - 1 - i
        labels_real[idx_from_bottom] = label_pool[i] if i < len(label_pool) else f'?{i}'

    vertebrae = [
        {'label': labels_real[i], 'center_row': cr, 'center_col': cc, 'inferred': False}
        for i, (cr, cc) in enumerate(vertebra_centers)
    ]

    # 推断头侧椎体：列坐标通过最头侧真实椎体的两终板连线斜率外推
    # 斜率 = (upper_col - lower_col) / (upper_row - lower_row)，即每行列变化量
    full_seq = ['T8', 'T9', 'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']
    top_label = labels_real[0]
    inferred_above = []
    if top_label in full_seq:
        top_idx = full_seq.index(top_label)

        # 取最头侧真实椎体的两终板坐标计算斜率
        top_vi        = vertebra_info[0]   # (cr, cc, lr, lc, ur, uc)
        top_lower_row = top_vi[2]
        top_lower_col = top_vi[3]
        top_upper_row = top_vi[4]
        top_upper_col = top_vi[5]
        row_span = top_upper_row - top_lower_row
        if row_span != 0:
            col_slope = (top_upper_col - top_lower_col) / row_span  # 列/行
        else:
            col_slope = 0.0

        # 基准点：最头侧真实椎体中心
        base_row = vertebra_centers[0][0]
        base_col = vertebra_centers[0][1]

        for k in range(1, top_idx + 1):
            inf_label = full_seq[top_idx - k]
            inf_row   = int(round(base_row - k * avg_spacing_px))
            if inf_row < 0:
                break
            # 列坐标：从基准中心点沿斤率外推
            inf_col   = int(round(base_col + col_slope * (inf_row - base_row)))
            inferred_above.append({
                'label':      inf_label,
                'center_row': inf_row,
                'center_col': inf_col,
                'inferred':   True,
            })

    # inferred_above 是从最近头侧往更头侧排列的，需反转得到从上到下顺序
    return inferred_above[::-1] + vertebrae


# ============ 可视化 ============
