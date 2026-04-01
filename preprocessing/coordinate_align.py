"""
坐标对齐模块
将压脂图扫描线坐标投影到压水图，做坐标对齐校验与映射
"""
import numpy as np


def align_scan_lines_to_f(scan_lines, w_meta: dict, f_meta: dict, f_img_2d):
    """
    将压脂图扫描线坐标投影到压水图，做坐标对齐校验与映射。

    参数：
        scan_lines  - 压脂图扫描线列表 [(offset_mm, cols, rows), ...]
        w_meta      - 压脂图 metadata dict
        f_meta      - 压水图 metadata dict
        f_img_2d    - 压水图 2D numpy 数组

    返回：
        调整后的 scan_lines_f（若完全对齐则原样返回）
    """
    def get_spacing(meta):
        acq = meta.get('acquisition_params', {})
        sp  = acq.get('pixel_spacing_mm', [0.9375, 0.9375])
        return float(sp[0]) if isinstance(sp, list) else float(sp)

    def get_origin(meta):
        acq = meta.get('acquisition_params', {})
        pos = acq.get('imagepositionpatient', [0, 0, 0])
        return [float(v) for v in pos]

    ps_w     = get_spacing(w_meta)
    ps_f     = get_spacing(f_meta)
    origin_w = get_origin(w_meta)
    origin_f = get_origin(f_meta)

    print(f"\n📊 坐标对齐校验:")
    print(f"   压脂: 像素间距={ps_w:.4f}mm, 原点={[round(v, 2) for v in origin_w]}")
    print(f"   压水: 像素间距={ps_f:.4f}mm, 原点={[round(v, 2) for v in origin_f]}")

    scale      = ps_w / ps_f if ps_f > 0 else 1.0
    col_offset = (origin_w[0] - origin_f[0]) / ps_f if ps_f > 0 else 0.0
    row_offset = (origin_w[1] - origin_f[1]) / ps_f if ps_f > 0 else 0.0

    print(f"   缩放比例: {scale:.4f}, 列偏移: {col_offset:.1f}px, 行偏移: {row_offset:.1f}px")

    if abs(scale - 1.0) < 0.01 and abs(col_offset) < 1 and abs(row_offset) < 1:
        print("   ✅ 坐标完全对齐，直接复用压脂图坐标")
        return scan_lines

    h_f, w_f = f_img_2d.shape
    scan_lines_f = []
    for off_mm, cols, rows in scan_lines:
        new_cols = np.clip(np.round(cols * scale + col_offset).astype(np.int32), 0, w_f - 1)
        new_rows = np.clip(np.round(np.array(rows) * scale + row_offset).astype(np.int32), 0, h_f - 1)
        scan_lines_f.append((off_mm, new_cols, new_rows))
    print("   ⚠️ 坐标已映射")
    return scan_lines_f
