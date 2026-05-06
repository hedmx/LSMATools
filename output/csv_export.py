"""
CSV 椎体几何形态导出模块

export_csv: 将椎体链路数据输出为几何形态 CSV 文件。
字段与 CR export.py 完全一致。
S2 行仍写入 CSV（仅掩模不保留）。
"""

import os
import csv
import math
import numpy as np
from skimage.draw import polygon


# CSV 字段名（与 CR export.py 一致）
_FIELDNAMES = [
    'slice_index',
    'level',
    'BP_top_row', 'BP_top_col', 'BP_top_row_mm', 'BP_top_col_mm',
    'AP_top_row', 'AP_top_col', 'AP_top_row_mm', 'AP_top_col_mm',
    'AP_bot_row', 'AP_bot_col', 'AP_bot_row_mm', 'AP_bot_col_mm',
    'BP_bot_row', 'BP_bot_col', 'BP_bot_row_mm', 'BP_bot_col_mm',
    'centroid_row', 'centroid_col',
    'area_px_count',
    'area_geo_mm2',
    'angle_sup_deg', 'angle_inf_deg', 'angle_c1_deg', 'angle_ant_deg',
    'ant_height_mm', 'pos_height_mm',
    'canal_area_px', 'canal_area_mm2',
    'canal_ap_min_mm', 'canal_ap_min_row',
    'canal_ap_max_mm', 'canal_ap_center_mm',
    'canal_centroid_row', 'canal_centroid_col',
]


def _f(v, ndigits=2):
    """格式化数值，NaN 返回空字符串。"""
    if v is None:
        return ''
    try:
        if math.isnan(v):
            return ''
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return ''


def export_csv(vertebrae_chain, img_shape, output_dir, stem,
               pixel_spacing, cord_mask_cut=None, best_slice_idx=None):
    """
    导出 CSV 几何形态文件。

    参数：
        vertebrae_chain  - 椎体链路列表（build_vertebra_chain 返回值）
        img_shape        - (H, W)
        output_dir       - 输出目录
        stem             - 文件名前缀
        pixel_spacing    - mm/pixel
        cord_mask_cut    - 椎管掩模布尔数组（可为 None）
        best_slice_idx   - 最佳切片索引（写入 slice_index 列）

    输出：{stem}_geom.csv
    """
    rows_out = []
    vb_px_counts = {}

    # 先计算各椎体像素计数（用四角点 polygon）
    if img_shape is not None:
        for entry in (vertebrae_chain or []):
            name = entry.get('name', 'Inc')
            q = entry.get('quad', {})
            sup_ant  = q.get('sup_ant')
            sup_post = q.get('sup_post')
            inf_ant  = q.get('inf_ant')
            inf_post = q.get('inf_post')
            if all(p is not None for p in (sup_ant, sup_post, inf_ant, inf_post)):
                rows_poly = [sup_post[0], sup_ant[0], inf_ant[0], inf_post[0]]
                cols_poly = [sup_post[1], sup_ant[1], inf_ant[1], inf_post[1]]
                rr, cc = polygon(rows_poly, cols_poly, shape=img_shape)
                vb_px_counts[name] = len(rr)

    for entry in (vertebrae_chain or []):
        name = entry.get('name', 'Inc')
        q = entry.get('quad', {})
        sup_ant  = q.get('sup_ant')   # AP_top（前缘上交点）
        sup_post = q.get('sup_post')  # BP_top（后缘上交点）
        inf_ant  = q.get('inf_ant')   # AP_bot（前缘下交点）
        inf_post = q.get('inf_post')  # BP_bot（后缘下交点）

        def _r(pt):
            return _f(pt[0]) if pt else ''
        def _c(pt):
            return _f(pt[1]) if pt else ''
        def _rmm(pt):
            return _f(pt[0] * pixel_spacing) if pt else ''
        def _cmm(pt):
            return _f(pt[1] * pixel_spacing) if pt else ''

        row_data = {
            'slice_index': best_slice_idx if best_slice_idx is not None else '',
            'level': name,
            'BP_top_row': _r(sup_post), 'BP_top_col': _c(sup_post),
            'BP_top_row_mm': _rmm(sup_post), 'BP_top_col_mm': _cmm(sup_post),
            'AP_top_row': _r(sup_ant),  'AP_top_col': _c(sup_ant),
            'AP_top_row_mm': _rmm(sup_ant), 'AP_top_col_mm': _cmm(sup_ant),
            'AP_bot_row': _r(inf_ant),  'AP_bot_col': _c(inf_ant),
            'AP_bot_row_mm': _rmm(inf_ant), 'AP_bot_col_mm': _cmm(inf_ant),
            'BP_bot_row': _r(inf_post), 'BP_bot_col': _c(inf_post),
            'BP_bot_row_mm': _rmm(inf_post), 'BP_bot_col_mm': _cmm(inf_post),
        }

        if all(p is not None for p in (sup_ant, sup_post, inf_ant, inf_post)):
            # 几何中心
            cent_r = (sup_post[0] + sup_ant[0] + inf_ant[0] + inf_post[0]) / 4
            cent_c = (sup_post[1] + sup_ant[1] + inf_ant[1] + inf_post[1]) / 4

            # 前缘/后缘高度
            ant_h = abs(inf_ant[0] - sup_ant[0]) * pixel_spacing
            pos_h = abs(inf_post[0] - sup_post[0]) * pixel_spacing

            # 上终板角度（AP_top_row - BP_top_row，前缘 vs 后缘）
            delta_row_sup = sup_ant[0] - sup_post[0]
            delta_col_sup = sup_ant[1] - sup_post[1]
            angle_sup = np.degrees(np.arctan2(delta_row_sup, abs(delta_col_sup) + 1e-9))
            if delta_col_sup < 0:
                angle_sup = -angle_sup

            # 下终板角度
            delta_row_inf = inf_ant[0] - inf_post[0]
            delta_col_inf = inf_ant[1] - inf_post[1]
            angle_inf = np.degrees(np.arctan2(delta_row_inf, abs(delta_col_inf) + 1e-9))
            if delta_col_inf < 0:
                angle_inf = -angle_inf

            # 后缘角度
            delta_row_c1 = inf_post[0] - sup_post[0]
            delta_col_c1 = inf_post[1] - sup_post[1]
            angle_c1 = np.degrees(np.arctan2(delta_col_c1, abs(delta_row_c1) + 1e-9))
            if delta_row_c1 < 0:
                angle_c1 = -angle_c1

            # 前缘角度
            delta_row_ant = inf_ant[0] - sup_ant[0]
            delta_col_ant = inf_ant[1] - sup_ant[1]
            angle_ant = np.degrees(np.arctan2(delta_col_ant, abs(delta_row_ant) + 1e-9))
            if delta_row_ant < 0:
                angle_ant = -angle_ant

            # 几何面积（斯绍斯公式）
            r_vals = [sup_post[0], sup_ant[0], inf_ant[0], inf_post[0]]
            c_vals = [sup_post[1], sup_ant[1], inf_ant[1], inf_post[1]]
            shoelace = 0.0
            n = 4
            for i in range(n):
                j = (i + 1) % n
                shoelace += r_vals[i] * c_vals[j] - r_vals[j] * c_vals[i]
            area_geo = abs(shoelace) / 2.0 * (pixel_spacing ** 2)

            row_data.update({
                'centroid_row': _f(cent_r),
                'centroid_col': _f(cent_c),
                'area_px_count': vb_px_counts.get(name, ''),
                'area_geo_mm2': _f(area_geo),
                'angle_sup_deg': _f(angle_sup),
                'angle_inf_deg': _f(angle_inf),
                'angle_c1_deg': _f(angle_c1),
                'angle_ant_deg': _f(angle_ant),
                'ant_height_mm': _f(ant_h),
                'pos_height_mm': _f(pos_h),
            })
        else:
            row_data.update({
                'centroid_row': '', 'centroid_col': '',
                'area_px_count': '', 'area_geo_mm2': '',
                'angle_sup_deg': '', 'angle_inf_deg': '',
                'angle_c1_deg': '', 'angle_ant_deg': '',
                'ant_height_mm': '', 'pos_height_mm': '',
            })

        rows_out.append(row_data)

    # 椎管行
    if cord_mask_cut is not None and np.any(cord_mask_cut):
        canal_area_px  = int(np.sum(cord_mask_cut))
        canal_area_mm2 = round(canal_area_px * (pixel_spacing ** 2), 2)

        rows_idx, cols_idx = np.where(cord_mask_cut)
        canal_centroid_row = round(float(np.mean(rows_idx)), 2)
        canal_centroid_col = round(float(np.mean(cols_idx)), 2)

        # AP 径（L5 下界以上）
        row_limit = None
        for entry in (vertebrae_chain or []):
            if entry.get('name') == 'L5':
                inf_post = entry.get('quad', {}).get('inf_post')
                if inf_post is not None:
                    row_limit = int(inf_post[0])
                break
        if row_limit is None:
            all_rows_u = np.unique(rows_idx)
            cutoff = int(len(all_rows_u) * 0.85)
            row_limit = int(all_rows_u[cutoff]) if cutoff < len(all_rows_u) else int(all_rows_u[-1])

        ap_per_row = {}
        for r in np.unique(rows_idx):
            if r > row_limit:
                continue
            cols_in_row = cols_idx[rows_idx == r]
            ap_per_row[r] = int(cols_in_row.max() - cols_in_row.min() + 1)

        if ap_per_row:
            min_ap_row = int(min(ap_per_row, key=ap_per_row.get))
            canal_ap_min_mm = round(ap_per_row[min_ap_row] * pixel_spacing, 2)
            max_ap_row = int(max(ap_per_row, key=ap_per_row.get))
            canal_ap_max_mm = round(ap_per_row[max_ap_row] * pixel_spacing, 2)
        else:
            min_ap_row = max_ap_row = 0
            canal_ap_min_mm = canal_ap_max_mm = 0.0

        centroid_row_int = int(round(canal_centroid_row))
        nearest = min(ap_per_row.keys(), key=lambda r: abs(r - centroid_row_int)) if ap_per_row else None
        canal_ap_center_mm = round(ap_per_row[nearest] * pixel_spacing, 2) if nearest is not None else 0.0

        rows_out.append({
            'slice_index': best_slice_idx if best_slice_idx is not None else '',
            'level': 'CANAL',
            'canal_area_px': canal_area_px,
            'canal_area_mm2': canal_area_mm2,
            'canal_ap_min_mm': canal_ap_min_mm,
            'canal_ap_min_row': min_ap_row,
            'canal_ap_max_mm': canal_ap_max_mm,
            'canal_ap_center_mm': canal_ap_center_mm,
            'canal_centroid_row': canal_centroid_row,
            'canal_centroid_col': canal_centroid_col,
        })

    if not rows_out:
        print("   [csv] 无数据，跳过 CSV 输出")
        return

    csv_path = os.path.join(output_dir, f"{stem}_geom.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"   [csv] 已保存: {os.path.basename(csv_path)}  ({len(rows_out)} 行)")
