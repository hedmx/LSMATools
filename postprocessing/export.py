"""
后处理导出模块
输出：
  1. 各椎体 2D ROI 掩模（多标签）→ {stem}_masks.nii.gz
  2. 几何坐标 + 角度 + 面积 → {stem}_coords.csv
"""
import os
import csv
import numpy as np
import nibabel as nib
from skimage.draw import polygon

# 椎体标签映射（S1=1, L5=2, L4=3, L3=4, L2=5, L1=6, T12=7, ...）
# S1 不一定每次都能识别到，但需纳入标签表避免 fallback 到 255 导致 colormap 失真
_LEVEL_LABEL = {
    'S1': 1, 'L5': 2, 'L4': 3, 'L3': 4, 'L2': 5, 'L1': 6,
    'T12': 7, 'T11': 8, 'T10': 9,
}


def export_vertebra_data(vertebrae_chain, img_shape, output_path, pixel_spacing,
                         orig_affine=None):
    """
    根据椎体链路数据生成 NIfTI 掩模文件和坐标 CSV 文件。

    参数
    ----
    vertebrae_chain : list[dict]
        visualize_results 返回的椎体链路，每个元素包含：
          'name'           : str  ('L5', 'L4', ...)
          'top_ix'         : dict {'c1': (row, col), 'front': (row, col)}
          'bot_ix'         : dict {'c1': (row, col), 'front': (row, col)}
          'area_mm2'       : float
          'angles'         : dict {'top': float, 'bot': float, 'c1': float, 'fr': float}
    img_shape : tuple
        压水图的 (H, W)，用于掩模尺寸。
    output_path : str
        PNG 输出路径，用于派生 stem（替换 _TRACED.png）。
    pixel_spacing : float
        像素间距 (mm)。
    orig_affine : np.ndarray, optional
        原始 NIfTI 图像的 affine 矩阵（4×4）。若传入，NIfTI 掩模将携带正确空间信息。
        若为 None，则使用以 pixel_spacing 为像素间距的单位 affine。
    """
    if not vertebrae_chain:
        return

    stem = output_path.replace('_TRACED.png', '')

    # ── 1. 生成多标签 2D 掩模 → NIfTI ─────────────────────────────────────
    label_map = np.zeros(img_shape, dtype=np.uint8)
    label_names = []

    for vb in vertebrae_chain:
        name = vb.get('name', 'unknown')
        top_ix = vb.get('top_ix', {})
        bot_ix = vb.get('bot_ix', {})
        pt_top_c1    = top_ix.get('c1')
        pt_top_front = top_ix.get('front')
        pt_bot_c1    = bot_ix.get('c1')
        pt_bot_front = bot_ix.get('front')

        if None in (pt_top_c1, pt_top_front, pt_bot_c1, pt_bot_front):
            continue  # 四角点不完整则跳过

        label = _LEVEL_LABEL.get(name, 255)  # 未知椎体用 255

        # ── 尝试用四条真实曲线拼合闭合轮廓 ──────────────────────────────
        top_pts   = vb.get('top_pts')    # {'rows': [...], 'cols': [...]}  上终板线
        bot_pts   = vb.get('bot_pts')    # 下终板线
        c1_pts    = vb.get('c1_pts')     # 皮质线1（后缘）
        front_pts = vb.get('front_pts')  # 前缘红线

        use_curves = (
            top_pts and len(top_pts.get('rows', [])) >= 2 and
            bot_pts and len(bot_pts.get('rows', [])) >= 2 and
            c1_pts  and len(c1_pts.get('rows', [])) >= 2 and
            front_pts and len(front_pts.get('rows', [])) >= 2
        )

        if use_curves:
            # 终板线点序已按 off_mm 升序排列（index 0=c1背侧端，index -1=front腹侧端）
            # 闭合轮廓拼合（顺时针）：
            #   上终板：原始顺序（c1端→front端）
            #   前缘：  row 升序（上→下）
            #   下终板：反转原始顺序（front端→c1端）
            #   皮质线1：row 降序（下→上）
            poly_rows = []
            poly_cols = []

            # 1. 上终板线：直接用原始顺序（c1端→front端，off_mm升序）
            top_r = np.array(top_pts['rows'], dtype=np.float64)
            top_c = np.array(top_pts['cols'], dtype=np.float64)
            poly_rows.extend(top_r.tolist())
            poly_cols.extend(top_c.tolist())

            # 2. 前缘红线：row 从小（上）到大（下）
            fr_r = np.array(front_pts['rows'], dtype=np.float64)
            fr_c = np.array(front_pts['cols'], dtype=np.float64)
            order_fr = np.argsort(fr_r)
            poly_rows.extend(fr_r[order_fr].tolist())
            poly_cols.extend(fr_c[order_fr].tolist())

            # 3. 下终板线：反转原始顺序（front端→c1端）
            bot_r = np.array(bot_pts['rows'], dtype=np.float64)
            bot_c = np.array(bot_pts['cols'], dtype=np.float64)
            poly_rows.extend(bot_r[::-1].tolist())
            poly_cols.extend(bot_c[::-1].tolist())

            # 4. 皮质线1（后缘）：row 从大（下）到小（上）
            c1_r = np.array(c1_pts['rows'], dtype=np.float64)
            c1_c = np.array(c1_pts['cols'], dtype=np.float64)
            order_c1 = np.argsort(c1_r)[::-1]
            poly_rows.extend(c1_r[order_c1].tolist())
            poly_cols.extend(c1_c[order_c1].tolist())

            rr, cc = polygon(poly_rows, poly_cols, shape=img_shape)
        else:
            # 降级：四角点直线连结（仅在曲线数据缺失时使用）
            rows_poly = [pt_top_c1[0], pt_top_front[0], pt_bot_front[0], pt_bot_c1[0]]
            cols_poly = [pt_top_c1[1], pt_top_front[1], pt_bot_front[1], pt_bot_c1[1]]
            rr, cc = polygon(rows_poly, cols_poly, shape=img_shape)

        label_map[rr, cc] = label
        label_names.append(f"{name}={label}")

    if label_names:
        # 构建 affine：若无原始 affine，使用 pixel_spacing 构建简单对角 affine
        if orig_affine is not None:
            affine_2d = orig_affine
        else:
            affine_2d = np.diag([pixel_spacing, pixel_spacing, 1.0, 1.0])

        # NIfTI 需要至少 3D 数组：shape (H, W, 1)
        nii_data = label_map[:, :, np.newaxis]
        nii_img = nib.Nifti1Image(nii_data, affine_2d)
        nii_img.header.set_zooms((pixel_spacing, pixel_spacing, 1.0))

        nii_path = stem + '_masks.nii.gz'
        nib.save(nii_img, nii_path)
        print(f"   [Export] NIfTI 椎体掩模已保存：{os.path.basename(nii_path)} "
              f"({len(label_names)} 节椎体: {', '.join(label_names)})")

    # ── 2. 生成几何坐标 CSV ───────────────────────────────────────────────
    csv_path = stem + '_coords.csv'
    fieldnames = [
        'level',
        'BP_top_row', 'BP_top_col', 'BP_top_row_mm', 'BP_top_col_mm',
        'AP_top_row', 'AP_top_col', 'AP_top_row_mm', 'AP_top_col_mm',
        'AP_bot_row', 'AP_bot_col', 'AP_bot_row_mm', 'AP_bot_col_mm',
        'BP_bot_row', 'BP_bot_col', 'BP_bot_row_mm', 'BP_bot_col_mm',
        'centroid_row', 'centroid_col',
        'area_mm2',
        'angle_sup_deg', 'angle_inf_deg', 'angle_c1_deg', 'angle_ant_deg',
        'ant_height_mm', 'pos_height_mm',
    ]

    rows_out = []
    for vb in vertebrae_chain:
        name = vb.get('name', 'unknown')
        top_ix = vb.get('top_ix', {})
        bot_ix = vb.get('bot_ix', {})
        pt_top_c1    = top_ix.get('c1')
        pt_top_front = top_ix.get('front')
        pt_bot_c1    = bot_ix.get('c1')
        pt_bot_front = bot_ix.get('front')
        angles = vb.get('angles', {})
        area   = vb.get('area_mm2', float('nan'))

        def _r(pt):
            return round(float(pt[0]), 2) if pt else ''

        def _c(pt):
            return round(float(pt[1]), 2) if pt else ''

        def _rmm(pt):
            return round(float(pt[0]) * pixel_spacing, 2) if pt else ''

        def _cmm(pt):
            return round(float(pt[1]) * pixel_spacing, 2) if pt else ''

        # 几何中心（四角点平均）
        if None not in (pt_top_c1, pt_top_front, pt_bot_c1, pt_bot_front):
            cent_r = (pt_top_c1[0] + pt_top_front[0] + pt_bot_c1[0] + pt_bot_front[0]) / 4
            cent_c = (pt_top_c1[1] + pt_top_front[1] + pt_bot_c1[1] + pt_bot_front[1]) / 4
            # 前缘高度 = 前上角到前下角的欧氏距离
            ant_h = (np.sqrt((pt_bot_front[0]-pt_top_front[0])**2 +
                             (pt_bot_front[1]-pt_top_front[1])**2) * pixel_spacing
                     if pt_top_front and pt_bot_front else float('nan'))
            # 后缘高度 = 后上角到后下角的欧氏距离
            pos_h = (np.sqrt((pt_bot_c1[0]-pt_top_c1[0])**2 +
                             (pt_bot_c1[1]-pt_top_c1[1])**2) * pixel_spacing
                     if pt_top_c1 and pt_bot_c1 else float('nan'))
        else:
            cent_r = cent_c = ant_h = pos_h = float('nan')

        rows_out.append({
            'level': name,
            'BP_top_row': _r(pt_top_c1),    'BP_top_col': _c(pt_top_c1),
            'BP_top_row_mm': _rmm(pt_top_c1), 'BP_top_col_mm': _cmm(pt_top_c1),
            'AP_top_row': _r(pt_top_front),  'AP_top_col': _c(pt_top_front),
            'AP_top_row_mm': _rmm(pt_top_front), 'AP_top_col_mm': _cmm(pt_top_front),
            'AP_bot_row': _r(pt_bot_front),  'AP_bot_col': _c(pt_bot_front),
            'AP_bot_row_mm': _rmm(pt_bot_front), 'AP_bot_col_mm': _cmm(pt_bot_front),
            'BP_bot_row': _r(pt_bot_c1),    'BP_bot_col': _c(pt_bot_c1),
            'BP_bot_row_mm': _rmm(pt_bot_c1), 'BP_bot_col_mm': _cmm(pt_bot_c1),
            'centroid_row': round(cent_r, 2) if cent_r == cent_r else '',
            'centroid_col': round(cent_c, 2) if cent_c == cent_c else '',
            'area_mm2': round(area, 2) if area == area else '',
            'angle_sup_deg': round(angles.get('top', float('nan')), 2),
            'angle_inf_deg': round(angles.get('bot', float('nan')), 2),
            'angle_c1_deg':  round(angles.get('c1',  float('nan')), 2),
            'angle_ant_deg': round(angles.get('fr',  float('nan')), 2),
            'ant_height_mm': round(ant_h, 2) if ant_h == ant_h else '',
            'pos_height_mm': round(pos_h, 2) if pos_h == pos_h else '',
        })

    if rows_out:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"   [Export] 几何坐标已保存：{os.path.basename(csv_path)} "
              f"({len(rows_out)} 行)")
