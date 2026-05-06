"""
掩模导出模块

export_masks: 将椎体链路输出为多标签 NIfTI 掩模文件。
椎体轮廓优先用真实曲线（上终板线+前缘线+下终板线+皮质线1）拼合闭合多边形，
四角点不完整或曲线缺失时降级为四角点直连。
S2 椎体不纳入掩模。
椎管掩模保留 label=15。
"""

import io
import os
import zipfile

import numpy as np
import nibabel as nib
from skimage.draw import polygon
from skimage import measure

try:
    import roifile as _roifile
    _HAS_ROIFILE = True
except ImportError:
    _HAS_ROIFILE = False

from config.params import LEVEL_LABEL as _LEVEL_LABEL, CANAL_LABEL as _CANAL_LABEL


def _clip_line_by_row(pts, r_min, r_max):
    """从点列表 [(r,c),...] 中取行范围在 [r_min, r_max] 内的点，按 row 升序返回。"""
    clipped = [(r, c) for r, c in pts if r_min <= r <= r_max]
    return sorted(clipped, key=lambda p: p[0])


def _interp_line_by_row(pts, r_min, r_max):
    """
    将稀疏点列表 [(r,c),...] 插值为逐行（整数行）一个点的密集序列。
    用 numpy 线性插值，覆盖 [r_min, r_max] 全行范围。
    返回 [(r, c),...] 按 row 升序，每 row 一个点。
    """
    if len(pts) < 2:
        return pts
    pts_sorted = sorted(pts, key=lambda p: p[0])
    rs = np.array([p[0] for p in pts_sorted], dtype=np.float64)
    cs = np.array([p[1] for p in pts_sorted], dtype=np.float64)
    # 插值到 [r_min, r_max] 整数行
    rows_dense = np.arange(int(r_min), int(r_max) + 1, dtype=np.float64)
    cols_dense = np.interp(rows_dense, rs, cs)
    return list(zip(rows_dense.tolist(), cols_dense.tolist()))


def _build_curve_polygon(entry, ant_line, c1_rows, c1_cols):
    """
    用四条真实曲线拼合闭合轮廓（顺时针）：
      1. 上终板线（sup_ext，col 升序：背侧→腹侧）
      2. 前缘线段（ant_line 行范围裁剪，row 升序：上→下）
      3. 下终板线（inf_ext 反转，col 降序：腹侧→背侧）
      4. 皮质线1段（c1 行范围裁剪，row 降序：下→上）

    任一曲线缺失则返回 None，调用方降级为四角点。
    """
    q        = entry.get('quad', {})
    sup_ant  = q.get('sup_ant')
    sup_post = q.get('sup_post')
    inf_ant  = q.get('inf_ant')
    inf_post = q.get('inf_post')

    if any(p is None for p in (sup_ant, sup_post, inf_ant, inf_post)):
        print(f"   [mask_debug] {entry.get('name','?')} 四角点缺失:"
              f" sup_ant={sup_ant is not None} sup_post={sup_post is not None}"
              f" inf_ant={inf_ant is not None} inf_post={inf_post is not None}")
        return None

    sup_ext = entry.get('sup_ext', [])
    inf_ext = entry.get('inf_ext', [])
    if len(sup_ext) < 2 or len(inf_ext) < 2:
        print(f"   [mask_debug] {entry.get('name','?')} 终板延伸线不足:"
              f" sup_ext={len(sup_ext)} inf_ext={len(inf_ext)}")
        return None
    if not ant_line or len(ant_line) < 2:
        print(f"   [mask_debug] {entry.get('name','?')} ant_line不足:"
              f" len={len(ant_line) if ant_line else 0}")
        return None
    if c1_rows is None or c1_cols is None or len(c1_rows) < 2:
        print(f"   [mask_debug] {entry.get('name','?')} c1_rows不足:"
              f" c1_rows={'OK' if c1_rows is not None else 'None'}"
              f" len={len(c1_rows) if c1_rows is not None else 0}")
        return None

    # 确定椎体行范围（取四角点 row 的全局 min/max，不假设 sup row < inf row）
    all_rows = [sup_ant[0], sup_post[0], inf_ant[0], inf_post[0]]
    r_top = min(all_rows)
    r_bot = max(all_rows)

    # ── 辅助：按 col 插值终板线，首尾强制为角点坐标 ──
    def _interp_ep_by_col(ep_pts, corner_l, corner_r):
        """
        将终板线点集按 col 插值为逐列一个点，覆盖 [corner_l[1], corner_r[1]]。
        首点强制为 corner_l，末点强制为 corner_r，保证与相邻段严格共点。
        返回按 col 升序的 [(r,c),...] 列表。
        """
        col_min = min(corner_l[1], corner_r[1])
        col_max = max(corner_l[1], corner_r[1])
        raw = [(r, c) for r, c in ep_pts if col_min - 1 <= c <= col_max + 1]
        if len(raw) < 2:
            raw = list(ep_pts)
        raw_csort = sorted(raw, key=lambda p: p[1])
        cs = np.array([p[1] for p in raw_csort], dtype=np.float64)
        rs = np.array([p[0] for p in raw_csort], dtype=np.float64)
        col_dense = np.arange(int(col_min), int(col_max) + 1, dtype=np.float64)
        row_dense = np.interp(col_dense, cs, rs)
        seg = list(zip(row_dense.tolist(), col_dense.tolist()))
        # 强制首尾为角点（保证拼合处严格共点，消除空隙）
        if corner_l[1] <= corner_r[1]:
            seg[0]  = (float(corner_l[0]), float(corner_l[1]))
            seg[-1] = (float(corner_r[0]), float(corner_r[1]))
        else:
            seg[0]  = (float(corner_r[0]), float(corner_r[1]))
            seg[-1] = (float(corner_l[0]), float(corner_l[1]))
        return seg  # col 升序

    # ── 辅助：按 row 插值纵向线，首尾强制为角点坐标 ──
    def _interp_vert_by_row(pts, corner_top, corner_bot):
        """
        将纵向线点集按 row 插值为逐行一个点，覆盖 [corner_top[0], corner_bot[0]]。
        首点强制为 corner_top，末点强制为 corner_bot。
        返回按 row 升序的 [(r,c),...] 列表。
        """
        r_min = min(corner_top[0], corner_bot[0])
        r_max = max(corner_top[0], corner_bot[0])
        raw = _clip_line_by_row(pts, r_min, r_max)
        if len(raw) < 2:
            raw = sorted(pts, key=lambda p: p[0])
        seg = _interp_line_by_row(raw, r_min, r_max)
        # 强制首尾为角点
        seg[0]  = (float(corner_top[0]), float(corner_top[1]))
        seg[-1] = (float(corner_bot[0]), float(corner_bot[1]))
        return seg  # row 升序

    # 1. 上终板线：col 插值，col 降序（sup_post 背侧→ sup_ant 腹侧）
    sup_seg_asc = _interp_ep_by_col(sup_ext, sup_post, sup_ant)
    # sup_seg_asc 按 col 升序（背侧col大→腹侧col小），需要降序
    # 但 col_min=sup_ant[1](腹侧), col_max=sup_post[1](背侧)，升序即腹→背，需反转为背→腹
    sup_seg = list(reversed(sup_seg_asc))

    # 2. 前缘线段：row 插值，row 升序（sup_ant 顶→ inf_ant 底）
    c1_pts_all = list(zip(c1_rows, c1_cols))
    front_raw = _clip_line_by_row(ant_line, r_top, r_bot)
    if len(front_raw) < 2:
        ant_rows = [p[0] for p in ant_line] if ant_line else []
        print(f"   [mask_debug] {entry.get('name','?')} 前缘裁剪为空:"
              f" r_top={r_top:.1f} r_bot={r_bot:.1f}"
              f" ant_line行范围=[{min(ant_rows):.1f},{max(ant_rows):.1f}]"
              if ant_rows else f"   [mask_debug] {entry.get('name','?')} 前缘为空")
        return None
    front_seg = _interp_vert_by_row(front_raw, sup_ant, inf_ant)

    # 3. 下终板线：col 插值，col 升序（inf_ant 腹侧→ inf_post 背侧）
    inf_seg = _interp_ep_by_col(inf_ext, inf_ant, inf_post)

    # 4. 皮质线1段：row 插值，row 降序（inf_post 底→ sup_post 顶），MAD 过滤
    c1_raw = _clip_line_by_row(c1_pts_all, r_top, r_bot)
    if len(c1_raw) < 2:
        c1_all_rows = [r for r, c in c1_pts_all]
        print(f"   [mask_debug] {entry.get('name','?')} 皮质线1裁剪为空:"
              f" r_top={r_top:.1f} r_bot={r_bot:.1f}"
              f" c1行范围=[{min(c1_all_rows):.1f},{max(c1_all_rows):.1f}]"
              if c1_all_rows else f"   [mask_debug] {entry.get('name','?')} 皮质线1为空")
        return None
    if len(c1_raw) >= 4:
        _c1_cs = np.array([p[1] for p in c1_raw], dtype=np.float64)
        _med = float(np.median(_c1_cs))
        _mad = float(np.median(np.abs(_c1_cs - _med))) + 1e-9
        _c1_filtered = [(r, c) for r, c in c1_raw if abs(c - _med) < 3.0 * _mad]
        if len(_c1_filtered) >= 2:
            c1_raw = _c1_filtered
    c1_seg_asc = _interp_vert_by_row(c1_raw, sup_post, inf_post)
    c1_seg = list(reversed(c1_seg_asc))  # row 降序（inf_post → sup_post）

    # 拼合（顺时针：sup_post → sup_ant → inf_ant → inf_post → sup_post）
    # 每段首尾已强制为角点，相邻段共享同一角点，轮廓严格闭合无空隙
    poly_rows = []
    poly_cols = []
    for r, c in sup_seg:        # sup_post → sup_ant（背侧→腹侧，上终板）
        poly_rows.append(r); poly_cols.append(c)
    for r, c in front_seg[1:]:  # sup_ant → inf_ant（跳过已有的 sup_ant 首点）
        poly_rows.append(r); poly_cols.append(c)
    for r, c in inf_seg[1:]:    # inf_ant → inf_post（跳过已有的 inf_ant 首点）
        poly_rows.append(r); poly_cols.append(c)
    for r, c in c1_seg[1:]:     # inf_post → sup_post（跳过已有的 inf_post 首点）
        poly_rows.append(r); poly_cols.append(c)

    return poly_rows, poly_cols


def _build_quad_polygon(entry):
    """降级方案：四角点直线连结。"""
    q        = entry.get('quad', {})
    sup_ant  = q.get('sup_ant')
    sup_post = q.get('sup_post')
    inf_ant  = q.get('inf_ant')
    inf_post = q.get('inf_post')
    if any(p is None for p in (sup_ant, sup_post, inf_ant, inf_post)):
        return None
    # 顺时针：sup_post → sup_ant → inf_ant → inf_post
    rows_poly = [sup_post[0], sup_ant[0], inf_ant[0], inf_post[0]]
    cols_poly = [sup_post[1], sup_ant[1], inf_ant[1], inf_post[1]]
    return rows_poly, cols_poly


def export_masks(vertebrae_chain, img_shape, output_dir, stem,
                 pixel_spacing, orig_affine=None, cord_mask_cut=None,
                 ant_line=None, c1_rows=None, c1_cols=None):
    """
    导出分割掩模 NIfTI 文件。

    椎体轮廓优先用真实曲线拼合（需提供 ant_line/c1_rows/c1_cols），
    缺失时降级为四角点直连。
    c1_rows/c1_cols 实际传入皮质线2-2坐标作为背侧边（参数名保持兼容）。

    参数：
        vertebrae_chain  - 椎体链路列表（build_vertebra_chain 返回值）
        img_shape        - (H, W) 图像尺寸
        output_dir       - 输出目录
        stem             - 文件名前缀
        pixel_spacing    - mm/pixel
        orig_affine      - 原始 NIfTI affine（可为 None）
        cord_mask_cut    - 椎管掩模布尔数组（可为 None）
        ant_line         - 全局前缘线点列表 [(r,c), ...]
        c1_rows/c1_cols  - 背侧线坐标（传入皮质线2-2）

    输出：{stem}_seg.nii.gz
    """
    label_map   = np.zeros(img_shape, dtype=np.uint8)
    label_names = []

    for entry in (vertebrae_chain or []):
        name = entry.get('name', 'Inc')
        if name == 'S2':
            continue
        label = _LEVEL_LABEL.get(name)
        if label is None:
            continue

        # 优先用真实曲线轮廓（背侧边使用皮质线2-2）
        result = _build_curve_polygon(entry, ant_line, c1_rows, c1_cols)
        used_curve = result is not None
        if result is None:
            result = _build_quad_polygon(entry)
        if result is None:
            continue

        rows_poly, cols_poly = result
        rr, cc = polygon(rows_poly, cols_poly, shape=img_shape)
        label_map[rr, cc] = label
        tag = 'curve' if used_curve else 'quad'
        label_names.append(f"{name}={label}({tag})")

    # 合并椎管掩模
    if cord_mask_cut is not None and np.any(cord_mask_cut):
        label_map[cord_mask_cut] = _CANAL_LABEL
        print(f"   [mask] 椎管掩模合并 label={_CANAL_LABEL}, area={int(np.sum(cord_mask_cut))}px")

    if not label_names and (cord_mask_cut is None or not np.any(cord_mask_cut)):
        print("   [mask] 无有效椎体或椎管掩模，跳过输出")
        return

    # 构建 affine
    if orig_affine is not None:
        affine_2d = orig_affine
    else:
        affine_2d = np.diag([pixel_spacing, pixel_spacing, 1.0, 1.0])

    nii_data = label_map[:, :, np.newaxis]
    nii_img  = nib.Nifti1Image(nii_data, affine_2d)
    nii_img.header.set_zooms((pixel_spacing, pixel_spacing, 1.0))

    nii_path = os.path.join(output_dir, f"{stem}_seg.nii.gz")
    nib.save(nii_img, nii_path)
    info_str = ', '.join(label_names) if label_names else '仅椎管'
    print(f"   [mask] 已保存: {os.path.basename(nii_path)}  ({info_str})")

    # 同步生成 ROI ZIP
    export_roi_zip(label_map, output_dir, stem)


# 椎体/椎管标签 → 名称映射（含 CANAL）
_LABEL_TO_NAME = {v: k for k, v in _LEVEL_LABEL.items()}
_LABEL_TO_NAME[15] = 'CANAL'


def export_roi_zip(label_map, output_dir, stem):
    """
    将多标签掩模（label_map 2D uint8）中每个标签的最大轮廓导出为 Fiji ROI ZIP。

    参数：
        label_map  - 2D np.ndarray (H, W)，每像素为标签值（0=背景）
        output_dir - 输出目录
        stem       - 文件名前缀，输出 {stem}_roi.zip
    """
    if not _HAS_ROIFILE:
        print("   [roi] 跳过ROI导出：roifile 未安装（pip install roifile）")
        return

    labels = np.unique(label_map)
    labels = labels[labels > 0]
    if len(labels) == 0:
        print("   [roi] 无有效标签，跳过ROI导出")
        return

    roi_entries = []  # [(name, roi_bytes), ...]

    for label in sorted(labels):
        single_mask = (label_map == label).astype(np.uint8)
        contours = measure.find_contours(single_mask, 0.5)
        if not contours:
            continue
        # 取最大轮廓
        largest = max(contours, key=len)
        # find_contours 返回 (row, col)，roifile 需要 (x=col, y=row)
        xy_coords = largest[:, [1, 0]].astype(np.float32)
        roi = _roifile.ImagejRoi.frompoints(xy_coords)
        roi_name = _LABEL_TO_NAME.get(int(label), f'Label_{label}')
        roi.name = roi_name
        roi_entries.append((roi_name, roi.tobytes()))

    if not roi_entries:
        print("   [roi] 未提取到有效轮廓，跳过ROI导出")
        return

    zip_path = os.path.join(output_dir, f"{stem}_roi.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for roi_name, roi_bytes in roi_entries:
            zf.writestr(f'{roi_name}.roi', roi_bytes)

    names_str = ', '.join(n for n, _ in roi_entries)
    print(f"   [roi] 已保存: {os.path.basename(zip_path)}  ({names_str})")

