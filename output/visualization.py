"""
可视化输出模块 – W图/IN图分割对比 PNG

visualize_wifs: 双图对比可视化（左：W图+链路，右：IN图+掩模叠加）。
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.draw import polygon

from config.params import LEVEL_LABEL as _LEVEL_LABEL, CANAL_LABEL as _CANAL_LABEL


# 掩模标签颜色映射（与 _LEVEL_LABEL 对应）
_LABEL_COLORS = {
    1:  (1.0,  0.2,  0.2,  0.5),   # S1 红
    2:  (1.0,  0.6,  0.0,  0.5),   # L5 橙
    3:  (1.0,  1.0,  0.0,  0.5),   # L4 黄
    4:  (0.2,  1.0,  0.2,  0.5),   # L3 绿
    5:  (0.0,  0.8,  1.0,  0.5),   # L2 青
    6:  (0.2,  0.4,  1.0,  0.5),   # L1 蓝
    7:  (0.8,  0.0,  1.0,  0.5),   # T12 紫
    8:  (1.0,  0.4,  0.8,  0.5),   # T11 粉
    9:  (0.6,  0.6,  0.6,  0.5),   # T10 灰
    10: (0.9,  0.8,  0.6,  0.5),   # T9 米
    15: (0.0,  1.0,  1.0,  0.3),   # 椎管 浅青
}



def _build_mask_overlay(vertebrae_chain, img_shape, cord_mask_cut,
                        ant_line=None, c1_rows=None, c1_cols=None,
                        c2_rows=None, c2_cols=None):
    """构建 RGBA 叠加图（H×W×4），供右图使用。
    优先用完整曲线轮廓（上终板线+前缘线+下终板线+皮质线2-2），缺失时降级为四角点。
    """
    from .mask_export import _build_curve_polygon, _build_quad_polygon

    overlay = np.zeros((*img_shape, 4), dtype=np.float32)

    # 使用皮质线2-2作为轮廓背侧边，无则降级用皮质线1
    back_rows = c2_rows if c2_rows is not None else c1_rows
    back_cols = c2_cols if c2_cols is not None else c1_cols

    # 椎管底层（先画）
    if cord_mask_cut is not None and np.any(cord_mask_cut):
        color = _LABEL_COLORS.get(15, (0, 1, 1, 0.3))
        for ch in range(4):
            overlay[..., ch][cord_mask_cut] = color[ch]

    # 椎体（后画覆盖椎管）
    for entry in (vertebrae_chain or []):
        name = entry.get('name', 'Inc')
        if name == 'S2':
            continue
        label = _LEVEL_LABEL.get(name)
        if label is None:
            continue

        # 优先用真实曲线轮廓（背侧边用皮质线2-2）
        result = _build_curve_polygon(entry, ant_line, back_rows, back_cols)
        if result is None:
            result = _build_quad_polygon(entry)
        if result is None:
            continue

        rows_poly, cols_poly = result
        rr, cc = polygon(rows_poly, cols_poly, shape=img_shape)
        color = _LABEL_COLORS.get(label, (0.5, 0.5, 0.5, 0.4))
        for ch in range(4):
            overlay[rr, cc, ch] = color[ch]

    return overlay


def _draw_chain_on_ax(ax, vertebrae_chain, ant_line, c1_rows, c1_cols,
                       junction_pts, disc_centers, pixel_spacing,
                       c2_rows=None, c2_cols=None):
    """在指定 ax 上绘制椎体链路元素（终板线、前缘线、四边形、名称标注）。"""
    # 皮质线1（白色，仅用于可视化参考）
    if c1_rows is not None and len(c1_rows) > 0:
        ax.plot(list(c1_cols), list(c1_rows), color='white', linewidth=0.8,
                alpha=0.8, zorder=5)

    # 终板汇合点（黄色圆点）
    for jp in (junction_pts or []):
        ax.plot(jp[1], jp[0], 'o', color='yellow', markersize=4,
                markeredgewidth=0, alpha=0.85, zorder=8)

    # 椎间盘中心（品红菱形）
    for dc in (disc_centers or []):
        ax.plot(dc[1], dc[0], 'D', color='magenta', markersize=5,
                markeredgewidth=0, alpha=0.8, zorder=8)

    # 前缘线（天蓝色）
    if ant_line and len(ant_line) >= 2:
        ax.plot([p[1] for p in ant_line], [p[0] for p in ant_line],
                '-', color='deepskyblue', linewidth=1.8, alpha=0.9, zorder=14)

    # 椎体链路
    for entry in (vertebrae_chain or []):
        q = entry.get('quad', {})
        sup_ant  = q.get('sup_ant')
        sup_post = q.get('sup_post')
        inf_ant  = q.get('inf_ant')
        inf_post = q.get('inf_post')

        # ── 终板延伸线截断到交叉点（仅绘制交点之间的实体段）──
        sup_ext = entry.get('sup_ext', [])
        inf_ext = entry.get('inf_ext', [])

        def _clip_ep_between(ep_pts, pt_l, pt_r):
            """将终板延伸线截断到两个交叉点之间（按col范围），返回裁剪后的点列表。"""
            if pt_l is None or pt_r is None or len(ep_pts) < 2:
                return ep_pts
            col_min = min(pt_l[1], pt_r[1])
            col_max = max(pt_l[1], pt_r[1])
            clipped = [(r, c) for r, c in ep_pts if col_min <= c <= col_max]
            if len(clipped) < 2:
                return ep_pts
            # 强制首尾为交叉点坐标
            clipped_sorted = sorted(clipped, key=lambda p: p[1])
            lpt = pt_l if pt_l[1] <= pt_r[1] else pt_r
            rpt = pt_r if pt_l[1] <= pt_r[1] else pt_l
            clipped_sorted[0]  = lpt
            clipped_sorted[-1] = rpt
            return clipped_sorted

        # 上终板延伸线截断（sup_post→sup_ant之间）
        if len(sup_ext) >= 2:
            sup_draw = _clip_ep_between(sup_ext, sup_post, sup_ant)
            if len(sup_draw) >= 2:
                ax.plot([p[1] for p in sup_draw], [p[0] for p in sup_draw],
                        '-', color='lawngreen', linewidth=1.2, alpha=0.85, zorder=12)

        # 下终板延伸线截断（inf_ant→inf_post之间）
        if len(inf_ext) >= 2:
            inf_draw = _clip_ep_between(inf_ext, inf_ant, inf_post)
            if len(inf_draw) >= 2:
                ax.plot([p[1] for p in inf_draw], [p[0] for p in inf_draw],
                        '-', color='tomato', linewidth=1.2, alpha=0.85, zorder=12)

        # ── 上下终板虚线延伸（交点向外各延伸20mm）──
        ext_px = 20.0 / pixel_spacing

        def _draw_ext_dashes(ax, ep_pts, corner, direction, color):
            """从 corner 沿 ep_pts 末端斜率延伸 ext_px，绘制虚线。"""
            if corner is None or len(ep_pts) < 2:
                return
            ep_sorted = sorted(ep_pts, key=lambda p: p[1])
            if direction == 'left':
                # 取最左侧几个点估算斜率
                seg = ep_sorted[:min(5, len(ep_sorted))]
            else:
                seg = ep_sorted[-min(5, len(ep_sorted)):]
            if len(seg) < 2:
                return
            cs = np.array([p[1] for p in seg], dtype=np.float64)
            rs = np.array([p[0] for p in seg], dtype=np.float64)
            try:
                slp, _ = np.polyfit(cs, rs, 1)
            except Exception:
                slp = (rs[-1] - rs[0]) / max(abs(cs[-1] - cs[0]), 1e-9)
            dc = 1.0 / np.sqrt(1.0 + slp**2)
            dr = slp * dc
            if direction == 'left':
                dc, dr = -dc, -dr
            cr, cc = float(corner[0]), float(corner[1])
            ext_r, ext_c = [cr], [cc]
            acc = 0.0
            while acc < ext_px:
                cr += dr; cc += dc
                acc += np.sqrt(dr**2 + dc**2)
                ext_r.append(cr); ext_c.append(cc)
            ax.plot(ext_c, ext_r, '--', color=color, linewidth=1.0,
                    alpha=0.6, zorder=11)

        # 上终板：col小的一侧（腹侧/左侧）向左延伸，col大的一侧（背侧/右侧）向右延伸
        if sup_ant is not None and sup_post is not None:
            if sup_ant[1] <= sup_post[1]:
                _draw_ext_dashes(ax, sup_ext, sup_ant,  'left',  'lawngreen')
                _draw_ext_dashes(ax, sup_ext, sup_post, 'right', 'lawngreen')
            else:
                _draw_ext_dashes(ax, sup_ext, sup_post, 'left',  'lawngreen')
                _draw_ext_dashes(ax, sup_ext, sup_ant,  'right', 'lawngreen')
        # 下终板：同理
        if inf_ant is not None and inf_post is not None:
            if inf_ant[1] <= inf_post[1]:
                _draw_ext_dashes(ax, inf_ext, inf_ant,  'left',  'tomato')
                _draw_ext_dashes(ax, inf_ext, inf_post, 'right', 'tomato')
            else:
                _draw_ext_dashes(ax, inf_ext, inf_post, 'left',  'tomato')
                _draw_ext_dashes(ax, inf_ext, inf_ant,  'right', 'tomato')

        corners = [sup_ant, sup_post, inf_post, inf_ant]
        if all(c is not None for c in corners):
            xs = [c[1] for c in corners] + [corners[0][1]]
            ys = [c[0] for c in corners] + [corners[0][0]]
            ax.plot(xs, ys, '-', color='yellow', linewidth=1.2,
                    alpha=0.85, zorder=13)
            # 角点 ×
            for ck in corners:
                ax.plot(ck[1], ck[0], 'x', color='orange',
                        markersize=5, markeredgewidth=1.0, zorder=15)

            # 名称标注
            vr, vc = entry.get('vert_center', (0, 0))
            name = entry.get('name', '?')
            ang_str = (f" {entry['ant_angle_deg']:.0f}°"
                       if entry.get('ant_angle_deg') is not None else '')
            sup_str = (f" s{entry['sup_angle_deg']:.0f}°"
                       if entry.get('sup_angle_deg') is not None else '')
            ax.text(10, vr, f"{name}{ang_str}{sup_str}",
                    color='white', fontsize=8, fontweight='bold',
                    va='center', ha='left', zorder=16,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='black',
                              alpha=0.5, edgecolor='none'))
            ax.plot([50, vc], [vr, vr],
                    '--', color='white', linewidth=0.6, alpha=0.4, zorder=10)
        else:
            vr, vc = entry.get('vert_center', (0, 0))
            ax.text(10, vr, f"{entry.get('name','?')}?",
                    color='gray', fontsize=7,
                    va='center', ha='left', zorder=16,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='black',
                              alpha=0.3, edgecolor='none'))


def visualize_wifs(w_img_2d, in_img_2d, vertebrae_chain, ant_line,
                   cord_mask_cut, output_dir, stem, pixel_spacing,
                   c1_rows=None, c1_cols=None, c2_rows=None, c2_cols=None,
                   junction_pts=None, disc_centers=None, vert_centers=None,
                   patient_label='', in_label='',
                   left_only=False,
                   scan_results=None, cluster_results=None, fan_params_list=None,
                   profile_pts=None):
    """
    W图/IN图分割对比可视化。

    左图：W图 + 皮质线 + 椎管轮廓 + 椎体链路（终板线/前缘/四边形/名称）
    中图：IN图 + 分割掩模叠加（彩色半透明，含椎管）
    右图（可选）：IN图 + 扇形扫描详细可视化（候选点/扇形框/聚类线/前缘线）

    参数：
        scan_results     - Step4 扇形扫描结果列表（含 sup_pts/inf_pts/ant_pts）
        cluster_results  - Step5 聚类结果列表（含 sup/inf/ant points）
        fan_params_list  - Step4 扇形参数列表（含 center/up/dn/ant 角度范围）
        其余参数同原版。

    输出：{stem}_vis.png
    """
    H, W = w_img_2d.shape
    H_in, W_in = in_img_2d.shape if in_img_2d is not None else (H, W)

    has_scan = (not left_only) and (scan_results is not None)

    if left_only:
        fig, ax_w = plt.subplots(1, 1, figsize=(10, 14))
        ax_in = None
        ax_scan = None
    elif has_scan:
        fig, (ax_w, ax_in, ax_scan) = plt.subplots(1, 3, figsize=(30, 14))
    else:
        fig, (ax_w, ax_in) = plt.subplots(1, 2, figsize=(20, 14))
        ax_scan = None

    # ── 左图：W图 + 链路 ──
    vmax_w = float(np.percentile(w_img_2d[w_img_2d > 0], 99)) \
             if np.any(w_img_2d > 0) else 1.0
    ax_w.imshow(w_img_2d, cmap='gray', aspect='equal', vmin=0, vmax=vmax_w)

    # 皮质线2（青色）
    if c2_rows is not None and len(c2_rows) > 0:
        ax_w.plot(list(c2_cols), list(c2_rows), color='cyan', linewidth=0.8,
                  alpha=0.7, zorder=5)

    # 椎管轮廓（青色虚线）
    if cord_mask_cut is not None and np.any(cord_mask_cut):
        from skimage import measure
        contours = measure.find_contours(cord_mask_cut.astype(np.uint8), 0.5)
        for contour in contours:
            ax_w.plot(contour[:, 1], contour[:, 0], '--',
                      color='cyan', linewidth=0.8, alpha=0.6, zorder=6)

    _draw_chain_on_ax(ax_w, vertebrae_chain, ant_line,
                      c1_rows, c1_cols, junction_pts, disc_centers, pixel_spacing,
                      c2_rows=c2_rows, c2_cols=c2_cols)

    n_complete = sum(
        1 for e in (vertebrae_chain or [])
        if all(e.get('quad', {}).get(k) is not None
               for k in ('sup_ant', 'sup_post', 'inf_ant', 'inf_post')))
    ax_w.set_title(
        f'W Image  |  {patient_label}\n'
        f'vertebrae={len(vertebrae_chain or [])}  complete={n_complete}',
        fontsize=11, color='white')
    ax_w.axis('off')

    # ── 中图：IN图 + 掩模叠加 ──
    if ax_in is not None:
        if in_img_2d is not None:
            vmax_in = float(np.percentile(in_img_2d[in_img_2d > 0], 99)) \
                      if np.any(in_img_2d > 0) else 1.0
            ax_in.imshow(in_img_2d, cmap='gray', aspect='equal', vmin=0, vmax=vmax_in)
        else:
            ax_in.imshow(np.zeros((H, W)), cmap='gray', aspect='equal')
            ax_in.text(W // 2, H // 2, 'IN image not available',
                       color='white', fontsize=12, ha='center', va='center')

        overlay = _build_mask_overlay(vertebrae_chain, (H_in, W_in), cord_mask_cut,
                                       ant_line=ant_line, c1_rows=c1_rows, c1_cols=c1_cols,
                                       c2_rows=c2_rows, c2_cols=c2_cols)
        ax_in.imshow(overlay, aspect='equal', zorder=4)

        for entry in (vertebrae_chain or []):
            q = entry.get('quad', {})
            if all(q.get(k) is not None for k in ('sup_ant', 'sup_post', 'inf_ant', 'inf_post')):
                vr, vc = entry.get('vert_center', (0, 0))
                name = entry.get('name', '?')
                ax_in.text(W_in - 10, vr, name,
                           color='white', fontsize=8, fontweight='bold',
                           va='center', ha='right', zorder=16,
                           bbox=dict(boxstyle='round,pad=0.1', facecolor='black',
                                     alpha=0.5, edgecolor='none'))

        ax_in.set_title(
            f'IN Image + Mask  |  {in_label or patient_label}\n'
            f'S2 mask excluded',
            fontsize=11, color='white')
        ax_in.axis('off')

    # ── 右图：IN图 + 扇形扫描详细可视化 ──
    if ax_scan is not None:
        if in_img_2d is not None:
            vmax_in = float(np.percentile(in_img_2d[in_img_2d > 0], 99)) \
                      if np.any(in_img_2d > 0) else 1.0
            ax_scan.imshow(in_img_2d, cmap='gray', aspect='equal', vmin=0, vmax=vmax_in)
        else:
            ax_scan.imshow(np.zeros((H_in, W_in)), cmap='gray', aspect='equal')

        # 皮质线1（白色）
        if c1_rows is not None and len(c1_rows) > 0:
            ax_scan.plot(list(c1_cols), list(c1_rows), color='white',
                         linewidth=0.8, alpha=0.8, zorder=5)
        # 皮质线2（青色）
        if c2_rows is not None and len(c2_rows) > 0:
            ax_scan.plot(list(c2_cols), list(c2_rows), color='cyan',
                         linewidth=0.8, alpha=0.7, zorder=5)

        # 信号剖面采样线（白色虚线，皮质线2前70%法线偏移20mm处）
        if profile_pts is not None and len(profile_pts) > 0:
            _pr = [p[0] for p in profile_pts]
            _pc = [p[1] for p in profile_pts]
            ax_scan.plot(_pc, _pr, '--', color='white',
                         linewidth=0.8, alpha=0.7, zorder=6)

        # 终板汇合点（黄色圆点）
        for jp in (junction_pts or []):
            ax_scan.plot(jp[1], jp[0], 'o', color='yellow', markersize=5,
                         markeredgewidth=0, alpha=0.9, zorder=8)
        # 椎间盘中心（品红菱形）
        for dc in (disc_centers or []):
            ax_scan.plot(dc[1], dc[0], 'D', color='magenta', markersize=6,
                         markeredgewidth=0, alpha=0.9, zorder=9)

        # disc模式矩阵扫描几何可视化
        # 当 fan_params 全为 None 时（ep_scan_mode='disc'），画出起点集和扫描方向
        _has_fan = any(fp is not None for fp in (fan_params_list or []))
        if not _has_fan and vert_centers is not None and junction_pts is not None and c2_rows is not None:
            c2r = np.array(c2_rows, dtype=np.float64)
            c2c = np.array(c2_cols, dtype=np.float64)
            
            n_vc = len(vert_centers)
            n_jp = len(junction_pts)
            for _vi in range(min(n_vc, n_jp - 1)):
                # 椎体中心（中点）
                vc = vert_centers[_vi]
                vc_r, vc_c = float(vc[0]), float(vc[1])
                
                # 上下两个junction_pt
                jt = junction_pts[_vi]
                jb = junction_pts[_vi + 1]
                jt_r, jt_c = float(jt[0]), float(jt[1])
                jb_r, jb_c = float(jb[0]), float(jb[1])
                
                # 找两个junction_pt在c2上的索引
                def _find_nearest_c2(tr, tc):
                    dists = (c2r - tr)**2 + (c2c - tc)**2
                    return int(np.argmin(dists))
                
                idx_top = _find_nearest_c2(jt_r, jt_c)
                idx_bot = _find_nearest_c2(jb_r, jb_c)
                if idx_top > idx_bot:
                    idx_top, idx_bot = idx_bot, idx_top
                
                # 沿c2计算路径距离
                seg_rs = c2r[idx_top:idx_bot+1]
                seg_cs = c2c[idx_top:idx_bot+1]
                cum_dist = np.zeros(len(seg_rs))
                for k in range(1, len(seg_rs)):
                    cum_dist[k] = cum_dist[k-1] + math.sqrt(
                        (seg_rs[k]-seg_rs[k-1])**2 + (seg_cs[k]-seg_cs[k-1])**2)
                total_dist = cum_dist[-1]
                mid_dist = total_dist / 2.0
                mid_idx = idx_top + int(np.searchsorted(cum_dist, mid_dist))
                mid_idx = min(mid_idx, len(c2r) - 1)
                right_r, right_c = float(c2r[mid_idx]), float(c2c[mid_idx])
                
                # 起点连线：椎体中心 → 右端点
                line_dr = right_r - vc_r
                line_dc = right_c - vc_c
                line_len = math.sqrt(line_dr**2 + line_dc**2) + 1e-9
                line_urn = line_dr / line_len
                line_uc = line_dc / line_len

                # 起点线完整宽度：右半段=line_len，左半段=1.5*line_len
                n_pts_right = int(round(line_len))
                n_pts_left  = int(round(line_len * 1.5))

                # 细白虚线覆盖完整起点线范围
                left_r = vc_r - n_pts_left  * line_urn
                left_c = vc_c - n_pts_left  * line_uc
                ax_scan.plot([left_c, right_c], [left_r, right_r], '--',
                             color='white', linewidth=0.5, alpha=0.55, zorder=4)
                
                # 扫描方向：垂直于起点连线
                scan_dr = -line_uc
                scan_dc = line_urn
                if scan_dr > 0:
                    scan_dr, scan_dc = -scan_dr, -scan_dc
                
                _arrow_len = 20.0 / pixel_spacing
                # 上终板方向（-scan_dir，向头部，绿色）
                ax_scan.annotate('', xy=(vc_c + (-scan_dc)*_arrow_len, vc_r + (-scan_dr)*_arrow_len),
                                 xytext=(vc_c, vc_r),
                                 arrowprops=dict(arrowstyle='->', color='lime',
                                                 lw=1.2), zorder=7)
                # 下终板方向（+scan_dir，向尾部，红色）
                ax_scan.annotate('', xy=(vc_c + scan_dc*_arrow_len, vc_r + scan_dr*_arrow_len),
                                 xytext=(vc_c, vc_r),
                                 arrowprops=dict(arrowstyle='->', color='salmon',
                                                 lw=1.2), zorder=7)

                # disc模式前缘扇形可视化（天蓝色虚线轮廓）
                _afp = None
                if scan_results is not None and _vi < len(scan_results):
                    _afp = scan_results[_vi].get('ant_fan_params')
                if _afp is not None:
                    _cr2, _cc2 = float(_afp['center'][0]), float(_afp['center'][1])
                    _base_ang = float(_afp['angle'])
                    _half_ang = float(_afp['half'])
                    _scan_px  = float(_afp['scan_mm']) / pixel_spacing
                    _ang1 = math.radians(_base_ang - _half_ang)
                    _ang2 = math.radians(_base_ang + _half_ang)
                    # 两条扇形边线
                    _r1 = _cr2 + _scan_px * math.sin(_ang1)
                    _c1x = _cc2 + _scan_px * math.cos(_ang1)
                    _r2 = _cr2 + _scan_px * math.sin(_ang2)
                    _c2x = _cc2 + _scan_px * math.cos(_ang2)
                    ax_scan.plot([_cc2, _c1x], [_cr2, _r1], '--', color='deepskyblue',
                                 linewidth=0.6, alpha=0.5, zorder=3)
                    ax_scan.plot([_cc2, _c2x], [_cr2, _r2], '--', color='deepskyblue',
                                 linewidth=0.6, alpha=0.5, zorder=3)
                    # 弧线
                    _arc_pts = []
                    _ang = _ang1
                    while _ang <= _ang2 + 1e-6:
                        _arc_pts.append((_cr2 + _scan_px * math.sin(_ang),
                                         _cc2 + _scan_px * math.cos(_ang)))
                        _ang += math.radians(5)
                    if len(_arc_pts) >= 2:
                        ax_scan.plot([_p2[1] for _p2 in _arc_pts],
                                     [_p2[0] for _p2 in _arc_pts],
                                     '--', color='deepskyblue', linewidth=0.6,
                                     alpha=0.5, zorder=3)

        # 扇形框（虚线边线 + 弧线）
        if fan_params_list:
            for fp in (fan_params_list or []):
                if not fp:
                    continue
                cr2, cc2 = fp['center']
                for key, color in [('up', 'lime'), ('dn', 'salmon'), ('ant', 'deepskyblue')]:
                    p = fp.get(key)
                    if not p:
                        continue
                    base_ang  = p['angle']
                    half_ang  = p['half']
                    scan_px   = p['scan_mm'] / pixel_spacing
                    ang1 = math.radians(base_ang - half_ang)
                    ang2 = math.radians(base_ang + half_ang)
                    # 两条边线
                    r1 = cr2 + scan_px * math.sin(ang1)
                    c1x = cc2 + scan_px * math.cos(ang1)
                    r2 = cr2 + scan_px * math.sin(ang2)
                    c2x = cc2 + scan_px * math.cos(ang2)
                    ax_scan.plot([cc2, c1x], [cr2, r1], '--', color=color,
                                 linewidth=0.6, alpha=0.5, zorder=3)
                    ax_scan.plot([cc2, c2x], [cr2, r2], '--', color=color,
                                 linewidth=0.6, alpha=0.5, zorder=3)
                    # 弧线
                    arc_pts = []
                    ang = ang1
                    while ang <= ang2 + 1e-6:
                        arc_pts.append((cr2 + scan_px * math.sin(ang),
                                        cc2 + scan_px * math.cos(ang)))
                        ang += math.radians(5)
                    if len(arc_pts) >= 2:
                        ax_scan.plot([p2[1] for p2 in arc_pts],
                                     [p2[0] for p2 in arc_pts],
                                     '--', color=color, linewidth=0.6,
                                     alpha=0.5, zorder=3)

        # 扇形候选点
        for sr in (scan_results or []):
            for (r, c) in sr.get('sup_pts', []):
                ax_scan.plot(c, r, '.', color='lime', markersize=2,
                             alpha=0.4, zorder=6)
            for (r, c) in sr.get('inf_pts', []):
                ax_scan.plot(c, r, '.', color='salmon', markersize=2,
                             alpha=0.4, zorder=6)
            for (r, c) in sr.get('ant_pts', []):
                ax_scan.plot(c, r, '.', color='deepskyblue', markersize=2,
                             alpha=0.4, zorder=6)

        # 聚类后终板线 & 前缘线
        for cr_item in (cluster_results or []):
            sup_cr = cr_item.get('sup') or {}
            if sup_cr.get('points'):
                pts = sorted(sup_cr['points'], key=lambda p: p[1])
                ax_scan.plot([p[1] for p in pts], [p[0] for p in pts],
                             '-', color='lawngreen', linewidth=1.8,
                             alpha=0.95, zorder=11)
            inf_cr = cr_item.get('inf') or {}
            if inf_cr.get('points'):
                pts = sorted(inf_cr['points'], key=lambda p: p[1])
                ax_scan.plot([p[1] for p in pts], [p[0] for p in pts],
                             '-', color='tomato', linewidth=1.8,
                             alpha=0.95, zorder=11)

        # 前缘线（天蓝色）
        if ant_line and len(ant_line) >= 2:
            ax_scan.plot([p[1] for p in ant_line], [p[0] for p in ant_line],
                         '-', color='deepskyblue', linewidth=2.0,
                         alpha=0.9, zorder=12)

        # 椎体名称（右侧标注）
        for entry in (vertebrae_chain or []):
            vr, vc = entry.get('vert_center', (0, 0))
            name = entry.get('name', '?')
            q = entry.get('quad', {})
            if all(q.get(k) is not None for k in ('sup_ant', 'sup_post', 'inf_ant', 'inf_post')):
                ax_scan.text(10, vr, name, color='white', fontsize=8,
                             fontweight='bold', va='center', ha='left',
                             zorder=16,
                             bbox=dict(boxstyle='round,pad=0.1', facecolor='black',
                                       alpha=0.5, edgecolor='none'))
                ax_scan.plot([50, vc], [vr, vr], '--', color='white',
                             linewidth=0.6, alpha=0.4, zorder=10)

        n_scan_complete = sum(
            1 for e in (vertebrae_chain or [])
            if all(e.get('quad', {}).get(k) is not None
                   for k in ('sup_ant', 'sup_post', 'inf_ant', 'inf_post')))
        ax_scan.set_title(
            f'IN Fan Scan  |  {in_label or patient_label}\n'
            f'lime=sup  salmon=inf  skyblue=ant  complete={n_scan_complete}',
            fontsize=11, color='white')
        ax_scan.axis('off')

    fig.patch.set_facecolor('black')
    plt.tight_layout()

    vis_path = os.path.join(output_dir, f"{stem}_vis.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f"   [vis] 已保存: {os.path.basename(vis_path)}")
