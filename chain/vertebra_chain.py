"""
椎体链路构建模块 – Step6

build_vertebra_chain: 从聚类结果构建椎体链路，含椎体命名（四分支规则）。

四分支命名规则（上终板角度阈值 20°）：
  last ≥20° 且 sec ≥20° → S2, S1, L5, L4...
  last ≥20° 且 sec <20° → S1, L5, L4...
  last <20° 且 sec ≥20° → S2, S1, L5...
  last <20° 且 sec <20°（或None） → L5, L4, L3...
"""

import math
import numpy as np
from config.params import SUP_ANGLE_THRESH, SUP_ANGLE_GRAY_LOW, WIDTH_RATIO_THRESH

# ─────────────────────────────────────────────────────────────────────────────
# 辅助工具
# ─────────────────────────────────────────────────────────────────────────────

def _build_lut_from_line(pts_rc):
    """从 [(row, col), ...] 构建整数 row→col 查找表（线性插值）。"""
    if not pts_rc:
        return {}
    pts_sorted = sorted(pts_rc, key=lambda p: p[0])
    lut = {}
    for i in range(len(pts_sorted) - 1):
        r0, c0 = pts_sorted[i]
        r1, c1 = pts_sorted[i + 1]
        ri0 = int(math.floor(r0)); ri1 = int(math.ceil(r1))
        if ri0 == ri1:
            lut[ri0] = c0
            continue
        for ri in range(ri0, ri1 + 1):
            t = (ri - r0) / (r1 - r0 + 1e-9)
            lut[ri] = c0 + t * (c1 - c0)
    r_last, c_last = pts_sorted[-1]
    lut[int(round(r_last))] = c_last
    return lut


def _seg_intersect(p1, p2, p3, p4):
    """
    计算线段 p1-p2 与线段 p3-p4 的真实交点。
    返回 (r, c) 或 None（平行/不相交）。
    所有点为 (row, col) 格式。
    """
    r1, c1 = float(p1[0]), float(p1[1])
    r2, c2 = float(p2[0]), float(p2[1])
    r3, c3 = float(p3[0]), float(p3[1])
    r4, c4 = float(p4[0]), float(p4[1])
    dr1 = r2 - r1; dc1 = c2 - c1
    dr2 = r4 - r3; dc2 = c4 - c3
    denom = dr1 * dc2 - dc1 * dr2
    if abs(denom) < 1e-9:
        return None
    t = ((r3 - r1) * dc2 - (c3 - c1) * dr2) / denom
    u = ((r3 - r1) * dc1 - (c3 - c1) * dr1) / denom
    if -1e-6 <= t <= 1.0 + 1e-6 and -1e-6 <= u <= 1.0 + 1e-6:
        ix_r = r1 + t * dr1
        ix_c = c1 + t * dc1
        return (float(ix_r), float(ix_c))
    return None


def _find_crossing(ep_pts, ref_lut):
    """
    符号变化法求交：终板延伸线（ep_pts）与参考线（ref_lut：row→col 查找表）的交点。
    逐点计算 diff = ep_col - lut[row]，检测符号变化（col 大小关系反转）即为穿越。
    与 CR _find_crossing 逻辑一致，对终板线已越过参考线的情况同样有效。
    ep_pts 为 [(row, col), ...]，ref_lut 为 {int_row: float_col}。
    返回 (row, col) 交点或 None。
    """
    if not ep_pts or not ref_lut:
        return None

    ep_sorted = sorted(ep_pts, key=lambda p: p[0])
    prev_diff = None
    prev_r = prev_c = None
    for (r, c) in ep_sorted:
        ri = int(round(r))
        if ri not in ref_lut:
            continue
        ref_c = ref_lut[ri]
        diff = c - ref_c
        if prev_diff is not None and prev_diff * diff <= 0:
            # 线性插值求精确交点
            frac = abs(prev_diff) / (abs(prev_diff) + abs(diff) + 1e-9)
            ix_r = prev_r + frac * (r - prev_r)
            ri2  = int(round(ix_r))
            ix_c = ref_lut[ri2] if ri2 in ref_lut else \
                   (ref_c + ref_lut.get(int(round(prev_r)), ref_c)) / 2.0
            return (float(ix_r), float(ix_c))
        prev_diff = diff
        prev_r, prev_c = r, c
    return None


def _smooth_ep_pts(pts_rc, smooth_mm, pixel_spacing):
    """
    对终板线点集做 MAD 过滤 + 移动均值平滑（row 维度），按 col 排序。
    终板线以 col 为自变量（近似水平骨皮质），col 排序才是空间连续顺序。
    """
    if len(pts_rc) < 2:
        return pts_rc
    pts = sorted(pts_rc, key=lambda p: p[1])   # 按 col 排序
    rs = np.array([p[0] for p in pts], dtype=np.float64)
    cs = np.array([p[1] for p in pts], dtype=np.float64)
    med = float(np.median(rs))
    mad = float(np.median(np.abs(rs - med))) + 1e-9
    keep = np.abs(rs - med) < 3.0 * mad        # MAD 过滤 row 离群点
    rs = rs[keep]; cs = cs[keep]
    if len(rs) < 2:
        return pts
    k = max(3, int(round(smooth_mm / pixel_spacing)))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    rs_sm = np.convolve(np.pad(rs, pad, mode='edge'), np.ones(k) / k, mode='valid')[:len(rs)]
    return [(float(r), float(c)) for r, c in zip(rs_sm, cs)]


def _extend_ep_line(ep_pts_rc, extend_mm, pixel_spacing, img_shape,
                    max_angle_deg=SUP_ANGLE_THRESH):
    """
    对终板线点集向两端延伸 extend_mm，夹角约束到 ±max_angle_deg（用于找交点）。
    返回密集点列表 [(row, col), ...]，已按 col 排序。
    """
    if len(ep_pts_rc) < 2:
        return ep_pts_rc
    H, W = img_shape
    ext_px = extend_mm / pixel_spacing

    pts = sorted(ep_pts_rc, key=lambda p: p[1])
    rs = np.array([p[0] for p in pts], dtype=np.float64)
    cs = np.array([p[1] for p in pts], dtype=np.float64)

    try:
        slp_h, _ = np.polyfit(cs, rs, 1)
    except Exception:
        slp_h = float(rs[-1] - rs[0]) / max(float(cs[-1] - cs[0]), 1e-9)

    angle_rad = math.atan(slp_h)
    max_rad   = math.radians(max_angle_deg)
    if abs(angle_rad) > max_rad:
        angle_rad = math.copysign(max_rad, angle_rad)
    slp_clamped = math.tan(angle_rad)

    dc_unit = 1.0 / math.sqrt(1.0 + slp_clamped ** 2)
    dr_unit = slp_clamped * dc_unit

    dr_l, dc_l = -dr_unit, -dc_unit
    left_pts = []
    cr, cc = float(rs[0]), float(cs[0])
    acc = 0.0
    while acc < ext_px:
        cr += dr_l; cc += dc_l
        acc += math.sqrt(dr_l**2 + dc_l**2)
        if 0 <= cr < H and 0 <= cc < W:
            left_pts.append((cr, cc))

    dr_r, dc_r = dr_unit, dc_unit
    right_pts = []
    cr, cc = float(rs[-1]), float(cs[-1])
    acc = 0.0
    while acc < ext_px:
        cr += dr_r; cc += dc_r
        acc += math.sqrt(dr_r**2 + dc_r**2)
        if 0 <= cr < H and 0 <= cc < W:
            right_pts.append((cr, cc))

    left_pts_sorted  = sorted(left_pts,  key=lambda p: p[1])
    right_pts_sorted = sorted(right_pts, key=lambda p: p[1])
    all_pts = left_pts_sorted + list(zip(rs, cs)) + right_pts_sorted
    return [(float(r), float(c)) for r, c in all_pts]


def _stitch_ant_line(cluster_results, pixel_spacing):
    """
    分段保留各椎体前缘线，相邻段之间直线连结过渡，全段插值填满。
    """
    def _smooth_seg(pts_rc, smooth_mm):
        if len(pts_rc) < 2:
            return pts_rc
        pts = sorted(pts_rc, key=lambda p: p[0])
        rs = np.array([p[0] for p in pts], dtype=np.float64)
        cs = np.array([p[1] for p in pts], dtype=np.float64)
        med = float(np.median(cs))
        mad = float(np.median(np.abs(cs - med))) + 1e-9
        keep = np.abs(cs - med) < 3.0 * mad
        rs = rs[keep]; cs = cs[keep]
        if len(rs) < 2:
            return pts
        k = max(3, int(round(smooth_mm / pixel_spacing)))
        if k % 2 == 0: k += 1
        pad = k // 2
        cs_sm = np.convolve(np.pad(cs, pad, mode='edge'), np.ones(k)/k, mode='valid')[:len(cs)]
        return [(float(r), float(c)) for r, c in zip(rs, cs_sm)]

    segs = []
    for cr in cluster_results:
        if cr['ant'] and cr['ant'].get('points'):
            raw = [(float(p[0]), float(p[1])) for p in cr['ant']['points']]
            seg = _smooth_seg(raw, smooth_mm=5.0)
            if len(seg) >= 2:
                segs.append(seg)

    if not segs:
        return []

    all_rows = []
    all_cols = []

    for i, seg in enumerate(segs):
        rs = np.array([p[0] for p in seg], dtype=np.float64)
        cs = np.array([p[1] for p in seg], dtype=np.float64)

        r0 = int(round(rs[0])); r1 = int(round(rs[-1]))
        row_int = np.arange(r0, r1 + 1, dtype=np.float64)
        col_int = np.interp(row_int, rs, cs)

        if i == 0:
            all_rows.extend(row_int.tolist())
            all_cols.extend(col_int.tolist())
        else:
            prev_r = all_rows[-1]; prev_c = all_cols[-1]
            cur_r  = float(row_int[0]); cur_c = float(col_int[0])
            gap = int(round(cur_r)) - int(round(prev_r))
            if gap > 1:
                for step in range(1, gap):
                    frac = step / gap
                    all_rows.append(prev_r + frac * (cur_r - prev_r))
                    all_cols.append(prev_c + frac * (cur_c - prev_c))
            all_rows.extend(row_int[1:].tolist())
            all_cols.extend(col_int[1:].tolist())

    print(f"   [前缘拼接] 共{len(segs)}段  总点数={len(all_rows)}")
    return [(float(r), float(c)) for r, c in zip(all_rows, all_cols)]


# ─────────────────────────────────────────────────────────────────────────────
# 已废弃辅助函数（保留备用）
# ─────────────────────────────────────────────────────────────────────────────

def _calc_geom_ratio(entry):
    """
    【已废弃，由上终板角度阈值命名机制取代，保留备用】
    计算椎体几何宽高比 C = W / H。
    """
    q = entry.get('quad', {})
    sa = q.get('sup_ant');  ia = q.get('inf_ant')
    sp = q.get('sup_post'); ip = q.get('inf_post')
    if any(p is None for p in (sa, ia, sp, ip)):
        return None
    mid_ant  = ((sa[0]+ia[0])/2.0, (sa[1]+ia[1])/2.0)
    mid_post = ((sp[0]+ip[0])/2.0, (sp[1]+ip[1])/2.0)
    mid_sup  = ((sa[0]+sp[0])/2.0, (sa[1]+sp[1])/2.0)
    mid_inf  = ((ia[0]+ip[0])/2.0, (ia[1]+ip[1])/2.0)
    W = math.sqrt((mid_ant[0]-mid_post[0])**2 + (mid_ant[1]-mid_post[1])**2)
    H = math.sqrt((mid_sup[0]-mid_inf[0])**2 + (mid_sup[1]-mid_inf[1])**2)
    if H < 1e-6:
        return None
    return W / H


def _score_vertebra_type(entry):
    """
    【已废弃，由上终板角度阈值命名机制取代，保留备用】
    计算椎体 L5/S1/S2 评分，返回得分最高的类型。
    """
    def _ls(val, zero_v, full_v):
        if val is None:
            return 0.0
        if abs(full_v - zero_v) < 1e-9:
            return 0.0
        return max(0.0, min(1.0, (val - zero_v) / (full_v - zero_v)))

    A = entry.get('ant_angle_deg')
    B = None
    if entry.get('ant_angle_deg') is not None and entry.get('post_angle_deg') is not None:
        B = abs(entry['ant_angle_deg'] - entry['post_angle_deg'])
    C = _calc_geom_ratio(entry)

    sA_L5 = _ls(A, 55.0, 80.0)
    sB_L5 = _ls(B, 10.0,  5.0)
    sC_L5 = _ls(C,  0.8,  1.3)
    score_L5 = (sA_L5 * 5 + sB_L5 * 3 + sC_L5 * 2) / 10.0

    sA_S1 = _ls(A, 50.0, 30.0)
    sB_S1 = _ls(B,  5.0, 20.0)
    sC_S1 = _ls(C,  1.0,  0.5)
    score_S1 = (sA_S1 * 5 + sB_S1 * 3 + sC_S1 * 2) / 10.0

    sC_S2 = _ls(C, 0.6, 0.3)
    score_S2 = sC_S2

    scores = {'L5': score_L5, 'S1': score_S1, 'S2': score_S2}
    best   = max(scores, key=lambda k: scores[k])
    return best, scores


# ─────────────────────────────────────────────────────────────────────────────
# 主函数：build_vertebra_chain
# ─────────────────────────────────────────────────────────────────────────────

def _project_pt_to_c2(pt_r, pt_c, c2r_arr, c2c_arr, c2_arc_arr):
    """
    将点 (pt_r, pt_c) 法线投影到皮质线2-2，返回 (arc_pos_mm, normal_offset_px, norm_dr, norm_dc)。
    arc_pos_mm: 投影点在皮质线2-2上的弧长坐标（像素单位）
    normal_offset_px: 点到皮质线2-2的法线距离（像素，正=腹侧/col减小方向）
    norm_dr, norm_dc: 法线方向单位向量（从皮质线2-2指向前缘侧）
    """
    n = len(c2r_arr)
    if n < 2:
        return None
    # 欧氏最近点
    dists = np.sqrt((c2r_arr - pt_r)**2 + (c2c_arr - pt_c)**2)
    idx = int(np.argmin(dists))
    # 该点切线
    if idx == 0:
        tdr = c2r_arr[1] - c2r_arr[0]; tdc = c2c_arr[1] - c2c_arr[0]
    elif idx == n - 1:
        tdr = c2r_arr[-1] - c2r_arr[-2]; tdc = c2c_arr[-1] - c2c_arr[-2]
    else:
        tdr = c2r_arr[idx+1] - c2r_arr[idx-1]; tdc = c2c_arr[idx+1] - c2c_arr[idx-1]
    t_len = math.sqrt(tdr**2 + tdc**2)
    if t_len < 1e-9:
        return None
    tdr /= t_len; tdc /= t_len
    # 法线方向（切线旋转90°，朝腹侧/col减小方向）
    ndr = tdc; ndc = -tdr   # 法线 = (tdc, -tdr)，调整符号使其指向前缘侧（col较小）
    # 点到投影点的向量
    vr = pt_r - float(c2r_arr[idx]); vc = pt_c - float(c2c_arr[idx])
    offset = vr * ndr + vc * ndc   # 法线分量（正=前缘侧）
    return (float(c2_arc_arr[idx]), offset, ndr, ndc)


def _calc_c2_arc(c2r_arr, c2c_arr, pixel_spacing):
    """计算皮质线2-2累积弧长数组（像素单位）。"""
    n = len(c2r_arr)
    arc = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        arc[i] = arc[i-1] + math.sqrt(
            (c2r_arr[i]-c2r_arr[i-1])**2 + (c2c_arr[i]-c2c_arr[i-1])**2)
    return arc


def build_vertebra_chain(cluster_results, vert_centers, c1_rows, c1_cols,
                          pixel_spacing, img_shape,
                          c2_rows=None, c2_cols=None,
                          junction_pts=None, disc_centers=None):
    """
    构建椎体链路：
      A. 前缘线拼接：所有椎体前缘点 → 分段平滑 → 两端法线延伸（基于皮质线2-2） → 建LUT
      B. 终板线延长：每条上/下终板线两端延长25mm，超界截断
      C. 四交点：上/下终板线 vs 前缘LUT + 皮质线1LUT
      D. 前缘夹角 + 后缘夹角 + 上终板角（sup_angle_deg）
      E. 椎体命名（四分支规则，阈值 20°）

    两端法线延伸逻辑：
      - 确定前缘线端点到皮质线2-2的法线偏移量 d_offset
      - 沿皮质线2-2走向延伸，每步保持 d_offset 横向偏移
      - 顶端终止：皮质线2-2弧长超过 disc_centers[0]+junction_pts[0] 投影点 + 5mm
      - 底端终止：皮质线2-2弧长超过 disc_centers[-1]+junction_pts[-1] 投影点 + 10mm
      - 回退：c2/junction_pts/disc_centers 缺失时退化为底端斜率延伸20mm

    返回：
        (chain, ant_line)
        chain: [{'name', 'sup_ext', 'inf_ext', 'quad', 'ant_angle_deg',
                 'post_angle_deg', 'sup_angle_deg', 'vert_center'}, ...]
        ant_line: [(r,c), ...]
    """
    H, W = img_shape

    # ── A. 前缘线拼接 + 两端法线延伸 ──
    ant_line = _stitch_ant_line(cluster_results, pixel_spacing=pixel_spacing)
    if ant_line:
        ant_sorted = sorted(ant_line, key=lambda p: p[0])
        rs_a = np.array([p[0] for p in ant_sorted], dtype=np.float64)
        cs_a = np.array([p[1] for p in ant_sorted], dtype=np.float64)

        # 判断是否具备法线延伸条件
        _can_normal_ext = (
            c2_rows is not None and c2_cols is not None
            and junction_pts is not None and disc_centers is not None
            and len(c2_rows) >= 4
            and len(junction_pts) >= 1 and len(disc_centers) >= 1
        )

        if _can_normal_ext:
            c2r_arr = np.array(c2_rows, dtype=np.float64)
            c2c_arr = np.array(c2_cols, dtype=np.float64)
            c2_arc  = _calc_c2_arc(c2r_arr, c2c_arr, pixel_spacing)
            ext_stop_mm = 10.0 / pixel_spacing   # 10mm，像素单位

            # ── 计算截止弧长：取 disc_center 和 junction_pt 各自投影弧长的均值 ──
            def _stop_arc(disc_pt, junc_pt):
                arcs = []
                for pt in [disc_pt, junc_pt]:
                    if pt is None:
                        continue
                    res = _project_pt_to_c2(float(pt[0]), float(pt[1]),
                                             c2r_arr, c2c_arr, c2_arc)
                    if res is not None:
                        arcs.append(res[0])
                return float(np.mean(arcs)) if arcs else None

            top_stop = _stop_arc(disc_centers[0], junction_pts[0])
            bot_stop = _stop_arc(disc_centers[-1], junction_pts[-1])

            # ── 顶端延伸（向上，沿c2往头侧） ──
            top_res = _project_pt_to_c2(float(rs_a[0]), float(cs_a[0]),
                                         c2r_arr, c2c_arr, c2_arc)
            head_ext = []
            if top_res is not None and top_stop is not None:
                top_arc0, d_top, ndr_t, ndc_t = top_res
                # 在皮质线2-2上从 top_arc0 往头侧（弧长减小方向）走
                # 找到 arc < top_arc0 且 arc >= top_stop - ext_stop_mm 的点
                stop_arc_top = top_stop - ext_stop_mm
                for i in range(len(c2_arc) - 1, -1, -1):
                    arc_i = c2_arc[i]
                    if arc_i >= top_arc0:
                        continue
                    if arc_i < stop_arc_top:
                        break
                    # 该皮质线2-2点沿法线偏移 d_top
                    er = float(c2r_arr[i]) + ndr_t * d_top
                    ec = float(c2c_arr[i]) + ndc_t * d_top
                    if 0 <= er < H and 0 <= ec < W:
                        head_ext.append((er, ec))
                head_ext = sorted(head_ext, key=lambda p: p[0])
                print(f"   [前缘顶端法线延伸] 补充 {len(head_ext)} 点，"
                      f"d_offset={d_top:.1f}px，截止弧长={stop_arc_top:.1f}px")

            # ── 底端延伸（向下，沿c2往尾侧） ──
            bot_res = _project_pt_to_c2(float(rs_a[-1]), float(cs_a[-1]),
                                         c2r_arr, c2c_arr, c2_arc)
            tail_ext = []
            if bot_res is not None and bot_stop is not None:
                bot_arc0, d_bot, ndr_b, ndc_b = bot_res
                stop_arc_bot = bot_stop + ext_stop_mm
                for i in range(len(c2_arc)):
                    arc_i = c2_arc[i]
                    if arc_i <= bot_arc0:
                        continue
                    if arc_i > stop_arc_bot:
                        break
                    er = float(c2r_arr[i]) + ndr_b * d_bot
                    ec = float(c2c_arr[i]) + ndc_b * d_bot
                    if 0 <= er < H and 0 <= ec < W:
                        tail_ext.append((er, ec))
                tail_ext = sorted(tail_ext, key=lambda p: p[0])
                print(f"   [前缘底端法线延伸] 补充 {len(tail_ext)} 点，"
                      f"d_offset={d_bot:.1f}px，截止弧长={stop_arc_bot:.1f}px")

            ant_line = head_ext + ant_sorted + tail_ext

            # ── 底端斜率外推：沿前缘线末端自身斜率再延伸 10mm ──
            tail_slope_mm = 10.0
            ext_base = ant_line  # 当前完整前缘线
            if len(ext_base) >= 3:
                # 取末端 20mm 路径计算平均斜率
                ext_cum = np.zeros(len(ext_base))
                for k in range(1, len(ext_base)):
                    ext_cum[k] = ext_cum[k-1] + math.sqrt(
                        (ext_base[k][0]-ext_base[k-1][0])**2 + (ext_base[k][1]-ext_base[k-1][1])**2)
                tail_dist_20 = 20.0 / pixel_spacing
                tail_mask_s = ext_cum >= (ext_cum[-1] - tail_dist_20)
                seg_rs = np.array([p[0] for p in ext_base])[tail_mask_s]
                seg_cs = np.array([p[1] for p in ext_base])[tail_mask_s]
                if len(seg_rs) < 2:
                    seg_rs = np.array([ext_base[-2][0], ext_base[-1][0]])
                    seg_cs = np.array([ext_base[-2][1], ext_base[-1][1]])
                try:
                    slp_s, _ = np.polyfit(seg_rs, seg_cs, 1)
                except Exception:
                    slp_s = float(seg_cs[-1]-seg_cs[0]) / max(float(seg_rs[-1]-seg_rs[0]), 1e-9)
                tlen_s = math.sqrt(1.0 + slp_s**2)
                dr_s = 1.0 / tlen_s
                dc_s = slp_s / tlen_s
                ext_px_s = tail_slope_mm / pixel_spacing
                cur_r_s = float(ext_base[-1][0])
                cur_c_s = float(ext_base[-1][1])
                acc_s = 0.0
                slope_ext = []
                while acc_s < ext_px_s:
                    cur_r_s += dr_s
                    cur_c_s += dc_s
                    acc_s += math.sqrt(dr_s**2 + dc_s**2)
                    if 0 <= cur_r_s < H and 0 <= cur_c_s < W:
                        slope_ext.append((cur_r_s, cur_c_s))
                if slope_ext:
                    ant_line = ant_line + slope_ext
                    print(f"   [前缘底端斜率外推] 补充 {len(slope_ext)} 点，延伸={tail_slope_mm:.1f}mm")

        else:
            # 回退：底端斜率延伸20mm
            tail_dist_px = 5.0 / pixel_spacing
            cum_a = np.zeros(len(rs_a))
            for k in range(1, len(rs_a)):
                cum_a[k] = cum_a[k-1] + math.sqrt(
                    (rs_a[k]-rs_a[k-1])**2 + (cs_a[k]-cs_a[k-1])**2)
            tail_mask = cum_a >= (cum_a[-1] - tail_dist_px)
            seg_r = rs_a[tail_mask]; seg_c = cs_a[tail_mask]
            if len(seg_r) < 2:
                seg_r = rs_a[-2:]; seg_c = cs_a[-2:]
            try:
                slp_a, icp_a = np.polyfit(seg_r, seg_c, 1)
            except Exception:
                slp_a = float(seg_c[-1]-seg_c[0]) / max(float(seg_r[-1]-seg_r[0]), 1e-9)
                icp_a = float(seg_c[-1]) - slp_a * float(seg_r[-1])
            tlen_a = math.sqrt(1.0 + slp_a**2)
            dr_a = 1.0 / tlen_a; dc_a = slp_a / tlen_a
            ext_px_a = 20.0 / pixel_spacing
            cur_r_a = float(rs_a[-1]); cur_c_a = float(cs_a[-1])
            acc_a = 0.0; ext_r = []; ext_c = []
            while acc_a < ext_px_a:
                cur_r_a += dr_a; cur_c_a += dc_a
                acc_a += math.sqrt(dr_a**2 + dc_a**2)
                if 0 <= cur_r_a < H and 0 <= cur_c_a < W:
                    ext_r.append(cur_r_a); ext_c.append(cur_c_a)
            ant_line = ant_sorted + [(r, c) for r, c in zip(ext_r, ext_c)]
            print(f"   [前缘延伸] 回退斜率延伸20mm（c2/junction_pts/disc_centers不足）")
    else:
        ant_line = []
    # ── 皮质线1 点列表 + LUT ──
    c1_pts = [(float(r), float(c)) for r, c in zip(c1_rows, c1_cols)]
    ant_lut = _build_lut_from_line(ant_line)
    c1_lut  = _build_lut_from_line(c1_pts)

    # ── B+C+D. 逐椎体处理 ──
    raw_chain = []
    for vi, (cr, vc) in enumerate(zip(cluster_results, vert_centers)):
        entry = {
            'name': f'V{vi}',
            'sup_ext': [], 'inf_ext': [],
            'quad': {'sup_ant': None, 'sup_post': None,
                     'inf_ant': None, 'inf_post': None},
            'ant_angle_deg': None,
            'post_angle_deg': None,
            'sup_angle_deg': None,
            'vert_center': vc,
        }

        if cr['sup'] and cr['sup'].get('points'):
            sup_pts = [(float(p[0]), float(p[1])) for p in cr['sup']['points']]
            sup_ext = _extend_ep_line(sup_pts, extend_mm=25.0,
                                       pixel_spacing=pixel_spacing,
                                       img_shape=img_shape)
            entry['sup_ext'] = sup_ext
            entry['quad']['sup_ant']  = _find_crossing(sup_ext, ant_lut)
            entry['quad']['sup_post'] = _find_crossing(sup_ext, c1_lut)

        if cr['inf'] and cr['inf'].get('points'):
            inf_pts = [(float(p[0]), float(p[1])) for p in cr['inf']['points']]
            inf_ext = _extend_ep_line(inf_pts, extend_mm=25.0,
                                       pixel_spacing=pixel_spacing,
                                       img_shape=img_shape)
            entry['inf_ext'] = inf_ext
            entry['quad']['inf_ant']  = _find_crossing(inf_ext, ant_lut)
            entry['quad']['inf_post'] = _find_crossing(inf_ext, c1_lut)

        # ── D. 前缘夹角 ──
        sa = entry['quad']['sup_ant']
        ia = entry['quad']['inf_ant']
        if sa is not None and ia is not None:
            dr = ia[0] - sa[0]
            dc = ia[1] - sa[1]
            vec_len = math.sqrt(dr*dr + dc*dc) + 1e-9
            cos_a = abs(dc * (-1.0)) / vec_len
            cos_a = float(np.clip(cos_a, 0.0, 1.0))
            _ang = math.degrees(math.acos(cos_a))
            entry['ant_angle_deg'] = min(_ang, 180.0 - _ang)

        # ── D2. 后缘夹角 ──
        sp = entry['quad']['sup_post']
        ip = entry['quad']['inf_post']
        if sp is not None and ip is not None:
            dr_p = ip[0] - sp[0]
            dc_p = ip[1] - sp[1]
            vec_len_p = math.sqrt(dr_p*dr_p + dc_p*dc_p) + 1e-9
            cos_p = abs(dc_p * (-1.0)) / vec_len_p
            cos_p = float(np.clip(cos_p, 0.0, 1.0))
            _ang_p = math.degrees(math.acos(cos_p))
            entry['post_angle_deg'] = min(_ang_p, 180.0 - _ang_p)

        # ── D3. 上终板角（sup_ant→sup_post 与水平线正夹角）──
        sa2 = entry['quad']['sup_ant']
        sp2 = entry['quad']['sup_post']
        if sa2 is not None and sp2 is not None:
            _dr_s = sp2[0] - sa2[0]
            _dc_s = sp2[1] - sa2[1]
            entry['sup_angle_deg'] = math.degrees(
                math.atan2(abs(_dr_s), max(_dc_s, 1e-9)))

        # ── E. 四边形几何中心 ──
        q = entry['quad']
        valid_corners = [v for v in [q['sup_ant'], q['sup_post'],
                                     q['inf_post'], q['inf_ant']] if v is not None]
        if len(valid_corners) >= 2:
            geom_r = float(np.mean([c[0] for c in valid_corners]))
            geom_c = float(np.mean([c[1] for c in valid_corners]))
            entry['vert_center'] = (geom_r, geom_c)

        raw_chain.append(entry)

    # ── E. 椎体命名（从下往上）──
    def _is_complete_entry(e):
        q = e['quad']
        return all(q[k] is not None for k in ('sup_ant', 'sup_post', 'inf_ant', 'inf_post'))

    for e in raw_chain:
        e['name'] = 'Inc'

    complete_for_naming = [e for e in raw_chain if _is_complete_entry(e)]

    if complete_for_naming:
        last_e     = complete_for_naming[-1]
        sec_last_e = complete_for_naming[-2] if len(complete_for_naming) >= 2 else None

        last_sup_ang     = last_e.get('sup_angle_deg')
        sec_last_sup_ang = sec_last_e.get('sup_angle_deg') if sec_last_e else None

        _last_s = f'{last_sup_ang:.1f}°'  if last_sup_ang  is not None else 'None'
        _sec_s  = f'{sec_last_sup_ang:.1f}°' if sec_last_sup_ang is not None else 'None'
        print(f"   [Step6 naming] last sup_angle={_last_s}  2nd sup_angle={_sec_s}")

        # ── 四分支命名规则 ──
        # _is_s: sup_angle >= 20° 直接为S；15°~20° 灰色区间用上/下终板宽度比判断
        def _is_s_vertebra(e):
            if e is None:
                return False
            sup_ang = e.get('sup_angle_deg')
            if sup_ang is None:
                return False
            if sup_ang >= SUP_ANGLE_THRESH:
                return True
            if sup_ang < SUP_ANGLE_GRAY_LOW:
                return False
            q = e.get('quad', {})
            sa = q.get('sup_ant');  sp = q.get('sup_post')
            ia = q.get('inf_ant');  ip = q.get('inf_post')
            if sa is None or sp is None or ia is None or ip is None:
                return False
            sup_w = math.sqrt((sp[0]-sa[0])**2 + (sp[1]-sa[1])**2)
            inf_w = math.sqrt((ip[0]-ia[0])**2 + (ip[1]-ia[1])**2)
            width_ratio = sup_w / inf_w if inf_w > 1e-6 else 0.0
            mid_sup_r = (sa[0] + sp[0]) / 2.0;  mid_sup_c = (sa[1] + sp[1]) / 2.0
            mid_inf_r = (ia[0] + ip[0]) / 2.0;  mid_inf_c = (ia[1] + ip[1]) / 2.0
            height    = math.sqrt((mid_inf_r - mid_sup_r)**2 + (mid_inf_c - mid_sup_c)**2)
            avg_width = (sup_w + inf_w) / 2.0
            hw_ratio  = height / avg_width if avg_width > 1e-6 else 0.0
            cond_w = width_ratio >= WIDTH_RATIO_THRESH
            cond_h = hw_ratio     >= WIDTH_RATIO_THRESH
            is_s   = cond_w or cond_h
            print("   [Step6 naming] 灰色区间 sup_angle=%.1f" % sup_ang
                  + "  w_ratio=%.3f  hw_ratio=%.3f" % (width_ratio, hw_ratio)
                  + "  -> " + ("S椎体" if is_s else "L5"))
            return is_s

        last_large = _is_s_vertebra(last_e)
        sec_large  = _is_s_vertebra(sec_last_e)

        if last_large and sec_large:
            bottom_seq = ['S2', 'S1', 'L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11']
        elif last_large and not sec_large:
            bottom_seq = ['S1', 'L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11']
        elif not last_large and sec_large:
            # last非S但sec为S：sec判为S1，last判为L5上一级
            bottom_seq = ['S2', 'S1', 'L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11']
        else:
            bottom_seq = ['L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11']

        print(f"   [Step6 naming] bottom_seq={bottom_seq[:4]}...")

        for i, entry in enumerate(reversed(complete_for_naming)):
            entry['name'] = bottom_seq[i] if i < len(bottom_seq) else f'T{10-i}'

    # ── F'. 前缘线宽度校验与修正（仅 L/T 椎体，独立处理不影响其他椎体）──
    _can_width_check = (
        c2_rows is not None and c2_cols is not None and len(c2_rows) >= 2
    )
    if _can_width_check:
        # 构建 c2 row→col 查找数组（用 row 插值）
        _c2r = np.array(c2_rows, dtype=np.float64)
        _c2c = np.array(c2_cols, dtype=np.float64)
        _c2_sort_idx = np.argsort(_c2r)
        _c2r_sorted  = _c2r[_c2_sort_idx]
        _c2c_sorted  = _c2c[_c2_sort_idx]

        def _c2_col_at_row(row_px):
            """按 row 插值得到 c2 对应 col（px）。"""
            return float(np.interp(row_px, _c2r_sorted, _c2c_sorted))

        _any_corrected = False

        # 计算各 L/T 椎体的前缘宽度 W_i (px)
        _width_list = []   # [(vi, W_i)]
        for _vi, (_cr, _entry) in enumerate(zip(cluster_results, raw_chain)):
            _name = _entry.get('name', '')
            if not (_name.startswith('L') or _name.startswith('T')):
                continue
            _ant = _cr.get('ant')
            if not _ant or not _ant.get('points'):
                continue
            _ant_rows = [float(p[0]) for p in _ant['points']]
            _ant_cols = [float(p[1]) for p in _ant['points']]
            if not _ant_cols:
                continue
            _ant_row_med = float(np.median(_ant_rows))
            _ant_col_med = float(np.median(_ant_cols))
            _c2_col = _c2_col_at_row(_ant_row_med)
            _W = _ant_col_med - _c2_col   # px，正值=前缘在c2右侧（正常）
            _width_list.append((_vi, abs(_W)))

        if len(_width_list) >= 3:
            _ws = sorted([w for _, w in _width_list])
            _ws_trim = _ws[1:-1]   # 去掉最大最小
            _width_ref = float(np.mean(_ws_trim)) if _ws_trim else float(np.mean(_ws))
            _width_thresh = 0.8 * _width_ref

            _w_summary = ', '.join(
                f"{raw_chain[_vi].get('name','V'+str(_vi))}:{_W:.1f}px"
                for _vi, _W in _width_list
            )
            print(f"   [Step6-校验] 宽度基准={_width_ref:.1f}px  阈值(80%)={_width_thresh:.1f}px  "
                  f"各椎体: {_w_summary}")

            for _vi, _W in _width_list:
                if _W >= _width_thresh:
                    continue
                _cr   = cluster_results[_vi]
                _entry = raw_chain[_vi]
                _name  = _entry.get('name', f'V{_vi}')

                # 边界：无上椎或无下椎则跳过
                _prev_ant = cluster_results[_vi - 1].get('ant') if _vi > 0 else None
                _next_ant = cluster_results[_vi + 1].get('ant') if _vi < len(cluster_results) - 1 else None
                if (not _prev_ant or not _prev_ant.get('points') or
                        not _next_ant or not _next_ant.get('points')):
                    print(f"   [Step6-校验] 椎体{_name}: W={_W:.1f}px(ref={_width_ref:.1f}px,"
                          f"70%={_width_thresh:.1f}px) → 宽度异常，无邻椎前缘线，跳过修正")
                    continue

                # 取上椎前缘尾端点、下椎前缘头端点
                _prev_pts = sorted(_prev_ant['points'], key=lambda p: p[0])
                _next_pts = sorted(_next_ant['points'], key=lambda p: p[0])
                P_top = _prev_pts[-1]   # 上椎前缘 row 最大的点
                P_bot = _next_pts[0]    # 下椎前缘 row 最小的点

                r_top, c_top = float(P_top[0]), float(P_top[1])
                r_bot, c_bot = float(P_bot[0]), float(P_bot[1])

                if r_bot <= r_top:
                    print(f"   [Step6-校验] 椎体{_name}: 上下端点row顺序异常，跳过修正")
                    continue

                # 线性插值生成修正前缘点列
                _new_pts = []
                for _r in range(int(round(r_top)), int(round(r_bot)) + 1):
                    _t = (_r - r_top) / (r_bot - r_top + 1e-9)
                    _c = c_top + _t * (c_bot - c_top)
                    _new_pts.append((_r, _c, 0.0, 0.0))   # 保持与原points格式一致

                _cr['ant']['points'] = _new_pts
                _cr['ant']['src_tag'] = 'predicted_interp'

                # 局部更新 ant_lut（仅该椎体 row 范围）
                for _r in range(int(round(r_top)), int(round(r_bot)) + 1):
                    _t = (_r - r_top) / (r_bot - r_top + 1e-9)
                    ant_lut[_r] = c_top + _t * (c_bot - c_top)

                # 重新计算该椎体的 sup_ant / inf_ant / ant_angle_deg
                if _entry['sup_ext']:
                    _entry['quad']['sup_ant'] = _find_crossing(_entry['sup_ext'], ant_lut)
                if _entry['inf_ext']:
                    _entry['quad']['inf_ant'] = _find_crossing(_entry['inf_ext'], ant_lut)

                _sa = _entry['quad']['sup_ant']
                _ia = _entry['quad']['inf_ant']
                if _sa is not None and _ia is not None:
                    _dr = _ia[0] - _sa[0]
                    _dc = _ia[1] - _sa[1]
                    _vl = math.sqrt(_dr * _dr + _dc * _dc) + 1e-9
                    _cos = float(np.clip(abs(_dc * (-1.0)) / _vl, 0.0, 1.0))
                    _ang = math.degrees(math.acos(_cos))
                    _entry['ant_angle_deg'] = min(_ang, 180.0 - _ang)

                print(f"   [Step6-校验] 椎体{_name}: W={_W:.1f}px"
                      f"(ref={_width_ref:.1f}px, 70%={_width_thresh:.1f}px)"
                      f" → 宽度异常，已修正前缘线（插值）")
                _any_corrected = True

        # F' 修正后：从 ant_lut 重建 ant_line，使可视化/掩膜/ROI同步更新
        if _any_corrected:
            ant_line = sorted(
                [(float(r), float(c)) for r, c in ant_lut.items()],
                key=lambda p: p[0]
            )
            print(f"   [Step6-校验] ant_line 已从 ant_lut 重建，共{len(ant_line)}点")

    # ── F. 统计 ──
    def _is_complete(entry):
        q = entry['quad']
        return all(q[k] is not None for k in ('sup_ant', 'sup_post', 'inf_ant', 'inf_post'))

    complete_chain = [e for e in raw_chain if _is_complete(e)]
    n_total    = len(raw_chain)
    n_complete = len(complete_chain)
    incomplete_names = [e['name'] for e in raw_chain if not _is_complete(e)]

    print(f"   [Step6链路] 检测椎体={n_total}  完整识别={n_complete}  "
          f"不完整={incomplete_names}  "
          f"前缘线={len(ant_line)}点  "
          f"命名: {[e['name'] for e in raw_chain]}")
    return raw_chain, ant_line
