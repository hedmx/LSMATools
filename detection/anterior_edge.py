"""
前缘线聚类模块 – Step5 / Step5b

cluster_anterior_edge  : 单椎体前缘线聚类（法线投影最密区间）
cluster_all_vertebrae  : 对所有椎体的扫描结果依次做聚类
_mad_smooth_line       : MAD 过滤 + 移动均值平滑
_smooth_ant_line       : 全段插值 + 三步移动均值平滑
"""

import math
import numpy as np
from ._scan_utils import _arc_coord, _project_to_c2


# ─────────────────────────────────────────────────────────────────────────────
# 去倾斜密集区间聚类（供 Step5 上/下终板聚类使用）
# ─────────────────────────────────────────────────────────────────────────────

def cluster_correction_pts(pts_list, ep_type, pixel_spacing,
                            win_row_mm=7.0, min_pts=15, smooth_mm=5.0,
                            col_span_thresh_mm=15.0):
    """
    弧长轴密集区间聚类 + 移动均值平滑。

    聚类窗口在 off_mm（皮质线3弧长坐标，mm）轴上滑动，用三层分级 score 选出最优窗口：
      - score = 0.5 × normalize(col_span) + 0.5 × normalize(count)
        col_span / count 各自除以全部候选窗口的最大值做归一化（相对评分）
        col_span 采用 col 的 5th~95th 百分位数范围（点数 ≥ 10 时），
        点数不足 10 时用 col 极差；与 row 分布无关，对倾斜终板友好
      - 主路径：col_span ≥ col_span_thresh AND count ≥ min_pts → score 最高
      - 降级1：count ≥ min_pts（放弃 col 门槛）→ score 最高
      - 降级2（兜底）：所有窗口均不足 min_pts，取点数最多的窗口

    参数：
        pts_list           - [(row, col, off_mm), ...]
        ep_type            - 'superior' 或 'inferior'
        pixel_spacing      - mm/pixel
        win_row_mm         - 弧长轴聚类窗口宽度（mm），默认 7.0
        min_pts            - 最少点数要求，默认 15
        smooth_mm          - 移动均值平滑窗口（mm），默认 5.0
        col_span_thresh_mm - col 跨度最低门槛（mm），默认 15.0

    返回：终板线 dict 或 None
    """
    if not pts_list:
        return None

    off_arr = np.array([p[2] for p in pts_list], dtype=np.float64)
    row_arr = np.array([p[0] for p in pts_list], dtype=np.float64)
    col_arr = np.array([p[1] for p in pts_list], dtype=np.float64)

    arc_min = float(off_arr.min())
    arc_max = float(off_arr.max())

    # ── 对每个窗口计算指标 ──
    def _window_metrics(mask):
        """给定布尔 mask，返回 (count, col_span_mm)。
        col_span 采用 col 的 5th~95th 百分位数范围（点数 ≥ 10 时），
        点数不足 10 时直接用 col 极差。与 row 分布无关，对倾斜终板友好。"""
        cnt = int(np.sum(mask))
        if cnt < 2:
            return cnt, 0.0
        f_col = col_arr[mask]
        if cnt >= 10:
            col_span = (float(np.percentile(f_col, 95)) - float(np.percentile(f_col, 5))) * pixel_spacing
        else:
            col_span = (float(f_col.max()) - float(f_col.min())) * pixel_spacing
        return cnt, col_span

    def _run_cluster(win_arc, round_label=''):
        """用指定窗口宽度 win_arc 跑一轮聚类，返回 (result_dict_or_None, tier_label)。"""
        step = 1.0

        # 枚举所有滑动窗口
        windows = []
        cur = arc_min
        while cur <= arc_max + 1e-6:
            mask = (off_arr >= cur) & (off_arr < cur + win_arc)
            center_off = cur + win_arc / 2.0
            cnt, col_span = _window_metrics(mask)
            windows.append([center_off, cnt, col_span, 0.0, mask])
            cur += step

        if not windows:
            return None, ''

        # 归一化后计算 score = 0.5×norm_col_span + 0.5×norm_count
        max_col_span = max(w[2] for w in windows) or 1.0
        max_count    = max(w[1] for w in windows) or 1.0
        for w in windows:
            norm_col = w[2] / max_col_span
            norm_cnt = w[1] / max_count
            w[3] = 0.5 * norm_col + 0.5 * norm_cnt

        # 三层分级选窗口
        def _best_by_score(cands):
            return max(cands, key=lambda w: w[3])

        main = [w for w in windows if w[1] >= min_pts and w[2] >= col_span_thresh_mm]
        if main:
            tier = _best_by_score(main)
            tier_label = f'主路径{round_label}'
        else:
            deg1 = [w for w in windows if w[1] >= min_pts]
            if deg1:
                tier = _best_by_score(deg1)
                tier_label = f'降级1(放弃col门槛){round_label}'
            else:
                tier = max(windows, key=lambda w: w[1])
                tier_label = f'降级2(兜底-最多点数){round_label}'

        best_center, best_count, best_col_span, best_score, _ = tier

        # 用选出的窗口取点
        sel_mask = (off_arr >= best_center - win_arc / 2.0) & \
                   (off_arr <  best_center + win_arc / 2.0)
        sel_off = off_arr[sel_mask]
        sel_row = row_arr[sel_mask]
        sel_col = col_arr[sel_mask]

        if len(sel_row) == 0:
            print(f"   [聚类] {ep_type} [{tier_label}] 窗口内无点，跳过")
            return None, tier_label

        # 按 col 排序后：MAD 过滤 row 离群点 → 移动均值平滑 row
        sort2   = np.argsort(sel_col)
        sel_off = sel_off[sort2]
        sel_row = sel_row[sort2]
        sel_col = sel_col[sort2]

        med_r = float(np.median(sel_row))
        mad_r = float(np.median(np.abs(sel_row - med_r))) + 1e-9
        keep  = np.abs(sel_row - med_r) < 3.0 * mad_r
        sel_off = sel_off[keep]
        sel_row = sel_row[keep]
        sel_col = sel_col[keep]

        if len(sel_row) == 0:
            print(f"   [聚类] {ep_type} [{tier_label}] MAD过滤后无点，跳过")
            return None, tier_label

        k_sm = max(3, int(round(smooth_mm / pixel_spacing)))
        if k_sm % 2 == 0:
            k_sm += 1
        pad = k_sm // 2
        row_sm = np.convolve(np.pad(sel_row, pad, mode='edge'), np.ones(k_sm) / k_sm, mode='valid')
        col_sm = sel_col

        points = [(float(r), float(c), 0.0, float(o))
                  for r, c, o in zip(row_sm, col_sm, sel_off)]

        row_center = float(np.mean(row_sm))
        arc_center = float(np.mean(sel_off))

        print(f"   [聚类] {ep_type} [{tier_label}]: 总候选={len(pts_list)} "
              f"弧长=[{arc_min:.1f},{arc_max:.1f}]mm 选{len(points)}点 "
              f"col_span={best_col_span:.1f}mm count={best_count} "
              f"score={best_score:.3f} arc_center={arc_center:.1f}mm")

        return {
            'ep_type':    ep_type,
            'points':     points,
            'row_center': row_center,
            'arc_center': arc_center,
        }, tier_label

    # ── 第一轮聚类 ──
    total_arc = arc_max - arc_min
    result, _ = _run_cluster(win_row_mm, '')

    # ── 弧长超限校验：总弧长 > 13mm 且第一轮结果有效，则用 40% 弧长窗口回退一次 ──
    if result is not None and total_arc > 13.0:
        fallback_win = total_arc * 0.4
        print(f"   [聚类回退] {ep_type} 总弧长={total_arc:.1f}mm>13mm，"
              f"回退窗口={fallback_win:.1f}mm")
        result_fb, _ = _run_cluster(fallback_win, '(回退40%)')
        if result_fb is not None:
            result = result_fb

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 5  前缘聚类
# ─────────────────────────────────────────────────────────────────────────────

def cluster_anterior_edge(ant_pts, junc_top, junc_bot,
                           pixel_spacing,
                           win_w_mm=5.0, min_pts=5, smooth_mm=3.0,
                           offset_min_mm=15.0, extend_baseline_mm=5.0):
    """
    椎体前缘线聚类：
      以上终板汇合点(junc_top)和下终板汇合点(junc_bot)的连线为基准，
      基准线两端各沿方向延伸 extend_baseline_mm，扩大行覆盖范围。
      取其法线方向（朝腹侧，col减小），将所有候选点投影到法线轴。
      仅保留投影值 >= offset_min_mm/pixel_spacing 的点，
      在这些点中滑动 win_w_mm 窗口找最密集区间，
      按 row 排序后做移动均值平滑。

    返回：
        {'points': [(row, col, 0.0, 0.0), ...], 'row_center': float} 或 None
    """
    if not ant_pts or len(ant_pts) < min_pts:
        return None

    at_raw = (float(junc_top[0]), float(junc_top[1]))
    ab_raw = (float(junc_bot[0]), float(junc_bot[1]))

    d_dr = ab_raw[0] - at_raw[0]
    d_dc = ab_raw[1] - at_raw[1]
    d_len = math.sqrt(d_dr * d_dr + d_dc * d_dc) + 1e-9
    d_dr /= d_len; d_dc /= d_len

    ext_px = extend_baseline_mm / pixel_spacing
    at = (at_raw[0] - d_dr * ext_px, at_raw[1] - d_dc * ext_px)

    p_row = d_dc; p_col = -d_dr
    if p_col > 0:
        p_row, p_col = -p_row, -p_col

    def proj_normal(r, c):
        return (r - at[0]) * p_row + (c - at[1]) * p_col

    proj_vals = np.array([proj_normal(p[0], p[1]) for p in ant_pts])

    offset_min_px = offset_min_mm / pixel_spacing
    valid_mask = proj_vals >= offset_min_px
    ant_valid = [p for p, m in zip(ant_pts, valid_mask) if m]
    proj_valid = proj_vals[valid_mask]

    if len(ant_valid) < min_pts:
        print(f"   [Step5前缘] offset≥{offset_min_mm}mm 后仅{len(ant_valid)}点 < {min_pts}，跳过")
        return None

    win_w_px  = win_w_mm / pixel_spacing
    p_min = float(proj_valid.min())
    p_max = float(proj_valid.max())

    best_count  = 0
    best_center = float(np.median(proj_valid))
    cur = p_min
    while cur <= p_max:
        mask = (proj_valid >= cur) & (proj_valid < cur + win_w_px)
        cnt = int(np.sum(mask))
        if cnt > best_count:
            best_count = cnt
            best_center = cur + win_w_px / 2.0
        cur += win_w_px / 4.0

    sel_mask = ((proj_valid >= best_center - win_w_px / 2.0) &
                (proj_valid <  best_center + win_w_px / 2.0))
    sel_pts = [p for p, m in zip(ant_valid, sel_mask) if m]

    if len(sel_pts) < min_pts:
        print(f"   [Step5前缘] 最密窗口仅{len(sel_pts)}点 < {min_pts}，跳过")
        return None

    sel_pts.sort(key=lambda p: p[0])
    sel_r = np.array([p[0] for p in sel_pts])
    sel_c = np.array([p[1] for p in sel_pts])

    k_sm = max(3, int(round(smooth_mm / pixel_spacing)))
    if k_sm % 2 == 0: k_sm += 1
    pad = k_sm // 2
    row_sm = np.convolve(np.pad(sel_r, pad, mode='edge'), np.ones(k_sm) / k_sm, mode='valid')[:len(sel_r)]
    col_sm = np.convolve(np.pad(sel_c, pad, mode='edge'), np.ones(k_sm) / k_sm, mode='valid')[:len(sel_c)]

    points = [(float(r), float(c), 0.0, 0.0) for r, c in zip(row_sm, col_sm)]
    print(f"   [Step5前缘] 总候选={len(ant_pts)}  offset过滤后={len(ant_valid)}  "
          f"选{len(points)}点  最密窗口{best_count}点")

    return {'points': points, 'row_center': float(np.mean(row_sm))}


def _make_inf_from_normal(junction_pt, c2_rows, c2_cols, pixel_spacing,
                          extend_right_mm=5.0, extend_left_mm=35.0):
    """
    当最后一个椎体后缘角 ≤ 65° 时，跳过聚类，直接用终板汇合点法线构造下终板线。

    从 junction_pt 投影到皮质线2，取法线方向（朝腹侧，n_col < 0）：
      - 右侧（背侧，反法线）延伸 extend_right_mm
      - 左侧（腹侧，法线方向）延伸 extend_left_mm
    沿法线方向每像素插值一点，生成与 cluster_correction_pts 输出格式一致的 dict。

    参数：
        junction_pt      - (row, col, val, c2_idx)
        c2_rows/c2_cols  - 皮质线2坐标数组
        pixel_spacing    - mm/pixel
        extend_right_mm  - 背侧延伸距离（mm），默认 5mm
        extend_left_mm   - 腹侧延伸距离（mm），默认 35mm

    返回：
        {'ep_type': 'inferior', 'points': [(r,c,0.,0.), ...],
         'row_center': float, 'arc_center': None}
        或 None（坐标越界等异常时）
    """
    jr = float(junction_pt[0])
    jc = float(junction_pt[1])

    r_c2, c_c2, n_row, n_col = _project_to_c2(jr, jc,
                                               np.array(c2_rows, dtype=np.float64),
                                               np.array(c2_cols, dtype=np.float64))
    # 确保法线朝腹侧（n_col < 0）
    if n_col > 0:
        n_row, n_col = -n_row, -n_col

    right_px = extend_right_mm / pixel_spacing   # 背侧（反法线）
    left_px  = extend_left_mm  / pixel_spacing   # 腹侧（法线方向）
    n_steps  = int(math.ceil(right_px + left_px))

    # 起点：沿反法线方向偏移 right_px
    start_r = jr + (-n_row) * right_px
    start_c = jc + (-n_col) * right_px

    points = []
    for i in range(n_steps + 1):
        t   = i / max(n_steps, 1)
        pr  = start_r + n_row * (right_px + left_px) * t
        pc  = start_c + n_col * (right_px + left_px) * t
        points.append((float(pr), float(pc), 0.0, 0.0))

    if not points:
        return None

    row_center = float(np.mean([p[0] for p in points]))
    print(f"   [法线下终板] junction=({jr:.1f},{jc:.1f}) "
          f"n=({n_row:.3f},{n_col:.3f}) 右{extend_right_mm}mm+左{extend_left_mm}mm "
          f"→ {len(points)}点")
    return {
        'ep_type':    'inferior',
        'points':     points,
        'row_center': row_center,
        'arc_center': None,
    }


def cluster_all_vertebrae(scan_results, disc_centers, pixel_spacing,
                           junction_pts=None,
                           c3_cols=None, c3_rows=None, arc_len_mm=None,
                           last_ant_angle_deg=None,
                           second_last_ant_angle_deg=None,
                           c2_rows=None, c2_cols=None):
    """
    对所有椎体的扇形/矩阵扫描结果依次做聚类。

    参数：
        scan_results          - [{'sup_pts', 'inf_pts', 'ant_pts'}, ...]
        disc_centers          - [(r, c), ...]
        pixel_spacing         - mm/pixel
        junction_pts          - [(row, col, val, c1_idx), ...]（可为 None）
        c3_cols               - 皮质线3列坐标数组（用于弧长投影，None 时降级为射线序号）
        c3_rows               - 皮质线3行坐标数组
        arc_len_mm            - 皮质线3弧长数组 (mm)
        last_ant_angle_deg    - 最后椎体后缘角（度），用于 is_last 下终板构造判断
        second_last_ant_angle_deg - 倒数第二椎体后缘角
        c2_rows / c2_cols     - 皮质线2坐标（angle ≤ 65° 时法线构造下终板线用）

    返回：
        results = [{'sup', 'inf', 'ant'}, ...]
    """
    _use_arc = (c3_cols is not None and c3_rows is not None and arc_len_mm is not None)
    print(f"[聚类弧长投影] 模式={'法线投影(c3)' if _use_arc else '射线序号(降级)'}")
    n_vert = len(scan_results)
    results = []
    for vi, sr in enumerate(scan_results):
        # 兼容两种格式：disc模式用 sr['sup']['points']，fan模式用 sr['sup_pts']
        sup_pts = (sr.get('sup') or {}).get('points', []) if 'sup' in sr else sr.get('sup_pts', [])
        inf_pts = (sr.get('inf') or {}).get('points', []) if 'inf' in sr else sr.get('inf_pts', [])
        
        if _use_arc:
            sup_raw = [(p[0], p[1], _arc_coord(p[0], p[1], c3_rows, c3_cols, arc_len_mm))
                       for p in sup_pts]
            inf_raw = [(p[0], p[1], _arc_coord(p[0], p[1], c3_rows, c3_cols, arc_len_mm))
                       for p in inf_pts]
        else:
            sup_raw = [(p[0], p[1], float(i)) for i, p in enumerate(sup_pts)]
            inf_raw = [(p[0], p[1], float(i)) for i, p in enumerate(inf_pts)]

        new_sup = cluster_correction_pts(
            sup_raw, 'superior', pixel_spacing,
            win_row_mm=5.0, min_pts=5, smooth_mm=5.0)

        is_last        = (vi == n_vert - 1)
        is_second_last = (vi == n_vert - 2)

        # ── 下终板：is_last 且后缘角 ≤ 65° → 法线构造，否则正常聚类 ──
        _use_normal_inf = (
            is_last
            and last_ant_angle_deg is not None
            and last_ant_angle_deg <= 50.0
            and junction_pts is not None
            and len(junction_pts) >= 1
            and c2_rows is not None
            and c2_cols is not None
        )
        if _use_normal_inf:
            print(f"   [Step5] 最后椎体后缘角={last_ant_angle_deg:.1f}°≤50° → 法线构造下终板线")
            new_inf = _make_inf_from_normal(
                junction_pts[-1], c2_rows, c2_cols, pixel_spacing,
                extend_right_mm=5.0, extend_left_mm=35.0)
        else:
            new_inf = cluster_correction_pts(
                inf_raw, 'inferior', pixel_spacing,
                win_row_mm=5.0, min_pts=5, smooth_mm=5.0)

        def _ant_win_from_angle(angle_deg, default_mm):
            if angle_deg is None:
                return default_mm
            a = float(np.clip(angle_deg, 40.0, 80.0))
            return 20.0 - (a - 40.0) / 40.0 * 15.0

        if is_last:
            ant_win_w_mm = _ant_win_from_angle(last_ant_angle_deg, 12.0)
        elif is_second_last:
            ant_win_w_mm = _ant_win_from_angle(second_last_ant_angle_deg, 8.0)
        else:
            ant_win_w_mm  = 5.0
        ant_smooth_mm = 3.0

        junc_top_pt = None
        junc_bot_pt = None
        if junction_pts is not None:
            if vi < len(junction_pts):
                junc_top_pt = junction_pts[vi]
            if vi + 1 < len(junction_pts):
                junc_bot_pt = junction_pts[vi + 1]

        new_ant = None
        # 兼容两种格式
        ant_pts = (sr.get('ant') or {}).get('points', []) if 'ant' in sr else sr.get('ant_pts', [])
        disc_top = disc_centers[vi]     if vi     < len(disc_centers) else None
        disc_bot = disc_centers[vi + 1] if vi + 1 < len(disc_centers) else None
        
        if ant_pts and junc_top_pt is not None and junc_bot_pt is not None:
            new_ant = cluster_anterior_edge(
                ant_pts, junc_top_pt, junc_bot_pt, pixel_spacing,
                win_w_mm=ant_win_w_mm, min_pts=5, smooth_mm=ant_smooth_mm,
                offset_min_mm=15.0)
        elif ant_pts and disc_top is not None and disc_bot is not None:
            new_ant = cluster_anterior_edge(
                ant_pts, disc_top, disc_bot, pixel_spacing,
                win_w_mm=ant_win_w_mm, min_pts=5, smooth_mm=ant_smooth_mm,
                offset_min_mm=15.0)

        results.append({'sup': new_sup, 'inf': new_inf, 'ant': new_ant})
        lbl = '(last)' if is_last else ('(2nd-last)' if is_second_last else '')
        print(f"   [Step5] 椎体{vi}{lbl}: "
              f"sup={'OK' if new_sup else 'FAIL'}  "
              f"inf={'OK' if new_inf else 'FAIL'}  "
              f"ant={'OK' if new_ant else 'FAIL'}  "
              f"win={ant_win_w_mm}mm")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 前缘平滑工具
# ─────────────────────────────────────────────────────────────────────────────

def _mad_smooth_line(points_rc, smooth_mm, pixel_spacing):
    """
    对按row排序的点集做 MAD 过滤 + 移动均值平滑。
    points_rc: [(row, col), ...]
    返回: [(row, col), ...]
    """
    if len(points_rc) < 3:
        return points_rc
    pts = sorted(points_rc, key=lambda p: p[0])
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


def _smooth_ant_line(points_rc, pixel_spacing):
    """
    将所有聚类后前缘散点重新连结平滑：
      1. 按 row 排序
      2. MAD 过滤（col维度，3σ）
      3. 线性插值填满所有整数行
      4. 三步移动均值平滑（15mm → 10mm → 5mm）
    points_rc: [(row, col), ...]
    返回: [(row, col), ...]
    """
    if len(points_rc) < 2:
        return points_rc

    pts = sorted(points_rc, key=lambda p: p[0])
    rs = np.array([p[0] for p in pts], dtype=np.float64)
    cs = np.array([p[1] for p in pts], dtype=np.float64)

    med = float(np.median(cs))
    mad = float(np.median(np.abs(cs - med))) + 1e-9
    keep = np.abs(cs - med) < 3.0 * mad
    rs = rs[keep]; cs = cs[keep]
    if len(rs) < 2:
        return pts

    row_int = np.arange(int(round(rs[0])), int(round(rs[-1])) + 1, dtype=np.float64)
    cs_interp = np.interp(row_int, rs, cs)

    def _ma(arr, mm):
        k = max(3, int(round(mm / pixel_spacing)))
        if k % 2 == 0: k += 1
        pad = k // 2
        return np.convolve(np.pad(arr, pad, mode='edge'), np.ones(k) / k, mode='valid')[:len(arr)]

    cs_sm = _ma(cs_interp, 15.0)
    cs_sm = _ma(cs_sm,     10.0)
    cs_sm = _ma(cs_sm,      5.0)

    return [(float(r), float(c)) for r, c in zip(row_int, cs_sm)]
