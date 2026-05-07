"""
聚类与终板线构建模块（V15）
包含 cluster_endplates_v15、build_endplate_line_v15、
     sliding_window_cluster_endplates、find_vertebra_right_edge_from_candidates
"""
import numpy as np
from .scan_lines_v15 import convert_to_arc_coord


def cluster_endplates_v15(raw_cands, scan_lines_v15, c2_cols, c2_rows,
                          arc_len_mm, pixel_spacing, win_mm=5.0, min_lines=16):
    """
    V15 终板点聚类：将候选点投影到皮质线2弧长坐标系，
    用弧长距离（即层面途径）替代原水平行号距离做聚类判断。
    即使 L5 严重倾斜，同一终板的点弧长坐标也会集中在同一小区间内。

    参数：
        raw_cands       - [(row, col, type_str, depth, line_idx), ...]
        scan_lines_v15  - build_scan_lines_v15 返回値
        c2_cols/c2_rows - 皮质线2
        arc_len_mm      - 皮质线2弧长数组 (mm)
        pixel_spacing   - 像素间距 mm
        win_mm          - 聚类窗口宽度 (mm)，默认 5mm
        min_lines       - 最少有效扫描线数，默认 7

    返回：
        与 sliding_window_cluster_endplates 相同格式：
        [{'ep_type': 'superior'/'inferior', 'points': [...], 'row_center': float,
          'arc_center': float}, ...]
    """
    if not raw_cands or not scan_lines_v15:
        return []

    n_v15 = len(scan_lines_v15)
    min_peak_sep_mm = 8.0   # 两条终板线弧长隔开最小距离

    # 每个候选点转换为弧长坐标
    # raw_cands 中 line_idx 现在对应 scan_lines_v15 的序号
    cands_arc = []  # [(s_mm, row, col, type_str, depth, line_idx), ...]
    for r, c, t, d, li in raw_cands:
        if li >= n_v15:
            continue
        s_mm, _ = convert_to_arc_coord(r, c, c2_cols, c2_rows, arc_len_mm)
        cands_arc.append((s_mm, r, c, t, d, li))

    if not cands_arc:
        return []

    s_all = np.array([x[0] for x in cands_arc])
    s_min, s_max = float(s_all.min()), float(s_all.max())
    print(f"   [V15聚类] 候选点弧长范围=[{s_min:.1f}, {s_max:.1f}]mm, 共{len(cands_arc)}个点")

    result_lines = []

    for ep_type in ('superior', 'inferior'):
        # 按扫描线分组弧长坐标
        # {line_idx: [s_mm, ...]}
        line_s = {}
        for s_mm, r, c, t, d, li in cands_arc:
            if t != ep_type:
                continue
            if li not in line_s:
                line_s[li] = []
            line_s[li].append(s_mm)

        if not line_s:
            continue

        # 建立弧长坐标覆盖统计：寻找 win_mm 内有≥ min_lines 条线的弧长区间
        # 在 [s_min, s_max] 内逐 0.5mm 扫描
        scan_step = 0.5  # mm
        s_probe = np.arange(s_min, s_max + scan_step, scan_step)

        cover_arr = {}  # {s: n_lines}
        for s in s_probe:
            cnt = sum(
                1 for li in line_s
                if any(s <= sv <= s + win_mm for sv in line_s[li])
            )
            cover_arr[s] = cnt

        # 找平建peak
        peaks = []
        in_peak = False
        seg_start = s_min
        probe_list = sorted(cover_arr.keys())
        for s in probe_list:
            if cover_arr[s] >= min_lines:
                if not in_peak:
                    seg_start = s
                    in_peak = True
            else:
                if in_peak:
                    pk = max(
                        (sv for sv in probe_list if seg_start <= sv < s),
                        key=lambda sv: cover_arr[sv]
                    )
                    peaks.append(pk)
                    in_peak = False
        if in_peak:
            pk = max(
                (sv for sv in probe_list if sv >= seg_start),
                key=lambda sv: cover_arr[sv]
            )
            peaks.append(pk)

        # 合并过近的peak
        merged_peaks = []
        for pk in sorted(peaks):
            if merged_peaks and abs(pk - merged_peaks[-1]) < min_peak_sep_mm:
                if cover_arr.get(pk, 0) > cover_arr.get(merged_peaks[-1], 0):
                    merged_peaks[-1] = pk
            else:
                merged_peaks.append(pk)

        print(f"   [V15聚类] {ep_type} 弧长峰内: {len(merged_peaks)} 个 @ {[f'{p:.1f}mm' for p in merged_peaks]}")

        # 对每个峰收集窗口内点构建终板线
        used_keys = set()
        for peak_s in merged_peaks:
            win_pts = []
            for s_mm, r, c, t, d, li in cands_arc:
                if t != ep_type:
                    continue
                key = (li, r)
                if key in used_keys:
                    continue
                if peak_s <= s_mm <= peak_s + win_mm:
                    off_mm = scan_lines_v15[li][0] if li < n_v15 else li * 1.0
                    win_pts.append((r, c, d, off_mm, s_mm))
                    used_keys.add(key)

            n_covered = len(set(p[3] for p in win_pts))
            if len(win_pts) < min_lines or n_covered < min_lines:
                continue

            row_center = float(np.mean([p[0] for p in win_pts]))
            arc_center = float(np.mean([p[4] for p in win_pts]))
            result_lines.append({
                'ep_type':    ep_type,
                'points':     [(p[0], p[1], p[2], p[3]) for p in win_pts],
                's_mm_list':  [p[4] for p in win_pts],
                'row_center': row_center,
                'arc_center': arc_center,
            })

    result_lines.sort(key=lambda x: x['arc_center'])
    print(f"   [V15聚类] 共识别 {len(result_lines)} 条终板线")
    return result_lines


def build_endplate_line_v15(ep, c2_cols, c2_rows, arc_len_mm, pixel_spacing):
    """
    V15终板分割直线生成。

    问题：26条扫描线平行于皮质线2，同一终板候选点的col坐标差异极小，
    不能直接用(row, col)做Theil-Sen斜率拟合。

    解决方案：把每个候选点的s_mm（弧长坐标）映射回皮质线2上的col坐标，
    用候选点的row 和 皮质线2投影点的col 做Theil-Sen斜率拟合。
    皮质线2本身在弧长范围内有足够的col跨度，能反映真实终板走向。

    参数：
        ep           - cluster_endplates_v15返回的单条终板，包含 's_mm_list'
        c2_cols/rows - 皮质线2
        arc_len_mm   - 皮质线2弧长数组
    返回：
        {'slope', 'intercept', 'center_row', 'center_col', 'normal_dr', 'normal_dc'}
    """
    pts    = ep['points']        # [(row, col, depth, off_mm), ...]
    s_list = ep.get('s_mm_list', [])
    if len(pts) < 3:
        return None

    rows_p = np.array([p[0] for p in pts], dtype=np.float64)
    cols_p = np.array([p[1] for p in pts], dtype=np.float64)
    center_row = float(np.median(rows_p))
    center_col = float(np.median(cols_p))

    if len(s_list) == len(pts):
        # 用弧长坐标查皮质线2对应col，做Theil-Sen
        c2_proj = np.array(
            [float(c2_cols[int(np.argmin(np.abs(arc_len_mm - s)))]) for s in s_list],
            dtype=np.float64)
        slopes = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dc = c2_proj[j] - c2_proj[i]
                if abs(dc) > 0.5:
                    slopes.append((rows_p[j] - rows_p[i]) / dc)
        slope = float(np.median(slopes)) if slopes else 0.0
    else:
        # fallback: 用arc_center处切线斜率
        arc_center = ep.get('arc_center', None)
        N = len(c2_rows)
        best_i = int(np.argmin(np.abs(arc_len_mm - arc_center))) if arc_center is not None else                  int(np.argmin(np.sqrt((c2_rows.astype(np.float64)-center_row)**2 +
                                      (c2_cols.astype(np.float64)-center_col)**2)))
        i_prev = max(0, best_i - 5)
        i_next = min(N - 1, best_i + 5)
        t_dr = float(c2_rows[i_next] - c2_rows[i_prev])
        t_dc = float(c2_cols[i_next] - c2_cols[i_prev])
        slope = t_dr / t_dc if abs(t_dc) > 1e-6 else 0.0

    intercept = center_row - slope * center_col

    t_len = np.sqrt(1.0 + slope * slope)
    normal_dr =  1.0 / t_len
    normal_dc = -slope / t_len
    if normal_dc > 0:
        normal_dr, normal_dc = -normal_dr, -normal_dc

    return {
        'slope':      slope,
        'intercept':  intercept,
        'center_row': center_row,
        'center_col': center_col,
        'normal_dr':  normal_dr,
        'normal_dc':  normal_dc,
    }


def find_vertebra_right_edge_from_candidates(raw_candidates, scan_lines_f, pixel_spacing,
                                               endplates_f=None):
    """
    V12.3 氿体右边缘检测：基于压水图扫描线候选点的双线共识验证。

    逻辑：
      1. 只取 scan_lines_f 中 offset 最小的3条线（最靠近皮质线）
      2. 从 raw_candidates 中过滤出这3条线上的候选点
      3. 行位置附近（当前行 ±tol_px）内有 ≥2 条线都有同类型候选点 → 真实标记
      4. 真实标记列坐标平移到皮质线（offset=2mm的列+2mm素素 = offset=0）
      5. 从下往上：红X(下终板 lower)到绿X(上终板 upper)直线中心点

    raw_candidates: [(row, col, type_str, depth, line_idx), ...]
    scan_lines_f:   [(offset_mm, cols_arr, rows_arr), ...]
    返回: dict
      'lower_edges': [(row, col), ...] 红X 氿体下终板
      'upper_edges': [(row, col), ...] 绿X 氿体上终板
      'centers':     [(row, col), ...] 氿体右边缘中心点
    """
    if not raw_candidates or not scan_lines_f:
        return {'lower_edges': [], 'upper_edges': [], 'centers': []}

    # 1. 取 offset 最小的3条线
    sorted_lines = sorted(scan_lines_f, key=lambda x: x[0])  # 按 offset_mm 升序
    top3_lines   = sorted_lines[:3]
    top3_offsets = {sl[0] for sl in top3_lines}              # offset_mm 集合

    # 2. 过滤候选点：line_idx 对应 scan_lines_f 的索引，需先建立 offset_mm 到 line_idx 的映射
    # raw_candidates 里存的 line_idx 是 scan_lines_f 列表索引
    top3_line_indices = set()
    for li, (off_mm, cols_, rows_) in enumerate(scan_lines_f):
        if off_mm in top3_offsets:
            top3_line_indices.add(li)

    # 将候选点按类型分组，只保留最小3条线上的
    # 格式: {row: {'upper': [col, ...], 'lower': [col, ...]}, ...}
    cands_by_row = {}
    for r, c, t, d, li in raw_candidates:
        if li not in top3_line_indices:
            continue
        if r not in cands_by_row:
            cands_by_row[r] = {'inferior': [], 'superior': []}
        cands_by_row[r][t].append((c, li))

    # 3. 行单位合并容差：±4mm
    tol_px = max(1, int(4.0 / pixel_spacing))  # ±4mm

    # 建立全行列表与皮质线坐标映射（用offset最小的第1条线 + 2mm平移）
    first_line_off, first_cols, first_rows = top3_lines[0]
    shift_px = max(1, int(2.0 / pixel_spacing))  # +2mm → offset=0（皮质线）
    row_to_cortical_col = {}
    for r, c in zip(first_rows, first_cols):
        row_to_cortical_col[int(r)] = int(c) + shift_px  # 平移到皮质线

    all_cand_rows = sorted(cands_by_row.keys())

    def get_cortical_col(row):
        """获取该行皮质线列坐标（从第1条线平移插值）"""
        rows_arr = np.array(sorted(row_to_cortical_col.keys()))
        nearest  = int(rows_arr[int(np.argmin(np.abs(rows_arr - row)))])
        return row_to_cortical_col[nearest]

    # 4. 双线共识验证：±tol_px 范围内 ≥2条线有同类型候选点 → 确认
    confirmed_inferior = []  # inferior（红色） 椎体下终板 Inferior EP
    confirmed_superior = []  # superior（绿色） 椎体上终板 Superior EP

    for row in all_cand_rows:
        for ep_type, color_label, confirm_list in [
            ('inferior', '红X(下终板 Inferior)', confirmed_inferior),
            ('superior', '绿X(上终板 Superior)', confirmed_superior),
        ]:
            contributing_lines = set()
            for r2 in all_cand_rows:
                if abs(r2 - row) <= tol_px:
                    for c2, li2 in cands_by_row.get(r2, {}).get(ep_type, []):
                        contributing_lines.add(li2)

            if len(contributing_lines) >= 2:
                cc = get_cortical_col(row)
                # 防止重复：±tol_px 内已有确认点则跳过
                if not any(abs(r3 - row) <= tol_px for r3, _ in confirm_list):
                    confirm_list.append((row, cc))
                    print(f"   [椎体后缘] {color_label} 共识 @ row={row}, col={cc} "
                          f"(支持线={len(contributing_lines)})")

    # 4.1 fallback：若某类型无共识结果，取该类型所有候选点中行号最小的点平移到皮质线
    for ep_type, color_label, confirm_list in [
        ('inferior', '红X(下终板 Inferior)', confirmed_inferior),
        ('superior', '绿X(上终板 Superior)', confirmed_superior),
    ]:
        if not confirm_list:
            # 收集该类型所有候选点
            all_type_cands = []
            for r2 in all_cand_rows:
                for c2, li2 in cands_by_row.get(r2, {}).get(ep_type, []):
                    all_type_cands.append((r2, c2))
            if all_type_cands:
                # 取行号最小的点（最靠头侧）
                fb_row, fb_col = min(all_type_cands, key=lambda x: x[0])
                fb_cc = get_cortical_col(fb_row)
                confirm_list.append((fb_row, fb_cc))
                print(f"   [椎体后缘 fallback] {color_label} "
                      f"无共识，取行号最小候选点 @ row={fb_row}, col={fb_cc}")

    # 4.5 终板锚点校验（与 endplates_f 比对）
    # 校验规则：
    #   step1 颜色校验：候选点在 ±tol_anchor_px 行内必须有同类型 endplates_f 锚点
    #   step2 空间去重：±tol_anchor_px 内同色多点只保留最近锚点的那个
    #   step3 交替校验：从下往上排列，强制红→绿→红→绿，连续同色剔除支持线少的
    tol_anchor_px = max(1, int(5.0 / pixel_spacing))  # ±5mm
    if endplates_f:
        # 建立锚点查找表 {ep_type: [row, ...]}
        anchor_inferior = sorted([ep[0] for ep in endplates_f if ep[3] == 'inferior'])
        anchor_superior = sorted([ep[0] for ep in endplates_f if ep[3] == 'superior'])

        def has_anchor(row, anchor_rows):
            return any(abs(row - ar) <= tol_anchor_px for ar in anchor_rows)

        # step1+2：颜色校验 + 空间去重（保留最近锚点的那个）
        def filter_by_anchor(cand_list, anchor_rows, label):
            kept = []
            # 按与锚点距离升序排列，贪心保留
            scored = []
            for row, col in cand_list:
                min_dist = min((abs(row - ar) for ar in anchor_rows), default=999)
                if min_dist <= tol_anchor_px:
                    scored.append((min_dist, row, col))
            scored.sort(key=lambda x: x[0])
            used_rows = []
            for dist, row, col in scored:
                # 空间去重：已保留点的 ±tol_anchor_px 内不再加入
                if not any(abs(row - ur) <= tol_anchor_px for ur in used_rows):
                    kept.append((row, col))
                    used_rows.append(row)
                    print(f"   [校验通过] {label} @ row={row} 锚点距={dist:.0f}px")
                else:
                    print(f"   [空间去重] {label} @ row={row} 已有近邻保留")
            removed = len(cand_list) - len(kept)
            if removed:
                print(f"   [颜色校验] {label} 剔除 {removed} 个无锚点匹配点")
            return kept

        confirmed_inferior = filter_by_anchor(confirmed_inferior, anchor_inferior, 'inferior(红)')
        confirmed_superior = filter_by_anchor(confirmed_superior, anchor_superior, 'superior(绿)')
        print(f"   [校验后] inferior={len(confirmed_inferior)} superior={len(confirmed_superior)}")
    
        # step3：交替校验 inferior（行大,下终板）→ superior（行小,上终板）交替
        # 合并排序：从下往上（行号降序），期望序列: inferior → superior → inferior → superior ...
        merged = sorted(
            [('inferior', r, c) for r, c in confirmed_inferior] +
            [('superior', r, c) for r, c in confirmed_superior],
            key=lambda x: -x[1]  # 行号降序 = 从下往上
        )
        # 统计每个点支持线数（用于同色竞争时剪除支持少的）
        def get_support(row, ep_type):
            cnt = 0
            for r2 in all_cand_rows:
                if abs(r2 - row) <= tol_px:
                    for c2, li2 in cands_by_row.get(r2, {}).get(ep_type, []):
                        cnt += 1
            return cnt
    
        changed = True
        while changed:
            changed = False
            for k in range(len(merged) - 1):
                t_a, r_a, c_a = merged[k]
                t_b, r_b, c_b = merged[k + 1]
                if t_a == t_b:  # 连续同色
                    sup_a = get_support(r_a, t_a)
                    sup_b = get_support(r_b, t_b)
                    drop_idx = k if sup_a <= sup_b else k + 1
                    t_d, r_d, _ = merged[drop_idx]
                    print(f"   [交替剪除] {t_d} @ row={r_d} 支持={get_support(r_d,t_d)}")
                    merged.pop(drop_idx)
                    changed = True
                    break
    
        # 重建校验后的列表
        confirmed_inferior = [(r, c) for t, r, c in merged if t == 'inferior']
        confirmed_superior = [(r, c) for t, r, c in merged if t == 'superior']
        print(f"   [交替校验后] inferior={len(confirmed_inferior)} superior={len(confirmed_superior)}")
    
    # 5. 配对：从下往上，每个 inferior（红色,在下）配其上方最近 superior（绿色,在上）
    confirmed_superior_s = sorted(confirmed_superior, key=lambda x: -x[0])  # 行号降序
    confirmed_inferior_s = sorted(confirmed_inferior, key=lambda x: -x[0])
    centers   = []
    used_superior = set()
    for ur, uc in confirmed_inferior_s:   # 红色，行号大，在下
        best_lr, best_lc, best_i = None, None, None
        best_dist = float('inf')
        for i, (lr, lc) in enumerate(confirmed_superior_s):
            if lr < ur and i not in used_superior:   # 绿色在红色上方
                dist = ur - lr
                if dist < best_dist:
                    best_dist, best_lr, best_lc, best_i = dist, lr, lc, i
        if best_lr is not None:
            used_superior.add(best_i)
            # 中心行号：两角点行号均値
            cr_float = (ur + best_lr) / 2.0
            # 中心列号：从皮质线插値（不是两列均値）
            cc_precise = get_cortical_col(cr_float)
            centers.append((cr_float, cc_precise))
            print(f"   [椎体后缘] 中心点 @ row={cr_float:.1f}, col={cc_precise} "
                  f"(BP_bot行={ur}, BP_top行={best_lr})")

    print(f"   [氿体右边缘] 共识别 {len(centers)} 个氿体右边缘")
    return {
        'inferior_edges': confirmed_inferior_s,
        'superior_edges': confirmed_superior_s,
        'centers':        centers,
    }



def sliding_window_cluster_endplates(raw_cands, scan_lines_f, pixel_spacing,
                                      win_mm=5.0, min_lines=7):
    """
    V12.4 压水图红绿三角候选点 → 滑动窗口聚类 → 终板线

    raw_cands 格式: [(row, col, type_str, depth, line_idx), ...]
      type_str: 'inferior'（红/下终板 Inferior EP）/ 'superior'（绿/上终板 Superior EP）
      line_idx: 对应 scan_lines_f 中的序号

    滑动窗口定义：
      - 高度 = win_mm（5mm），转像素＝ win_px
      - 窗口左边界 = 最左扫描线（offset最小）列坐标 - 28mm安全余量
      - 窗口右边界 = 最大offset扫描线列坐标最大值
      - 从图像最小行到最大行逐行向下滑动
      - 窗口内按 upper/lower 分别统计有多少条扫描线有候选点
      - 有效扫描线数 >= min_lines(7) → 确认为一条终板线
      - 相同的点只能被归属一条终板线

    返回: [{'ep_type': 'superior'/'inferior',
              'points': [(row, col, depth, off_mm), ...],
              'row_center': float}, ...]
    """
    if not raw_cands or not scan_lines_f:
        return []

    win_px = max(3, int(win_mm / pixel_spacing))
    left_extra_px  = max(1, int(28.0 / pixel_spacing))  # 左側 28mm 安全余量
    min_peak_sep_px = max(4, int(8.0 / pixel_spacing))   # 峰间最小间距 8mm

    # 计算列窗口范围
    # 扫描线方向：皮质线 → 腔内（offset增大 = 列坐标减小）
    # offset最小（2mm）= 最靠近皮质线 = 列坐标最大（右边界）
    # offset最大（26mm）= 最远离皮质线 = 列坐标最小（左边界 - 28mm）
    offsets_all = [sl[0] for sl in scan_lines_f]
    min_off_idx = int(np.argmin(offsets_all))  # offset最小 = 最靠近皮质线 = 列坐标最大
    max_off_idx = int(np.argmax(offsets_all))  # offset最大 = 最远离皮质线 = 列坐标最小
    col_right = int(np.max(scan_lines_f[min_off_idx][1]))   # 右边界：最靠近皮质线的最大列
    col_left  = int(np.min(scan_lines_f[max_off_idx][1])) - left_extra_px  # 左边界：最远离皮质线的最小列 - 28mm
    col_left  = max(0, col_left)
    print(f"   [V12.4列窗口] col=[{col_left}, {col_right}] 左余量=28mm")

    # 按 type 分组候选点，过滤列范围外的点
    pts_by_type_line = {'upper': {}, 'lower': {}}
    for r, c, t, d, li in raw_cands:
        if t not in pts_by_type_line:
            continue
        if int(c) < col_left or int(c) > col_right:
            continue  # 列坐标超出窗口范围，过滤
        if li not in pts_by_type_line[t]:
            pts_by_type_line[t][li] = []
        pts_by_type_line[t][li].append((int(r), int(c), float(d)))

    all_rows_all = [r for r, c, t, d, li in raw_cands]
    if not all_rows_all:
        return []
    r_min = int(min(all_rows_all))
    r_max = int(max(all_rows_all))

    result_lines = []

    for ep_type in ('superior', 'inferior'):
        line_pts = pts_by_type_line[ep_type]  # {line_idx: [(row,col,depth), ...]}
        if not line_pts:
            continue

        # 建立各扫描线排序行号数组
        rows_index = {}
        for li, pts in line_pts.items():
            rows_index[li] = sorted([p[0] for p in pts])

        # 计算每个起始行 r 的覆盖线数
        cover_arr = {}
        for r in range(r_min, r_max + 1):
            cnt = sum(
                1 for li in rows_index
                if any(r <= row <= r + win_px for row in rows_index[li])
            )
            cover_arr[r] = cnt

        # 找满足 >= min_lines 的峰区段
        peaks = []
        in_peak = False
        seg_start = r_min
        for r in range(r_min, r_max + 1):
            if cover_arr[r] >= min_lines:
                if not in_peak:
                    seg_start = r
                    in_peak = True
            else:
                if in_peak:
                    peak_r = max(range(seg_start, r), key=lambda x: cover_arr[x])
                    peaks.append(peak_r)
                    in_peak = False
        if in_peak:
            peak_r = max(range(seg_start, r_max + 1), key=lambda x: cover_arr.get(x, 0))
            peaks.append(peak_r)

        # 合并过近的峰
        merged_peaks = []
        for pk in sorted(peaks):
            if merged_peaks and abs(pk - merged_peaks[-1]) < min_peak_sep_px:
                if cover_arr.get(pk, 0) > cover_arr.get(merged_peaks[-1], 0):
                    merged_peaks[-1] = pk
            else:
                merged_peaks.append(pk)

        print(f"   [V12.4滑动窗口] {ep_type} 峰: {len(merged_peaks)} 个, 行号={merged_peaks}")

        # 对每个峰收集窗口内点构建终板线
        # 相同的点（line_idx, row）只能被归属一条终板线
        used_pt_keys = set()
        for peak_row in merged_peaks:
            win_pts = []  # [(row, col, depth, off_mm), ...]
            for li, pts in line_pts.items():
                off_mm = scan_lines_f[li][0] if li < len(scan_lines_f) else li * 2.0
                for (row, col, depth) in pts:
                    key = (li, row)
                    if (row >= peak_row
                            and row <= peak_row + win_px   # 与 cover_arr 统计窗口完全一致
                            and key not in used_pt_keys):
                        win_pts.append((row, col, depth, off_mm))
                        used_pt_keys.add(key)

            covered = len(set(p[3] for p in win_pts))  # 不同 off_mm 数
            if len(win_pts) < min_lines or covered < min_lines:
                continue

            row_center = float(np.mean([p[0] for p in win_pts]))
            result_lines.append({
                'ep_type':    ep_type,
                'points':     win_pts,
                'row_center': row_center,
                'covered':    covered,
            })

    result_lines.sort(key=lambda x: x['row_center'])
    return result_lines


