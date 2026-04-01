"""
解剖校正模块
包含 enforce_alternating、align_to_midline、
     anatomical_gap_correction、fill_missing_endplates
（这些函数原为 find_endplates_on_water_image 的内嵌函数，已提取为模块级函数）
"""
import numpy as np
from scipy import signal


def enforce_alternating(eps):
    """
    eps: [(rep_row, avg_col, avg_dep, ep_type, n_pts, curve_pts), ...] 按 rep_row 升序
    同类型冲突时，保留行号更小（更靠头侧）的，因为第一个遇到的信号更可能是真实终板
    """
    if not eps:
        return []
    result = [eps[0]]
    for cur in eps[1:]:
        prev = result[-1]
        if cur[3] == prev[3]:  # 同类型，冲突
            # 保留行号更小（更靠头侧）的终板，丢弃行号更大的
            # prev 行号 < cur 行号，所以默认保留 prev
            print(f"   交替校验: 丢弃同类型({cur[3]}) row={cur[0]}，保留头侧 row={prev[0]}")
        else:
            result.append(cur)
    return result

def align_to_midline(endplates, scan_lines_f):
    """
    将所有终板点的列坐标对齐到第7条（中心）扫描线对应行的列坐标。
    第7条线距皮质线约14mm，最接近椎体/椎间盘中心深度。
    对齐后所有点在同一条扫描线上，间距测量不含横向偏移误差。
    endplates: [(row, col, depth, ep_type), ...]
    返回:      [(row, col_aligned, depth, ep_type), ...]
    """
    if not scan_lines_f:
        return endplates
    mid_idx = len(scan_lines_f) // 2          # 13条线 → index=6（第7条）
    _, mid_cols, mid_rows = scan_lines_f[mid_idx]
    mid_rows_arr = np.array(mid_rows, dtype=np.int32)
    print(f"   [中心线对齐] 使用第{mid_idx+1}条扫描线 "
          f"(offset={scan_lines_f[mid_idx][0]:.0f}mm) 作为列坐标基准")
    aligned = []
    for (row, col, dep, ep_type) in endplates:
        # 在 mid_rows 中找最近的行索引
        idx = int(np.argmin(np.abs(mid_rows_arr - row)))
        new_col = int(mid_cols[idx])
        aligned.append((row, new_col, dep, ep_type))
    return aligned

def anatomical_gap_correction(endplates, all_sigs, scan_lines_f, pixel_spacing):
    """
    解剖间距复核与预测修正（V12.1重写）
    流程：
      1. MAD鲁棒统计 avg_disc_mm / avg_vert_mm（直线距离，mm）
      2. 对每个中间点做前后双侧检查：
         - 前后都异常 → 本点可疑，预测修正
         - 仅一侧异常 → 留给另一侧的点处理（本点不动）
      3. 修正：预测行 = 前邻稳定点行 + avg_gap_px
              在预测行±search_half范围内各扫描线找信号最低点，取中位数
      4. 修正点的列坐标从中心扫描线取（与 align_to_midline 一致）
      5. 修正后的点标记 predicted=True
    endplates: [(row, col, depth, ep_type), ...] 已对齐到中心线，严格交替
    返回: [(row, col, depth, ep_type, predicted), ...]
    """
    if len(endplates) < 3:
        return [(r, c, d, t, False) for r, c, d, t in endplates]

    # ---- 直线距离辅助函数 ----
    def dist_mm(ep_a, ep_b):
        dr = float(ep_b[0] - ep_a[0])
        dc = float(ep_b[1] - ep_a[1])
        return float(np.sqrt(dr**2 + dc**2)) * pixel_spacing

    # ---- 计算相邻间距并分类 ----
    disc_dists = []   # (i+1, dist_mm)  superior→inferior
    vert_dists = []   # (i+1, dist_mm)  inferior→superior
    for i in range(len(endplates) - 1):
        d = dist_mm(endplates[i], endplates[i + 1])
        if endplates[i][3] == 'superior' and endplates[i+1][3] == 'inferior':
            disc_dists.append((i + 1, d))
        elif endplates[i][3] == 'inferior' and endplates[i+1][3] == 'superior':
            vert_dists.append((i + 1, d))

    def robust_mean_mm(dist_list):
        """MAD鲁棒均值（mm），去除2.5σ离群后取均值"""
        vals = [d for _, d in dist_list]
        if not vals:
            return None
        med = float(np.median(vals))
        mad = float(np.median([abs(v - med) for v in vals]))
        valid = [v for v in vals if abs(v - med) <= 2.5 * (mad + 1e-6)]
        return float(np.mean(valid)) if valid else med

    avg_disc_mm = robust_mean_mm(disc_dists)
    avg_vert_mm = robust_mean_mm(vert_dists)

    if avg_disc_mm is None or avg_vert_mm is None:
        return [(r, c, d, t, False) for r, c, d, t in endplates]

    avg_disc_px = avg_disc_mm / pixel_spacing
    avg_vert_px = avg_vert_mm / pixel_spacing
    print(f"   [解剖间距] 椎间盘均距={avg_disc_mm:.1f}mm  "
          f"椎体均距={avg_vert_mm:.1f}mm（直线距离）")

    # 异常判定阈值（收紧）
    disc_lo, disc_hi = 0.40, 1.80
    vert_lo, vert_hi = 0.40, 1.80

    # ---- 获取中心扫描线用于列坐标赋值 ----
    mid_idx = len(scan_lines_f) // 2
    _, mid_cols, mid_rows = scan_lines_f[mid_idx]
    mid_rows_arr = np.array(mid_rows, dtype=np.int32)

    def col_at_row(target_row):
        idx = int(np.argmin(np.abs(mid_rows_arr - target_row)))
        return int(mid_cols[idx])

    # ---- 在信号上查找极值（不聚类，取各扫描线最低点中位数行）----
    n_rows_scan = len(scan_lines_f[0][2]) if scan_lines_f else 0

    def find_signal_min_row(target_row, search_half_px):
        if n_rows_scan == 0:
            return target_row
        r_lo = max(0, target_row - search_half_px)
        r_hi = min(n_rows_scan - 1, target_row + search_half_px)
        if r_lo >= r_hi:
            return target_row
        found = []
        for sig in all_sigs:
            seg = sig[r_lo: r_hi + 1]
            if len(seg) > 0:
                found.append(r_lo + int(np.argmin(seg)))
        return int(np.median(found)) if found else target_row

    # ---- 构建可变结果列表 ----
    result    = list(endplates)
    predicted = [False] * len(result)
    n = len(result)

    def gap_type(i):
        """返回 i→i+1 的间距类型: 'disc' / 'vert' / None"""
        if result[i][3] == 'superior' and result[i+1][3] == 'inferior':
            return 'disc'
        if result[i][3] == 'inferior' and result[i+1][3] == 'superior':
            return 'vert'
        return None

    def gap_ratio(i):
        """计算 i→i+1 的间距比（直线距离 / 对应均值）"""
        gt = gap_type(i)
        if gt is None:
            return 1.0
        d = dist_mm(result[i], result[i+1])
        avg = avg_disc_mm if gt == 'disc' else avg_vert_mm
        return d / (avg + 1e-6)

    def is_abnormal(ratio, gt):
        if gt == 'disc':
            return ratio > disc_hi or ratio < disc_lo
        if gt == 'vert':
            return ratio > vert_hi or ratio < vert_lo
        return False

    # ---- 逐点双侧检查（跳过首尾） ----
    for i in range(1, n - 1):
        gt_before = gap_type(i - 1)   # (i-1)→i 的间距类型
        gt_after  = gap_type(i)       # i→(i+1) 的间距类型
        if gt_before is None or gt_after is None:
            continue

        ratio_before = gap_ratio(i - 1)
        ratio_after  = gap_ratio(i)

        ab_before = is_abnormal(ratio_before, gt_before)
        ab_after  = is_abnormal(ratio_after,  gt_after)

        if ab_before and ab_after:
            # 本点前后间距都异常 → 本点自身可疑，预测修正
            # 用前邻稳定点推算期望位置
            avg_px = avg_disc_px if gt_before == 'disc' else avg_vert_px
            predicted_row = result[i - 1][0] + int(round(avg_px))
            search_half = max(int(avg_px * 0.4),
                              max(3, int(5.0 / pixel_spacing)))
            refined_row = find_signal_min_row(predicted_row, search_half)
            new_col     = col_at_row(refined_row)
            ep_type     = result[i][3]
            old_row     = result[i][0]
            print(f"   [解剖修正] {ep_type} row {old_row}→{refined_row} "
                  f"(前比={ratio_before:.2f} 后比={ratio_after:.2f})")
            result[i]    = (refined_row, new_col, result[i][2], ep_type)
            predicted[i] = True
        # 仅一侧异常 → 不动本点，留给对侧的点在它自己轮次处理

    # ---- 拼合 predicted 标记 ----
    return [(r, c, d, t, predicted[idx])
            for idx, (r, c, d, t) in enumerate(result)]

def fill_missing_endplates(eps_in, scan_lines_f, pixel_spacing):
    """
    检测相邻终板间距异常，推断并插入缺失点。
    eps_in: [(row,col,depth,ep_type,predicted), ...] 已按行号升序
    返回: 插补后的列表，新增点 predicted=True
    """
    if len(eps_in) < 2:
        return eps_in

    # 重新统计 disc/vert 间距均値
    disc_dists, vert_dists = [], []
    for i in range(len(eps_in) - 1):
        a, b = eps_in[i], eps_in[i + 1]
        dr = float(b[0] - a[0])
        dc = float(b[1] - a[1])
        d_mm = float(np.sqrt(dr**2 + dc**2)) * pixel_spacing
        if a[3] == 'superior' and b[3] == 'inferior':
            disc_dists.append(d_mm)
        elif a[3] == 'inferior' and b[3] == 'superior':
            vert_dists.append(d_mm)

    avg_disc = float(np.median(disc_dists)) if disc_dists else 8.0
    avg_vert = float(np.median(vert_dists)) if vert_dists else 28.0
    step_disc_px = avg_disc / pixel_spacing
    step_vert_px = avg_vert / pixel_spacing
    full_cycle_mm = avg_disc + avg_vert
    full_cycle_px = full_cycle_mm / pixel_spacing

    # 中心扫描线列坐标插値
    mid_idx = len(scan_lines_f) // 2
    _, mid_cols, mid_rows = scan_lines_f[mid_idx]
    mid_rows_arr = np.array(mid_rows, dtype=np.int32)

    def col_at_row(target_row):
        idx = int(np.argmin(np.abs(mid_rows_arr - target_row)))
        return int(mid_cols[idx])

    result = list(eps_in)
    i = 0
    while i < len(result) - 1:
        a, b = result[i], result[i + 1]
        dr = float(b[0] - a[0])
        dc = float(b[1] - a[1])
        d_mm = float(np.sqrt(dr**2 + dc**2)) * pixel_spacing

        if a[3] != b[3]:
            # 正常交替：检查间距是否过大（缺了整对）
            if d_mm > 1.8 * full_cycle_mm:
                n_missing = max(1, int(round(d_mm / full_cycle_mm)) - 1)
                for k in range(1, n_missing + 1):
                    # 按 a 的类型决定插入顺序
                    if a[3] == 'superior':
                        # superior→[缺 inferior]→[缺 superior]→inferior(b)
                        ins_row_1 = int(round(a[0] + k * full_cycle_px - step_disc_px / 2))
                        ins_row_2 = int(round(a[0] + k * full_cycle_px + step_disc_px / 2))
                        t1, t2 = 'inferior', 'superior'
                    else:
                        # inferior→[缺 superior]→[缺 inferior]→superior(b)
                        ins_row_1 = int(round(a[0] + k * full_cycle_px - step_vert_px / 2))
                        ins_row_2 = int(round(a[0] + k * full_cycle_px + step_vert_px / 2))
                        t1, t2 = 'superior', 'inferior'
                    offset = (k - 1) * 2
                    result.insert(i + offset + 1,
                                  (ins_row_1, col_at_row(ins_row_1), 0.0, t1, True))
                    result.insert(i + offset + 2,
                                  (ins_row_2, col_at_row(ins_row_2), 0.0, t2, True))
                    print(f"   [缺失插补] 对#{k}: {t1} row={ins_row_1}, {t2} row={ins_row_2}")
                i += 1 + n_missing * 2
                continue
        else:
            # 同类型相邻：中间缺一个对立类型
            missing_type = 'superior' if a[3] == 'inferior' else 'inferior'
            exp_px = step_vert_px if missing_type == 'superior' else step_disc_px
            ins_row = int(round(a[0] + exp_px))
            ins_col = col_at_row(ins_row)
            result.insert(i + 1, (ins_row, ins_col, 0.0, missing_type, True))
            color_ch = '红(下终板 Inferior)' if missing_type == 'inferior' else '绿(上终板 Superior)'
            print(f"   [缺失插补] {color_ch} predicted @ row={ins_row} "
                  f"(同类型相邻: {a[3]}→{b[3]})")
            continue  # 重新检查 (a, ins_pt)

        i += 1

    return result

