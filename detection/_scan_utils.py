"""
detection 模块公共工具：皮质线2投影和法线扫描函数
"""
import numpy as np
import math


def _project_to_c2(r0, c0, c2_rows, c2_cols):
    """
    将点 (r0, c0) 投影到皮质线2，返回最近点和该点的局部法线方向。

    法线方向 = 投影点处局部切线顺时针旋转90°，与 build_scan_lines_v15 一致：
      切线 (t_dr, t_dc) → 法线 (nx, ny) = (t_dc, -t_dr)
      法线朝向腹侧（ny < 0，即列减小方向）

    参数：
        r0, c0   - 点坐标
        c2_rows  - 皮质线2行坐标数组
        c2_cols  - 皮质线2列坐标数组

    返回：
        (r_c2, c_c2, n_row, n_col)
    """
    c2r = np.array(c2_rows, dtype=np.float64)
    c2c = np.array(c2_cols, dtype=np.float64)

    dists = np.sqrt((c2r - r0) ** 2 + (c2c - c0) ** 2)
    idx   = int(np.argmin(dists))
    r_c2  = float(c2r[idx])
    c_c2  = float(c2c[idx])

    i_prev = max(0, idx - 1)
    i_next = min(len(c2r) - 1, idx + 1)
    t_dr = float(c2r[i_next] - c2r[i_prev])
    t_dc = float(c2c[i_next] - c2c[i_prev])
    t_len = float(np.sqrt(t_dr * t_dr + t_dc * t_dc))
    if t_len < 1e-6:
        t_dr, t_dc, t_len = 1.0, 0.0, 1.0
    t_dr /= t_len
    t_dc /= t_len

    n_row = t_dc
    n_col = -t_dr
    if n_col > 0:
        n_row, n_col = -n_row, -n_col

    return r_c2, c_c2, n_row, n_col


def _arc_coord(r0, c0, c3_rows, c3_cols, arc_len_mm):
    """
    将点 (r0, c0) 投影到皮质线3，返回法线投影点处的弧长值 (mm)。

    法线投影：对皮质线3上每个点计算局部切线，取点在切线方向上的偏差（切线分量）
    绝对值最小的点作为法线投影点。若皮质线3点不足3个则退化为欧氏最近点。

    算法：
        对第 i 个皮质线3点，切线方向 (t_dr, t_dc) 由相邻点差分得到。
        切线分量 = (r0 - c3r[i]) * t_dr + (c0 - c3c[i]) * t_dc
        |切线分量| 最小 → 法线投影点最近。

    参数：
        r0, c0      - 点坐标
        c3_rows     - 皮质线3行坐标数组
        c3_cols     - 皮质线3列坐标数组
        arc_len_mm  - 皮质线3弧长数组 (mm)，与 c3_rows/c3_cols 等长

    返回：
        float  弧长值 (mm)
    """
    c3r = np.array(c3_rows, dtype=np.float64)
    c3c = np.array(c3_cols, dtype=np.float64)
    n = len(c3r)

    if n < 3:
        # 退化：欧氏最近点
        dists = np.sqrt((c3r - r0) ** 2 + (c3c - c0) ** 2)
        return float(arc_len_mm[int(np.argmin(dists))])

    # 计算每个点的局部切线方向（差分，两端用单侧）
    t_dr = np.empty(n, dtype=np.float64)
    t_dc = np.empty(n, dtype=np.float64)
    t_dr[1:-1] = c3r[2:] - c3r[:-2]
    t_dc[1:-1] = c3c[2:] - c3c[:-2]
    t_dr[0]    = c3r[1]  - c3r[0]
    t_dc[0]    = c3c[1]  - c3c[0]
    t_dr[-1]   = c3r[-1] - c3r[-2]
    t_dc[-1]   = c3c[-1] - c3c[-2]
    t_len = np.sqrt(t_dr ** 2 + t_dc ** 2)
    t_len = np.where(t_len < 1e-9, 1.0, t_len)
    t_dr /= t_len
    t_dc /= t_len

    # 切线分量（点到皮质线3各点的向量在切线方向的投影）
    dr = r0 - c3r
    dc = c0 - c3c
    tang_proj = np.abs(dr * t_dr + dc * t_dc)

    idx = int(np.argmin(tang_proj))
    return float(arc_len_mm[idx])


def _scan_normal_descent(in_img_2d, start_r, start_c,
                         dr, dc, scan_mm, pixel_spacing,
                         in_low_mean, in_high_mean,
                         drop_ratio=0.25, low_ratio=1.3, smooth_win=2):
    """
    从 (start_r, start_c) 沿 (dr, dc) 方向扫描 scan_mm，找第一个信号下降沿。
    Step1：cur < ref*(1-drop_ratio)，ref=前2点滑动均值（下降沿触发）
    Step2确认：取当前像素左右两列（col-1,col 与 col,col+1 两组），任一满足
              region_mean < in_high_mean*0.55 OR region_mean < in_low_mean*low_ratio

    返回：(found_r, found_c) 或 None
    """
    H, W = in_img_2d.shape
    scan_px = int(round(scan_mm / pixel_spacing))
    n_samples = scan_px + 1

    seg_raw = np.zeros(n_samples, dtype=np.float32)
    for t in range(n_samples):
        rf = start_r + dr * t
        cf = start_c + dc * t
        ri0 = int(np.clip(int(np.floor(rf)), 0, H - 1))
        ri1 = int(np.clip(ri0 + 1, 0, H - 1))
        ci0 = int(np.clip(int(np.floor(cf)), 0, W - 1))
        ci1 = int(np.clip(ci0 + 1, 0, W - 1))
        wr = rf - np.floor(rf)
        wc = cf - np.floor(cf)
        seg_raw[t] = (float(in_img_2d[ri0, ci0]) * (1 - wr) * (1 - wc)
                    + float(in_img_2d[ri1, ci0]) * wr * (1 - wc)
                    + float(in_img_2d[ri0, ci1]) * (1 - wr) * wc
                    + float(in_img_2d[ri1, ci1]) * wr * wc)

    if n_samples < smooth_win + 2:
        return None

    seg_sm = np.convolve(seg_raw, np.ones(smooth_win) / smooth_win, mode='same')

    ref_locked = None  # Step2失败时锁定ref，防止ref随低值漂移
    for t in range(1, len(seg_sm)):
        cur = float(seg_sm[t])
        # Step1: 下降沿触发，ref = 锁定值 或 前2点滑动均值
        if ref_locked is None:
            ref = float(np.mean(seg_sm[max(0, t - 2):t])) if t >= 2 else float(seg_sm[0])
            ref = max(ref, 1.0)
        else:
            # 锁定模式：信号回升超过 ref_locked*(1-drop_ratio/2) 则解锁
            if cur >= ref_locked * (1.0 - drop_ratio / 2.0):
                ref_locked = None
                ref = float(np.mean(seg_sm[max(0, t - 2):t])) if t >= 2 else float(seg_sm[0])
                ref = max(ref, 1.0)
            else:
                ref = ref_locked
        if not (cur < ref * (1.0 - drop_ratio)):
            continue
        found_r = float(np.clip(start_r + dr * t, 0, H - 1))
        found_c = float(np.clip(start_c + dc * t, 0, W - 1))
        r_int = int(round(found_r))
        c_int = int(round(found_c))
        c_lo = max(0, c_int - 1)
        c_hi = min(W - 1, c_int + 1)
        region_mean_a = float(np.mean([in_img_2d[r_int, c_lo], in_img_2d[r_int, c_int]]))
        region_mean_b = float(np.mean([in_img_2d[r_int, c_int], in_img_2d[r_int, c_hi]]))
        # Step2: 横向两点组合任一满足
        ok_a = (region_mean_a < in_high_mean * 0.55 or region_mean_a < in_low_mean * low_ratio)
        ok_b = (region_mean_b < in_high_mean * 0.55 or region_mean_b < in_low_mean * low_ratio)
        if ok_a or ok_b:
            return (found_r, found_c)
        # Step2失败：锁定当前ref，防止基准随低值漂移
        ref_locked = ref

    return None


def _scan_normal_descent_diag(in_img_2d, start_r, start_c,
                               dr, dc, scan_mm, pixel_spacing,
                               in_low_mean, in_high_mean,
                               drop_ratio=0.25, low_ratio=1.3, smooth_win=2):
    """
    终板专用下降沿扫描（对角信号确认版）。
    Step1：cur < ref*(1-drop_ratio)，ref=前2点滑动均值（下降沿触发）
    Step2确认：取当前像素 (r,c) 配右上 (r-1,c+1) 或 左下 (r+1,c-1)，任一满足
              region_mean < in_high_mean*0.55 OR region_mean < in_low_mean*low_ratio
    用于最后一个椎体后缘夹角 < 65° 时的上/下终板扫描。

    返回：(found_r, found_c) 或 None
    """
    H, W = in_img_2d.shape
    scan_px = int(round(scan_mm / pixel_spacing))
    n_samples = scan_px + 1

    seg_raw = np.zeros(n_samples, dtype=np.float32)
    for t in range(n_samples):
        rf = start_r + dr * t
        cf = start_c + dc * t
        ri0 = int(np.clip(int(np.floor(rf)), 0, H - 1))
        ri1 = int(np.clip(ri0 + 1, 0, H - 1))
        ci0 = int(np.clip(int(np.floor(cf)), 0, W - 1))
        ci1 = int(np.clip(ci0 + 1, 0, W - 1))
        wr = rf - np.floor(rf)
        wc = cf - np.floor(cf)
        seg_raw[t] = (float(in_img_2d[ri0, ci0]) * (1 - wr) * (1 - wc)
                    + float(in_img_2d[ri1, ci0]) * wr * (1 - wc)
                    + float(in_img_2d[ri0, ci1]) * (1 - wr) * wc
                    + float(in_img_2d[ri1, ci1]) * wr * wc)

    if n_samples < smooth_win + 2:
        return None

    seg_sm = np.convolve(seg_raw, np.ones(smooth_win) / smooth_win, mode='same')

    ref_locked = None  # Step2失败时锁定ref，防止ref随低值漂移
    for t in range(1, len(seg_sm)):
        cur = float(seg_sm[t])
        # Step1: 下降沿触发，ref = 锁定值 或 前2点滑动均值
        if ref_locked is None:
            ref = float(np.mean(seg_sm[max(0, t - 2):t])) if t >= 2 else float(seg_sm[0])
            ref = max(ref, 1.0)
        else:
            # 锁定模式：信号回升超过 ref_locked*(1-drop_ratio/2) 则解锁
            if cur >= ref_locked * (1.0 - drop_ratio / 2.0):
                ref_locked = None
                ref = float(np.mean(seg_sm[max(0, t - 2):t])) if t >= 2 else float(seg_sm[0])
                ref = max(ref, 1.0)
            else:
                ref = ref_locked
        if not (cur < ref * (1.0 - drop_ratio)):
            continue
        found_r = float(np.clip(start_r + dr * t, 0, H - 1))
        found_c = float(np.clip(start_c + dc * t, 0, W - 1))
        r_int = int(round(found_r))
        c_int = int(round(found_c))
        # / 方向对角：当前点 (r,c) 配右上 (r-1,c+1) 或 左下 (r+1,c-1)，任一满足
        r_up = max(0, r_int - 1);  c_up = min(W - 1, c_int + 1)
        r_dn = min(H - 1, r_int + 1); c_dn = max(0, c_int - 1)
        region_mean_a = float(np.mean([in_img_2d[r_int, c_int], in_img_2d[r_up, c_up]]))
        region_mean_b = float(np.mean([in_img_2d[r_int, c_int], in_img_2d[r_dn, c_dn]]))
        ok_a = (region_mean_a < in_high_mean * 0.55 or region_mean_a < in_low_mean * low_ratio)
        ok_b = (region_mean_b < in_high_mean * 0.55 or region_mean_b < in_low_mean * low_ratio)
        if ok_a or ok_b:
            return (found_r, found_c)
        # Step2失败：锁定当前ref，防止基准随低值漂移
        ref_locked = ref

    return None


def _scan_normal_descent_ant_diag(in_img_2d, start_r, start_c,
                                   dr, dc, scan_mm, pixel_spacing,
                                   in_low_mean, in_high_mean,
                                   drop_ratio=0.25, low_ratio=1.3, smooth_win=2):
    """
    前缘专用下降沿扫描（单信号确认版）。
    Step1：cur < ref*(1-drop_ratio)，ref=前2点滑动均值（下降沿触发）
    Step2确认：当前像素信号值 < in_high_mean*0.55 OR < in_low_mean*low_ratio
    用于最后一个椎体后缘夹角 < 65° 时的前缘扫描。

    返回：(found_r, found_c) 或 None
    """
    H, W = in_img_2d.shape
    scan_px = int(round(scan_mm / pixel_spacing))
    n_samples = scan_px + 1

    seg_raw = np.zeros(n_samples, dtype=np.float32)
    for t in range(n_samples):
        rf = start_r + dr * t
        cf = start_c + dc * t
        ri0 = int(np.clip(int(np.floor(rf)), 0, H - 1))
        ri1 = int(np.clip(ri0 + 1, 0, H - 1))
        ci0 = int(np.clip(int(np.floor(cf)), 0, W - 1))
        ci1 = int(np.clip(ci0 + 1, 0, W - 1))
        wr = rf - np.floor(rf)
        wc = cf - np.floor(cf)
        seg_raw[t] = (float(in_img_2d[ri0, ci0]) * (1 - wr) * (1 - wc)
                    + float(in_img_2d[ri1, ci0]) * wr * (1 - wc)
                    + float(in_img_2d[ri0, ci1]) * (1 - wr) * wc
                    + float(in_img_2d[ri1, ci1]) * wr * wc)

    if n_samples < smooth_win + 2:
        return None

    seg_sm = np.convolve(seg_raw, np.ones(smooth_win) / smooth_win, mode='same')

    ref_locked = None  # Step2失败时锁定ref，防止ref随低值漂移
    for t in range(1, len(seg_sm)):
        cur = float(seg_sm[t])
        # Step1: 下降沿触发，ref = 锁定值 或 前2点滑动均值
        if ref_locked is None:
            ref = float(np.mean(seg_sm[max(0, t - 2):t])) if t >= 2 else float(seg_sm[0])
            ref = max(ref, 1.0)
        else:
            # 锁定模式：信号回升超过 ref_locked*(1-drop_ratio/2) 则解锁
            if cur >= ref_locked * (1.0 - drop_ratio / 2.0):
                ref_locked = None
                ref = float(np.mean(seg_sm[max(0, t - 2):t])) if t >= 2 else float(seg_sm[0])
                ref = max(ref, 1.0)
            else:
                ref = ref_locked
        if not (cur < ref * (1.0 - drop_ratio)):
            continue
        found_r = float(np.clip(start_r + dr * t, 0, H - 1))
        found_c = float(np.clip(start_c + dc * t, 0, W - 1))
        r_int = int(round(found_r))
        c_int = int(round(found_c))
        # 单信号确认：仅检查当前像素
        region_val = float(in_img_2d[r_int, c_int])
        if region_val < in_high_mean * 0.55 or region_val < in_low_mean * low_ratio:
            return (found_r, found_c)
        # Step2失败：锁定当前ref，防止基准随低值漂移
        ref_locked = ref

    return None


def _scan_normal_descent_ant(in_img_2d, start_r, start_c,
                              dr, dc, scan_mm, pixel_spacing,
                              in_low_mean, in_high_mean,
                              drop_ratio=0.25, low_ratio=1.3, smooth_win=2):
    """
    前缘专用下降沿扫描。
    Step1：cur < ref*(1-drop_ratio)，ref=前2点滑动均值（下降沿触发）
    Step2确认：取当前像素上下各1行（row-1,row 与 row,row+1 两组），任一满足
              region_mean < in_high_mean*0.55 OR region_mean < in_low_mean*low_ratio

    返回：(found_r, found_c) 或 None
    """
    H, W = in_img_2d.shape
    scan_px = int(round(scan_mm / pixel_spacing))
    n_samples = scan_px + 1

    seg_raw = np.zeros(n_samples, dtype=np.float32)
    for t in range(n_samples):
        rf = start_r + dr * t
        cf = start_c + dc * t
        ri0 = int(np.clip(int(np.floor(rf)), 0, H - 1))
        ri1 = int(np.clip(ri0 + 1, 0, H - 1))
        ci0 = int(np.clip(int(np.floor(cf)), 0, W - 1))
        ci1 = int(np.clip(ci0 + 1, 0, W - 1))
        wr = rf - np.floor(rf)
        wc = cf - np.floor(cf)
        seg_raw[t] = (float(in_img_2d[ri0, ci0]) * (1 - wr) * (1 - wc)
                    + float(in_img_2d[ri1, ci0]) * wr * (1 - wc)
                    + float(in_img_2d[ri0, ci1]) * (1 - wr) * wc
                    + float(in_img_2d[ri1, ci1]) * wr * wc)

    if n_samples < smooth_win + 2:
        return None

    seg_sm = np.convolve(seg_raw, np.ones(smooth_win) / smooth_win, mode='same')

    ref_locked = None  # Step2失败时锁定ref，防止ref随低值漂移
    for t in range(1, len(seg_sm)):
        cur = float(seg_sm[t])
        # Step1: 下降沿触发，ref = 锁定值 或 前2点滑动均值
        if ref_locked is None:
            ref = float(np.mean(seg_sm[max(0, t - 2):t])) if t >= 2 else float(seg_sm[0])
            ref = max(ref, 1.0)
        else:
            # 锁定模式：信号回升超过 ref_locked*(1-drop_ratio/2) 则解锁
            if cur >= ref_locked * (1.0 - drop_ratio / 2.0):
                ref_locked = None
                ref = float(np.mean(seg_sm[max(0, t - 2):t])) if t >= 2 else float(seg_sm[0])
                ref = max(ref, 1.0)
            else:
                ref = ref_locked
        if not (cur < ref * (1.0 - drop_ratio)):
            continue
        found_r = float(np.clip(start_r + dr * t, 0, H - 1))
        found_c = float(np.clip(start_c + dc * t, 0, W - 1))
        r_int = int(round(found_r))
        c_int = int(round(found_c))
        r_lo = max(0, r_int - 1)
        r_hi = min(H - 1, r_int + 1)
        region_mean_a = float(np.mean([in_img_2d[r_lo, c_int], in_img_2d[r_int, c_int]]))
        region_mean_b = float(np.mean([in_img_2d[r_int, c_int], in_img_2d[r_hi, c_int]]))
        # Step2: 纵向两点组合任一满足
        ok_a = (region_mean_a < in_high_mean * 0.55 or region_mean_a < in_low_mean * low_ratio)
        ok_b = (region_mean_b < in_high_mean * 0.55 or region_mean_b < in_low_mean * low_ratio)
        if ok_a or ok_b:
            return (found_r, found_c)
        # Step2失败：锁定当前ref，防止基准随低值漂移
        ref_locked = ref

    return None


def _scan_rise_ascent(in_img_2d, start_r, start_c,
                       dr, dc, scan_mm, pixel_spacing,
                       in_high_mean,
                       drop_ratio=0.25, smooth_win=2):
    """
    从 (start_r, start_c) 沿 (dr, dc) 方向扫描 scan_mm，找终板信号谷底。

    Step1：滑动窗口（大小=smooth_win）找局部极小值：
           seg[t] <= seg[t-1] 且 seg[t] <= seg[t+1]
    Step2确认：从极小值点继续向前取3px，均值 > in_high_mean * 0.4
    满足则返回极小值点坐标，否则继续找下一个极小值。

    返回：(found_r, found_c) 或 None
    """
    H, W = in_img_2d.shape
    scan_px = int(round(scan_mm / pixel_spacing))
    n_samples = scan_px + 1

    seg_raw = np.zeros(n_samples, dtype=np.float32)
    for t in range(n_samples):
        rf = start_r + dr * t
        cf = start_c + dc * t
        ri0 = int(np.clip(int(np.floor(rf)), 0, H - 1))
        ri1 = int(np.clip(ri0 + 1, 0, H - 1))
        ci0 = int(np.clip(int(np.floor(cf)), 0, W - 1))
        ci1 = int(np.clip(ci0 + 1, 0, W - 1))
        wr = rf - np.floor(rf)
        wc = cf - np.floor(cf)
        seg_raw[t] = (float(in_img_2d[ri0, ci0]) * (1 - wr) * (1 - wc)
                    + float(in_img_2d[ri1, ci0]) * wr * (1 - wc)
                    + float(in_img_2d[ri0, ci1]) * (1 - wr) * wc
                    + float(in_img_2d[ri1, ci1]) * wr * wc)

    if n_samples < smooth_win + 4:
        return None

    seg_sm = np.convolve(seg_raw, np.ones(smooth_win) / smooth_win, mode='same')

    # 遍历找局部极小值（排除首尾各1个点，确保有前后邻居）
    for t in range(1, len(seg_sm) - 1):
        # Step1：局部极小值判断
        if not (seg_sm[t] <= seg_sm[t - 1] and seg_sm[t] <= seg_sm[t + 1]):
            continue
        # Step2：向前取3px（沿扫描方向继续），确认为高信号椎体区域
        t3_end = min(t + 4, len(seg_sm))   # t+1, t+2, t+3
        if t3_end - (t + 1) < 1:
            continue
        forward_mean = float(np.mean(seg_sm[t + 1:t3_end]))
        if forward_mean > in_high_mean * 0.4:
            found_r = float(np.clip(start_r + dr * t, 0, H - 1))
            found_c = float(np.clip(start_c + dc * t, 0, W - 1))
            return (found_r, found_c)

    return None
