"""
signal_ref.py - Step1 信号参考值计算
"""
import math
import numpy as np
from ._scan_utils import _project_to_c2
from config.params import OFFSET_MM_SIGNAL


def compute_signal_references(in_img_2d, c2_rows, c2_cols, pixel_spacing,
                               offset_mm=OFFSET_MM_SIGNAL):
    """
    沿皮质线2每个点，沿法线方向偏移 offset_mm 处采样信号，
    统计 low_mean（低于中位数的均值）和 high_mean（90百分位）。

    返回：
        (low_mean, high_mean, profile_pts)
        profile_pts = [(r, c), ...] 剖面线坐标，用于可视化
    """
    H, W = in_img_2d.shape
    c2r = np.array(c2_rows, dtype=np.float64)
    c2c = np.array(c2_cols, dtype=np.float64)
    off_px = offset_mm / pixel_spacing

    profile_vals = []
    profile_pts  = []

    # 只取皮质线2前70%的点（头侧段），排除最后30%（S椎体/骶骨高信号区）
    n_c2 = len(c2r)
    n_use = max(1, int(round(n_c2 * 0.7)))

    for i in range(n_use):
        i_prev = max(0, i - 1)
        i_next = min(len(c2r) - 1, i + 1)
        t_dr = float(c2r[i_next] - c2r[i_prev])
        t_dc = float(c2c[i_next] - c2c[i_prev])
        t_len = math.sqrt(t_dr * t_dr + t_dc * t_dc) + 1e-9
        t_dr /= t_len; t_dc /= t_len
        n_row = t_dc; n_col = -t_dr
        if n_col > 0:
            n_row, n_col = -n_row, -n_col

        sr = float(c2r[i]) + n_row * off_px
        sc = float(c2c[i]) + n_col * off_px
        ri = int(np.clip(int(round(sr)), 0, H - 1))
        ci = int(np.clip(int(round(sc)), 0, W - 1))
        v = float(in_img_2d[ri, ci])
        profile_pts.append((sr, sc))
        if v > 0:
            profile_vals.append(v)

    if len(profile_vals) == 0:
        high_mean = 200.0
        low_mean  = 70.0
    else:
        nonzero_all = in_img_2d[in_img_2d > 0]
        high_mean = float(np.percentile(nonzero_all, 90)) if len(nonzero_all) > 0 else 200.0
        med = float(np.median(profile_vals))
        lows = [v for v in profile_vals if v < med]
        low_mean = float(np.mean(lows)) if lows else med * 0.5

    print(f"   [Step1] 剖面采样={len(profile_vals)}点(c2前70%={n_use}/{n_c2})  "
          f"low_mean={low_mean:.1f}  high_mean={high_mean:.1f}")
    return low_mean, high_mean, profile_pts
