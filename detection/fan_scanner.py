"""
fan_scanner.py - Step4 扇形扫描（上/下终板 + 前缘）
"""
import math
import numpy as np
from ._scan_utils import (_project_to_c2, _scan_normal_descent,
                          _scan_normal_descent_diag,
                          _scan_normal_descent_ant, _scan_normal_descent_ant_diag,
                          _scan_rise_ascent)


def _sample_ant_local_signal(in_img_2d, center_r, center_c,
                              ant_angle_deg, fan_step_deg, scan_mm,
                              pixel_spacing, start_radius_mm=2.0):
    """
    对前缘扇形中间 3 条射线逐像素采样，返回局部 (ant_low_mean, ant_high_mean)。
    """
    H, W = in_img_2d.shape
    scan_px = scan_mm / pixel_spacing
    start_px = start_radius_mm / pixel_spacing
    vals = []
    for ang_deg in [ant_angle_deg - fan_step_deg,
                    ant_angle_deg,
                    ant_angle_deg + fan_step_deg]:
        rad = math.radians(ang_deg)
        dr = math.sin(rad); dc = math.cos(rad)
        length = math.sqrt(dr*dr + dc*dc) + 1e-9
        dr /= length; dc /= length
        n_steps = int(round(scan_px - start_px))
        for t in range(n_steps + 1):
            rf = center_r + dr * (start_px + t)
            cf = center_c + dc * (start_px + t)
            ri = int(np.clip(int(round(rf)), 0, H - 1))
            ci = int(np.clip(int(round(cf)), 0, W - 1))
            v = float(in_img_2d[ri, ci])
            if v > 0:
                vals.append(v)
    if len(vals) < 4:
        return None, None
    arr = np.array(vals, dtype=np.float64)
    med = float(np.median(arr))
    lows = arr[arr < med]
    ant_low_mean  = float(np.mean(lows)) if len(lows) > 0 else med * 0.5
    ant_high_mean = float(np.percentile(arr, 90))
    return ant_low_mean, ant_high_mean


def _fan_scan_direction(center_r, center_c, base_angle_deg, fan_half_deg,
                        fan_step_deg, scan_mm, pixel_spacing,
                        in_img_2d, low_mean, high_mean,
                        drop_ratio=0.25, low_ratio=1.3,
                        scan_fn=None, start_radius_mm=2.0):
    """
    以 (center_r, center_c) 为原点扫描一个扇形，每 fan_step_deg 一条线。

    返回：(pts, dirs)
    """
    if scan_fn is None:
        scan_fn = _scan_normal_descent
    start_radius_px = start_radius_mm / pixel_spacing
    pts = []
    dirs = []
    angle_min = base_angle_deg - fan_half_deg
    angle_max = base_angle_deg + fan_half_deg
    angle = angle_min
    while angle <= angle_max + 1e-6:
        rad = math.radians(angle)
        dr = math.sin(rad)
        dc = math.cos(rad)
        length = math.sqrt(dr * dr + dc * dc) + 1e-9
        dr /= length; dc /= length

        start_r = center_r + dr * start_radius_px
        start_c = center_c + dc * start_radius_px

        result = scan_fn(
            in_img_2d, start_r, start_c,
            dr, dc, scan_mm, pixel_spacing,
            in_low_mean=low_mean, in_high_mean=high_mean,
            drop_ratio=drop_ratio, low_ratio=low_ratio, smooth_win=2)
        if result is not None:
            pts.append(result)
            dirs.append((dr, dc))
        angle += fan_step_deg
    return pts, dirs


def _calc_ant_angle_deg(vert_center, c2_rows, c2_cols, disc_top=None, disc_bot=None):
    """
    计算椎体前缘扇形中心射线角度（ant_angle_deg）。
    """
    cr, cc = float(vert_center[0]), float(vert_center[1])
    c2r = np.array(c2_rows, dtype=np.float64)
    c2c = np.array(c2_cols, dtype=np.float64)
    if disc_top is not None and disc_bot is not None:
        d_dr = float(disc_bot[0]) - float(disc_top[0])
        d_dc = float(disc_bot[1]) - float(disc_top[1])
        d_len = math.sqrt(d_dr**2 + d_dc**2) + 1e-9
        d_dr /= d_len; d_dc /= d_len
        if d_dr < 0:
            d_dr, d_dc = -d_dr, -d_dc
        n_row_a = -d_dc; n_col_a = d_dr
        if n_col_a > 0:
            n_row_a, n_col_a = -n_row_a, -n_col_a
        return math.degrees(math.atan2(n_row_a, n_col_a))
    else:
        _, _, n_row, n_col = _project_to_c2(cr, cc, c2r, c2c)
        if n_col > 0:
            n_row, n_col = -n_row, -n_col
        return math.degrees(math.atan2(n_row, n_col))


def fan_scan_vertebra(in_img_2d, vert_center, c2_rows, c2_cols,
                       pixel_spacing, low_mean,
                       fan_half_deg=50.0, fan_step_deg=2.0,
                       scan_up_mm=30.0, scan_dn_mm=30.0, scan_ant_mm=40.0,
                       low_ratio=1.3,
                       high_mean=None,
                       drop_ratio=0.25,
                       ant_drop_ratio=0.35,
                       disc_top=None, disc_bot=None,
                       fan_half_ep_deg=None,
                       ant_low_mean=None,
                       ant_high_mean=None,
                       ant_diag_confirm=False,
                       ep_diag_confirm=False):
    """
    以椎体中心为原点，分三个方向扇形扫描：
      - 上方扇形 → 上终板候选点（sup_pts）
      - 下方扇形 → 下终板候选点（inf_pts）
      - 腹侧扇形 → 前缘候选点（ant_pts）

    返回：
        {'sup_pts': [...], 'inf_pts': [...], 'ant_pts': [...], 'ant_dirs': [...], 'fan_params': {...}}
    """
    cr, cc = float(vert_center[0]), float(vert_center[1])
    c2r = np.array(c2_rows, dtype=np.float64)
    c2c = np.array(c2_cols, dtype=np.float64)

    if disc_top is not None and disc_bot is not None:
        d_dr = float(disc_bot[0]) - float(disc_top[0])
        d_dc = float(disc_bot[1]) - float(disc_top[1])
        d_len = math.sqrt(d_dr**2 + d_dc**2) + 1e-9
        d_dr /= d_len; d_dc /= d_len
        if d_dr < 0:
            d_dr, d_dc = -d_dr, -d_dc

        up_angle_deg   = math.degrees(math.atan2(-d_dr, -d_dc))
        down_angle_deg = math.degrees(math.atan2(d_dr, d_dc))
        n_row_a = -d_dc; n_col_a = d_dr
        if n_col_a > 0:
            n_row_a, n_col_a = -n_row_a, -n_col_a
        ant_angle_deg  = math.degrees(math.atan2(n_row_a, n_col_a))
    else:
        _, _, n_row, n_col = _project_to_c2(cr, cc, c2r, c2c)
        t_dr = -n_col; t_dc = n_row
        t_len = math.sqrt(t_dr**2 + t_dc**2) + 1e-9
        t_dr /= t_len; t_dc /= t_len
        if t_dr > 0:
            t_dr, t_dc = -t_dr, -t_dc
        up_angle_deg   = math.degrees(math.atan2(t_dr, t_dc))
        down_angle_deg = math.degrees(math.atan2(-t_dr, -t_dc))
        if n_col > 0:
            n_row, n_col = -n_row, -n_col
        ant_angle_deg  = math.degrees(math.atan2(n_row, n_col))

    _ep_half = fan_half_ep_deg if fan_half_ep_deg is not None else fan_half_deg
    _ep_scan_fn = _scan_normal_descent_diag if ep_diag_confirm else _scan_normal_descent
    _ep_hm = high_mean if high_mean is not None else low_mean * 3.0

    sup_pts, _ = _fan_scan_direction(
        cr, cc, up_angle_deg, _ep_half, fan_step_deg,
        scan_up_mm, pixel_spacing, in_img_2d, low_mean, _ep_hm,
        drop_ratio=drop_ratio, low_ratio=low_ratio,
        scan_fn=_ep_scan_fn)

    inf_pts, _ = _fan_scan_direction(
        cr, cc, down_angle_deg, _ep_half, fan_step_deg,
        scan_dn_mm, pixel_spacing, in_img_2d, low_mean, _ep_hm,
        drop_ratio=drop_ratio, low_ratio=low_ratio,
        scan_fn=_ep_scan_fn)

    _ant_lm = ant_low_mean if ant_low_mean is not None else low_mean
    _ant_hm = ant_high_mean if ant_high_mean is not None else _ep_hm
    _ant_scan_fn = _scan_normal_descent_ant_diag if ant_diag_confirm else _scan_normal_descent_ant
    ant_pts, ant_dirs = _fan_scan_direction(
        cr, cc, ant_angle_deg, fan_half_deg, fan_step_deg,
        scan_ant_mm, pixel_spacing, in_img_2d, _ant_lm, _ant_hm,
        drop_ratio=ant_drop_ratio, low_ratio=low_ratio,
        scan_fn=_ant_scan_fn)

    print(f"   [Step4] 椎体中心({cr:.0f},{cc:.0f}): "
          f"上终板={len(sup_pts)} 下终板={len(inf_pts)} 前缘={len(ant_pts)}")

    fan_params = {
        'center': (cr, cc),
        'up':  {'angle': up_angle_deg,   'half': _ep_half,      'scan_mm': scan_up_mm},
        'dn':  {'angle': down_angle_deg, 'half': _ep_half,      'scan_mm': scan_dn_mm},
        'ant': {'angle': ant_angle_deg,  'half': fan_half_deg,  'scan_mm': scan_ant_mm},
    }
    return {'sup_pts': sup_pts, 'inf_pts': inf_pts, 'ant_pts': ant_pts,
            'ant_dirs': ant_dirs,
            'fan_params': fan_params}


# ─────────────────────────────────────────────────────────────────────────────
# Step4c  前缘候选点二次校验
# ─────────────────────────────────────────────────────────────────────────────

def _verify_ant_pts_forward(ant_pts, ant_dirs, in_img_2d, pixel_spacing,
                             high_mean, forward_mm=5.0,
                             ant_angle_deg=None,
                             upper_filter_start=25.0, upper_filter_end=50.0):
    """
    对前缘候选点做二次校验：

    角度范围处理（相对扇形中心角 ant_angle_deg 的偏移 delta）：
        - delta ∈ [upper_filter_start, upper_filter_end]（默认 [+25°, +50°]，头颅侧上25°区间）
          → 直接删除，不做任何信号判断
        - 其余点：直接保留
        - ant_angle_deg=None：对所有点直接保留（不过滤）

    参数：
        ant_pts              - [(r, c), ...]
        ant_dirs             - [(dr, dc), ...]
        in_img_2d            - 2D 图像数组
        pixel_spacing        - mm/pixel
        high_mean            - 保留参数（兼容旧签名，当前逻辑不使用）
        forward_mm           - 保留参数（兼容旧签名，当前逻辑不使用）
        ant_angle_deg        - 扇形中心射线角度（度），None 表示不过滤
        upper_filter_start   - 过滤区间起始偏移（默认 25°）
        upper_filter_end     - 过滤区间结束偏移（默认 50°）

    返回：
        (kept_pts, kept_dirs)
    """
    kept_pts  = []
    kept_dirs = []
    n_filtered = 0
    n_bypass   = 0

    for (r, c), (dr, dc) in zip(ant_pts, ant_dirs):
        in_filter_zone = False
        if ant_angle_deg is not None:
            ray_angle = math.degrees(math.atan2(dr, dc))
            delta = ray_angle - ant_angle_deg
            while delta > 180.0:  delta -= 360.0
            while delta <= -180.0: delta += 360.0
            in_filter_zone = (upper_filter_start - 1e-6 <= delta <= upper_filter_end + 1e-6)

        if in_filter_zone:
            # 直接删除，不做信号判断
            n_filtered += 1
        else:
            kept_pts.append((r, c))
            kept_dirs.append((dr, dc))
            n_bypass += 1

    if ant_angle_deg is not None:
        print(f"      [Step4c] 直接删除={n_filtered} 保留={n_bypass}")

    return kept_pts, kept_dirs


# ─────────────────────────────────────────────────────────────────────────────
# Step4-Alternative: 椎间盘终板矩阵扫描（上升沿，与扇形扫描并行）
# ─────────────────────────────────────────────────────────────────────────────

def scan_disc_endplates(in_img_2d, vert_center, junc_top, junc_bot,
                        c2_rows, c2_cols, pixel_spacing,
                        low_mean, high_mean,
                        scan_up_mm=30.0, scan_dn_mm=30.0,
                        drop_ratio=0.25):
    """
    矩阵扫描（下降沿检测终板）。

    几何逻辑：
        起点连线：
            中点 = vert_center（椎体中心）
            右端点 = junction_pts[vi] 与 junction_pts[vi+1] 在皮质线2上的路径中点
            左端点 = 以中点为中心，向左（腹侧）延伸为右半段的1.5倍
            起点集 = 左端点→右端点，每隔 1px 一个起点
        扫描方向：
            垂直于起点连线
            上终板：向头部（row减小）
            下终板：向尾部（row增大）
        信号检测：下降沿检测（与扇形扫描一致）

    参数：
        in_img_2d      - IN序列2D图像
        vert_center    - 椎体中心 (r, c)
        junc_top       - 上椎间盘对应的终板汇合点 (jr, jc, jval, jidx)
        junc_bot       - 下椎间盘对应的终板汇合点 (jr, jc, jval, jidx)
        c2_rows, c2_cols - 皮质线2坐标
        pixel_spacing  - mm/px
        low_mean       - 低信号参考值
        high_mean      - 高信号参考值
        scan_up_mm     - 上终板扫描距离
        scan_dn_mm     - 下终板扫描距离
        drop_ratio     - 下降沿阈值参数

    返回：
        {'sup_pts': [...], 'inf_pts': [...], 'ant_pts': [], 'ant_dirs': [], 'fan_params': None}
    """
    H, W = in_img_2d.shape

    vc_r, vc_c = float(vert_center[0]), float(vert_center[1])
    jt_r, jt_c = float(junc_top[0]), float(junc_top[1])
    jb_r, jb_c = float(junc_bot[0]), float(junc_bot[1])

    # 计算两个junction_pt在皮质线2上的路径距离中点（右端点）
    c2r = np.array(c2_rows, dtype=np.float64)
    c2c = np.array(c2_cols, dtype=np.float64)

    def _find_nearest_c2_index(target_r, target_c):
        dists = (c2r - target_r)**2 + (c2c - target_c)**2
        return int(np.argmin(dists))

    idx_top = _find_nearest_c2_index(jt_r, jt_c)
    idx_bot = _find_nearest_c2_index(jb_r, jb_c)

    if idx_top > idx_bot:
        idx_top, idx_bot = idx_bot, idx_top

    # 沿皮质线2计算路径距离
    seg_rs = c2r[idx_top:idx_bot+1]
    seg_cs = c2c[idx_top:idx_bot+1]
    cum_dist = np.zeros(len(seg_rs))
    for k in range(1, len(seg_rs)):
        cum_dist[k] = cum_dist[k-1] + math.sqrt(
            (seg_rs[k]-seg_rs[k-1])**2 + (seg_cs[k]-seg_cs[k-1])**2)
    total_dist = cum_dist[-1]
    mid_dist = total_dist / 2.0

    # 找路径中点对应的皮质线坐标
    mid_idx = idx_top + int(np.searchsorted(cum_dist, mid_dist))
    mid_idx = min(mid_idx, len(c2r) - 1)
    right_r = float(c2r[mid_idx])
    right_c = float(c2c[mid_idx])

    # 起点连线方向：椎体中心 → 右端点
    line_dr = right_r - vc_r
    line_dc = right_c - vc_c
    line_len = math.sqrt(line_dr**2 + line_dc**2) + 1e-9
    line_urn = line_dr / line_len
    line_uc = line_dc / line_len

    # 起点集：以椎体中心为中点，左半段（腹侧）延伸为右半段的1.5倍
    right_len = line_len
    left_len = right_len * 1.5
    n_pts_right = int(round(right_len))
    n_pts_left = int(round(left_len))
    if n_pts_right < 1:
        n_pts_right = 1
    if n_pts_left < 1:
        n_pts_left = 1

    # 扫描方向：垂直于起点连线
    # 正方向定义为"向尾部（row增大）"，上终板用 -scan_dir，下终板用 +scan_dir
    scan_dr = -line_uc
    scan_dc = line_urn
    if scan_dr < 0:
        scan_dr, scan_dc = -scan_dr, -scan_dc

    sup_pts = []
    inf_pts = []

    for k in range(-n_pts_left, n_pts_right + 1):
        sk_r = vc_r + k * line_urn
        sk_c = vc_c + k * line_uc

        # 上终板：沿 -scan_dir（向头部）
        pt_up = _scan_normal_descent(
            in_img_2d, sk_r, sk_c, -scan_dr, -scan_dc,
            scan_up_mm, pixel_spacing,
            in_low_mean=low_mean, in_high_mean=high_mean,
            drop_ratio=drop_ratio, low_ratio=1.3, smooth_win=2)
        if pt_up is not None:
            sup_pts.append(pt_up)

        # 下终板：沿 +scan_dir（向尾部）
        pt_dn = _scan_normal_descent(
            in_img_2d, sk_r, sk_c, scan_dr, scan_dc,
            scan_dn_mm, pixel_spacing,
            in_low_mean=low_mean, in_high_mean=high_mean,
            drop_ratio=drop_ratio, low_ratio=1.3, smooth_win=2)
        if pt_dn is not None:
            inf_pts.append(pt_dn)

    print(f"   [Step4-矩阵] vert=({vc_r:.0f},{vc_c:.0f}) junc_mid=({right_r:.0f},{right_c:.0f}) "
          f"右半段={right_len:.1f}px 左半段={left_len:.1f}px 起点集={n_pts_left+n_pts_right+1} "
          f"上终板={len(sup_pts)} 下终板={len(inf_pts)}")

    return {'sup_pts': sup_pts, 'inf_pts': inf_pts, 'ant_pts': [],
            'ant_dirs': [], 'fan_params': None}
