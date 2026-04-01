"""
前缘检测模块（V15）
包含 build_anterior_scan_lines_v15、find_anterior_min_points_h/v15、
     refine_anterior_edge_v15、scan_anterior_edge_v15、
     cluster_anterior_edge_v15、smooth_anterior_edge_v15、
     find_anterior_corner_v15、find_anterior_edge_by_descent、
     find_arc_roi_min_points、refine_arc_roi_to_anterior_edge、
     filter_arc_roi_by_dense_offset
"""
import numpy as np
from scipy import signal


def build_anterior_scan_lines_v15(c2_cols, c2_rows, img_shape,
                                   pixel_spacing, step_mm=1.0, depth_mm=50.0):
    """
    皮质线2前缘扫描线生成。
    沿皮质线2弧长方向每隔 step_mm 取一个采样点，
    每个点沿该处法线方向延伸 depth_mm，生成密集前缘扫描线组。
    返回：[(s_mm, row_arr, col_arr, nx_arr, ny_arr), ...]
      s_mm    : 该采样点在皮质线2上的弧长坐标 (mm)
      row_arr : 沿法线方向 depth_mm 个像素的行坐标
      col_arr : 沿法线方向 depth_mm 个像素的列坐标
      nx_arr  : 各像素处法线行分量（恒定，等于采样点法线）
      ny_arr  : 各像素处法线列分量（恒定，等于采样点法线）
    """
    H, W = img_shape
    N = len(c2_rows)

    # 1. 计算皮质线2弧长
    arc_mm = np.zeros(N, dtype=np.float64)
    for i in range(1, N):
        dr = float(c2_rows[i] - c2_rows[i - 1])
        dc = float(c2_cols[i] - c2_cols[i - 1])
        arc_mm[i] = arc_mm[i - 1] + np.sqrt(dr * dr + dc * dc) * pixel_spacing

    total_arc = arc_mm[-1]

    # 2. 逐点计算法线方向
    nx_full = np.zeros(N, dtype=np.float64)
    ny_full = np.zeros(N, dtype=np.float64)
    for i in range(N):
        i0 = max(0, i - 2)
        i1 = min(N - 1, i + 2)
        t_dr = float(c2_rows[i1] - c2_rows[i0])
        t_dc = float(c2_cols[i1] - c2_cols[i0])
        t_len = np.sqrt(t_dr * t_dr + t_dc * t_dc)
        if t_len < 1e-6:
            t_dr, t_dc = 1.0, 0.0; t_len = 1.0
        t_dr /= t_len; t_dc /= t_len
        # 法线 = 切线顺时针旋转90°: (t_dc, -t_dr)
        # 确保指向椎体方向（列增大方向，ny > 0 → 向左即ny < 0 ... 这里取ny<0朝前缘）
        nx = t_dc; ny = -t_dr
        # 保证朝向前方（列减小方向，脊柱前方在左图左侧，col更小）
        if ny > 0:
            nx, ny = -nx, -ny
        nx_full[i] = nx
        ny_full[i] = ny

    # 3. 沿弧长每隔 step_mm 采样
    depth_px = int(round(depth_mm / pixel_spacing))
    scan_lines = []
    s_target = 0.0
    while s_target <= total_arc + 1e-6:
        # 找最近索引
        idx = int(np.argmin(np.abs(arc_mm - s_target)))
        base_r = float(c2_rows[idx])
        base_c = float(c2_cols[idx])
        nx = nx_full[idx]
        ny = ny_full[idx]

        rows_arr = []
        cols_arr = []
        for k in range(depth_px):
            r = int(np.clip(round(base_r + nx * k), 0, H - 1))
            c = int(np.clip(round(base_c + ny * k), 0, W - 1))
            rows_arr.append(r)
            cols_arr.append(c)

        scan_lines.append((
            float(s_target),
            np.array(rows_arr, dtype=np.int32),
            np.array(cols_arr, dtype=np.int32),
            float(nx), float(ny),
        ))
        s_target = round(s_target + step_mm, 4)

    return scan_lines


def find_anterior_min_points_h(c2_cols, c2_rows, f_img_2d, pixel_spacing,
                               near_mm=20.0, far_mm=40.0,
                               expand_ratio=1.5):
    """
    水平offset方式谷底查找（对标V12_4 find_arc_roi_min_points）。
    不使用法线方向，避免皮质线2定位误差沿法线深度方向被放大。

    对皮质线2每一行，在水平方向 [near_mm, far_mm] 偏移区间内找最低信号点。
    坐标系：皮质线2在右侧，offset向左增大，col = base_col - offset_px。
    near_mm : 右边界（靠皮质线2侧），固定不变。
    far_mm  : 左边界（远离皮质线2侧），随行号从顶到底线性扩展到 expand_ratio 倍。

    返回: [(row, col, offset_mm, base_col), ...]
    """
    if c2_cols is None or c2_rows is None or len(c2_cols) == 0:
        return []
    H, W = f_img_2d.shape
    n_rows = len(c2_rows)
    near_px = int(round(near_mm / pixel_spacing))

    row_min  = float(c2_rows[0])
    row_max  = float(c2_rows[-1])
    row_span = max(row_max - row_min, 1.0)

    result = []
    for idx in range(n_rows):
        row      = int(c2_rows[idx])
        base_col = int(c2_cols[idx])
        if row < 0 or row >= H:
            continue
        # far_mm 随行号线性扩展：顶行=far_mm，底行=far_mm*expand_ratio
        t = (float(row) - row_min) / row_span
        far_mm_row = far_mm * (1.0 + (expand_ratio - 1.0) * t)
        far_px = int(round(far_mm_row / pixel_spacing))

        col_right = max(0, min(W - 1, base_col - near_px))
        col_left  = max(0, min(W - 1, base_col - far_px))
        if col_right <= col_left:
            continue
        roi_sig = f_img_2d[row, col_left:col_right + 1].astype(np.float32)
        if len(roi_sig) == 0:
            continue
        min_idx   = int(np.argmin(roi_sig))
        min_col   = col_left + min_idx
        offset_mm = (base_col - min_col) * pixel_spacing
        result.append((row, min_col, offset_mm, base_col))

    print(f"   [V15前缘谷底-H] 水平offset {near_mm:.0f}~{far_mm:.0f}mm "
          f"(底行扩展至{far_mm*expand_ratio:.0f}mm) 共 {len(result)} 行最低点")
    return result


def find_anterior_min_points_v15(scan_lines, f_img_2d, pixel_spacing,
                                  near_mm=20.0, far_mm=40.0,
                                  expand_ratio=1.5):
    """
    法线方向ROI谷底查找（对标V12_4的find_arc_roi_min_points，但改用法线方向）。
    对每条皮质线2法线扫描线，取 [near_mm, far_mm] 深度区间内的信号最低点。
    这个最低点通常是椎体前缘皮质骨（低信号），作为上升沿检测的起始点。
    expand_ratio: far_mm随行号增大线性扩展（顶行=far_mm，底行=far_mm*expand_ratio），
                  对标V12_4按行号从顶到底扩展。
    返回: [(s_mm, row, col, depth_mm), ...]
    """
    if not scan_lines:
        return []
    H, W = f_img_2d.shape
    near_px = int(round(near_mm / pixel_spacing))

    # 计算行号总范围（用于动态扩展缩放，对标V12_4按行号从顶到底扩展）
    rows_first = [float(sl[1][0]) for sl in scan_lines if len(sl[1]) > 0]
    row_min = float(min(rows_first)) if rows_first else 0.0
    row_max = float(max(rows_first)) if rows_first else 1.0
    row_span = max(row_max - row_min, 1.0)

    result = []
    for sl in scan_lines:
        s_mm, rows_a, cols_a, nx, ny = sl
        n = len(rows_a)
        if n <= near_px:
            continue
        # far_mm 随行号增大线性扩展：顶行=far_mm，底行=far_mm*expand_ratio（对标V12_4）
        t = (float(rows_a[0]) - row_min) / row_span
        far_mm_row = far_mm * (1.0 + (expand_ratio - 1.0) * t)
        far_px = int(round(far_mm_row / pixel_spacing))
        end_px = min(far_px, n - 1)
        if end_px <= near_px:
            continue
        seg = np.array([f_img_2d[r, c]
                        for r, c in zip(rows_a[near_px:end_px + 1],
                                        cols_a[near_px:end_px + 1])],
                       dtype=np.float32)
        if len(seg) == 0:
            continue
        min_idx = int(np.argmin(seg))
        abs_i   = near_px + min_idx
        result.append((
            float(s_mm),
            int(rows_a[abs_i]),
            int(cols_a[abs_i]),
            float(abs_i) * pixel_spacing,
        ))
    print(f"   [V15前缘谷底] {len(result)} 个法线ROI最低点"
          f"（{near_mm:.0f}~{far_mm:.0f}mm，末端扩展至{far_mm*expand_ratio:.0f}mm）")
    return result


def refine_anterior_edge_v15(min_pts, scan_lines, f_img_2d, pixel_spacing,
                              high_mean=200.0, low_mean=50.0,
                              scan_mm=20.0, rise_ratio=0.50,
                              probe_ratio=0.6, texture_mode=False,
                              skip_px=0):
    """
    法线方向上升沿精修（对标V12_4的refine_arc_roi_to_anterior_edge）。

    几何方向：
      法线从皮质线2出发，depth增大=朝前（远离皮质线2）
      谷底点在大depth处（前缘皮质骨/低信号）
      上升沿扫描方向：从谷底向小depth方向回扫（回向骨髓）
      对应V12_4：从谷底小col向大col（向右）扫回骨髓

    扫描段（倒序读取）：
      [start_abs - scan_px, start_abs - skip_px]
      倒序 → 信号从"谷底低信号"逐渐升高进入骨髓

    texture_mode=False（第一轮）：
      上升沿后脂肪过滤（邻域均值 > high_mean*1.3 跳过），
      后续段60%高于相对阈值 且 绝对均值>=high_mean*0.75 → refined

    texture_mode=True（第二轮）：
      跳过skip_px像素后再找上升沿；
      模式C：全段>50%像素 < low_mean → kept_low
      模式B：后续连续>=2像素 < low_mean → 放弃此沿继续找
      模式A：后续>=60%像素 >= high_mean*0.75 → refined

    返回: [(s_mm, row, col, depth_mm, flag), ...]
      flag: 'refined'|'kept'|'kept_low'
    """
    if not min_pts or not scan_lines:
        return []

    H, W = f_img_2d.shape
    win_px   = 2
    scan_px  = max(2, int(round(scan_mm / pixel_spacing)))
    high_thr = high_mean * 0.75

    # 建立 s_mm → scan_line 快速查找
    sl_map = {}
    for sl in scan_lines:
        key = round(float(sl[0]), 1)
        sl_map[key] = sl

    result = []
    n_refined = 0

    for pt in min_pts:
        s_mm, row0, col0, depth_mm0 = float(pt[0]), int(pt[1]), int(pt[2]), float(pt[3])

        # 找对应扫描线
        key = round(s_mm, 1)
        sl = sl_map.get(key)
        if sl is None:
            best_key = min(sl_map.keys(), key=lambda k: abs(k - s_mm))
            sl = sl_map[best_key]

        _, rows_a, cols_a, nx, ny = sl
        n_total = len(rows_a)

        # 谷底在扫描线中的索引
        start_abs = int(round(depth_mm0 / pixel_spacing))
        start_abs = max(0, min(start_abs, n_total - 1))

        # 回扫段：从谷底向小depth方向，跳过skip_px像素
        seg_end   = max(0, start_abs - skip_px)
        seg_start = max(0, start_abs - skip_px - scan_px)

        if seg_end <= seg_start + win_px:
            result.append((s_mm, row0, col0, depth_mm0, 'kept'))
            continue

        # 提取倒序信号：index 0=谷底侧（低信号），末尾=骨髓侧（高信号）
        seg_raw = np.array([f_img_2d[r, c]
                            for r, c in zip(rows_a[seg_start:seg_end + 1],
                                            cols_a[seg_start:seg_end + 1])],
                           dtype=np.float32)[::-1]

        if len(seg_raw) < win_px + 2:
            result.append((s_mm, row0, col0, depth_mm0, 'kept'))
            continue

        seg_sm = np.convolve(seg_raw, np.ones(win_px) / win_px, mode='same')

        # texture_mode: 模式C（全段>50%低信号）
        if texture_mode:
            low_ratio_total = float(np.mean(seg_sm < low_mean))
            if low_ratio_total > 0.50:
                result.append((s_mm, row0, col0, depth_mm0, 'kept_low'))
                continue

        found_edge = False
        for j in range(1, len(seg_sm)):
            ref = float(seg_sm[j - 1])
            cur = float(seg_sm[j])

            if cur > ref * (1.0 + rise_ratio):
                if not texture_mode:
                    # 脂肪信号过滤：上升沿位置在原始坐标中的索引
                    # 倒序第j个 → 原始索引 seg_end - j
                    abs_j = seg_end - j
                    fat_vals = []
                    for dc in (0, 1, 2, 3):
                        abs_nb = int(np.clip(abs_j - dc, 0, n_total - 1))
                        c_nb = int(np.clip(cols_a[abs_nb], 0, W - 1))
                        r_nb = int(rows_a[abs_nb])
                        for dr in (-1, 0, 1):
                            r2 = int(np.clip(r_nb + dr, 0, H - 1))
                            fat_vals.append(float(f_img_2d[r2, c_nb]))
                    fat_mean = float(np.mean(fat_vals)) if fat_vals else 0.0
                    if fat_mean > high_mean * 1.3:
                        continue

                    rest = seg_sm[j + 1:]
                    if len(rest) == 0:
                        break
                    high_thresh_rel = ref * (1.0 + rise_ratio * 0.5)
                    high_ratio = float(np.mean(rest > high_thresh_rel))
                    rest_abs_mean = float(np.mean(rest))
                    if high_ratio >= probe_ratio and rest_abs_mean >= high_thr:
                        # 倒序第j个 → 原始depth索引 seg_end - j
                        new_abs = int(np.clip(seg_end - j, 0, n_total - 1))
                        result.append((
                            s_mm,
                            int(rows_a[new_abs]),
                            int(cols_a[new_abs]),
                            float(new_abs) * pixel_spacing,
                            'refined',
                        ))
                        found_edge = True
                        n_refined += 1
                        break
                else:
                    # 纹理模式
                    rest = seg_sm[j + 1:]
                    if len(rest) == 0:
                        break
                    tex_win = 3
                    tex_pad = tex_win // 2
                    rest_sm = np.convolve(
                        np.pad(rest, tex_pad, mode='edge'),
                        np.ones(tex_win) / tex_win, mode='valid')

                    # 模式B：连续>=2像素 < low_mean → 放弃
                    is_B = False
                    low_cnt = 0
                    for k in range(len(rest_sm)):
                        if rest_sm[k] < low_mean:
                            low_cnt += 1
                            if low_cnt >= 2:
                                is_B = True
                                break
                        else:
                            low_cnt = 0
                    if is_B:
                        continue

                    # 模式A：高信号占>=60%
                    high_ratio = float(np.mean(rest_sm >= high_thr))
                    if high_ratio >= probe_ratio:
                        new_abs = int(np.clip(seg_end - j, 0, n_total - 1))
                        result.append((
                            s_mm,
                            int(rows_a[new_abs]),
                            int(cols_a[new_abs]),
                            float(new_abs) * pixel_spacing,
                            'refined',
                        ))
                        found_edge = True
                        n_refined += 1
                        break

        if not found_edge:
            result.append((s_mm, row0, col0, depth_mm0, 'kept'))

    mode_str = '纹理模式' if texture_mode else '普通模式'
    print(f"   [V15前缘精修({mode_str})] {n_refined}/{len(min_pts)} 点更新到上升沿")
    return result


def scan_anterior_edge_v15(scan_lines, f_img_2d, pixel_spacing,
                           skip_mm=20.0, high_mean=None, low_mean=None):
    """
    皮质线2法线前缘扫描：从 skip_mm 处开始找下降沿（椎体前缘皮质骨）。

    解剖路径（皮质线2是椎管前壁，法线朝前）：
        皮质线2 -> 椎体骨髓（高信号）-> 椎体前缘皮质骨（低信号）-> 前方软组织
        或：皮质线2 -> 椎间盘（低信号）-> 前方软组织

    初始信号校验（20mm处往前3mm均值）：
        - 3mm均值 >= low_mean：正常椎体骨髓，从20mm找下降沿
        - 3mm均值 <  low_mean：该线落在椎间盘，直接跳过
          （3mm均值校验用于排除椎体骨髓内单点噪声的误判）

    下降沿检测（2mm局部参考窗口，drop_ratio=0.50）：
        cur < ref*(1-0.50)
        且后续4mm探测区>=60%低于 ref*(1-0.25)
        且探测区绝对均值 < low_mean

    high_mean / low_mean：从26条终板扫描线的 find_endplates_on_water_image 返回值传入，
                          与终板检测共用同一套Otsu分组标准。
    返回: [(s_mm, row, col, depth_mm), ...]
    """
    if not scan_lines:
        return []

    H, W = f_img_2d.shape
    win_mm      = 2.0
    probe_ratio = 0.6
    drop_ratio  = 0.50
    win_px      = max(2, int(win_mm / pixel_spacing))
    skip_px     = int(round(skip_mm / pixel_spacing))
    probe_px    = max(2, int(4.0 / pixel_spacing))
    probe3_px   = max(1, int(3.0 / pixel_spacing))  # 初始校验3mm

    # high_mean / low_mean: 优先使用外部传入值（26条终板扫描线Otsu分组）
    if high_mean is None or low_mean is None:
        from skimage.filters import threshold_otsu
        mid_sigs = []
        for sl in scan_lines[::max(1, len(scan_lines)//10)]:
            s_mm, rows_a, cols_a, nx, ny = sl
            sig = np.array([f_img_2d[r, c] for r, c in zip(rows_a, cols_a)],
                           dtype=np.float32)
            mid_sigs.extend(sig.tolist())
        mid_arr = np.array(mid_sigs, dtype=np.float32)
        sm = np.convolve(mid_arr, np.ones(win_px)/win_px, mode='same')
        otsu_thresh = float(threshold_otsu(sm))
        high_vals = sm[sm >= otsu_thresh]
        low_vals  = sm[sm <  otsu_thresh]
        if high_mean is None:
            high_mean = float(np.mean(high_vals)) if len(high_vals) > 0 else float(np.max(sm))
        if low_mean is None:
            low_mean  = float(np.mean(low_vals))  if len(low_vals)  > 0 else float(np.min(sm))

    kernel = np.ones(win_px) / win_px

    result = []
    for sl in scan_lines:
        s_mm, rows_a, cols_a, nx, ny = sl
        n = len(rows_a)
        if n <= skip_px + probe_px + win_px:
            continue

        full_sig = np.array([f_img_2d[r, c] for r, c in zip(rows_a, cols_a)],
                            dtype=np.float32)
        full_sm  = np.convolve(full_sig, kernel, mode='same')

        # 初始信号校验：20mm处往前3mm均值，低于low_mean则为椎间盘，跳过
        probe3_end = min(skip_px + probe3_px, len(full_sm))
        init_mean  = float(np.mean(full_sm[skip_px:probe3_end]))
        if init_mean < low_mean:
            continue

        # 从 skip_px 开始扫描子段
        sub_rows = rows_a[skip_px:]
        sub_cols = cols_a[skip_px:]
        sig = np.array([f_img_2d[r, c] for r, c in zip(sub_rows, sub_cols)],
                       dtype=np.float32)
        if len(sig) < win_px * 2 + 4:
            continue

        sig_sm     = np.convolve(sig, kernel, mode='same')
        global_med = float(np.median(sig_sm))

        i = win_px
        while i < len(sig_sm) - probe_px - 1:
            cur = float(sig_sm[i])
            ref = float(np.mean(sig_sm[max(0, i - win_px):i])) if i >= win_px else global_med
            ref = max(ref, 1.0)
            if cur < ref * (1.0 - drop_ratio):
                probe_slice = sig_sm[i + 1: i + probe_px + 1]
                low_thresh  = ref * (1.0 - drop_ratio * 0.5)
                if (len(probe_slice) > 0
                        and np.mean(probe_slice < low_thresh) >= probe_ratio):
                    abs_i    = min(skip_px + i, len(rows_a) - 1)
                    depth_mm = float(abs_i) * pixel_spacing
                    result.append((
                        float(s_mm),
                        int(rows_a[abs_i]),
                        int(cols_a[abs_i]),
                        depth_mm,
                    ))
                    break
            i += 1

    return result

def cluster_anterior_edge_v15(ant_edge_pts, pixel_spacing,
                              win_mm=5.0, smooth_mm=5.0,
                              d_start_mm=20.0, d_end_mm=50.0,
                              expand_ratio=1.5):
    """
    在法线d轴方向用win_mm滑动窗口找最密深度区间，再对该区间内的全段点做三步平滑。

    聚类空间几何：
      - 形状：沿皮质线2弧度走向的"弯曲长条带"
      - 长度：皮质线2全长（s轴，覆盖所有红点）
      - 宽度：法线方向win_mm（d轴，从 d 到 d+win_mm）
      - 滑动：在d轴从 d_start_mm 到 d_end_mm-win_mm，每次移动1mm
      - 动态扩展：每个点的d_end_mm按行号row线性扩展，顶行=d_end_mm，底行=d_end_mm*expand_ratio
      - 找红点最多的那个d区间 [best_d, best_d+win_mm]
      - 取该d区间内所有红点，按s_mm排序，每个s_mm取col中位数为代表点
    三步平滑（皮质线1方式）：MAD过滤 → 线性插值 → 移动均值(~5mm)

    参数：
      ant_edge_pts: [(s_mm, row, col, depth_mm), ...]
      expand_ratio: d_end_mm随行号线性扩展倍数，默认1.5（对标V12_4）
    返回：
      {"smoothed": [(row, col), ...], "raw_pts": [(row, col), ...],
       "best_depth_mm": float}  # 最密深度窗口起点
    """
    if not ant_edge_pts:
        return {"smoothed": [], "raw_pts": [], "best_depth_mm": d_start_mm}

    pts = sorted(ant_edge_pts, key=lambda x: x[0])  # 按 s_mm 排序
    s_arr   = np.array([p[0] for p in pts], dtype=np.float64)
    row_arr = np.array([p[1] for p in pts], dtype=np.float64)
    col_arr = np.array([p[2] for p in pts], dtype=np.float64)
    d_arr   = np.array([p[3] for p in pts], dtype=np.float64)

    # 行号总范围（用于动态扩展缩放，对标V12_4按行号从顶到底扩展）
    row_min_c = float(row_arr.min()) if len(row_arr) > 0 else 0.0
    row_max_c = float(row_arr.max()) if len(row_arr) > 0 else 1.0
    row_span_c = max(row_max_c - row_min_c, 1.0)
    # 每个点的动态d_end_mm：顶行=d_end_mm，底行=d_end_mm*expand_ratio
    t_arr = (row_arr - row_min_c) / row_span_c
    d_end_arr = d_end_mm * (1.0 + (expand_ratio - 1.0) * t_arr)

    # ---------- 1. d轴滑动窗口找最密深度（含动态扩展）----------
    best_count = 0
    best_d = d_start_mm
    d_cur = d_start_mm
    d_end_max = float(d_end_arr.max()) if len(d_end_arr) > 0 else d_end_mm
    while d_cur <= d_end_max - win_mm + 1e-6:
        # 窗口内的点：d在[d_cur, d_cur+win_mm]且未超出其动态d_end上限
        mask = (d_arr >= d_cur) & (d_arr < d_cur + win_mm) & (d_cur + win_mm <= d_end_arr)
        cnt = int(np.sum(mask))
        if cnt > best_count:
            best_count = cnt
            best_d = d_cur
        d_cur = round(d_cur + 1.0, 4)

    # 取最密d区间内所有点
    sel_mask = (d_arr >= best_d) & (d_arr < best_d + win_mm)
    sel_s   = s_arr[sel_mask]
    sel_row = row_arr[sel_mask]
    sel_col = col_arr[sel_mask]

    if len(sel_row) == 0:
        return {"smoothed": [], "raw_pts": [], "best_depth_mm": best_d}

    print(f"   [V15前缘聚类] 最密深度窗口: {best_d:.1f}~{best_d+win_mm:.1f}mm, "
          f"点数={best_count}, s范围={sel_s.min():.1f}~{sel_s.max():.1f}mm"
          f"（d_end末端扩展至{d_end_mm*expand_ratio:.1f}mm）")

    # 按 s_mm 去重：同1mm弧长内取col中位数
    unique_s = np.unique(np.round(sel_s, 0))
    rep_rows, rep_cols = [], []
    for us in unique_s:
        m = np.abs(sel_s - us) < 0.5
        rep_rows.append(float(np.median(sel_row[m])))
        rep_cols.append(float(np.median(sel_col[m])))
    rep_rows = np.array(rep_rows, dtype=np.float64)
    rep_cols = np.array(rep_cols, dtype=np.float64)

    raw_pts = [(int(round(r)), int(round(c)))
               for r, c in zip(rep_rows, rep_cols)]

    # ---------- 2. 三步平滑（皮质线1方式）----------
    rows_sorted_idx = np.argsort(rep_rows)
    xs = rep_rows[rows_sorted_idx]
    ys = rep_cols[rows_sorted_idx]

    # Step A: MAD 过滤（窗口11，阈值2.0σ）
    n = len(ys)
    window = min(11, n)
    if window % 2 == 0:
        window -= 1
    valid_mask = np.ones(n, dtype=bool)
    half = window // 2
    for i in range(n):
        s0 = max(0, i - half); s1 = min(n, i + half + 1)
        w = ys[s0:s1]
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if mad > 0 and np.abs(ys[i] - med) / (mad * 1.4826) > 2.0:
            valid_mask[i] = False
    clean_xs = xs[valid_mask]
    clean_ys = ys[valid_mask]

    if len(clean_xs) < 4:
        return {"smoothed": raw_pts, "raw_pts": raw_pts, "best_depth_mm": best_d}

    # Step B: 线性插值到整数行范围
    all_rows_interp = np.arange(int(round(clean_xs[0])),
                                int(round(clean_xs[-1])) + 1)
    interp_cols = np.interp(all_rows_interp, clean_xs, clean_ys)

    # Step C: 移动均值平滑（~5mm 窗口）
    k = max(3, int(smooth_mm / pixel_spacing))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(interp_cols, pad, mode="edge")
    kernel = np.ones(k) / k
    smooth_ys = np.convolve(padded, kernel, mode="valid")

    smoothed_pts = [(int(r), int(round(c)))
                    for r, c in zip(all_rows_interp, smooth_ys)]

    return {"smoothed": smoothed_pts, "raw_pts": raw_pts, "best_depth_mm": best_d}

def smooth_anterior_edge_v15(ant_pts, pixel_spacing, smooth_mm=5.0):
    """
    纯三步平滑（不含密集窗口过滤），对全量前缘点做：
      Step A: MAD过滤（窗口11，阈值2.0σ）
      Step B: 线性插值到整数行范围
      Step C: 移动均值平滑（smooth_mm窗口）
    输入: [(s_mm, row, col, depth_mm), ...]  或  [(row, col, ...), ...]
    返回: [(row, col), ...]
    """
    if not ant_pts:
        return []

    # 兼容4元组(s_mm,row,col,depth_mm)和2元组(row,col)
    if len(ant_pts[0]) >= 4:
        pts_rc = [(float(p[1]), float(p[2])) for p in ant_pts]
    else:
        pts_rc = [(float(p[0]), float(p[1])) for p in ant_pts]

    pts_rc.sort(key=lambda x: x[0])
    xs = np.array([p[0] for p in pts_rc], dtype=np.float64)
    ys = np.array([p[1] for p in pts_rc], dtype=np.float64)

    n = len(ys)
    if n < 4:
        return [(int(round(r)), int(round(c))) for r, c in pts_rc]

    # Step A: MAD过滤
    window = min(11, n)
    if window % 2 == 0:
        window -= 1
    valid_mask = np.ones(n, dtype=bool)
    half = window // 2
    for i in range(n):
        s0 = max(0, i - half); s1 = min(n, i + half + 1)
        w = ys[s0:s1]
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if mad > 0 and np.abs(ys[i] - med) / (mad * 1.4826) > 2.0:
            valid_mask[i] = False
    clean_xs = xs[valid_mask]
    clean_ys = ys[valid_mask]

    if len(clean_xs) < 4:
        return [(int(round(r)), int(round(c))) for r, c in pts_rc]

    # Step B: 线性插值到整数行范围
    all_rows = np.arange(int(round(clean_xs[0])), int(round(clean_xs[-1])) + 1)
    interp_cols = np.interp(all_rows, clean_xs, clean_ys)

    # Step C: 移动均值平滑
    k = max(3, int(smooth_mm / pixel_spacing))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(interp_cols, pad, mode='edge')
    smooth_ys = np.convolve(padded, np.ones(k) / k, mode='valid')

    return [(int(r), int(round(c))) for r, c in zip(all_rows, smooth_ys)]


def find_anterior_corner_v15(ep_line, img_2d, pixel_spacing,
                             cortical1_cols, cortical1_rows,
                             max_search_mm=30.0, probe_mm=3.0,
                             low_ratio=0.5):
    """
    V15 前缘角定位：
    沿终板分割直线延伸线逐像素向腿侧扫描，
    在每个扩展点的终板法线方向做 3mm 双侧探测（红色=向上右，绿色=向下左），
    连续两点探测信号均为低信号则确认前缘角。

    参数：
        ep_line         - build_endplate_line_v15 的返回値
        img_2d          - 压水图 F二维数组
        pixel_spacing   - 像素间距 mm
        cortical1_cols  - 皮质线1列坐标
        cortical1_rows  - 皮质线1行坐标
        max_search_mm   - 最大延伸搜索距离 mm
        probe_mm        - 双侧探测长度 mm
        low_ratio       - 低信号判定比例：信号 ≤ 皮质线引用均値 * low_ratio

    返回：
        (row, col) 前缘角坐标，或 None
    """
    H, W = img_2d.shape
    slope      = ep_line['slope']
    intercept  = ep_line['intercept']
    center_row = ep_line['center_row']
    center_col = ep_line['center_col']
    normal_dr  = ep_line['normal_dr']
    normal_dc  = ep_line['normal_dc']

    # 分割直线方向单位向量（朝腿侧，列坐标减小）
    t_len = np.sqrt(1.0 + slope * slope)
    line_dr = slope / t_len   # 分割直线延伸行分量
    line_dc = -1.0 / t_len    # 分割直线延伸列分量（朝腿侧，列减小）

    # 获取皮质线1在 center_row 处的信号平均値（作为低信号基准）
    def get_cortical1_signal_ref(row):
        rows_arr = np.array(cortical1_rows)
        nearest  = int(rows_arr[int(np.argmin(np.abs(rows_arr - row)))])
        c_idx    = np.searchsorted(cortical1_rows, nearest)
        c_idx    = int(np.clip(c_idx, 0, len(cortical1_cols) - 1))
        cort_col = int(cortical1_cols[c_idx])
        # 取皮质线周围列 ±2个像素平均
        sig_vals = [float(img_2d[nearest, max(0, cort_col + dc)])
                    for dc in range(-2, 3)
                    if 0 <= cort_col + dc < W]
        return float(np.mean(sig_vals)) if sig_vals else 1.0

    ref_signal = get_cortical1_signal_ref(center_row)
    ref_signal = max(ref_signal, 1.0)
    low_thresh = ref_signal * low_ratio

    max_steps = int(round(max_search_mm / pixel_spacing))
    probe_steps = int(round(probe_mm / pixel_spacing))

    # 沿延伸线逆行（从中心点向腿侧扩展）
    # 起始点：皮质线1与分割直线的交点
    # 简化：直接从 center_col 开始延伸
    consec_low = 0  # 连续低信号点数
    prev_low = False
    candidate = None

    for step in range(1, max_steps + 1):
        r_f = center_row + line_dr * step
        c_f = center_col + line_dc * step
        ri  = int(round(r_f))
        ci  = int(round(c_f))
        if not (0 <= ri < H and 0 <= ci < W):
            break

        # 法线方向 3mm 双侧探测
        probe_low_count = 0
        for sign in (+1, -1):  # +1=向上(head)，-1=向下(tail)
            for pb in range(1, probe_steps + 1):
                pr = int(round(ri + sign * normal_dr * pb))
                pc = int(round(ci + sign * normal_dc * pb))
                if 0 <= pr < H and 0 <= pc < W:
                    if float(img_2d[pr, pc]) <= low_thresh:
                        probe_low_count += 1

        # 双侧均触及低信号（至少一侧 probe_steps 个像素都低）
        this_low = (probe_low_count >= probe_steps)

        if this_low:
            consec_low += 1
            if prev_low and consec_low >= 2:   # 连续两点低信号
                # 取两点平均位置为前缘角
                r_prev = center_row + line_dr * (step - 1)
                c_prev = center_col + line_dc * (step - 1)
                candidate = ((r_f + r_prev) / 2.0, (c_f + c_prev) / 2.0)
                break
        else:
            consec_low = 0
        prev_low = this_low

    if candidate:
        print(f"   [V15前缘角] 终板 {ep_line['ep_type'] if 'ep_type' in ep_line else '?'} "
              f"@ row={candidate[0]:.1f}, col={candidate[1]:.1f}")
    else:
        print(f"   [V15前缘角] 未找到，返回 None")
    return candidate



def find_anterior_edge_by_descent(c2_cols, c2_rows, f_img_2d, pixel_spacing,
                                   consensus_endplates,
                                   high_mean3, low_mean3, drop_ratio3,
                                   nx_arr=None, ny_arr=None,
                                   scan_lines_v15=None):
    """
    腹侧下降沿法找椎体前缘（水平向左扫描，V13_2重构版）。

    扫描起点几何（与 build_scan_lines_v15 统一）：
      - 20mm基线 = scan_lines_v15[19]，即皮质线2法线偏移20mm的曲线
      - 33mm边界 = scan_lines_v15[32]

    主扫描段（椎体骨髓区）：
      - 范围：绿色终板线(upper)与20mm基线交叉点行号 → 橙色终板线(lower)与20mm基线交叉点行号
      - 每行从 20mm基线对应列坐标出发，水平向左扫描最多25mm

    补充扫描段（终板弯曲延伸，offset 20~33mm区间）：
      - 绿线补充：upper终板线各点中，行号 < 绿线交叉点行号的点，
        计算该点到20mm基线的水平列距（≤13mm），从该点水平向左扫 (25 - 列距_mm) mm
      - 橙线补充：lower终板线各点中，行号 > 橙线交叉点行号的点，逻辑对称

    信号检测（与原版一致）：
      4a. 下降沿：cur < running_max*(1-dr) AND cur < low_mean3 → flag='confirmed'
      4b. 上升沿：连续3点 > high_mean3*1.3，标注回退2像素 → flag='confirmed_rise'
      先到先得，break退出

    返回: [(row, col, flag), ...]
      flag: 'confirmed' / 'confirmed_rise' / 'not_found'
    """
    if c2_cols is None or c2_rows is None or f_img_2d is None:
        return []
    if not consensus_endplates:
        return []

    h_f, w_f = f_img_2d.shape
    scan_mm     = 25.0                               # 最大水平扫描距离mm
    scan_px     = int(round(scan_mm / pixel_spacing)) # 对应像素数
    max_supp_mm = 13.0                               # 补充扫描最大列距（33-20mm）
    win_px      = 2                                  # 平滑窗口2px
    dr          = float(drop_ratio3) if drop_ratio3 else 0.35
    rise_thr    = high_mean3 * 1.3                   # 上升沿触发阈值
    cross_tol   = 2                                  # 交叉点列差容差（像素）

    # ── 建立皮质线2逐行插值查找表（用于行号覆盖）
    _c2r_arr = np.array(c2_rows, dtype=np.float64)
    _c2c_arr = np.array(c2_cols, dtype=np.float64)

    # ── 从 scan_lines_v15 建立 row→col 查找表（20mm基线 & 33mm边界）
    # scan_lines_v15 格式: [(off_mm, rows_arr, cols_arr, nx_arr, ny_arr), ...]
    # 索引19=20mm，索引32=33mm
    row2col_20 = {}  # row → col at 20mm offset
    row2col_33 = {}  # row → col at 33mm offset
    if scan_lines_v15 and len(scan_lines_v15) >= 33:
        _l20 = scan_lines_v15[19]
        _l33 = scan_lines_v15[32]
        for _r, _c in zip(_l20[1], _l20[2]):
            row2col_20[int(_r)] = int(_c)
        for _r, _c in zip(_l33[1], _l33[2]):
            row2col_33[int(_r)] = int(_c)
    else:
        # fallback：用皮质线2水平偏移20mm/33mm
        if len(c2_rows) >= 2:
            _r_min = int(np.floor(_c2r_arr.min()))
            _r_max = int(np.ceil(_c2r_arr.max()))
            off20_px = int(round(20.0 / pixel_spacing))
            off33_px = int(round(33.0 / pixel_spacing))
            for _r in range(_r_min, _r_max + 1):
                _bc = float(np.interp(_r, _c2r_arr, _c2c_arr))
                row2col_20[_r] = int(round(_bc - off20_px))
                row2col_33[_r] = int(round(_bc - off33_px))

    # ── 解析 consensus_endplates，提取 superior/inferior 终板各点
    # 格式支持 dict（含 'points', 'ep_type', 'row_center'）
    superior_eps = []  # 上终板 Superior EP（绿色 lawngreen），列表of dict
    inferior_eps = []  # 下终板 Inferior EP（红色 red），列表of dict
    for ep in consensus_endplates:
        if isinstance(ep, dict):
            etype = ep.get('ep_type', 'superior')
            if etype == 'superior':
                superior_eps.append(ep)
            else:
                inferior_eps.append(ep)
    superior_eps.sort(key=lambda e: e['row_center'])
    inferior_eps.sort(key=lambda e: e['row_center'])

    # ── 构建椎体段列表：每个段对应一条 superior 终板 + 紧随其后的一条 inferior 终板
    # 同时记录该段用到的终板 ep dict（用于补充扫描）
    # ep_list 用于段划分（行号+类型）
    ep_list = []
    for ep in consensus_endplates:
        if isinstance(ep, dict):
            ep_list.append((int(ep['row_center']), ep.get('ep_type', 'superior'), ep))
        elif isinstance(ep, (list, tuple)) and len(ep) >= 4:
            ep_list.append((int(ep[0]), ep[3], None))
    ep_list.sort(key=lambda x: x[0])

    min_vert_px = int(round(8.0 / pixel_spacing))
    vert_segments = []  # (seg_r0, seg_r1, superior_ep_dict, inferior_ep_dict)
    for i in range(len(ep_list) - 1):
        r0, t0, ep0 = ep_list[i]
        r1, t1, ep1 = ep_list[i + 1]
        if t0 == 'superior' and t1 == 'inferior' and (r1 - r0) >= min_vert_px:
            vert_segments.append((r0, r1, ep0, ep1))

    if not vert_segments:
        print("   [下降沿法V2] 无有效椎体段，跳过")
        return []

    # ── 内部工具：对单行执行水平向左扫描并返回 (row, col, flag, src_tag, base_col)
    def _scan_row(row, start_col, scan_px_local, src_tag='main'):
        """从 (row, start_col) 水平向左采样 scan_px_local 点，检测前缘。
        src_tag: 'main' 主扫描段 / 'supp_upper' 上终板补允充 / 'supp_lower' 下终板补充
        返回格式: (row, col, flag, src_tag, base_col)
        base_col = 该行对应皮质线2的列坐标，用于密集窗口offset计算（与上升沿点统一）
        """
        # 计算该行对应的皮质线2列坐标
        _base_col = int(round(float(np.interp(float(row), _c2r_arr, _c2c_arr))))
        n_samples = scan_px_local + 1
        seg_raw = np.zeros(n_samples, dtype=np.float32)
        for t in range(n_samples):
            cf = float(start_col) - t   # 水平向左：col减小
            rf = float(row)
            ci0 = int(np.clip(int(np.floor(cf)), 0, w_f - 1))
            ci1 = int(np.clip(ci0 + 1,           0, w_f - 1))
            ri0 = int(np.clip(int(np.floor(rf)), 0, h_f - 1))
            ri1 = int(np.clip(ri0 + 1,           0, h_f - 1))
            wc = cf - np.floor(cf)
            wr = rf - np.floor(rf)
            seg_raw[t] = (float(f_img_2d[ri0, ci0]) * (1 - wr) * (1 - wc)
                        + float(f_img_2d[ri1, ci0]) * wr       * (1 - wc)
                        + float(f_img_2d[ri0, ci1]) * (1 - wr) * wc
                        + float(f_img_2d[ri1, ci1]) * wr       * wc)

        if n_samples < win_px + 2:
            return (row, int(round(start_col)), 'not_found', src_tag, _base_col)

        seg_sm = np.convolve(seg_raw, np.ones(win_px) / win_px, mode='same')
        ref = float(seg_sm[0])
        running_max = ref

        for t in range(1, len(seg_sm)):
            cur = float(seg_sm[t])
            if cur > running_max:
                running_max = cur
            # 下降沿检测（cur < low_mean3 * 1.3，允许皮质骨信号略高于椎间盘均值也能触发）
            if running_max > 1.0 and cur < running_max * (1.0 - dr) and cur < low_mean3 * 1.3:
                abs_col = int(np.clip(int(round(start_col - t)), 0, w_f - 1))
                return (row, abs_col, 'confirmed', src_tag, _base_col)
            # 上升沿检测（连续3点）—— 暂时注释掉，仅使用下降沿
            # if (cur > rise_thr
            #         and t + 2 < len(seg_sm)
            #         and float(seg_sm[t + 1]) > rise_thr
            #         and float(seg_sm[t + 2]) > rise_thr):
            #     t_mark  = max(t - 2, 0)
            #     abs_col = int(np.clip(int(round(start_col - t_mark)), 0, w_f - 1))
            #     return (row, abs_col, 'confirmed_rise', src_tag)

        return (row, int(round(start_col)), 'not_found', src_tag, _base_col)

    result = []
    n_confirmed = 0
    for (seg_r0, seg_r1, lower_ep, upper_ep) in vert_segments:

        # ── Step1: 找20mm基线与终板的交叉点行号
        # 上终板 Superior EP（绿色 lawngreen）交叉点：从 superior_ep 中统计
        row_g = None  # 绿线交叉点行号
        if lower_ep and isinstance(lower_ep, dict):
            _pts_u = lower_ep.get('points', [])
            _best_diff = cross_tol + 1
            for _p in _pts_u:
                _pr, _pc = int(_p[0]), int(_p[1])
                _base20 = row2col_20.get(_pr)
                if _base20 is None:
                    continue
                _diff = abs(_pc - _base20)
                if _diff < _best_diff:
                    _best_diff = _diff
                    row_g = _pr

        # 下终板 Inferior EP（红色 red）交叉点：从 inferior_ep 中统计
        row_o = None  # 橙线交叉点行号
        if upper_ep and isinstance(upper_ep, dict):
            _pts_l = upper_ep.get('points', [])
            _best_diff = cross_tol + 1
            for _p in _pts_l:
                _pr, _pc = int(_p[0]), int(_p[1])
                _base20 = row2col_20.get(_pr)
                if _base20 is None:
                    continue
                _diff = abs(_pc - _base20)
                if _diff < _best_diff:
                    _best_diff = _diff
                    row_o = _pr

        # 若找不到交叉点，fallback 用 seg_r0/seg_r1
        if row_g is None:
            row_g = seg_r1  # upper 终板行（椎体上缘）
        if row_o is None:
            row_o = seg_r0  # lower 终板行（椎体下缘）

        # 确保主扫描方向正确（row_g < row_o，即从上到下）
        main_r_top = min(row_g, row_o)
        main_r_bot = max(row_g, row_o)

        # ── Step2: 主扫描段（main_r_top ~ main_r_bot）
        for row in range(main_r_top, main_r_bot + 1):
            start_col = row2col_20.get(row)
            if start_col is None:
                continue
            res = _scan_row(row, start_col, scan_px, src_tag='main')
            result.append(res)
            if res[2] in ('confirmed', 'confirmed_rise'):
                n_confirmed += 1

        # ── Step3: 绿线补充（椎体上终板，对应 lower_ep，行号 < row_g）
        if lower_ep and isinstance(lower_ep, dict):
            _pts_u = lower_ep.get('points', [])
            for _p in _pts_u:
                _pr, _pc = int(_p[0]), int(_p[1])
                if _pr >= row_g:
                    continue  # 只处理行号 < 绿线交叉点的点（更靠上的上终板）
                _base20 = row2col_20.get(_pr)
                if _base20 is None:
                    continue
                # 列差：终板点比20mm基线偏右多少像素
                _col_diff_px = _base20 - _pc
                if _col_diff_px < 0:
                    continue  # 终板点已在20mm基线左侧，不做补充
                _col_diff_mm = _col_diff_px * pixel_spacing
                if _col_diff_mm > max_supp_mm:
                    continue  # 超过33mm范围，跳过
                _supp_mm = scan_mm - _col_diff_mm
                if _supp_mm <= 0:
                    continue
                _supp_px = int(round(_supp_mm / pixel_spacing))
                res = _scan_row(_pr, _pc, _supp_px, src_tag='supp_upper')
                result.append(res)
                if res[2] in ('confirmed', 'confirmed_rise'):
                    n_confirmed += 1
        
        # ── Step4: 橙线补充（椎体下终板，对应 upper_ep，行号 > row_o）
        if upper_ep and isinstance(upper_ep, dict):
            _pts_l = upper_ep.get('points', [])
            for _p in _pts_l:
                _pr, _pc = int(_p[0]), int(_p[1])
                if _pr <= row_o:
                    continue  # 只处理行号 > 橙线交叉点的点（更靠下的下终板）
                _base20 = row2col_20.get(_pr)
                if _base20 is None:
                    continue
                _col_diff_px = _base20 - _pc
                if _col_diff_px < 0:
                    continue
                _col_diff_mm = _col_diff_px * pixel_spacing
                if _col_diff_mm > max_supp_mm:
                    continue
                _supp_mm = scan_mm - _col_diff_mm
                if _supp_mm <= 0:
                    continue
                _supp_px = int(round(_supp_mm / pixel_spacing))
                res = _scan_row(_pr, _pc, _supp_px, src_tag='supp_lower')
                result.append(res)
                if res[2] in ('confirmed', 'confirmed_rise'):
                    n_confirmed += 1

    n_confirmed_desc = sum(1 for r in result if r[2] == 'confirmed')
    n_confirmed_rise = sum(1 for r in result if r[2] == 'confirmed_rise')
    print(f"   [下降沿法V2] 椎体段: {len(vert_segments)}, "
          f"确认前缘点: {n_confirmed}/{len(result)} "
          f"(下降沿:{n_confirmed_desc} 上升沿:{n_confirmed_rise})")
    return result


def find_arc_roi_min_points(f_img_2d, smooth_cols, all_rows, pixel_spacing,
                             left_off_mm=30.0, right_off_mm=50.0,
                             expand_ratio=1.2):
    """
    在弧形ROI（皮质线 offset left_off_mm ~ right_off_mm）内逐行找最低信号点。
    坐标系：皮质线在图像右边，offset向左增大，col = base_col - offset_px。
    left_off_mm ：ROI右边界（offset小，靠皮质线，列坐标大），固定不变。
    right_off_mm：ROI左边界（offset大，远离皮质线，列坐标小），随行号从顶到底线性扩展到 expand_ratio 倍。
    """
    h_f, w_f = f_img_2d.shape
    left_px  = int(round(left_off_mm  / pixel_spacing))  # 右边界，固定不变

    n_rows = len(all_rows)
    row_min = float(all_rows[0])  if n_rows > 0 else 0.0
    row_max = float(all_rows[-1]) if n_rows > 0 else 1.0
    row_span = max(row_max - row_min, 1.0)

    min_points = []
    for idx, row in enumerate(all_rows):
        row = int(row)
        if row < 0 or row >= h_f:
            continue
        # 左边界（right_off_mm）随行号增大：offset增大，列坐标减小，向左扩展
        # 顶行=right_off_mm, 底行=right_off_mm*expand_ratio
        t = (float(row) - row_min) / row_span
        right_off_row = right_off_mm * (1.0 + (expand_ratio - 1.0) * t)
        right_px = int(round(right_off_row / pixel_spacing))

        base_col = int(smooth_cols[idx])
        # 皮质线向左（前方）偏移：列坐标减小
        col_right = max(0, min(w_f - 1, base_col - left_px))
        col_left  = max(0, min(w_f - 1, base_col - right_px))
        if col_right <= col_left:
            continue
        roi_sig = f_img_2d[row, col_left:col_right + 1].astype(np.float32)
        if len(roi_sig) == 0:
            continue
        # 若同一行有多个低谷，取最靠近搜索区间中点的那个
        # 物理含义：不偏向皮质线侧也不偏向远端，取中间位置的谷底
        min_val = float(roi_sig.min())
        min_thr = min_val + 5.0  # 绝对值容差5，适配皮质骨低信号区
        cand_idxs = [i for i in range(len(roi_sig)) if float(roi_sig[i]) <= min_thr]
        mid_idx = (len(roi_sig) - 1) / 2.0  # 搜索区间中点索引
        min_idx = min(cand_idxs, key=lambda i: abs(i - mid_idx))  # 取最靠近中心的谷底
        min_col = col_left + min_idx
        min_val = float(roi_sig[min_idx])
        min_points.append((row, min_col, min_val, base_col))

    print(f"   [弧形ROI] offset {left_off_mm:.0f}~{right_off_mm:.0f}mm "
          f"(底行左边界扩展到{right_off_mm*expand_ratio:.0f}mm) "
          f"共 {len(min_points)} 行最低点")
    return min_points


def refine_arc_roi_to_anterior_edge(filtered_pts, f_img_2d, pixel_spacing,
                                     high_mean, rise_ratio=0.50,
                                     probe_ratio=0.6,
                                     scan_mm=20.0, smooth_win=2,
                                     skip_px=2,
                                     texture_mode=False, low_mean=50.0):
    """
    对弧形ROI谷底点，逐点向右（椎体方向）扫描 scan_mm，
    检测上升沿并确认后续持续高信号，定位椎体前沿。

    合并双轮精修为单轮（skip_px=2默认生效，texture_mode参数保留但不影响主逻辑）：
      skip_px=2   : 跳过皮质骨薄层，从骨髓内部开始判断
      模式C前置   : 全段信号<low_mean比例>50% → kept_low（混浊区/椎间盘）
      脂肪过滤    : 上升沿后邻域>high_mean×1.3 → 跳过
      模式B       : 触发上升沿后连续≥2px低信号 → 放弃此沿，继续找
      模式A       : 后续段高信号比例≥60% + 绝对均值≥high_thr → confirmed refined
    """
    if not filtered_pts:
        return []

    h_f, w_f = f_img_2d.shape
    scan_px  = max(2, int(round(scan_mm / pixel_spacing)))
    win_px   = 2
    high_thr = high_mean * 0.75

    result    = []
    n_refined = 0

    for (row, col, val, base_col) in filtered_pts:
        row = int(row)
        col = int(col)
        col_end   = min(w_f - 1, col + scan_px)
        col_start = min(col + skip_px, col_end)

        if col_end <= col_start + win_px:
            result.append((row, col, val, base_col, 'kept'))
            continue

        seg    = f_img_2d[row, col_start: col_end + 1].astype(float)
        seg_sm = np.convolve(seg, np.ones(win_px) / win_px, mode='same')

        # 模式C前置：全段低信号 → kept_low
        if float(np.mean(seg_sm < low_mean)) > 0.50:
            result.append((row, col, val, base_col, 'kept_low'))
            continue

        found_edge = False
        for j in range(1, len(seg_sm)):
            ref = float(seg_sm[j - 1])
            cur = float(seg_sm[j])
            if cur <= ref * (1.0 + rise_ratio):
                continue

            # 脂肪过滤
            abs_col_j1 = col_start + j + 1
            fat_vals = [float(f_img_2d[int(np.clip(row + dr, 0, h_f - 1)),
                                        int(np.clip(abs_col_j1 + dc, 0, w_f - 1))])
                        for dc in (0, 1, 2, 3) for dr in (-1, 0, 1)]
            if float(np.mean(fat_vals)) > high_mean * 1.3:
                continue

            rest = seg_sm[j + 1:]
            if len(rest) == 0:
                break

            # 后续段3px平滑
            rest_sm = np.convolve(np.pad(rest, 1, mode='edge'),
                                  np.ones(3) / 3, mode='valid')

            # 模式B：高低高 → 放弃，继续
            low_count, is_B = 0, False
            for k in range(len(rest_sm)):
                if rest_sm[k] < low_mean:
                    low_count += 1
                    if low_count >= 2:
                        is_B = True; break
                else:
                    low_count = 0
            if is_B:
                continue

            # 模式A：高信号占多数 + 绝对均值达标
            if (float(np.mean(rest_sm >= high_thr)) >= probe_ratio
                    and float(np.mean(rest_sm)) >= high_thr):
                new_col = col_start + j
                result.append((row, new_col, val, base_col, 'refined'))
                found_edge = True
                n_refined += 1
                break

        if not found_edge:
            result.append((row, col, val, base_col, 'kept'))

    print(f"   [前缘精修-单轮] {n_refined}/{len(filtered_pts)} 点更新到椎体前沿")
    result.sort(key=lambda p: p[0])
    return result


def filter_arc_roi_by_dense_offset(min_points, pixel_spacing,
                                    window_mm=4.0, step_mm=0.5,
                                    expand_ratio=1.0):
    """
    对弧形ROI最低点进行 offset 集中度过滤，并做平滑处理。

    算法：
      1. 计算每个点相对皮质线的 offset_mm = (base_col - col) * pixel_spacing
      2. 用滑动窗口（宽 window_mm，步长 step_mm）扫描 offset 轴，
         找点数最多的窗口 [best_lo, best_lo + window_mm]
      3. 只保留 offset 落在该窗口内的点
      4. 对保留点按行排序，用移动均值（窗口5行）平滑列坐标

    返回：
      filtered  : [(row, col_smooth, val, base_col), ...]  过滤+平滑后的点
      best_range: (lo_mm, hi_mm)  最密集窗口范围
    """
    if not min_points:
        return [], (0.0, window_mm)

    # 计算每点 offset（元组可能是4元或5元）
    offsets = np.array([(pt[3] - pt[1]) * pixel_spacing
                        for pt in min_points], dtype=np.float32)

    rows_arr   = np.array([pt[0] for pt in min_points], dtype=np.float32)
    row_min_f  = float(rows_arr.min()) if len(rows_arr) > 0 else 0.0
    row_max_f  = float(rows_arr.max()) if len(rows_arr) > 0 else 1.0
    row_span_f = max(row_max_f - row_min_f, 1.0)

    lo_min = float(offsets.min())
    lo_max = float(offsets.max()) - window_mm
    if lo_max < lo_min:
        lo_max = lo_min  # 范围本身小于窗口，整体保留

    # 滑动扫描找最密集窗口
    # 搜索阶段与过滤阶段几何一致：用梯形（右边界固定，左边界随行号动态扩展）统计点数
    step = step_mm
    best_lo    = lo_min
    best_count = 0
    cur = lo_min
    while cur <= lo_max + 1e-6:
        cnt = 0
        for i in range(len(min_points)):
            t_i   = (float(rows_arr[i]) - row_min_f) / row_span_f
            hi_dyn = cur + window_mm * (1.0 + (expand_ratio - 1.0) * t_i)
            if offsets[i] >= cur - 1e-6 and offsets[i] <= hi_dyn + 1e-6:
                cnt += 1
        if cnt > best_count:
            best_count = cnt
            best_lo    = cur
        cur += step

    best_hi = best_lo + window_mm
    print(f"   [offset过滤] 最密集窗口 {best_lo:.1f}~{best_hi:.1f}mm，"
          f"梯形统计 {best_count}/{len(min_points)} 点")

    # 过滤：右边界 best_lo 固定（offset小，靠皮质线），左边界随行号动态扩展（offset大，向左）
    kept = []
    for i, (pt, off) in enumerate(zip(min_points, offsets)):
        t = (float(pt[0]) - row_min_f) / row_span_f
        best_hi_row = best_lo + window_mm * (1.0 + (expand_ratio - 1.0) * t)
        if off >= best_lo - 1e-6 and off <= best_hi_row + 1e-6:
            kept.append((pt, off))
    if not kept:
        return [], (best_lo, best_hi)

    n_pts = len(kept)
    print(f"   [offset过滤] 动态窗口保留 {n_pts} 点（左边界底行扩展到{best_hi*expand_ratio:.1f}mm）")

    kept_pts = [item[0] for item in kept]
    # 按行号排序
    kept_pts.sort(key=lambda p: p[0])

    rows_raw = np.array([p[0] for p in kept_pts], dtype=np.float32)
    cols_raw = np.array([p[1] for p in kept_pts], dtype=np.float32)

    # ── 第一步：MAD过滤（窗口11行，阈值2.0σ）去除粗大偏差点 ──
    mad_win = min(11, len(cols_raw))
    if mad_win % 2 == 0:
        mad_win -= 1
    mad_win = max(1, mad_win)
    valid_mask = np.ones(len(cols_raw), dtype=bool)
    half_m = mad_win // 2
    for i in range(len(cols_raw)):
        s = max(0, i - half_m)
        e = min(len(cols_raw), i + half_m + 1)
        w = cols_raw[s:e]
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if mad > 0 and np.abs(cols_raw[i] - med) / (mad * 1.4826) > 2.0:
            valid_mask[i] = False
    clean_cols = cols_raw[valid_mask]
    clean_rows = rows_raw[valid_mask]
    n_removed = int(np.sum(~valid_mask))
    if n_removed > 0:
        print(f"   [前缘平滑] MAD过滤去除 {n_removed} 个粗大偏差点")

    if len(clean_rows) < 4:
        # 清洗后点太少，跳过插值直接用原始
        clean_rows = rows_raw
        clean_cols = cols_raw

    # ── 第二步：线性插值 → 填满所有行（连续无缺行）──
    all_rows_interp = np.arange(int(clean_rows[0]), int(clean_rows[-1]) + 1)
    interp_cols = np.interp(all_rows_interp, clean_rows, clean_cols)

    # ── 第三步：移动均值平滑（5mm，按 pixel_spacing 动态换算）──
    k = max(3, int(round(5.0 / pixel_spacing)))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(interp_cols, pad, mode='edge')
    kernel = np.ones(k) / k
    cols_sm = np.convolve(padded, kernel, mode='valid')

    # ── 将插值平滑后的所有行号（含插补行）全部输出，供二次精修使用 ──
    # base_col 用最近邻的原始点估算
    row_to_basecol = {int(pt[0]): pt[3] for pt in kept_pts}
    # 对插补行，用最近邻 kept_pts 的 base_col
    kept_rows_sorted = sorted(row_to_basecol.keys())
    # 按行号建立 flag 查找表（五元组才有 flag）
    row_to_flag = {int(pt[0]): pt[4] for pt in kept_pts if len(pt) > 4}

    filtered = []
    for r, c in zip(all_rows_interp, cols_sm):
        r_int = int(r)
        if r_int in row_to_basecol:
            base_col = row_to_basecol[r_int]
        else:
            # 最近邻
            idx = int(np.argmin(np.abs(np.array(kept_rows_sorted, dtype=np.float32) - r_int)))
            base_col = row_to_basecol[kept_rows_sorted[idx]]
        flag = row_to_flag.get(r_int, 'kept')
        filtered.append((float(r_int), float(c), 0.0, float(base_col), flag))
    
    return filtered, (best_lo, best_hi)


# ============ \u9aa8\u9ad3\u5b9a\u4f4d ============
