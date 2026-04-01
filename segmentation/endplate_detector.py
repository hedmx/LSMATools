"""
终板检测模块（V15）
包含 find_endplates_on_water_image（含全局回退与点数回退机制）
依赖：clustering.cluster_endplates_v15
"""
import numpy as np
from scipy import signal


def find_endplates_on_water_image(scan_lines_f, f_img_2d, pixel_spacing,
                                   drop_ratio_override=None, rise_ratio_override=None,
                                   use_half_global_med=False):
    """
    在压水图上识别终板线 (V15 扫描线逐行扫描版)
    信号特征：椎体=中高信号 (亮)，椎间盘/终板=低信号 (暗)
    流程：
      每条扫描线从头 (上) 向尾 (下) 逐行扫描:
        - 遇到第一个「下降沿」(信号从高变低 >= drop_ratio):
            向下探测 probe_mm，若持续低信号 >= probe_ratio → 标注为「上终板」
        - 遇到「上升沿」(信号从低变高 >= rise_ratio):
            向下探测 probe_mm，若持续高信号 >= probe_ratio → 标注为「下终板」
        - 继续向下扫描，找下一个变化
      最终：同空间行号范围内的候选点连成终板线（不做投票，直接按行号聚类连线）
    返回：{'endplates':[], 'all_candidates':[], 'raw_candidates':[], ...}
    
    参数：
        use_half_global_med - 全局回退模式：True 时所有扫描线使用 global_med × 0.5 作为初始判断基准
    """
    if not scan_lines_f:
        return {'endplates': [], 'vert_segments': [], 'all_candidates': []}

    h_f, w_f = f_img_2d.shape
    n_rows = len(scan_lines_f[0][2])  # 所有扫描线共享同一行坐标数组

    # ---- 固定基础参数 ----
    win_mm      = 3.0   # 局部参考窗口（mm）
    probe_ratio = 0.6   # 探测区域中低/高信号占比阈值
    min_sig_mm  = 5.0   # 最短有效信号区段（过滤噪声小波动）

    win_px     = max(2, int(win_mm    / pixel_spacing))
    min_sig_px = max(3, int(min_sig_mm / pixel_spacing))

    # 用于连线的聚类容差：同行号 ±cluster_px 内视为同一终板
    cluster_px = max(2, int(3.0 / pixel_spacing))

    # ---- 采集各扫描线信号 ----
    all_sigs = []
    for off_mm, cols, rows in scan_lines_f:
        sig = np.array([f_img_2d[r, c] for r, c in zip(rows, cols)], dtype=np.float32)
        all_sigs.append(sig)

    # ---- 三区信号统计：后区/中区/前区各取3条扫描线拼接后做Otsu分割 ----
    # 扫描线从皮质线2向前排列（index 0-based），共33条（0~32）
    # 后区代表线：第4~6条(index 3~5)  → high_mean1/low_mean1 → 适用第1~11条(0~10)
    # 中区代表线：第14~16条(index 13~15) → high_mean2/low_mean2 → 适用第12~21条(11~20)
    # 前区代表线：第24~26条(index 23~25) → high_mean3/low_mean3 → 适用第22~33条(21~32)
    from skimage.filters import threshold_otsu
    
    def _calc_zone_stats(sigs, idxs, wp):
        """将多条扫描线信号各自取中间50%再拼接后Otsu，返回(high_mean, low_mean, drop_ratio, rise_ratio)
        每条线单独取中间50%：去掉头尾各25%（头尾椎旁脂肪），再拼接做Otsu。
        """
        valid_idxs = [i for i in idxs if i < len(sigs)]
        if not valid_idxs:
            return 200.0, 50.0, 0.35, 0.32
        # 每条线单独平滑后取中间50%再拼接
        _segs = []
        for _i in valid_idxs:
            _s = np.convolve(sigs[_i], np.ones(wp)/wp, mode='same')
            _n = len(_s)
            _lo = int(_n * 0.25)
            _hi = int(_n * 0.75)
            _seg = _s[_lo:_hi] if _hi > _lo + 4 else _s
            _segs.append(_seg)
        combined_mid = np.concatenate(_segs)
        try:
            _ot = float(threshold_otsu(combined_mid))
        except Exception:
            _ot = float(np.median(combined_mid))
        _hv = combined_mid[combined_mid >= _ot]
        _lv = combined_mid[combined_mid <  _ot]
        _hm = float(np.mean(_hv)) if len(_hv) > 0 else float(np.max(combined_mid))
        _lm = float(np.mean(_lv)) if len(_lv) > 0 else float(np.min(combined_mid))
        # 局部极值统计（用于 drop_ratio 计算，仍在中间70%段内）
        _lmax = []
        _half = max(2, wp)
        for _i in range(_half, len(combined_mid) - _half):
            _win = combined_mid[_i - _half: _i + _half + 1]
            if combined_mid[_i] == np.max(_win) and combined_mid[_i] > _ot:
                _lmax.append(float(combined_mid[_i]))
        _pk  = float(np.median(_lmax)) if _lmax else _hm
        _drop_otsu   = (_hm - _ot) / (_hm + 1e-6)
        _drop_stable = (_pk - _ot) / (_pk + 1e-6)
        _dr = float(np.clip((_drop_otsu + _drop_stable) / 2.0, 0.25, 0.60))
        _rr = float(np.clip(_dr * 0.9, 0.25, 0.55))
        return _hm, _lm, _dr, _rr
    
    high_mean1, low_mean1, drop_ratio1, rise_ratio1 = _calc_zone_stats(all_sigs, [3, 4, 5],   win_px)
    high_mean2, low_mean2, drop_ratio2, rise_ratio2 = _calc_zone_stats(all_sigs, [13, 14, 15], win_px)
    high_mean3, low_mean3, drop_ratio3, rise_ratio3 = _calc_zone_stats(all_sigs, [23, 24, 25], win_px)
    
    # 兼容旧接口：全局 high_mean/low_mean/drop_ratio/rise_ratio 取中区（第2区）
    high_mean  = high_mean2
    low_mean   = low_mean2
    drop_ratio = drop_ratio2
    rise_ratio = rise_ratio2
    
    # probe_mm 自适应：沿用中区扫描线估算（mid_idx≈16，属中区）
    mid_idx = len(all_sigs) // 2
    mid_sig = all_sigs[mid_idx]
    mid_sm  = np.convolve(mid_sig, np.ones(win_px) / win_px, mode='same')
    # Otsu 分割（仅用于 probe_mm 估算和日志）
    otsu_thresh = float(threshold_otsu(mid_sm))
    lmax_vals, lmin_vals = [], []
    half = max(2, win_px)
    for _i in range(half, len(mid_sm) - half):
        window = mid_sm[_i - half: _i + half + 1]
        if mid_sm[_i] == np.max(window) and mid_sm[_i] > otsu_thresh:
            lmax_vals.append(float(mid_sm[_i]))
        if mid_sm[_i] == np.min(window) and mid_sm[_i] < otsu_thresh:
            lmin_vals.append(float(mid_sm[_i]))
    peak_med   = float(np.median(lmax_vals)) if lmax_vals else high_mean
    valley_med = float(np.median(lmin_vals)) if lmin_vals else low_mean
    drop_otsu        = (high_mean - otsu_thresh) / (high_mean + 1e-6)
    rise_otsu        = (otsu_thresh - low_mean)  / (otsu_thresh + 1e-6)
    drop_local_stable = (peak_med - otsu_thresh) / (peak_med + 1e-6)
    rise_local_stable = drop_local_stable * 0.85
    
    # ── 外部覆盖（回退策略用）── 三区同步覆盖
    if drop_ratio_override is not None:
        drop_ratio1 = drop_ratio2 = drop_ratio3 = float(np.clip(drop_ratio_override, 0.10, 0.60))
        drop_ratio  = drop_ratio2
        print(f"   [回退扫描] drop_ratio 全区覆盖为 {drop_ratio:.3f}")
    if rise_ratio_override is not None:
        rise_ratio1 = rise_ratio2 = rise_ratio3 = float(np.clip(rise_ratio_override, 0.10, 0.55))
        rise_ratio  = rise_ratio2
        print(f"   [回退扫描] rise_ratio 全区覆盖为 {rise_ratio:.3f}")
    
    # ===== probe_mm 自适应：基于局部极値间距估算椎间盘厉度 =====
    # 相邻极小值间距 ≈ 一个椎体+椎间盘周期，椎间盘约占 1/4
    if len(lmin_vals) >= 2:
        # 重新找极小值的位置
        lmin_pos = [_i for _i in range(half, len(mid_sm) - half)
                    if mid_sm[_i] == np.min(mid_sm[_i-half:_i+half+1])
                    and mid_sm[_i] < otsu_thresh]
        if len(lmin_pos) >= 2:
            gaps = [lmin_pos[k+1] - lmin_pos[k] for k in range(len(lmin_pos)-1)]
            period_px   = float(np.median(gaps))
            disc_est_mm = (period_px / 4.0) * pixel_spacing
            probe_mm    = float(np.clip(disc_est_mm, 1.0, 3.0))
        else:
            probe_mm = 2.5
    else:
        probe_mm = 2.5

    probe_px = max(2, int(probe_mm / pixel_spacing))

    print(f"   [三区统计] 后区(1~11): high={high_mean1:.0f} low={low_mean1:.0f} "
          f"drop={drop_ratio1:.2f} rise={rise_ratio1:.2f}")
    print(f"   [三区统计] 中区(12~21): high={high_mean2:.0f} low={low_mean2:.0f} "
          f"drop={drop_ratio2:.2f} rise={rise_ratio2:.2f}")
    print(f"   [三区统计] 前区(22~33): high={high_mean3:.0f} low={low_mean3:.0f} "
          f"drop={drop_ratio3:.2f} rise={rise_ratio3:.2f}")
    print(f"   [自适应] probe_mm={probe_mm:.1f}mm, 中区 Otsu={otsu_thresh:.0f}")
    
    # all_cand_detail: [(row, col, type_str, depth, line_idx), ...]
    # type_str: 'inferior'(下终板 Inferior EP=下降沿) 或 'superior'(上终板 Superior EP=上升沿)
    all_cand_detail = []
    
    for line_idx, (off_mm, cols, rows) in enumerate(scan_lines_f):
        sig  = all_sigs[line_idx]
        n    = len(sig)
        if n < win_px * 2 + 4:
            continue
    
        # 平滑信号，减少噪声影响（移动均值）
        kernel = np.ones(win_px) / win_px
        sig_sm = np.convolve(sig, kernel, mode='same')
            
        i = win_px
            
        # === 第一轮：正常阈值 ===
        min_line_candidates = 5  # 每条线至少需要 5 个候选点
        global_med = float(np.median(sig_sm))
        state = 'high' if sig_sm[i] > global_med else 'low'
        looking_for = 'drop' if state == 'high' else 'rise'
            
        print(f"   [Line {line_idx+1}] global_med={global_med:.1f}, start_state={state}")
            
        # 根据 line_idx 选对应区的阈值
        if line_idx <= 12:    # 后区：第 1~13 条
            _z_hm, _z_lm = high_mean1, low_mean1
            _z_dr, _z_rr = drop_ratio1, rise_ratio1
        elif line_idx <= 25:  # 中区：第 14~26 条
            _z_hm, _z_lm = high_mean2, low_mean2
            _z_dr, _z_rr = drop_ratio2, rise_ratio2
        else:                 # 前区：第 27~40 条
            _z_hm, _z_lm = high_mean3, low_mean3
            _z_dr, _z_rr = drop_ratio3, rise_ratio3
            
        line_candidates = []  # 当前线的候选点
            
        while i < n - probe_px - 1:
            cur = float(sig_sm[i])
            ref = float(np.mean(sig_sm[max(0, i - win_px):i])) if i >= win_px else global_med
            ref = max(ref, 1.0)
                
            if looking_for == 'drop':
                if cur < ref * (1.0 - _z_dr):
                    probe_slice = sig_sm[i + 1: i + probe_px + 1]
                    low_thresh  = ref * (1.0 - _z_dr * 0.5)
                    if len(probe_slice) > 0 and np.mean(probe_slice < low_thresh) >= probe_ratio:
                        probe_abs_mean = float(np.mean(probe_slice))
                        _ok1 = probe_abs_mean < _z_hm * 0.55
                        _ok2 = probe_abs_mean < _z_lm * 1.5
                        if not (_ok1 or _ok2):
                            i += 1
                            continue
                        r_idx = min(i, len(rows) - 1)
                        c_idx = min(i, len(cols) - 1)
                        line_candidates.append((
                            int(rows[r_idx]), int(cols[c_idx]), 'inferior',
                            float((ref - cur) / (ref + 1e-6)),
                            line_idx
                        ))
                        state       = 'low'
                        looking_for = 'rise'
                        i += probe_px
                        continue
                
            else:  # looking_for == 'rise'
                if cur > ref * (1.0 + _z_rr):
                    probe_slice = sig_sm[i + 1: i + probe_px + 1]
                    high_thresh = ref * (1.0 + _z_rr * 0.5)
                    if len(probe_slice) > 0 and np.mean(probe_slice > high_thresh) >= probe_ratio:
                        probe_abs_mean = float(np.mean(probe_slice))
                        if probe_abs_mean < _z_hm * 0.70:
                            i += 1
                            continue
                        r_idx = min(i, len(rows) - 1)
                        c_idx = min(i, len(cols) - 1)
                        line_candidates.append((
                            int(rows[r_idx]), int(cols[c_idx]), 'superior',
                            float((cur - ref) / (ref + 1e-6)),
                            line_idx
                        ))
                        state       = 'high'
                        looking_for = 'drop'
                        i += probe_px
                        continue
                
            i += 1
            
        # 加入总候选点
        all_cand_detail.extend(line_candidates)
    
    if not all_cand_detail:
        return {'endplates': [], 'all_candidates': all_cand_detail,
                'raw_candidates': [], 'all_sigs': all_sigs, 'scan_lines_f': scan_lines_f,
                'pixel_spacing': pixel_spacing, 'high_mean': high_mean, 'low_mean': low_mean,
                'high_mean1': high_mean1, 'low_mean1': low_mean1,
                'high_mean2': high_mean2, 'low_mean2': low_mean2,
                'high_mean3': high_mean3, 'low_mean3': low_mean3,
                'drop_ratio': drop_ratio, 'rise_ratio': rise_ratio,
                'drop_ratio3': drop_ratio3, 'rise_ratio3': rise_ratio3,
                'endplate_point_groups': []}
    
    # ---- 按行号聚类，同一空间范围内连成终板线 ----
    # 先按 row 排序
    all_cand_detail.sort(key=lambda x: x[0])
        
    # 将候选点按 ±cluster_px 分组，每组取中位行号作为代表行
    groups = []   # [(rep_row, [( row,col,type,depth), ...]), ...]
    for pt in all_cand_detail:
        row_pt = pt[0]
        placed = False
        for g in groups:
            if abs(row_pt - g[0]) <= cluster_px:
                g[1].append(pt)
                # 更新代表行为组内中位数
                g[0] = int(np.median([p[0] for p in g[1]]))
                placed = True
                break
        if not placed:
            groups.append([row_pt, [pt]])
    
    # ---- 构建终板线（endplates）和 all_candidates ----
    min_pts   = 5  # 至少 5 个候选点支持（去重后）；宽松阅値预防侧弯病例被过滤
    min_cover = 3  # 必须来自 ≥ 3 条不同扫描线（主要靠 min_pts+去重双重防假）
    raw_eps = []  # [(rep_row, avg_col, avg_dep, ep_type, n_pts, curve_pts), ...]
    for rep_row, pts in groups:
        # ---- 同一扫描线内去重：只保留 depth 最大的点 ----
        by_line = {}
        for p in pts:
            lidx = p[4]  # line_idx
            if lidx not in by_line or p[3] > by_line[lidx][3]:
                by_line[lidx] = p
        deduped_pts = list(by_line.values())
        n_cover = len(by_line)  # 覆盖的不同扫描线数
        if len(deduped_pts) < min_pts:
            continue
        if n_cover < min_cover:
            continue
        pts = deduped_pts  # 后续用去重后的点集
        n_superior = sum(1 for p in pts if p[2] == 'superior')
        n_inferior = sum(1 for p in pts if p[2] == 'inferior')
        ep_type = 'superior' if n_superior >= n_inferior else 'inferior'
        avg_col = int(np.mean([p[1] for p in pts]))
        avg_dep = float(np.mean([p[3] for p in pts]))
        curve_pts = sorted([(p[1], p[0]) for p in pts], key=lambda x: x[0])  # (col, row)
        raw_eps.append((rep_row, avg_col, avg_dep, ep_type, len(pts), curve_pts))

    # ---- 严格交替校验：上下终板必须 upper → lower → upper 交替 ----
    # 如果出现连续同类型，保留支持点数更多的（avg_dep 更大的），丢弃另一个
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

    raw_eps.sort(key=lambda x: x[0])  # 按行号升序
    filtered_eps = enforce_alternating(raw_eps)
    
    endplates = []
    endplate_point_groups = []
    for rep_row, avg_col, avg_dep, ep_type, n_pts, curve_pts in filtered_eps:
        endplates.append((rep_row, avg_col, avg_dep, ep_type))
        endplate_point_groups.append((ep_type, curve_pts))

    # ---- 列坐标对齐到中心扫描线 ----
    endplates = align_to_midline(endplates, scan_lines_f)

    # ---- 解剖间距复核与预测修正 ----
    # endplates 格式: (row, col, depth, ep_type)
    endplates_corrected = anatomical_gap_correction(
        endplates, all_sigs, scan_lines_f, pixel_spacing
    )
    # endplates_corrected 格式：(row, col, depth, ep_type, predicted)
    # 打印预测修正结果（仅保留解剖修正的日志）
    for idx, (r, c, d, t, pred) in enumerate(endplates_corrected):
        if pred:
            color_ch = '红 (下终板 Inferior)±' if t == 'inferior' else '绿 (上终板 Superior)±'
            print(f"   [解剖修正] {color_ch} @ row={r}, col={c}")

    # ---- V12.3 缺失终板插补 ----
    # 对 anatomical_gap_correction 输出后再做一道扫描：
    #   情况1：同类型相邻（upper→upper 或 lower→lower）→ 中间缺一个对立类型
    #   情况2：正常交替但间距 > 1.8×(avg_disc+avg_vert) → 中间缺整对
    # 插入的点 predicted=True，和真实点一样参与后续椎体编号计算
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

    endplates_corrected = fill_missing_endplates(
        endplates_corrected, scan_lines_f, pixel_spacing
    )

    all_cand_vis = [(r, c, d) for r, c, t, d, _li in all_cand_detail]

    return {
        'endplates':             endplates_corrected,  # (row,col,depth,ep_type,predicted)
        'endplate_point_groups': endplate_point_groups,
        'all_candidates':        all_cand_vis,
        'raw_candidates':        all_cand_detail,
        'all_sigs':              all_sigs,
        'scan_lines_f':          scan_lines_f,
        'pixel_spacing':         pixel_spacing,
        'high_mean':             high_mean,
        'low_mean':              low_mean,
        'high_mean1':            high_mean1,
        'low_mean1':             low_mean1,
        'high_mean2':            high_mean2,
        'low_mean2':             low_mean2,
        'high_mean3':            high_mean3,
        'low_mean3':             low_mean3,
        'drop_ratio':            drop_ratio,
        'rise_ratio':            rise_ratio,
        'drop_ratio3':           drop_ratio3,
        'rise_ratio3':           rise_ratio3,
    }


