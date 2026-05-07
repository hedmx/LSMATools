"""
junction_detector.py - Step2 终板汇合点扫描 + Step2b 间距校验补全
"""
import math
import numpy as np
from ._scan_utils import _project_to_c2


def scan_endplate_junction_points(in_img_2d, c2_rows, c2_cols,
                                   pixel_spacing, low_mean,
                                   win_h_mm=2.0, win_w_mm=2.0,
                                   low_ratio=0.8, smooth_win=3,
                                   suppress_mm=8.0):
    """
    沿皮质线2-2从顶向下逐点滑动（遍历基准改为c2，几何稳定性更高），
    滑动窗口切线/法线方向跟随皮质线2局部切线，
    窗口锚点为c2当前点沿法线腹侧偏移，偏移量分段线性渐变：
      - 0%~60% 行程：offset 从 1mm 线性渐变到 3mm
      - 60%~100% 行程：offset 固定 3mm

    返回：
        (junction_pts, anchor_pts_out)
        junction_pts = [(row, col, mean_val, idx_in_c2), ...] 终板汇合点列表
        anchor_pts_out = [(ar, ac), ...] 锚点坐标列表
    """
    H, W = in_img_2d.shape
    c2r = np.array(c2_rows, dtype=np.float64)
    c2c = np.array(c2_cols, dtype=np.float64)

    win_h_px    = max(1, int(round(win_h_mm  / pixel_spacing)))
    win_w_px    = max(1, int(round(win_w_mm  / pixel_spacing)))
    suppress_px = suppress_mm / pixel_spacing

    r_min = float(c2r[0]); r_max = float(c2r[-1])
    r_span = max(r_max - r_min, 1e-9)

    step_dists = np.zeros(len(c2r), dtype=np.float64)
    for i in range(1, len(c2r)):
        step_dists[i] = math.sqrt(
            (c2r[i] - c2r[i - 1]) ** 2 + (c2c[i] - c2c[i - 1]) ** 2)

    win_means = []
    anchor_pts = []
    for i in range(len(c2r)):
        t = float(np.clip((c2r[i] - r_min) / r_span, 0.0, 1.0))
        if t <= 0.6:
            frac = t / 0.6
            offset_mm = 1.0 + frac * (3.0 - 1.0)
        else:
            offset_mm = 3.0
        offset_px = offset_mm / pixel_spacing

        _, _, n_row, n_col = _project_to_c2(
            float(c2r[i]), float(c2c[i]), c2r, c2c)
        if n_col > 0:
            n_row, n_col = -n_row, -n_col
        t_dr = -n_col; t_dc = n_row
        t_len = math.sqrt(t_dr * t_dr + t_dc * t_dc) + 1e-9
        t_dr /= t_len; t_dc /= t_len

        ar = float(c2r[i]) + n_row * offset_px
        ac = float(c2c[i]) + n_col * offset_px
        anchor_pts.append((ar, ac))

        samples = []
        for dh in range(-win_h_px // 2, win_h_px // 2 + 1):
            for dw in range(0, win_w_px + 1):
                sr = ar + t_dr * dh + n_row * dw
                sc = ac + t_dc * dh + n_col * dw
                ri = int(np.clip(int(round(sr)), 0, H - 1))
                ci = int(np.clip(int(round(sc)), 0, W - 1))
                samples.append(float(in_img_2d[ri, ci]))
        win_means.append(float(np.mean(samples)) if samples else 0.0)

    if len(win_means) >= smooth_win:
        pad = smooth_win // 2
        sm = np.convolve(
            np.pad(win_means, pad, mode='edge'),
            np.ones(smooth_win) / smooth_win, mode='valid')
        sm = sm[:len(win_means)]
    else:
        sm = np.array(win_means)

    thresh = low_mean * low_ratio
    junction_pts = []
    anchor_pts_out = []
    suppress_accum = 0.0
    in_candidate = False
    candidate_seg = []

    def _flush_candidate(seg):
        if not seg:
            return
        mid_i = seg[len(seg) // 2]
        junction_pts.append(
            (float(c2r[mid_i]), float(c2c[mid_i]), float(sm[mid_i]), mid_i))
        ar, ac = anchor_pts[mid_i]
        anchor_pts_out.append((float(ar), float(ac)))

    for i in range(1, len(sm) - 1):
        suppress_accum += float(step_dists[i])
        if suppress_accum < suppress_px:
            if in_candidate:
                in_candidate = False
                candidate_seg = []
            continue

        cur  = float(sm[i])
        prev = float(sm[i - 1])
        nxt  = float(sm[i + 1])

        if cur < thresh:
            if not in_candidate:
                if cur < prev:
                    in_candidate = True
                    candidate_seg = [i]
            else:
                candidate_seg.append(i)
        else:
            if in_candidate:
                _flush_candidate(candidate_seg)
                suppress_accum = 0.0
                in_candidate = False
                candidate_seg = []

    if in_candidate and candidate_seg:
        _flush_candidate(candidate_seg)

    print(f"   [Step2] 皮质线2-2共{len(c2r)}点，"
          f"窗口{win_h_mm}×{win_w_mm}mm，静默{suppress_mm}mm，"
          f"offset分段1→3mm，"
          f"找到终板汇合点 {len(junction_pts)} 个")
    for _i, _jp in enumerate(junction_pts):
        print(f"   [Step2] 点[{_i}] c2=({_jp[0]:.1f},{_jp[1]:.1f}) mean={_jp[2]:.1f}")
    return junction_pts, anchor_pts_out


def repair_junction_pts(junction_pts, anchor_pts_list,
                        c2_rows, c2_cols,
                        pixel_spacing,
                        in_img_2d=None,
                        c2_rows_scan=None, c2_cols_scan=None,
                        low_mean=None, low_ratio=0.8,
                        win_h_mm=2.0, win_w_mm=2.0):
    """
    对终板汇合点列表做间距双向校验：
    A. 去多（假阳性）
    B. 补少（漏检）
    C. 末尾实扫补全

    路径弧长基准统一使用皮质线2-2（c2_rows/c2_cols）。

    返回：
        new_junction_pts, new_anchor_pts
    """
    if len(junction_pts) < 2:
        return junction_pts, anchor_pts_list

    c2r = np.array(c2_rows, dtype=np.float64)
    c2c = np.array(c2_cols, dtype=np.float64)

    c2_cum = np.zeros(len(c2r), dtype=np.float64)
    for k in range(1, len(c2r)):
        c2_cum[k] = c2_cum[k - 1] + math.sqrt(
            (c2r[k] - c2r[k - 1]) ** 2 + (c2c[k] - c2c[k - 1]) ** 2)

    # 将每个汇合点投影到c2，获取idx_in_c2
    def _proj_to_c2_idx(pr, pc):
        dists = (c2r - pr) ** 2 + (c2c - pc) ** 2
        return int(np.argmin(dists))

    paired = sorted(zip(junction_pts, anchor_pts_list), key=lambda x: x[0][0])
    jpts   = [p[0] for p in paired]
    apts   = [p[1] for p in paired]

    # 重建每个汇合点的c2索引（替换原 idx_in_c1 字段）
    jpts = [(jr, jc, jv, _proj_to_c2_idx(jr, jc)) for (jr, jc, jv, _) in jpts]

    def _calc_gaps(jpts_):
        gs = []
        for i in range(len(jpts_) - 1):
            ia = int(np.clip(int(jpts_[i][3]),     0, len(c2_cum) - 1))
            ib = int(np.clip(int(jpts_[i + 1][3]), 0, len(c2_cum) - 1))
            gs.append(c2_cum[ib] - c2_cum[ia])
        return gs

    # ── A. 去多（假阳性）──
    changed = True
    while changed and len(jpts) >= 3:
        changed = False
        gaps = _calc_gaps(jpts)
        normal_gap = float(np.median(gaps))
        if normal_gap < 1e-3:
            break
        thresh_sum  = normal_gap * 1.5   # 两侧之和触发二次校验的阈值
        thresh_half = normal_gap * 0.75  # 单侧判断真假阳性的阈值
        for i in range(1, len(jpts) - 1):
            gap_prev = gaps[i - 1]
            gap_next = gaps[i]
            if gap_prev + gap_next < thresh_sum:
                # 二次校验：根据两侧间距大小决定删哪个点
                prev_short = gap_prev <= thresh_half
                next_short = gap_next <= thresh_half
                if prev_short and next_short:
                    # 两侧都短：当前点(i)确认假阳
                    del_idx = i
                    reason = (f"两侧均≤0.75×normal "
                              f"({gap_prev:.1f}+{gap_next:.1f})px，删除点[{del_idx}]")
                elif prev_short and not next_short:
                    # gap_prev短、gap_next正常：当前点(i)为假阳
                    del_idx = i
                    reason = (f"gap_prev={gap_prev:.1f}≤0.75×normal，"
                              f"gap_next={gap_next:.1f}>0.75×normal，删除点[{del_idx}]")
                else:
                    # gap_prev正常、gap_next短：下一个点(i+1)为假阳
                    del_idx = i + 1
                    reason = (f"gap_prev={gap_prev:.1f}>0.75×normal，"
                              f"gap_next={gap_next:.1f}≤0.75×normal，删除点[{del_idx}]")
                print(f"   [Step2b-去多] 两侧间距之和"
                      f"({gap_prev:.1f}+{gap_next:.1f}={gap_prev+gap_next:.1f})px"
                      f" < 1.5×normal({normal_gap:.1f})px → {reason}")
                jpts.pop(del_idx)
                apts.pop(del_idx)
                changed = True
                break

    # ── B. 补少（漏检）──
    if len(jpts) < 2:
        return jpts, apts

    gaps = _calc_gaps(jpts)
    normal_gap = float(np.median(gaps))
    if normal_gap < 1e-3:
        return jpts, apts

    new_jpts = [jpts[0]]
    new_apts = [apts[0]]

    for i, gap in enumerate(gaps):
        n_insert = int(round(gap / normal_gap)) - 1
        if n_insert > 0:
            idx_a = int(np.clip(int(jpts[i][3]),     0, len(c2_cum) - 1))
            idx_b = int(np.clip(int(jpts[i + 1][3]), 0, len(c2_cum) - 1))
            dist_a = c2_cum[idx_a]
            dist_b = c2_cum[idx_b]
            for k in range(1, n_insert + 1):
                frac = k / (n_insert + 1)
                target_dist = dist_a + frac * (dist_b - dist_a)
                c2_idx = int(np.argmin(np.abs(c2_cum - target_dist)))
                ins_r = float(c2r[c2_idx])
                ins_c = float(c2c[c2_idx])
                new_jpts.append((ins_r, ins_c, 0.0, c2_idx))
                new_apts.append((ins_r, ins_c))
            print(f"   [Step2b-补少] 区间[{i}→{i+1}] 间距{gap:.1f}px"
                  f" > 1.5×normal({normal_gap:.1f}px)，插入{n_insert}个补全点")
        new_jpts.append(jpts[i + 1])
        new_apts.append(apts[i + 1])

    paired_out = sorted(zip(new_jpts, new_apts), key=lambda x: x[0][0])
    out_jpts = [p[0] for p in paired_out]
    out_apts = [p[1] for p in paired_out]

    # ── B'. 末尾点偏移修正 ──
    # 若末尾点与前一个点的间距 < 0.8×median，认为末尾点过早触发（假阳性/偏移），循环删除
    if len(out_jpts) >= 2:
        gaps_b2 = _calc_gaps(out_jpts)
        median_b2 = float(np.median(gaps_b2)) if gaps_b2 else 0.0
        if median_b2 > 1e-3:
            while len(out_jpts) >= 2:
                ia_ = int(np.clip(int(out_jpts[-2][3]), 0, len(c2_cum) - 1))
                ib_ = int(np.clip(int(out_jpts[-1][3]), 0, len(c2_cum) - 1))
                gap_last = c2_cum[ib_] - c2_cum[ia_]
                if gap_last < 0.7 * median_b2:
                    print(f"   [Step2b'-末尾修正] 末尾间距{gap_last:.1f}px"
                          f" < 0.7×median({median_b2:.1f}px)，删除末尾点"
                          f" c2=({out_jpts[-1][0]:.1f},{out_jpts[-1][1]:.1f})")
                    out_jpts.pop()
                    out_apts.pop()
                else:
                    break

    # ── C. 末尾补偿扫描 ──
    if len(out_jpts) >= 1:
        if len(out_jpts) >= 2:
            gaps_final = []
            for i in range(len(out_jpts) - 1):
                ia_ = int(np.clip(int(out_jpts[i][3]),     0, len(c2_cum) - 1))
                ib_ = int(np.clip(int(out_jpts[i + 1][3]), 0, len(c2_cum) - 1))
                gaps_final.append(c2_cum[ib_] - c2_cum[ia_])
            tail_normal_gap = float(np.median(gaps_final))
        else:
            tail_normal_gap = normal_gap

        if tail_normal_gap > 1e-3:
            last_c2_idx = int(np.clip(int(out_jpts[-1][3]), 0, len(c2_cum) - 1))
            tail_remaining = c2_cum[-1] - c2_cum[last_c2_idx]

            n_tail_floor = int(tail_remaining // tail_normal_gap)
            n_tail = min(4, n_tail_floor)
            if n_tail_floor == 0:
                print(f"   [Step2b-末尾] tail_remaining={tail_remaining:.1f}px"
                      f" < gap={tail_normal_gap:.1f}px，剩余不足一个间距，跳过补偿扫描")

            can_scan = (in_img_2d is not None and c2_rows_scan is not None
                        and c2_cols_scan is not None and low_mean is not None)

            if n_tail > 0 and can_scan:
                H_img, W_img = in_img_2d.shape
                c2r_t = np.array(c2_rows_scan, dtype=np.float64)
                c2c_t = np.array(c2_cols_scan, dtype=np.float64)
                win_h_px = max(1, int(round(win_h_mm / pixel_spacing)))
                win_w_px = max(1, int(round(win_w_mm / pixel_spacing)))
                thresh   = low_mean * low_ratio

                # 末尾补偿扫描：沿c2路径滑动，静默距离20mm
                suppress_path_px = 20.0 / pixel_spacing

                found_count = 0
                suppress_accum = 0.0
                in_candidate = False
                candidate_seg_tail = []

                # 当前基准c2_idx（用于距离校验，随每次采纳/插入预测点更新）
                base_c2_idx = last_c2_idx

                for i in range(last_c2_idx + 1, len(c2r)):
                    suppress_accum += math.sqrt(
                        (c2r[i] - c2r[i-1])**2 + (c2c[i] - c2c[i-1])**2)

                    # 主动超距检测：累积弧长超过1.2×median时，不等封口直接插预测点
                    if suppress_accum > 1.2 * tail_normal_gap:
                        pred_dist = c2_cum[base_c2_idx] + tail_normal_gap
                        if pred_dist <= c2_cum[-1]:
                            pred_idx = int(np.argmin(np.abs(c2_cum - pred_dist)))
                            pred_r = float(c2r[pred_idx])
                            pred_c = float(c2c[pred_idx])
                            out_jpts.append((pred_r, pred_c, 0.0, pred_idx))
                            out_apts.append((pred_r, pred_c))
                            print(f"   [Step2b-补偿扫描] 累积超过1.2×，插入预测点[{found_count+1}]:"
                                  f" c2=({pred_r:.1f},{pred_c:.1f})")
                            found_count += 1
                            base_c2_idx = pred_idx
                            suppress_accum = 0.0
                            in_candidate = False
                            candidate_seg_tail = []
                            if found_count >= n_tail:
                                break
                        continue

                    if suppress_accum < suppress_path_px:
                        if in_candidate:
                            in_candidate = False
                            candidate_seg_tail = []
                        continue

                    # 计算该c2点的法线方向
                    _, _, n_row, n_col = _project_to_c2(
                        float(c2r[i]), float(c2c[i]), c2r_t, c2c_t)
                    if n_col > 0:
                        n_row, n_col = -n_row, -n_col
                    t_dr = -n_col; t_dc = n_row
                    t_len = math.sqrt(t_dr*t_dr + t_dc*t_dc) + 1e-9
                    t_dr /= t_len; t_dc /= t_len

                    # offset -2mm～16mm 步进1mm 逐档扫描，找到低信号即停止
                    ar, ac, mean_val = None, None, 9999.0
                    for off_mm in range(-2, 17):
                        off_px = off_mm / pixel_spacing
                        ar_cand = float(c2r[i]) + n_row * off_px
                        ac_cand = float(c2c[i]) + n_col * off_px
                        samples = []
                        for dh in range(-win_h_px // 2, win_h_px // 2 + 1):
                            for dw in range(0, win_w_px + 1):
                                sr = ar_cand + t_dr * dh + n_row * dw
                                sc = ac_cand + t_dc * dh + n_col * dw
                                ri = int(np.clip(int(round(sr)), 0, H_img - 1))
                                ci = int(np.clip(int(round(sc)), 0, W_img - 1))
                                samples.append(float(in_img_2d[ri, ci]))
                        mv = float(np.mean(samples)) if samples else 9999.0
                        if mv < thresh:
                            ar, ac, mean_val = ar_cand, ac_cand, mv
                            break

                    if ar is None:
                        # 该c2点所有offset均未命中低信号 → 封口候选段
                        if in_candidate and candidate_seg_tail:
                            mid = candidate_seg_tail[len(candidate_seg_tail) // 2]
                            mid_i, mid_ar, mid_ac, mid_mv = mid
                            j_r = float(c2r[mid_i])
                            j_c = float(c2c[mid_i])

                            # 距离校验：候选点与当前基准的间距须在 [0.7×, 1.2×] 内
                            gap_cand = c2_cum[mid_i] - c2_cum[base_c2_idx]
                            if 0.7 * tail_normal_gap <= gap_cand <= 1.2 * tail_normal_gap:
                                # 采纳
                                out_jpts.append((j_r, j_c, mid_mv, mid_i))
                                out_apts.append((mid_ar, mid_ac))
                                print(f"   [Step2b-补偿扫描] 采纳点[{found_count+1}]:"
                                      f" c2=({j_r:.1f},{j_c:.1f})"
                                      f" anchor=({mid_ar:.1f},{mid_ac:.1f})"
                                      f" mean={mid_mv:.1f} gap={gap_cand:.1f}px")
                                found_count += 1
                                base_c2_idx = mid_i
                                suppress_accum = 0.0
                            elif gap_cand > 1.2 * tail_normal_gap:
                                # 超过1.2×：插入预测点（按median_gap等分）
                                pred_dist = c2_cum[base_c2_idx] + tail_normal_gap
                                if pred_dist <= c2_cum[-1]:
                                    pred_idx = int(np.argmin(np.abs(c2_cum - pred_dist)))
                                    pred_r = float(c2r[pred_idx])
                                    pred_c = float(c2c[pred_idx])
                                    out_jpts.append((pred_r, pred_c, 0.0, pred_idx))
                                    out_apts.append((pred_r, pred_c))
                                    print(f"   [Step2b-补偿扫描] 超过1.2×，插入预测点[{found_count+1}]:"
                                          f" c2=({pred_r:.1f},{pred_c:.1f})")
                                    found_count += 1
                                    base_c2_idx = pred_idx
                                    suppress_accum = 0.0
                            else:
                                # gap < 0.8×：过早命中，丢弃，不重置suppress_accum
                                print(f"   [Step2b-补偿扫描] 丢弃过早点:"
                                      f" c2=({j_r:.1f},{j_c:.1f})"
                                      f" gap={gap_cand:.1f}px < 0.7×{tail_normal_gap:.1f}px")

                            in_candidate = False
                            candidate_seg_tail = []
                            if found_count >= n_tail:
                                break
                        else:
                            in_candidate = False
                            candidate_seg_tail = []
                        continue

                    # 命中低信号，累积候选段
                    if not in_candidate:
                        in_candidate = True
                        candidate_seg_tail = [(i, ar, ac, mean_val)]
                    else:
                        candidate_seg_tail.append((i, ar, ac, mean_val))

                # 处理未封口的候选段（c2末尾）
                if in_candidate and candidate_seg_tail and found_count < n_tail:
                    mid = candidate_seg_tail[len(candidate_seg_tail) // 2]
                    mid_i, mid_ar, mid_ac, mid_mv = mid
                    j_r = float(c2r[mid_i])
                    j_c = float(c2c[mid_i])
                    gap_cand = c2_cum[mid_i] - c2_cum[base_c2_idx]
                    if 0.7 * tail_normal_gap <= gap_cand <= 1.2 * tail_normal_gap:
                        out_jpts.append((j_r, j_c, mid_mv, mid_i))
                        out_apts.append((mid_ar, mid_ac))
                        print(f"   [Step2b-补偿扫描] 采纳末端点[{found_count+1}]:"
                              f" c2=({j_r:.1f},{j_c:.1f})"
                              f" anchor=({mid_ar:.1f},{mid_ac:.1f})"
                              f" mean={mid_mv:.1f} gap={gap_cand:.1f}px")
                        found_count += 1
                    elif gap_cand > 1.2 * tail_normal_gap:
                        pred_dist = c2_cum[base_c2_idx] + tail_normal_gap
                        if pred_dist <= c2_cum[-1]:
                            pred_idx = int(np.argmin(np.abs(c2_cum - pred_dist)))
                            pred_r = float(c2r[pred_idx])
                            pred_c = float(c2c[pred_idx])
                            out_jpts.append((pred_r, pred_c, 0.0, pred_idx))
                            out_apts.append((pred_r, pred_c))
                            print(f"   [Step2b-补偿扫描] 末端超过1.2×，插入预测点[{found_count+1}]:"
                                  f" c2=({pred_r:.1f},{pred_c:.1f})")
                            found_count += 1
                    else:
                        print(f"   [Step2b-补偿扫描] 末端过早点丢弃:"
                              f" c2=({j_r:.1f},{j_c:.1f}) gap={gap_cand:.1f}px")

                if found_count == 0:
                    print(f"   [Step2b-补偿扫描] 未找到补全点"
                          f" (last_c2_idx={last_c2_idx} 剩余路径={tail_remaining:.1f}px)")

                paired_tail = sorted(zip(out_jpts, out_apts), key=lambda x: x[0][0])
                out_jpts = [p[0] for p in paired_tail]
                out_apts = [p[1] for p in paired_tail]

            elif n_tail > 0 and not can_scan:
                for k in range(1, n_tail + 1):
                    target_dist = c2_cum[last_c2_idx] + k * tail_normal_gap
                    if target_dist > c2_cum[-1]:
                        break
                    c2_idx_new = int(np.argmin(np.abs(c2_cum - target_dist)))
                    ins_r = float(c2r[c2_idx_new])
                    ins_c = float(c2c[c2_idx_new])
                    out_jpts.append((ins_r, ins_c, 0.0, c2_idx_new))
                    out_apts.append((ins_r, ins_c))
                    print(f"   [Step2b-末尾预测(兜底)] 插入点[{k}]:"
                          f" row={ins_r:.1f}")
                paired_tail = sorted(zip(out_jpts, out_apts), key=lambda x: x[0][0])
                out_jpts = [p[0] for p in paired_tail]
                out_apts = [p[1] for p in paired_tail]

    if len(out_jpts) != len(junction_pts):
        print(f"   [Step2b] 补全后汇合点: {len(junction_pts)} → {len(out_jpts)}")
    for _i, _jp in enumerate(out_jpts):
        print(f"   [Step2b] 最终点[{_i}] c2=({_jp[0]:.1f},{_jp[1]:.1f}) mean={_jp[2]:.1f}")
    return out_jpts, out_apts
