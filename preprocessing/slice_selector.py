"""
切片优选模块（V15 升级版）
三步掩模合并 + 相对评分，自动选择最优矢状位中间切片
"""
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation as sk_binary_dilation, disk
from skimage.measure import label as sk_label
from skimage import measure
from skimage.draw import line
from scipy.ndimage import binary_closing

from config.params import CANAL_MIN_AREA_MM2, MAX_CANAL_WIDTH_MM, SLICE_CANDIDATES


def segment_initial_enhanced_v2(img_raw, pixel_spacing):
    """
    独立版双区域椎管初始化（V14_2 增强版），返回三元组供 select_best_slice 使用。
    - 下区域（覆盖腰骶尾椎）：行 30%~95%，列 30%~70%
    - 上区域（覆盖胸腰段）  ：行  0%~65%，列 30%~70%
    各自 Otsu + 保留所有合理连通域（非仅最大），合并 OR 得到完整椎管掩模。

    返回：(canal_seed, upper_mask, lower_mask)
        canal_seed - 完整合并掩模（upper | lower）
        upper_mask - 上区域掩模（0%~65%）
        lower_mask - 下区域掩模（30%~95%）
    """
    h, w = img_raw.shape
    col_start, col_end = int(w * 0.30), int(w * 0.70)
    canal_seed = np.zeros_like(img_raw, dtype=bool)

    min_area_px = int(CANAL_MIN_AREA_MM2 / (pixel_spacing ** 2))  # ~300mm²

    def _extract_best_region(r0, r1):
        """在 [r0,r1) 行、[col_start,col_end) 列内做 Otsu + 保留所有合理连通域"""
        r0 = max(0, min(r0, h - 10))
        r1 = max(r0 + 10, min(r1, h))
        region_img = img_raw[r0:r1, col_start:col_end]
        if region_img.size == 0:
            return None
        otsu_thresh = threshold_otsu(region_img)
        mask = region_img > otsu_thresh
        mask = sk_binary_dilation(mask, footprint=disk(1))
        labeled = sk_label(mask)

        out = np.zeros_like(img_raw, dtype=bool)
        kept_count = 0
        for reg in measure.regionprops(labeled):
            if reg.area >= min_area_px:
                minr, minc, maxr, maxc = reg.bbox
                out[r0 + minr: r0 + maxr,
                    col_start + minc: col_start + maxc] = reg.image
                kept_count += 1

        if kept_count > 1:
            print(f"   [Segment Initial] Region [{r0}-{r1}] kept {kept_count} components")

        return out if np.any(out) else None

    # ── 下区域：覆盖腰骶尾椎（行 30%~95%）──
    lower = _extract_best_region(int(h * 0.30), int(h * 0.95))
    if lower is not None:
        canal_seed |= lower

    # ── 上区域：覆盖胸腰段（行 0%~65%）──
    upper = _extract_best_region(int(h * 0.00), int(h * 0.65))
    if upper is not None:
        canal_seed |= upper

    if not np.any(canal_seed):
        return None, None, None

    upper_out = upper if upper is not None else np.zeros_like(img_raw, dtype=bool)
    lower_out = lower if lower is not None else np.zeros_like(img_raw, dtype=bool)
    return canal_seed, upper_out, lower_out


def _process_single_slice(data, h_img, w_img, idx, pixel_spacing):
    """
    Process one candidate slice: load → segment → Step1/2/3 → artifact check → width check.
    Returns result dict (scores set to 0.0) or None if no valid mask.
    """
    slc = data[:, :, idx].astype(np.float32)
    seed, upper_mask, lower_raw = segment_initial_enhanced_v2(slc, pixel_spacing)
    if seed is None or not np.any(seed):
        print(f"   切片 [第{idx+1}张]: 无有效掩模，跳过")
        return None

    labeled = sk_label(seed)
    n_components = int(labeled.max()) if labeled.max() > 0 else 0
    if n_components == 0:
        print(f"   切片 [第{idx+1}张]: 无连通域，跳过")
        return None

    green_mask = None
    green_area_px = 0
    green_bottom_row = 0
    green_col_center = 0
    green_width_px = 0
    lower_regions = []

    # ── Step 1: 上区域最大连通域作为核心绿色掩模 ──
    if np.any(upper_mask):
        labeled_upper = sk_label(upper_mask)
        upper_regions = []
        for i in range(1, int(labeled_upper.max()) + 1):
            rm = (labeled_upper == i)
            ra = np.sum(rm)
            rows = np.where(np.any(rm, axis=1))[0]
            cols = np.where(np.any(rm, axis=0))[0]
            if len(rows) > 0 and len(cols) > 0:
                upper_regions.append({'area': ra, 'mask': rm,
                                      'row_range': (rows[0], rows[-1]),
                                      'col_range': (cols[0], cols[-1])})
        upper_regions.sort(key=lambda x: x['area'], reverse=True)

        if upper_regions:
            core = upper_regions[0]
            green_mask = core['mask'].copy()
            print(f"   切片 [第{idx+1}张] Step1 核心: area={int(np.sum(green_mask))}px, "
                  f"row=[{core['row_range'][0]}-{core['row_range'][1]}], "
                  f"col=[{core['col_range'][0]}-{core['col_range'][1]}]")

            # ── Step 2: 下区域与核心掩模有像素重叠的连通域合并 ──
            if np.any(lower_raw):
                labeled_lower = sk_label(lower_raw)
                for i in range(1, int(labeled_lower.max()) + 1):
                    rm = (labeled_lower == i)
                    ra = np.sum(rm)
                    rows = np.where(np.any(rm, axis=1))[0]
                    cols = np.where(np.any(rm, axis=0))[0]
                    if len(rows) > 0 and len(cols) > 0:
                        lower_regions.append({'area': ra, 'mask': rm,
                                              'row_range': (rows[0], rows[-1]),
                                              'col_range': (cols[0], cols[-1])})
                lower_regions.sort(key=lambda x: x['area'], reverse=True)

                for lr in lower_regions:
                    if np.any(lr['mask'] & green_mask):
                        green_mask |= lr['mask']
                        print(f"   切片 [第{idx+1}张] Step2 合并下区域: area={lr['area']}px, "
                              f"row=[{lr['row_range'][0]}-{lr['row_range'][1]}]")
                        break

            green_rows = np.where(np.any(green_mask, axis=1))[0]
            green_cols = np.where(np.any(green_mask, axis=0))[0]
            green_area_px = int(np.sum(green_mask))
            green_bottom_row = int(green_rows[-1])
            green_col_center = (green_cols[0] + green_cols[-1]) / 2
            green_width_px = green_cols[-1] - green_cols[0] + 1
            print(f"   切片 [第{idx+1}张] Step1+2 后: area={green_area_px}px, bottom={green_bottom_row}")

    # Fallback: upper_mask 为空时，用完整 seed 最大连通域
    if green_mask is None:
        regions_all = []
        for i in range(1, n_components + 1):
            rm = (labeled == i)
            ra = np.sum(rm)
            rows = np.where(np.any(rm, axis=1))[0]
            cols = np.where(np.any(rm, axis=0))[0]
            if len(rows) > 0 and len(cols) > 0:
                regions_all.append({'area': ra, 'mask': rm,
                                    'row_range': (rows[0], rows[-1]),
                                    'col_range': (cols[0], cols[-1])})
        regions_all.sort(key=lambda x: x['area'], reverse=True)
        if regions_all:
            fb = regions_all[0]
            green_mask = fb['mask'].copy()
            green_area_px = fb['area']
            green_rows = np.where(np.any(green_mask, axis=1))[0]
            green_cols = np.where(np.any(green_mask, axis=0))[0]
            green_bottom_row = int(green_rows[-1])
            green_col_center = (green_cols[0] + green_cols[-1]) / 2
            green_width_px = green_cols[-1] - green_cols[0] + 1
            print(f"   切片 [第{idx+1}张] Fallback 最大连通域: area={green_area_px}px, bottom={green_bottom_row}")

    if green_mask is None:
        return None

    core_green_mask = green_mask.copy()

    # ── Step 3: 掩膜 < 80% 时向下探索（形态校验 + 桥接）──
    merged_regions = []
    remaining_regions = []
    for lr in lower_regions:
        if not np.any(lr['mask'] & core_green_mask):
            remaining_regions.append(lr)

    _step3_trigger = green_bottom_row < int(h_img * 0.80)
    if _step3_trigger and remaining_regions:
        remaining_regions.sort(key=lambda x: x['area'], reverse=True)

        _gm_widths = []
        for _r in range(h_img):
            _cols = np.where(green_mask[_r, :])[0]
            if len(_cols) > 0:
                _gm_widths.append(_cols[-1] - _cols[0] + 1)
        _med_green_w = float(np.median(_gm_widths)) if _gm_widths else 0.0

        max_gap_px = int(round(20.0 / pixel_spacing))
        for region in remaining_regions:
            gap_px = region['row_range'][0] - green_bottom_row
            if gap_px <= 0:
                continue
            if gap_px > max_gap_px:
                print(f"   切片 [第{idx+1}张] Step3 候选: area={region['area']}px, "
                      f"gap={gap_px * pixel_spacing:.1f}mm > 20mm, 跳过")
                continue

            _reg_widths = []
            for _r in range(h_img):
                _cols = np.where(region['mask'][_r, :])[0]
                if len(_cols) > 0:
                    _reg_widths.append(_cols[-1] - _cols[0] + 1)
            _med_reg_w = float(np.median(_reg_widths)) if _reg_widths else 0.0

            _ratio_str = ""
            if _med_green_w > 0 and _med_reg_w > 0:
                _ratio = _med_reg_w / _med_green_w
                _ratio_str = f", ratio={_ratio:.2f}"
                if _ratio < 0.5 or _ratio > 1.5:
                    print(f"   切片 [第{idx+1}张] Step3 形态不符(宽度比): "
                          f"green中位数宽={_med_green_w:.1f}px, "
                          f"region中位数宽={_med_reg_w:.1f}px{_ratio_str}, 跳过")
                    continue

            # ── ② 左边界自洽（候选区域内部渐变，不依赖 green 尾部） ──
            _reg_left_cols = []
            for _rr in range(region['row_range'][0], region['row_range'][1] + 1):
                _rc = np.where(region['mask'][_rr, :])[0]
                if len(_rc) > 0:
                    _reg_left_cols.append(int(_rc[0]))
            _left_ok = True
            if len(_reg_left_cols) >= 3:
                _max_jump = max(abs(_reg_left_cols[i] - _reg_left_cols[i - 1])
                               for i in range(1, len(_reg_left_cols)))
                _jump_limit = max(3, 0.20 * _med_reg_w)
                if _max_jump > _jump_limit:
                    _left_ok = False
                    print(f"   切片 [第{idx+1}张] Step3 左边界跳变: "
                          f"max跳变={_max_jump}px > {_jump_limit:.0f}px, 跳过")
            if not _left_ok:
                continue

            # ── ③ 宽度一致性 ──
            if len(_reg_widths) >= 3:
                _reg_w_mean = float(np.mean(_reg_widths))
                _reg_w_std = float(np.std(_reg_widths))
                _cv = _reg_w_std / _reg_w_mean if _reg_w_mean > 0 else 999
                if _cv > 0.50:
                    print(f"   切片 [第{idx+1}张] Step3 宽度波动大: "
                          f"cv={_cv:.2f} > 0.50, 跳过")
                    continue

            # ── ④ 细长比 ──
            _reg_h = region['row_range'][1] - region['row_range'][0] + 1
            if _med_reg_w > 0:
                _ar = _reg_h / _med_reg_w
                if _ar < 1.5:
                    print(f"   切片 [第{idx+1}张] Step3 非细长形: "
                          f"h={_reg_h}, w={_med_reg_w:.1f}, ar={_ar:.2f} < 1.5, 跳过")
                    continue

            _gbr_cols = np.where(green_mask[green_bottom_row, :])[0]
            _rtr_cols = np.where(region['mask'][region['row_range'][0], :])[0]
            if len(_gbr_cols) > 0 and len(_rtr_cols) > 0:
                _pt_a_c = int(round((_gbr_cols[0] + _gbr_cols[-1]) / 2))
                _pt_b_c = int(round((_rtr_cols[0] + _rtr_cols[-1]) / 2))
                _br, _bc = line(green_bottom_row, _pt_a_c,
                                int(region['row_range'][0]), _pt_b_c)
                _valid = (_br >= 0) & (_br < h_img) & (_bc >= 0) & (_bc < w_img)
                green_mask[_br[_valid], _bc[_valid]] = True

            green_mask |= region['mask']
            merged_regions.append(id(region['mask']))
            print(f"   ✅ 切片 [第{idx+1}张] Step3 桥接合并: area={region['area']}px, "
                  f"gap={gap_px * pixel_spacing:.1f}mm{_ratio_str}")
            break
    elif _step3_trigger:
        print(f"   切片 [第{idx+1}张] Step3: 无剩余候选区域")

    if merged_regions:
        green_area_px = int(np.sum(green_mask))
        green_rows = np.where(np.any(green_mask, axis=1))[0]
        green_cols = np.where(np.any(green_mask, axis=0))[0]
        if len(green_rows) > 0:
            green_bottom_row = int(green_rows[-1])
            green_col_center = (green_cols[0] + green_cols[-1]) / 2
            green_width_px = green_cols[-1] - green_cols[0] + 1
        print(f"   切片 [第{idx+1}张] Step3 合并后: area={green_area_px}px, bottom={green_bottom_row}")

    # ── Left-border artifact check (metal implant detection) ──
    total_gap_px = 0
    lb_start_row = int(h_img * 0.70)
    lb_end_row = green_bottom_row

    cut_row = None
    prev_gap_row = None
    for r in range(lb_start_row, lb_end_row + 1):
        px = np.where(green_mask[r, :])[0]
        if len(px) >= 2:
            span = px[-1] - px[0] + 1
            internal_gaps = span - len(px)
            total_gap_px += internal_gaps
            if internal_gaps > 2:
                if prev_gap_row is not None and r == prev_gap_row + 1:
                    cut_row = prev_gap_row
                    break
                prev_gap_row = r
            else:
                prev_gap_row = None

    if cut_row is not None:
        lb_left_cols = {}
        lb_right_cols = {}
        last_left = None
        last_right = None
        for r in range(cut_row, lb_end_row + 1):
            px = np.where(green_mask[r, :])[0]
            if len(px) > 0:
                last_left = int(px[0])
                last_right = int(px[-1])
                lb_left_cols[r] = last_left
                lb_right_cols[r] = last_right
            else:
                if last_left is not None:
                    lb_left_cols[r] = last_left
                    lb_right_cols[r] = last_right

        rows_in_range = sorted(lb_left_cols.keys())
        path_len = 0.0
        for ri in range(len(rows_in_range) - 1):
            r0, r1 = rows_in_range[ri], rows_in_range[ri + 1]
            dc = lb_left_cols[r1] - lb_left_cols[r0]
            dr = r1 - r0
            path_len += (dc ** 2 + dr ** 2) ** 0.5

        cut_row_px = np.where(green_mask[cut_row, :])[0]
        cut_col = int(cut_row_px[-1]) if len(cut_row_px) > 0 else (
            lb_right_cols.get(rows_in_range[0], 0) if rows_in_range else 0)

        right_of_cut = sum(1 for r in rows_in_range if lb_right_cols.get(r, cut_col) < cut_col)
        right_ratio = right_of_cut / len(rows_in_range) if rows_in_range else 0.0

        if path_len > 10.0 and right_ratio > 0.30:
            old_bottom = green_bottom_row
            green_mask[cut_row:, :] = False
            green_rows2 = np.where(np.any(green_mask, axis=1))[0]
            green_cols2 = np.where(np.any(green_mask, axis=0))[0]
            if len(green_rows2) > 0 and len(green_cols2) > 0:
                green_area_px    = int(np.sum(green_mask))
                green_bottom_row = int(green_rows2[-1])
                green_col_center = (green_cols2[0] + green_cols2[-1]) / 2
                green_width_px   = green_cols2[-1] - green_cols2[0] + 1
            print(f"   ⚠️ 切片 [第{idx+1}张] 伪影裁切: cut_row={cut_row}, "
                  f"path_len={path_len:.1f}px, right_ratio={right_ratio:.0%} "
                  f"(bottom {old_bottom}→{green_bottom_row})")
        else:
            print(f"   切片 [第{idx+1}张] 伪影检查 OK: cut_row={cut_row}, "
                  f"path_len={path_len:.1f}px, right_ratio={right_ratio:.0%}")
    else:
        print(f"   切片 [第{idx+1}张] 伪影检查: 70% 以下无连续空洞，跳过")

    # 宽度异常检测（V15.5: 前60%异常时先修剪再评分）
    width_valid = True
    max_normal_width_px = int(round(MAX_CANAL_WIDTH_MM / pixel_spacing))
    abnormal_row_count = 0
    abnormal_rows_list = []
    for row in range(h_img):
        cols = np.where(green_mask[row, :])[0]
        if len(cols) > 0:
            if cols[-1] - cols[0] + 1 > max_normal_width_px:
                abnormal_row_count += 1
                abnormal_rows_list.append(row)
    if abnormal_row_count >= 5:
        mask_rows_all = np.where(np.any(green_mask, axis=1))[0]
        if len(mask_rows_all) > 0:
            mask_top = mask_rows_all[0]
            mask_bottom = mask_rows_all[-1]
            mask_height = mask_bottom - mask_top + 1
            # 死亡评分：异常行数 > 总掩膜行数的 30%，直接判 0
            if abnormal_row_count / mask_height > 0.30:
                width_valid = False
                print(f"   ☠️ 切片 [第{idx+1}张] 死亡评分: "
                      f"异常行{abnormal_row_count}/{mask_height}行 > 30%, 直接判0")
            else:
                threshold_60_row = mask_top + int(mask_height * 0.60)
                abnormal_in_first60 = sum(1 for r in abnormal_rows_list if r <= threshold_60_row)

                if abnormal_in_first60 / abnormal_row_count > 0.60:
                    row_widths = []
                    for row in range(h_img):
                        cols = np.where(green_mask[row, :])[0]
                        if len(cols) > 0:
                            row_widths.append(cols[-1] - cols[0] + 1)
                    if row_widths:
                        med_w = float(np.median(row_widths))
                        trim_w = int(round(med_w))
                        trimmed_count = 0
                        for row in abnormal_rows_list:
                            cols = np.where(green_mask[row, :])[0]
                            if len(cols) > 0:
                                col_right = cols[-1]
                                col_left_new = col_right - trim_w + 1
                                if col_left_new > 0:
                                    green_mask[row, :col_left_new] = False
                                    trimmed_count += 1
                        green_mask = binary_closing(green_mask, iterations=2)
                        green_area_px = int(np.sum(green_mask))
                        green_rows = np.where(np.any(green_mask, axis=1))[0]
                        green_cols = np.where(np.any(green_mask, axis=0))[0]
                        if len(green_rows) > 0:
                            green_bottom_row = int(green_rows[-1])
                            green_col_center = (green_cols[0] + green_cols[-1]) / 2
                            green_width_px = green_cols[-1] - green_cols[0] + 1
                        print(f"   🔧 切片 [第{idx+1}张] 前60%宽度修剪: "
                              f"中位数宽={med_w:.1f}px, 修剪{trimmed_count}/{abnormal_row_count}行, "
                              f"修剪后面积={green_area_px}px")
                    else:
                        width_valid = False
                        print(f"   ⚠️ 切片 [第{idx+1}张] 宽度异常(无有效行宽): {abnormal_row_count}行")
                else:
                    width_valid = False
                    print(f"   ⚠️ 切片 [第{idx+1}张] 宽度异常(后40%): {abnormal_row_count}行超过30mm")
        else:
            width_valid = False
            print(f"   ⚠️ 切片 [第{idx+1}张] 宽度异常: {abnormal_row_count}行超过30mm")

    area_mm2 = green_area_px * (pixel_spacing ** 2)
    return {
        'slice_idx': idx,
        'area_mm2': area_mm2,
        'green_mask': green_mask,
        'canal_seed': seed,
        'merged_regions': merged_regions if merged_regions else None,
        'green_bottom_row': green_bottom_row,
        'width_valid': width_valid,
        'total_gap_px': total_gap_px,
        'area_score': 0.0,
        'bottom_score': 0.0,
        'gap_score': 0.0,
        'total_score': 0.0,
    }


def select_best_slice(data, pixel_spacing):
    """
    从 3D NIfTI 数据中自动选择最优矢状位中间切片（V15：五候选直接遍历，三步掩模合并 + 相对评分）。

    策略：以中间切片为基准，向前后各扩展 2 张，共 5 张候选切片（n-2, n-1, n, n+1, n+2）。
    使用 segment_initial_enhanced_v2() 返回 (canal_seed, upper_mask, lower_mask) 三元组。

    【三步掩模合并算法】
      Step1: 上区域（0%~65%）最大连通域作为核心绿色掩模（锁定椎管身份，排除盆腔）
      Step2: 下区域（30%~95%）与核心掩模有像素重叠的连通域合并（同一椎管延伸）
      Step3: 若合并后底行 < 80% 图像高度且有未合并区域（gap ≤ 20mm），
             经形态校验（宽度比 + 左边界自洽 + 宽度一致性 + 细长比）后桥接合并

    【评分机制】（45% 面积 + 45% 底部 + 10% 空洞惩罚，均为相对评分）
      1. 面积评分：绿色掩模面积最大者得 1 分，其他按比例
      2. 底部位置评分：底行最低者得 1 分，其他按比例
      宽度异常检测：异常行 > 30% 掩膜行数 → 死亡评分直接判 0；
                    否则 ≥5 行宽度超过 30mm → 前 60% 则修剪再评分，后 40% 则置零
      全零兜底：所有切片得 0 分时，选中间候选切片

    参数：
        data          - NIfTI 3D 数组，shape=(H, W, N_slices)
        pixel_spacing - 像素间距（mm）

    返回：
        best_idx        - 最优切片索引（int）
        best_green_mask - 最佳切片的绿色掩模（合并后完整掩模，numpy array）
        best_area       - 对应掩模像素面积（int，调试用）
        merged_regions  - Step3 合并的区域列表（list or None）
        best_canal_seed - 完整初始化掩模（用于后续追踪）
    """
    n_slices = data.shape[2]
    h_img, w_img = data.shape[0], data.shape[1]
    mid = n_slices // 2
    _half = SLICE_CANDIDATES // 2
    candidates = [i for i in range(mid - _half, mid + _half + 1) if 0 <= i < n_slices]

    results = []
    cand_display = [i + 1 for i in candidates]
    print(f"\n🔍 切片优选 (V15 五候选三步合并 + 相对评分): 候选第{cand_display}张 (总切片数={n_slices})")

    for idx in candidates:
        result = _process_single_slice(data, h_img, w_img, idx, pixel_spacing)
        if result is not None:
            results.append(result)

    # 计算相对评分（45% 面积 + 45% 底部 + 10% 空洞惩罚）
    if results:
        max_area = max(r['area_mm2'] for r in results)
        max_bottom_row = max(r['green_bottom_row'] for r in results)
        min_gap_px = min(r['total_gap_px'] for r in results)

        for r in results:
            if not r.get('width_valid', True):
                r['area_score'] = 0.0
                r['bottom_score'] = 0.0
                r['gap_score'] = 0.0
            else:
                r['area_score'] = r['area_mm2'] / max_area if max_area > 0 else 0.0
                r['bottom_score'] = r['green_bottom_row'] / max_bottom_row if max_bottom_row > 0 else 0.0
                gap_px = r['total_gap_px']
                if gap_px == 0:
                    r['gap_score'] = 1.0
                else:
                    r['gap_score'] = max(0.1, min_gap_px / gap_px)

            r['total_score'] = (0.45 * r['area_score']
                                + 0.45 * r['bottom_score']
                                + 0.10 * r['gap_score'])

            width_flag = " [宽度异常]" if not r.get('width_valid', True) else ""
            merge_flag = " [Step3合并]" if r['merged_regions'] else ""
            print(f"   切片 [第{r['slice_idx']+1}张]{width_flag}{merge_flag}: "
                  f"面积={r['area_mm2']:.0f}mm² ({r['area_score']:.2f})  "
                  f"底部={r['green_bottom_row']}px ({r['bottom_score']:.2f})  "
                  f"空洞={r['total_gap_px']}px (gap={r['gap_score']:.2f})  "
                  f"→ 总分={r['total_score']:.2f}")

    # 选择最佳切片
    if results:
        max_score = max(r['total_score'] for r in results)

        all_zero = all(r['total_score'] == 0.0 for r in results)
        if all_zero:
            mid_idx = len(candidates) // 2
            best_idx = candidates[mid_idx]
            print(f"   ⚠️ 所有切片得 0 分（宽度异常），选中间候选第{best_idx+1}张")
            best_result = next((r for r in results if r['slice_idx'] == best_idx), results[mid_idx])

            # ── 全零回退后对 green_mask 做宽度修正 ──
            # 保留背侧（col 右侧）锚点，按行截断超过中位数宽度×1.5的腹侧异常延伸
            raw_mask = best_result['green_mask']
            h_m, w_m = raw_mask.shape
            max_w_px = int(round(MAX_CANAL_WIDTH_MM / pixel_spacing))

            # 1. 计算每行宽度和背侧锚点（col_right）
            row_widths = []
            row_col_right = {}
            for rr in range(h_m):
                cols = np.where(raw_mask[rr, :])[0]
                if len(cols) > 0:
                    row_widths.append(cols[-1] - cols[0] + 1)
                    row_col_right[rr] = int(cols[-1])

            csf_hints = {}
            if row_widths:
                med_w = float(np.median(row_widths))
                limit_w = max(max_w_px, int(round(med_w * 1.2)))
                fixed_mask = raw_mask.copy()
                clipped_rows = 0

                # 2. 逐行截断：超过 limit_w 的行，从背侧锚点向腹侧保留 limit_w 列
                for rr in range(h_m):
                    cols = np.where(fixed_mask[rr, :])[0]
                    if len(cols) == 0:
                        continue
                    w_row = cols[-1] - cols[0] + 1
                    if w_row > limit_w:
                        col_right = cols[-1]
                        col_left_new = col_right - limit_w + 1
                        # 清除 col_left_new 左侧的像素
                        if col_left_new > 0:
                            fixed_mask[rr, :col_left_new] = False
                        # 记录 CSF 中心锚点：背侧向腹侧 1/3 处
                        csf_hints[rr] = col_right - limit_w // 3
                        clipped_rows += 1

                # 3. 修正后重建连通：用形态学闭运算填补因截断产生的小空洞
                fixed_mask = binary_closing(fixed_mask, iterations=2)

                new_area = int(np.sum(fixed_mask))
                print(f"   [宽度修正] 中位数宽={med_w:.1f}px  限制={limit_w}px"
                      f"  截断行数={clipped_rows}  修正后面积={new_area}px"
                      f"  csf锚点行数={len(csf_hints)}")
                best_result = dict(best_result)
                best_result['green_mask'] = fixed_mask
                best_result['csf_hints'] = csf_hints
        else:
            best_candidates = [r for r in results if abs(r['total_score'] - max_score) < 0.001]
            if len(best_candidates) == 1:
                best_result = best_candidates[0]
            else:
                # 总分平局 → 优先选底部评分最高
                max_bottom = max(r['bottom_score'] for r in best_candidates)
                bottom_best = [r for r in best_candidates if abs(r['bottom_score'] - max_bottom) < 0.001]
                if len(bottom_best) == 1:
                    best_result = bottom_best[0]
                else:
                    # 底部评分也平局 → 选最靠中间
                    mid_candidate = candidates[len(candidates) // 2]
                    best_result = min(bottom_best, key=lambda r: abs(r['slice_idx'] - mid_candidate))
            best_idx = best_result['slice_idx']

        best_green_mask = best_result['green_mask']
        best_canal_seed = best_result['canal_seed']
        best_area = int(np.sum(best_green_mask))
        best_merged = best_result.get('merged_regions')
        best_csf_hints = best_result.get('csf_hints', {})

        merge_info = f" [Step3合并:{len(best_merged)}]" if best_merged else ""
        print(f"   ✅ 选定第{best_idx+1}张切片{merge_info}，绿色掩模面积={best_area}px，评分={best_result['total_score']:.2f}")
        return best_idx, best_green_mask, best_area, best_merged, best_canal_seed, best_csf_hints
    else:
        print(f"   ⚠️ 无有效切片，回退到中间第{mid+1}张")
        return mid, None, 0, None, None, {}
