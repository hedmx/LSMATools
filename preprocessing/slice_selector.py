"""
切片优选模块（V14_2 升级版）
三步掩模合并 + 相对评分，自动选择最优矢状位中间切片
"""
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation as sk_binary_dilation, disk
from skimage.measure import label as sk_label
from skimage import measure


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

    min_area_px = int(300 / (pixel_spacing ** 2))  # ~300mm²

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


def select_best_slice(data, pixel_spacing):
    """
    从 3D NIfTI 数据中自动选择最优矢状位中间切片（V14_2 升级版：三步掩模合并 + 相对评分）。

    策略：以中间切片为基准，向前后各扩展 2 张，共 5 张候选切片。
    使用 segment_initial_enhanced_v2() 返回 (canal_seed, upper_mask, lower_mask) 三元组。

    【三步掩模合并算法】
      Step1: 上区域（0%~65%）最大连通域作为核心绿色掩模（锁定椎管身份，排除盆腔）
      Step2: 下区域（30%~95%）与核心掩模有像素重叠的连通域合并（同一椎管延伸）
      Step3: 若合并后底行 < 60% 图像高度，向下搜索 20mm×10mm 窗口（像素级重叠判断）

    【评分机制】（各 50% 权重，均为相对评分）
      1. 面积评分：绿色掩模面积最大者得 1 分，其他按比例
      2. 底部位置评分：底行最低者得 1 分，其他按比例
      宽度异常检测：逐行扫描，≥5行宽度超过 60mm → 该切片所有分数强制为 0
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
    candidates = [i for i in range(mid - 2, mid + 3) if 0 <= i < n_slices]

    results = []
    print(f"\n🔍 切片优选 (V14_2 三步合并 + 相对评分): 候选索引 {candidates} (总切片数={n_slices})")

    for idx in candidates:
        slc = data[:, :, idx].astype(np.float32)

        seed, upper_mask, lower_raw = segment_initial_enhanced_v2(slc, pixel_spacing)

        if seed is None or not np.any(seed):
            print(f"   切片 [{idx}]: 无有效掩模，跳过")
            continue

        labeled = sk_label(seed)
        n_components = int(labeled.max()) if labeled.max() > 0 else 0

        if n_components == 0:
            print(f"   切片 [{idx}]: 无连通域，跳过")
            continue

        green_mask = None
        green_area_px = 0
        green_bottom_row = 0
        green_col_center = 0
        green_width_px = 0
        lower_regions = []
        step2_merged = False

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
                print(f"   切片 [{idx}] Step1 核心: area={int(np.sum(green_mask))}px, "
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
                            step2_merged = True
                            print(f"   切片 [{idx}] Step2 合并下区域: area={lr['area']}px, "
                                  f"row=[{lr['row_range'][0]}-{lr['row_range'][1]}]")
                            break

                green_rows = np.where(np.any(green_mask, axis=1))[0]
                green_cols = np.where(np.any(green_mask, axis=0))[0]
                green_area_px = int(np.sum(green_mask))
                green_bottom_row = int(green_rows[-1])
                green_col_center = (green_cols[0] + green_cols[-1]) / 2
                green_width_px = green_cols[-1] - green_cols[0] + 1
                print(f"   切片 [{idx}] Step1+2 后: area={green_area_px}px, bottom={green_bottom_row}")

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
                print(f"   切片 [{idx}] Fallback 最大连通域: area={green_area_px}px, bottom={green_bottom_row}")

        if green_mask is None:
            continue

        core_green_mask = green_mask.copy()

        # ── Step 3: 底行 < 60% 时向下搜索 20mm×10mm ──
        merged_regions = []
        remaining_regions = []
        for lr in lower_regions:
            if not np.any(lr['mask'] & core_green_mask):
                remaining_regions.append(lr)

        if green_bottom_row < int(h_img * 0.60):
            search_range_px = int(round(20.0 / pixel_spacing))
            search_start_row = green_bottom_row
            search_end_row = min(green_bottom_row + search_range_px, h_img - 1)

            bottom_row_cols = np.where(core_green_mask[green_bottom_row, :])[0]
            half_width_px = int(round(5.0 / pixel_spacing))
            if len(bottom_row_cols) > 0:
                center_col = (bottom_row_cols[0] + bottom_row_cols[-1]) / 2
                search_col_start = max(0, int(center_col - half_width_px))
                search_col_end = min(w_img - 1, int(center_col + half_width_px))
            else:
                search_col_start = max(0, int(green_col_center) - half_width_px)
                search_col_end = min(w_img - 1, int(green_col_center) + half_width_px)

            print(f"   切片 [{idx}] Step3 搜索窗: row[{search_start_row}-{search_end_row}], "
                  f"col[{search_col_start}-{search_col_end}] (20mm×10mm)")

            if remaining_regions:
                remaining_regions.sort(key=lambda x: x['area'], reverse=True)
                search_box_mask = np.zeros_like(core_green_mask, dtype=bool)
                search_box_mask[search_start_row:search_end_row + 1,
                                search_col_start:search_col_end + 1] = True

                for region in remaining_regions:
                    overlap_pixels = int(np.sum(region['mask'] & search_box_mask))
                    print(f"   切片 [{idx}] Step3 候选: area={region['area']}px, "
                          f"row=[{region['row_range'][0]}-{region['row_range'][1]}], "
                          f"col=[{region['col_range'][0]}-{region['col_range'][1]}] | overlap={overlap_pixels}px")
                    if overlap_pixels > 0:
                        green_mask |= region['mask']
                        merged_regions.append(id(region['mask']))
                        print(f"   ✅ 切片 [{idx}] Step3 合并成功 (area={region['area']}px)")
                        break
            else:
                print(f"   切片 [{idx}] Step3: 无剩余候选区域")

            if merged_regions:
                green_area_px = int(np.sum(green_mask))
                green_rows = np.where(np.any(green_mask, axis=1))[0]
                green_cols = np.where(np.any(green_mask, axis=0))[0]
                if len(green_rows) > 0:
                    green_bottom_row = int(green_rows[-1])
                    green_col_center = (green_cols[0] + green_cols[-1]) / 2
                    green_width_px = green_cols[-1] - green_cols[0] + 1
                print(f"   切片 [{idx}] Step3 合并后: area={green_area_px}px, bottom={green_bottom_row}")

        # 宽度异常检测
        width_valid = True
        max_normal_width_px = int(round(60.0 / pixel_spacing))
        abnormal_row_count = 0
        for row in range(h_img):
            cols = np.where(green_mask[row, :])[0]
            if len(cols) > 0:
                if cols[-1] - cols[0] + 1 > max_normal_width_px:
                    abnormal_row_count += 1
        if abnormal_row_count >= 5:
            width_valid = False
            print(f"   ⚠️ 切片 [{idx}] 宽度异常：{abnormal_row_count} 行超过 60mm ({max_normal_width_px}px)")

        area_mm2 = green_area_px * (pixel_spacing ** 2)

        results.append({
            'slice_idx': idx,
            'area_mm2': area_mm2,
            'green_mask': green_mask,
            'canal_seed': seed,
            'merged_regions': merged_regions if merged_regions else None,
            'green_bottom_row': green_bottom_row,
            'width_valid': width_valid,
            'area_score': 0.0,
            'bottom_score': 0.0,
            'total_score': 0.0
        })

    # 计算相对评分
    if results:
        max_area = max(r['area_mm2'] for r in results)
        max_bottom_row = max(r['green_bottom_row'] for r in results)

        for r in results:
            if not r.get('width_valid', True):
                r['area_score'] = 0.0
                r['bottom_score'] = 0.0
            else:
                r['area_score'] = r['area_mm2'] / max_area if max_area > 0 else 0.0
                r['bottom_score'] = r['green_bottom_row'] / max_bottom_row if max_bottom_row > 0 else 0.0

            r['total_score'] = 0.50 * r['area_score'] + 0.50 * r['bottom_score']

            width_flag = " [宽度异常]" if not r.get('width_valid', True) else ""
            merge_flag = " [Step3合并]" if r['merged_regions'] else ""
            print(f"   切片 [{r['slice_idx']}]{width_flag}{merge_flag}: "
                  f"面积={r['area_mm2']:.0f}mm² ({r['area_score']:.2f})  "
                  f"底部={r['green_bottom_row']}px ({r['bottom_score']:.2f})  "
                  f"→ 总分={r['total_score']:.2f}")

    # 选择最佳切片
    if results:
        max_score = max(r['total_score'] for r in results)

        all_zero = all(r['total_score'] == 0.0 for r in results)
        if all_zero:
            mid_idx = len(candidates) // 2
            best_idx = candidates[mid_idx]
            print(f"   ⚠️ 所有切片得 0 分（宽度异常），选中间候选切片: {best_idx}")
            best_result = next((r for r in results if r['slice_idx'] == best_idx), results[mid_idx])
        else:
            best_candidates = [r for r in results if abs(r['total_score'] - max_score) < 0.001]
            if len(best_candidates) == 1:
                best_result = best_candidates[0]
            else:
                mid_candidate = candidates[len(candidates) // 2]
                best_result = min(best_candidates, key=lambda r: abs(r['slice_idx'] - mid_candidate))
            best_idx = best_result['slice_idx']

        best_green_mask = best_result['green_mask']
        best_canal_seed = best_result['canal_seed']
        best_area = int(np.sum(best_green_mask))
        best_merged = best_result.get('merged_regions')

        merge_info = f" [Step3合并:{len(best_merged)}]" if best_merged else ""
        print(f"   ✅ 选定切片索引={best_idx}{merge_info}，绿色掩模面积={best_area}px，评分={best_result['total_score']:.2f}")
        return best_idx, best_green_mask, best_area, best_merged, best_canal_seed
    else:
        print(f"   ⚠️ 无有效切片，回退到中间切片 {mid}")
        return mid, None, 0, None, None
