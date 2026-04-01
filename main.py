#!/usr/bin/env python3
"""
主流程编排器（LSMATOOLS_CP V15）
入口函数：
  test_single_image(nifti_path, ...) - 处理单张图像
  process_batch(parent_dir, output_dir) - 批量处理
  main() - 命令行入口
"""
import os
import sys
import io
import json
import numpy as np
import nibabel as nib
import warnings
warnings.filterwarnings('ignore')

from config.metadata_parser import load_metadata, parse_pixel_spacing
from preprocessing.series_utils import _get_series_type
from preprocessing.image_loader import find_fat_water_image
from preprocessing.slice_selector import select_best_slice
from preprocessing.coordinate_align import align_scan_lines_to_f
from segmentation.canal_processor import SpinalCanalProcessor, SpinalCordLocator
from segmentation.clustering import cluster_endplates_v15
from segmentation.endplate_detector import find_endplates_on_water_image
from segmentation.anterior_edge import (
    scan_anterior_edge_v15, find_anterior_edge_by_descent,
    filter_arc_roi_by_dense_offset, refine_arc_roi_to_anterior_edge,
    find_anterior_corner_v15, find_arc_roi_min_points,
)
from postprocessing.geometric_center import compute_vertebra_geometry
from postprocessing.visualization import visualize_results
from postprocessing.export import export_vertebra_data


def test_single_image(nifti_path, metadata_path=None, output_dir=None, 
                     patient_dir=None, seq_dir=None):
    """测试单张图像"""
    
    import io
    import sys
    
    # 捕获日志输出
    log_capture = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = log_capture
    
    print("="*80)
    print("单张图像测试 - 滑动窗口法")
    print("="*80)
    
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "test_output")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📂 加载图像: {nifti_path}")
    try:
        img = nib.load(nifti_path)
    except Exception as e:
        print(f"❌ 无法加载图像: {e}")
        return
    
    data = img.get_fdata()
    n_slices = data.shape[2]
    
    # 初始化变量（用于 JSON 输出）
    best_slice_idx = None
    merged_regions = None
    status = None
    pixel_spacing_val = 0.9375
    
    pixel_spacing = 0.9375
    meta = {}
    if metadata_path and os.path.exists(metadata_path):
        import json
        try:
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
                acq  = meta.get('acquisition_params', {})
                si   = meta.get('series_info', {})
                spacing = acq.get('pixel_spacing_mm', [0.9375, 0.9375])
                pixel_spacing = float(spacing[0]) if isinstance(spacing, list) else float(spacing)
                pixel_spacing_val = pixel_spacing  # 保存用于 JSON
            print(f"   📋 像素间距: {pixel_spacing:.4f}mm")
            # 校验输入序列必须是 T2 Dixon W 序列
            w_desc = si.get('series_description', '')
            w_type = _get_series_type(w_desc)
            if w_desc:
                is_t2     = 't2'    in w_desc.lower()
                is_dixon  = 'dixon' in w_desc.lower()
                is_w_type = w_type == 'W'
                if not (is_t2 and is_dixon and is_w_type):
                    print(f"   ⚠️ 序列校验: series='{w_desc}'")
                    if not is_t2:
                        print("      ❌ 不是T2序列")
                    if not is_dixon:
                        print("      ❌ 不是Dixon序列")
                    if not is_w_type:
                        print(f"      ❌ 序列类型为'{w_type}'，不是W(压脂)序列")
                    print("   ⚠️ 继续处理，但结果可能不可靠")
                else:
                    print(f"   ✅ 序列校验通过: T2 Dixon W | '{w_desc}'")
        except:
            print(f"   ⚠️ 使用默认像素间距: {pixel_spacing:.4f}mm")

    # ── 切片优选：先读 meta 获得真实 pixel_spacing，再取 mid±2 中椎管种子面积最大切片 ──
    best_slice_idx, best_green_mask, _, merged_regions, best_canal_seed = select_best_slice(data, pixel_spacing=pixel_spacing)
    slice_2d = data[:, :, best_slice_idx].astype(np.float32)
    print(f"   📍 压脂图使用切片索引={best_slice_idx+1} / {data.shape[2]}")

    print("\n🔍 正在处理椎管...")
    processor = SpinalCanalProcessor(pixel_spacing, meta=meta)
    
    # 如果最佳切片的完整种子掩模有效，用它作为追踪种子
    if best_canal_seed is not None and np.any(best_canal_seed):
        print(f"   ✅ 使用最佳切片的绿色掩模（合并后）进行后续处理")
        traced, roi_points, valid_rows, status, v9_data = processor.process_with_mask(
            slice_2d, best_green_mask, best_canal_seed,
            merged_regions=merged_regions
        )
    else:
        # 回退到正常流程
        traced, roi_points, valid_rows, status, v9_data = processor.process(slice_2d)
    
    if traced is None:
        print(f"   ❌ 椎管处理失败: {status}")
        return
    
    print(f"   ✅ {status}")
    if roi_points and roi_points.get('csf'):
        print(f"   📊 找到 {len(roi_points['csf'])} 个脑脊液中心点")
    
    # 定位骨髓
    print("\n🔍 正在定位骨髓...")
    locator = SpinalCordLocator(pixel_spacing)
    success, cord_mask, roi_points = locator.locate(slice_2d, traced, roi_points)
    
    if success:
        print(f"   ✅ 骨髓定位成功")
    else:
        print(f"   ⚠️ 骨髓定位失败，只显示椎管")
    
    # 生成输出文件名
    if patient_dir and seq_dir:
        output_filename = f"{patient_dir}_{seq_dir}_TRACED.png"
    else:
        base_name = os.path.splitext(os.path.basename(nifti_path))[0]
        if base_name.endswith('.nii'):
            base_name = base_name[:-4]
        output_filename = f"{base_name}_TRACED.png"
    
    output_path = os.path.join(output_dir, output_filename)

    # ===== V12：加载压水图 + 坐标对齐 + 终板检测 =====
    f_img_2d = None
    f_meta   = {}
    f_data   = None
    if v9_data and v9_data.get('scan_lines'):
        f_img_2d, f_meta, f_nii_path = find_fat_water_image(nifti_path, slice_idx=best_slice_idx)
        if f_img_2d is not None:
            scan_lines_f = align_scan_lines_to_f(
                v9_data['scan_lines'], meta, f_meta, f_img_2d)
            v9_data['scan_lines_f'] = scan_lines_f
    
            # ── V13 切线方向扫描线 → 弧长坐标系聚类 → 终板线 ──
            _c2_cols       = v9_data.get('c2_cols')
            _c2_rows       = v9_data.get('c2_rows')
            _arc_len_mm_v  = v9_data.get('arc_len_mm')
            _scan_lines_v15= v9_data.get('scan_lines_v15', [])
            if (_c2_cols is not None and _c2_rows is not None
                    and _arc_len_mm_v is not None and _scan_lines_v15):
                # 用全部 40 条扫描线在压水图上计算 high_mean/low_mean 及终板候选点
                print("\n🔍 V15 扫描线终板检测（全部 40 条，用于 high_mean/low_mean 及候选点）")
                f_data_v15 = find_endplates_on_water_image(
                    [(off_mm, cols_a, rows_a) for off_mm, rows_a, cols_a, nx_a, ny_a in _scan_lines_v15],
                    f_img_2d, pixel_spacing)
                raw_cands_v15 = f_data_v15.get('raw_candidates', []) if f_data_v15 else []
                print(f"   [V15] V15扫描线候选点: {len(raw_cands_v15)} 个")
                if f_data_v15:
                    _hm1 = f_data_v15.get('high_mean1', 0); _lm1 = f_data_v15.get('low_mean1', 0)
                    _hm2 = f_data_v15.get('high_mean2', 0); _lm2 = f_data_v15.get('low_mean2', 0)
                    _hm3 = f_data_v15.get('high_mean3', 0); _lm3 = f_data_v15.get('low_mean3', 0)
                    print(f"   [三区信号] 后区: high={_hm1:.0f} low={_lm1:.0f} | "
                          f"中区: high={_hm2:.0f} low={_lm2:.0f} | "
                          f"前区: high={_hm3:.0f} low={_lm3:.0f}")
                # 弧长坐标系初始聚类
                consensus_v15 = cluster_endplates_v15(
                    raw_cands_v15, _scan_lines_v15,
                    _c2_cols, _c2_rows, _arc_len_mm_v,
                    pixel_spacing, win_mm=5.0, min_lines=15)
                print(f"   [V15聚类] 初始终板线：{len(consensus_v15)} 条")
                
                # ── 全局回退策略：上下终板线只要有一个 < 5 条，就用 global_med × 0.5 重扫整个流程 ──
                _v15_scan_input = [(sl[0], sl[2], sl[1]) for sl in _scan_lines_v15]
                _min_ep_count  = 5  # 上下终板线最少各需 5 条
                _retry_added_keys = set()   # {(type_flag, line_idx), ...}
                
                # 统计上下终板线数量
                _sup_lines = [ep for ep in consensus_v15 if ep.get('ep_type') == 'superior']
                _inf_lines = [ep for ep in consensus_v15 if ep.get('ep_type') == 'inferior']
                _sup_count = len(_sup_lines)
                _inf_count = len(_inf_lines)
                print(f"   [V15] 上终板线：{_sup_count} 条，下终板线：{_inf_count} 条")
                
                # 检查是否需要全局回退
                if _sup_count < _min_ep_count or _inf_count < _min_ep_count:
                    print(f"   [V15 全局回退] 触发条件：上={_sup_count} 下={_inf_count} (< {_min_ep_count})")
                    # 重新扫描全 40 条线，同时降低 global_med、drop/rise 阈值 和 min_lines
                    _base_drop = f_data_v15.get('drop_ratio') if f_data_v15 else drop_ratio2
                    _base_rise = f_data_v15.get('rise_ratio') if f_data_v15 else rise_ratio2
                    _global_drop = _base_drop * 0.7  # 降低 30%
                    _global_rise = _base_rise * 0.7  # 降低 30%
                    _global_min_lines = max(10, int(15 * 0.7))  # 降低 30%，但不低于 10
                    print(f"   [V15 全局回退] 阈值调整：drop={_global_drop:.3f}, rise={_global_rise:.3f}, min_lines={_global_min_lines}")
                    _global_retry_data = find_endplates_on_water_image(
                        _v15_scan_input, f_img_2d, pixel_spacing,
                        use_half_global_med=True,
                        drop_ratio_override=_global_drop,
                        rise_ratio_override=_global_rise)
                    _new_c = _global_retry_data.get('raw_candidates', []) if _global_retry_data else []
                    # 去重合并
                    _exist_keys = {(p[0], p[4]) for p in raw_cands_v15}
                    _added_cnt  = 0
                    for _nc in _new_c:
                        _k = (_nc[0], _nc[4])
                        if _k not in _exist_keys:
                            raw_cands_v15.append(_nc)
                            _exist_keys.add(_k)
                            _retry_added_keys.add(_k)
                            _added_cnt += 1
                    print(f"   [V15全局回退] 新增 {_added_cnt} 个候选点")
                    if _added_cnt > 0:
                        # 重新聚类（使用降低后的 min_lines）
                        consensus_v15 = cluster_endplates_v15(
                            raw_cands_v15, _scan_lines_v15,
                            _c2_cols, _c2_rows, _arc_len_mm_v,
                            pixel_spacing, win_mm=5.0, min_lines=_global_min_lines)
                        # 交替校验：过滤掉连续同类型的终板线，保留行号更小的（更靠头侧）
                        _valid_eps = []
                        for ep in sorted(consensus_v15, key=lambda x: x['row_center']):
                            if not _valid_eps or _valid_eps[-1]['ep_type'] != ep['ep_type']:
                                _valid_eps.append(ep)
                            else:
                                print(f"   [V15 交替校验] 丢弃同类型 ({ep['ep_type']}) row={ep['row_center']:.1f}，保留头侧 row={_valid_eps[-1]['row_center']:.1f}")
                        consensus_v15 = _valid_eps
                        _sup_lines = [ep for ep in consensus_v15 if ep.get('ep_type') == 'superior']
                        _inf_lines = [ep for ep in consensus_v15 if ep.get('ep_type') == 'inferior']
                        print(f"   [V15 全局回退后] 上终板线：{len(_sup_lines)} 条，下终板线：{len(_inf_lines)} 条")
                else:
                    print(f"   [V15] 上下终板线数量满足要求，无需全局回退")
                
                # ── 回退校验策略：点数<28的终板线降低阈值重扫，最多2轮 ──
                _base_drop     = f_data_v15.get('drop_ratio') if f_data_v15 else None
                _base_rise     = f_data_v15.get('rise_ratio') if f_data_v15 else None
                _retry_factors = [0.80, 0.65]
                _min_pts_ep_base = 28  # 原始点数标准（min_lines=15时）
                # 如果已经执行过全局回退，则按 min_lines 比例调整点数标准；否则使用默认值 28
                if '_global_min_lines' in locals():
                    # min_lines 从 15 降到 10，比例 = 10/15 = 0.67，点数标准也同步降低
                    _pts_ratio = _global_min_lines / 15.0
                    _min_pts_ep = int(_min_pts_ep_base * _pts_ratio)  # 28 * 0.67 ≈ 19
                    print(f"   [V15 点数校验] 全局回退后点数标准调整：{_min_pts_ep_base} → {_min_pts_ep} (min_lines={_global_min_lines})")
                else:
                    _min_pts_ep = _min_pts_ep_base
                # 记录回退新增的候选点 key 集合，供可视化区分
                _retry_added_keys_pt = set()   # {(type_flag, line_idx), ...} 点数校验回退新增
                # 如果已经执行过全局回退，则继承其 min_lines；否则使用默认值 15
                _current_min_lines = _global_min_lines if '_global_min_lines' in locals() else 15
                
                for _retry_round, _factor in enumerate(_retry_factors, 1):
                    _weak_eps = [ep for ep in consensus_v15
                                 if len(ep.get('points', [])) < _min_pts_ep]
                    if not _weak_eps:
                        print(f"   [V15 回退] 第{_retry_round}轮：所有终板线点数已满足，停止回退")
                        break
                    print(f"   [V15 回退] 第{_retry_round}轮：{len(_weak_eps)}条点数<{_min_pts_ep}，"
                          f"阈值因子×{_factor}")
                    _rd = (_base_drop * _factor) if _base_drop else (0.25 * _factor)
                    _rr = (_base_rise * _factor) if _base_rise else (_rd * 0.9)
                    print(f"   [V15 回退] drop_ratio={_rd:.3f}, rise_ratio={_rr:.3f}")
                    # 对全部扫描线用降低后的阈值重扫，合并新找到的候选点（去重）
                    _retry_data = find_endplates_on_water_image(
                        _v15_scan_input, f_img_2d, pixel_spacing,
                        drop_ratio_override=_rd, rise_ratio_override=_rr)
                    _new_c = _retry_data.get('raw_candidates', []) if _retry_data else []
                    # 去重合并：以 (ep_type_flag, line_idx) 为 key
                    _exist_keys = {(p[0], p[4]) for p in raw_cands_v15}
                    _added_cnt  = 0
                    for _nc in _new_c:
                        _k = (_nc[0], _nc[4])
                        if _k not in _exist_keys:
                            raw_cands_v15.append(_nc)
                            _exist_keys.add(_k)
                            _retry_added_keys_pt.add(_k)   # 标记为点数校验回退新增
                            _added_cnt += 1
                    print(f"   [V15 点数回退] 第{_retry_round}轮：新增 {_added_cnt} 个候选点")
                    if _added_cnt == 0:
                        print(f"   [V15 点数回退] 第{_retry_round}轮：无新增候选点，跳过重聚类")
                        continue
                    consensus_v15 = cluster_endplates_v15(
                        raw_cands_v15, _scan_lines_v15,
                        _c2_cols, _c2_rows, _arc_len_mm_v,
                        pixel_spacing, win_mm=5.0, min_lines=_current_min_lines)
                    # 交替校验：过滤掉连续同类型的终板线，保留行号更小的（更靠头侧）
                    _valid_eps = []
                    for ep in sorted(consensus_v15, key=lambda x: x['row_center']):
                        if not _valid_eps or _valid_eps[-1]['ep_type'] != ep['ep_type']:
                            _valid_eps.append(ep)
                        else:
                            print(f"   [V15 交替校验] 丢弃同类型 ({ep['ep_type']}) row={ep['row_center']:.1f}，保留头侧 row={_valid_eps[-1]['row_center']:.1f}")
                    consensus_v15 = _valid_eps
                    print(f"   [V15 点数回退] 第{_retry_round}轮聚类后：{len(consensus_v15)} 条，"
                          f"点数满足 (>={_min_pts_ep}): "
                          f"{sum(1 for ep in consensus_v15 if len(ep.get('points',[]))>=_min_pts_ep)} 条")
                
                v9_data['raw_candidates_v15']      = raw_cands_v15
                v9_data['consensus_endplates_v15']  = consensus_v15
                # 合并全局回退 + 点数校验回退的 key 集合
                _all_retry_keys = _retry_added_keys.union(_retry_added_keys_pt)
                v9_data['retry_added_keys_v15']     = _all_retry_keys  # 全局回退 + 点数校验回退新增点 key 集合
                print(f"   [V15聚类最终] 终板线：{len(consensus_v15)} 条")
                _ant_high_mean = f_data_v15.get('high_mean3') if f_data_v15 else None
                _ant_low_mean  = f_data_v15.get('low_mean3')  if f_data_v15 else None
                # ── 完全对标V12_4逻辑，以皮质线2(c2_cols/c2_rows)为基准 ──
                # Step1: 水平offset谷底查找（皮质线2做base_col）
                arc_min_pts_v13 = find_arc_roi_min_points(
                    f_img_2d      = f_img_2d,
                    smooth_cols   = _c2_cols,
                    all_rows      = _c2_rows,
                    pixel_spacing = pixel_spacing,
                    left_off_mm   = 20.0,
                    right_off_mm  = 40.0,
                    expand_ratio  = 1.5,
                )
                # Step2: 第一轮上升沿精修（水平向右扫回骨髓，与V12_4完全一致）
                arc_refined1_v13 = refine_arc_roi_to_anterior_edge(
                    arc_min_pts_v13, f_img_2d, pixel_spacing,
                    high_mean   = _ant_high_mean if _ant_high_mean else 200.0,
                    low_mean    = _ant_low_mean  if _ant_low_mean  else 50.0,
                    rise_ratio  = 0.50,
                    probe_ratio = 0.6,
                    scan_mm     = 40.0,
                    smooth_win  = 2,
                )
                # Step3.6: 定义密集空间扩展参数（算法端与可视化端共用）
                _dense_expand_v13 = 3.0
                if v9_data is not None:
                    v9_data['dense_expand_v13'] = _dense_expand_v13

                # Step3.6: filter（第一次密集窗口：纯上升沿）
                arc_filtered_v13, arc_best_range_v13 = filter_arc_roi_by_dense_offset(
                    list(arc_refined1_v13), pixel_spacing, window_mm=6.0, step_mm=0.5,
                    expand_ratio=_dense_expand_v13)
                # 保存第一次密集窗口位置与过滤结果供可视化对比用
                arc_best_range1_v13  = arc_best_range_v13
                arc_filtered1_v13    = list(arc_filtered_v13)  # 第一次密集窗口输出点，供第二次聚类输入

                # Step4：平滑（与青线完全一致：MAD过滤+插値+5mm移动均値）
                # 数据源：arc_refined1_v13（未经平滑的原始精修点）中 offset 在密集窗口内的子集
                # 再加密集窗口右侧补充点（offset < best_lo，顶部50mm内）
                _best_lo_mm = arc_best_range_v13[0] if arc_best_range_v13 else 0.0
                _best_hi_mm = arc_best_range_v13[1] if arc_best_range_v13 else 999.0
                
                # 密集窗口内的原始精修点
                # 与可视化虚线框完全一致：expand=3.0，行号范围用皮质线2（_c2r）全范围
                _window_mm      = _best_hi_mm - _best_lo_mm
                _expand_ratio_s = v9_data.get('dense_expand_v13', 3.0) if v9_data else 3.0
                _c2r_vis = v9_data.get('c2_rows') if v9_data else None
                if _c2r_vis is not None and len(_c2r_vis) > 0:
                    _row_min_s  = float(min(_c2r_vis))
                    _row_max_s  = float(max(_c2r_vis))
                else:
                    _ref2_rows_input = [float(p[0]) for p in arc_refined1_v13]
                    _row_min_s = min(_ref2_rows_input) if _ref2_rows_input else 0.0
                    _row_max_s = max(_ref2_rows_input) if _ref2_rows_input else 0.0
                _row_span_s = max(_row_max_s - _row_min_s, 1.0)
                _ref2_v13 = []
                for _ps in arc_refined1_v13:
                    if (len(_ps) > 4 and _ps[4] == 'kept_low'):
                        continue
                    _t_s    = (float(_ps[0]) - _row_min_s) / _row_span_s
                    _hi_row_s = _best_lo_mm + _window_mm * (1.0 + (_expand_ratio_s - 1.0) * _t_s)
                    _off_s  = (float(_ps[3]) - float(_ps[1])) * pixel_spacing
                    if _best_lo_mm - 1e-6 <= _off_s <= _hi_row_s + 1e-6:
                        _ref2_v13.append(_ps)
                _ref2_v13 = sorted(_ref2_v13, key=lambda p: p[0])
                
                # 密集窗口右侧补充点（offset < best_lo，皮质线2总高度一半范围内）
                _filtered_rows = {int(p[0]) for p in _ref2_v13}
                _added_right = 0
                if arc_best_range_v13 is not None and _ref2_v13:
                    _top_row   = min(int(p[0]) for p in _ref2_v13)
                    # 用皮质线2总高度的一半作为限制（替代固定50mm）
                    if _c2r_vis is not None and len(_c2r_vis) > 0:
                        _c2_half_rows = (float(max(_c2r_vis)) - float(min(_c2r_vis))) / 2.0
                    else:
                        _c2_half_rows = 50.0 / pixel_spacing
                    _row_limit = _top_row + int(round(_c2_half_rows))
                    for _rp in arc_refined1_v13:
                        _rp_flag = _rp[4] if len(_rp) > 4 else 'kept'
                        if _rp_flag == 'kept_low':
                            continue
                        _rp_off = (float(_rp[3]) - float(_rp[1])) * pixel_spacing
                        _rp_row = int(_rp[0])
                        if (_rp_off < _best_lo_mm
                                and _rp_row not in _filtered_rows
                                and _rp_row <= _row_limit):
                            _ref2_v13.append(_rp)
                            _filtered_rows.add(_rp_row)
                            _added_right += 1
                    if _added_right > 0:
                        _ref2_v13 = sorted(_ref2_v13, key=lambda p: p[0])
                        print(f"   [V13前缘] 右侧补充点数: {_added_right}（限制在皮质线2总高度一半以内）")
                
                # 全局平滑：MAD过滤 → 插値 → 5mm移动均値（与青线完全一致）
                arc_final_v13 = []
                if _ref2_v13 and len(_ref2_v13) >= 2:
                    _rr2 = np.array([p[0] for p in _ref2_v13], dtype=np.float32)
                    _cc2 = np.array([p[1] for p in _ref2_v13], dtype=np.float32)
                    _mw2 = min(11, len(_cc2))
                    if _mw2 % 2 == 0: _mw2 -= 1
                    _mw2 = max(1, _mw2)
                    _vm2 = np.ones(len(_cc2), dtype=bool)
                    _hm2 = _mw2 // 2
                    for _i2 in range(len(_cc2)):
                        _s2 = max(0, _i2 - _hm2); _e2 = min(len(_cc2), _i2 + _hm2 + 1)
                        _w2 = _cc2[_s2:_e2]
                        _med2 = np.median(_w2); _mad2 = np.median(np.abs(_w2 - _med2))
                        if _mad2 > 0 and np.abs(_cc2[_i2] - _med2) / (_mad2 * 1.4826) > 2.0:
                            _vm2[_i2] = False
                    _cr2 = _cc2[_vm2]; _rrr2 = _rr2[_vm2]
                    if len(_rrr2) < 4: _cr2, _rrr2 = _cc2, _rr2
                    _ari2 = np.arange(int(_rrr2[0]), int(_rrr2[-1]) + 1)
                    _ic2  = np.interp(_ari2, _rrr2, _cr2)
                    _k2   = max(3, int(round(8.0 / pixel_spacing)))
                    if _k2 % 2 == 0: _k2 += 1
                    _csm2 = np.convolve(np.pad(_ic2, _k2 // 2, mode='edge'),
                                        np.ones(_k2) / _k2, mode='valid')
                    _rm2  = {int(r): float(c) for r, c in zip(_ari2, _csm2)}
                    # 输出：只输出_ref2_v13中的点（用平滑坐标）
                    for p in _ref2_v13:
                        arc_final_v13.append((p[0],
                                              _rm2.get(int(p[0]), float(p[1])),
                                              p[2], p[3],
                                              p[4] if len(p) > 4 else 'kept'))
                if not arc_final_v13:
                    arc_final_v13 = arc_filtered_v13

                # ── 头尾外推补充：将 arc_final_v13 延伸到 c2_rows 完整范围 ──
                # 目标：保证前缘线覆盖皮质线2全行（含延伸段），支持末端倾斜椎体闭合
                _c2r_ext = _c2_rows   # 直接用已取局部变量，包含延伸段
                _c2c_ext = _c2_cols
                if (_c2r_ext is not None and _c2c_ext is not None
                        and len(arc_final_v13) >= 4):
                    _af_sorted = sorted(arc_final_v13, key=lambda p: p[0])
                    _af_rows   = np.array([float(p[0]) for p in _af_sorted])
                    _af_cols   = np.array([float(p[1]) for p in _af_sorted])
                    _tgt_r0    = int(_c2r_ext[0])
                    _tgt_r1    = int(_c2r_ext[-1])
                    _ext_tail_px = max(4, int(round(10.0 / pixel_spacing)))  # 10mm参考段
                    _img_W       = f_img_2d.shape[1]

                    # ── 头部外推（向上）──
                    _cur_r0 = int(_af_rows[0])
                    if _cur_r0 > _tgt_r0:
                        _n = min(_ext_tail_px, len(_af_rows))
                        _seg_r = _af_rows[:_n]
                        _seg_c = _af_cols[:_n]
                        try:
                            _slp_h, _icp_h = np.polyfit(_seg_r, _seg_c, 1)
                        except Exception:
                            _slp_h = float(_seg_c[-1] - _seg_c[0]) / max(float(_seg_r[-1] - _seg_r[0]), 1.0)
                            _icp_h = float(_seg_c[0]) - _slp_h * float(_seg_r[0])
                        _head_pts = []
                        _ex_rows_h = set(int(p[0]) for p in _af_sorted)
                        for _er in range(_tgt_r0, _cur_r0):
                            if _er in _ex_rows_h:
                                continue
                            _ec = float(np.clip(_slp_h * _er + _icp_h, 0, _img_W - 1))
                            # 不超过同行皮质线2列坐标（保持在椎体侧）
                            _c2_col_h = float(np.interp(_er, _c2r_ext, _c2c_ext))
                            _ec = min(_ec, _c2_col_h)
                            _head_pts.append((_er, _ec, 0.0, _c2_col_h, 'extrapolated'))
                        if _head_pts:
                            arc_final_v13 = _head_pts + _af_sorted
                            arc_final_v13 = sorted(arc_final_v13, key=lambda p: p[0])
                            print(f"   [前缘头部外推] 补充 {len(_head_pts)} 行 "
                                  f"({_tgt_r0}→{_cur_r0-1}), 斜率dc/dr={_slp_h:.4f}")

                    # ── 尾部补充（向下）：从 arc_min_pts_v13 查表，取密集空间每行谷底 ──
                    _af_sorted2 = sorted(arc_final_v13, key=lambda p: p[0])
                    _cur_r1 = int(float(_af_sorted2[-1][0]))
                    if _cur_r1 < _tgt_r1:
                        # 建立谷底查找表：row -> (row, col, val, base_col)
                        _min_lut = {int(mp[0]): mp for mp in arc_min_pts_v13}
                        _tail_pts = []
                        _ex_rows_t = set(int(p[0]) for p in _af_sorted2)
                        for _er in range(_cur_r1 + 1, _tgt_r1 + 1):
                            if _er in _ex_rows_t:
                                continue
                            if _er in _min_lut:
                                _mp = _min_lut[_er]
                                _tail_pts.append((_mp[0], _mp[1], _mp[2], _mp[3], 'kept'))
                        if _tail_pts:
                            arc_final_v13 = _af_sorted2 + _tail_pts
                            arc_final_v13 = sorted(arc_final_v13, key=lambda p: p[0])
                            print(f"   [前缘尾部谷底补充] 补充 {len(_tail_pts)} 行 "
                                  f"({_cur_r1+1}→{_tgt_r1})，来源=arc_min_pts_v13")

                # 存入v9_data供可视化使用
                v9_data['arc_min_pts_v13']       = arc_min_pts_v13
                v9_data['arc_refined1_v13']      = arc_refined1_v13
                v9_data['arc_filtered_v13']      = arc_filtered_v13
                v9_data['arc_best_range1_v13']   = arc_best_range1_v13   # 第一次密集窗口（纯上升沿）
                v9_data['arc_best_range_v13']    = arc_best_range_v13
                v9_data['arc_refined2_raw_v13'] = arc_filtered_v13   # 单轮：filtered即为最终精修点
                v9_data['arc_refined_v13']      = arc_final_v13
                v9_data['arc_left_off_mm_v13']  = 20.0
                v9_data['arc_right_off_mm_v13'] = 40.0
                v9_data['arc_off_expand_v13']   = 1.5
                print(f"   [V13前缘-单轮] 谷底:{len(arc_min_pts_v13)}, "
                      f"精修:{len(arc_refined1_v13)}, 密集窗口:{len(arc_filtered_v13)}, "
                      f"最终:{len(arc_final_v13)}")

                # Step5: 新方案：腹侧下降沿法找前缘（并行运行）
                _consensus_eps = v9_data.get('consensus_endplates_v15', [])
                _dr3 = f_data_v15.get('drop_ratio3') if f_data_v15 else None
                # 从 scan_lines_v15 第一条提取法线方向（各条扫描线 nx/ny相同）
                _sl_v15 = v9_data.get('scan_lines_v15', [])
                _nx_arr_d = _sl_v15[0][3] if _sl_v15 else None
                _ny_arr_d = _sl_v15[0][4] if _sl_v15 else None
                arc_descent_v13 = find_anterior_edge_by_descent(
                    c2_cols          = _c2_cols,
                    c2_rows          = _c2_rows,
                    f_img_2d         = f_img_2d,
                    pixel_spacing    = pixel_spacing,
                    consensus_endplates = _consensus_eps,
                    high_mean3       = _ant_high_mean if _ant_high_mean else 200.0,
                    low_mean3        = _ant_low_mean  if _ant_low_mean  else 50.0,
                    drop_ratio3      = _dr3 if _dr3 else 0.35,
                    nx_arr           = _nx_arr_d,
                    ny_arr           = _ny_arr_d,
                    scan_lines_v15   = _sl_v15,
                )
                v9_data['arc_descent_v13'] = arc_descent_v13

                # ── V14_1: 椎间盘区域前缘点过滤（最后两组椎间盘）──
                # 需要过滤：最后两组椎间盘区域内部的点
                # 椎间盘区域 = inferior → superior 之间的行号范围
                _ax_cv15 = v9_data.get('consensus_endplates_v15', [])
                _scan_lines_v15 = v9_data.get('scan_lines_v15', [])
                
                print(f"   [V14_1 调试] consensus_endplates_v15: {len(_ax_cv15)} 条")
                print(f"   [V14_1 调试] scan_lines_v15: {len(_scan_lines_v15)} 条")
                print(f"   [V14_1 调试] arc_descent_v13: {len(arc_descent_v13) if arc_descent_v13 else 0} 点")
                
                # 打印所有终板线信息
                print(f"   [V14_1 调试] 全部终板线列表:")
                for idx, ep in enumerate(_ax_cv15):
                    print(f"     [{idx}] ep_type={ep.get('ep_type')}, row={ep.get('row_center'):.1f}, points={len(ep.get('points', []))}")
                
                if len(_ax_cv15) >= 4 and _scan_lines_v15 and arc_descent_v13:
                    # ── V14_1: 从后往前扫描，找到最后两组椎间盘 ──
                    # 解剖学关系：inferior → superior = 椎间盘 (5-15mm)
                    # 扫描策略：如果最后一条是 inferior 则丢弃，从第一个 superior 开始交替查找
                    
                    print(f"   [V14_1 扫描] 检查最后一条终板线类型...")
                    last_ep_type = _ax_cv15[-1].get('ep_type')
                    print(f"     最后一条 [{len(_ax_cv15)-1}] ep_type={last_ep_type}")
                    
                    # 确定起始索引
                    start_idx = len(_ax_cv15) - 1
                    if last_ep_type == 'inferior':
                        print(f"     → 最后一条是 inferior，丢弃，从倒数第二条开始")
                        start_idx = len(_ax_cv15) - 2
                    
                    # 从后往前扫描，收集 4 条有效的终板线（2 个 superior + 2 个 inferior，交替）
                    valid_endplates = []  # [(index, ep), ...]
                    expect_superior = True  # 期望找到 superior（因为从后往前是先遇到 superior）
                    
                    print(f"   [V14_1 扫描] 开始从后往前扫描，start_idx={start_idx}, 期望先找 superior")
                    for i in range(start_idx, -1, -1):  # 从最后一个往前
                        ep = _ax_cv15[i]
                        ep_type = ep.get('ep_type')
                        
                        print(f"     检查 [{i}] ep_type={ep_type}, expect_superior={expect_superior}")
                        
                        # 按期望类型匹配（交替查找）
                        if expect_superior and ep_type == 'superior':
                            valid_endplates.append((i, ep))
                            expect_superior = False  # 下一个期望 inferior
                            print(f"       → 选中 superior (第{len(valid_endplates)}条)")
                        elif not expect_superior and ep_type == 'inferior':
                            valid_endplates.append((i, ep))
                            expect_superior = True  # 下一个期望 superior
                            print(f"       → 选中 inferior (第{len(valid_endplates)}条)")
                        else:
                            print(f"       → 跳过 (类型不符)")
                        
                        # 找到 4 条就停止
                        if len(valid_endplates) == 4:
                            print(f"   [V14_1 扫描] 已找到 4 条，停止扫描")
                            break
                    
                    print(f"   [V14_1 调试] 从后往前找到 {len(valid_endplates)} 条有效终板线")
                    for idx, (orig_idx, ep) in enumerate(valid_endplates):
                        print(f"     [{idx}] 原索引={orig_idx}, ep_type={ep.get('ep_type')}, "
                              f"row={ep.get('row_center')}, points={len(ep.get('points', []))}")
                    
                    # 需要至少找到 4 条（2 个 inferior + 2 个 superior）
                    if len(valid_endplates) >= 4:
                        # 从后往前找到的顺序：
                        # valid_endplates[0] = 最后一个 inferior（最靠近尾部）
                        # valid_endplates[1] = 最后一个 superior
                        # valid_endplates[2] = 倒数第二个 inferior
                        # valid_endplates[3] = 倒数第二个 superior（最靠近头部）
                        
                        # 最后一组椎间盘：最后一个 inferior → 最后一个 superior
                        last_disc_inferior = valid_endplates[0][1]   # 最后面的 inferior
                        last_disc_superior = valid_endplates[1][1]   # 最后面的 superior
                        
                        # 倒数第二组椎间盘：倒数第二个 inferior → 倒数第二个 superior
                        second_last_disc_inferior = valid_endplates[2][1]
                        second_last_disc_superior = valid_endplates[3][1]
                        
                        print(f"   [V14_1 调试] 最后一组椎间盘：inferior row={last_disc_inferior.get('row_center')}, "
                              f"superior row={last_disc_superior.get('row_center')}")
                        print(f"   [V14_1 调试] 倒数第二组椎间盘：inferior row={second_last_disc_inferior.get('row_center')}, "
                              f"superior row={second_last_disc_superior.get('row_center')}")
                    else:
                        print(f"   [V14_1 警告] 未找到足够的终板线构建最后两组椎间盘")
                        last_disc_inferior = None
                        last_disc_superior = None
                        second_last_disc_inferior = None
                        second_last_disc_superior = None
                    
                    # ── V14_1: 为每个椎间盘单独计算基准线并过滤 ──
                    # 每个椎间盘用自己的基准线过滤自己区域内的点
                    # 椎间盘行号范围 = 同一条扫描线上与上下终板标记点的行号
                                                            
                    disc_baselines = {}  # {disc_idx: baseline_offset}
                    disc_row_ranges = []  # [(row_min, row_max), ...]
                                        
                    # 获取终板候选点（包含 line_idx）
                    raw_cands_v15 = v9_data.get('raw_candidates_v15', [])
                                                            
                    for disc_idx, (inf_ep, sup_ep) in enumerate([
                        (last_disc_inferior, last_disc_superior),      # 最后一组
                        (second_last_disc_inferior, second_last_disc_superior)  # 倒数第二组
                    ]):
                        if inf_ep is None or sup_ep is None:
                            continue
                                            
                        # 从 offset 最大（最左/腹侧）的扫描线开始向右（背侧）扫描
                        scan_lines_sorted = sorted(_scan_lines_v15, key=lambda x: x[0], reverse=True)
                                            
                        baseline_offset = None
                        disc_top_row = None    # 椎间盘顶部行号（上终板标记点）
                        disc_bottom_row = None # 椎间盘底部行号（下终板标记点）
                                            
                        for sl in scan_lines_sorted:
                            offset_mm, rows_arr, cols_arr, nx_arr, ny_arr = sl
                                                    
                            # 在这条扫描线上找上下终板的标记点
                            has_inferior_on_line = False
                            has_superior_on_line = False
                            row_inf = None
                            row_sup = None
                                                    
                            # 检查下终板标记点是否在这条扫描线上
                            inf_pts = inf_ep.get('points', [])
                            for inf_pt in inf_pts:
                                inf_row, inf_col = int(inf_pt[0]), int(inf_pt[1])
                                # 检查是否在扫描线上
                                for ri, ci in zip(rows_arr, cols_arr):
                                    if abs(ri - inf_row) <= 1 and abs(ci - inf_col) <= 2:
                                        has_inferior_on_line = True
                                        row_inf = float(ri)
                                        break
                                if has_inferior_on_line:
                                    break
                                                    
                            # 检查上终板标记点是否在这条扫描线上
                            sup_pts = sup_ep.get('points', [])
                            for sup_pt in sup_pts:
                                sup_row, sup_col = int(sup_pt[0]), int(sup_pt[1])
                                for ri, ci in zip(rows_arr, cols_arr):
                                    if abs(ri - sup_row) <= 1 and abs(ci - sup_col) <= 2:
                                        has_superior_on_line = True
                                        row_sup = float(ri)
                                        break
                                if has_superior_on_line:
                                    break
                                                    
                            # 如果这条扫描线上同时有上下终板标记点，就是基准线
                            if has_inferior_on_line and has_superior_on_line:
                                baseline_offset = offset_mm
                                disc_top_row = row_sup
                                disc_bottom_row = row_inf
                                print(f"   [椎间盘{disc_idx+1}] 基准线 offset={baseline_offset:.1f}mm, "
                                      f"扫描线上终板标记点行号：{disc_top_row:.1f}~{disc_bottom_row:.1f}")
                                break
                                                
                        if baseline_offset is None:
                            print(f"   [警告] 椎间盘{disc_idx+1}未找到基准线！")
                            print(f"     下终板点数：{len(inf_ep.get('points', []))}, "
                                  f"上终板点数：{len(sup_ep.get('points', []))}")
                                            
                        if baseline_offset is not None:
                            disc_baselines[disc_idx] = baseline_offset
                            if disc_top_row is not None and disc_bottom_row is not None:
                                row_min = min(disc_top_row, disc_bottom_row)
                                row_max = max(disc_top_row, disc_bottom_row)
                                disc_row_ranges.append((row_min, row_max))
                                        
                    print(f"   [椎间盘过滤] 各椎间盘基准线：{disc_baselines}")
                    print(f"   [椎间盘过滤] 椎间盘行号范围（扫描线标记点）: {disc_row_ranges}")
                                        
                    # 存入 v9_data 供可视化使用
                    v9_data['disc_row_ranges'] = disc_row_ranges
                    v9_data['disc_baselines'] = disc_baselines
                                        
                    # 过滤 arc_descent_v13 中位于椎间盘区域内的点
                    filtered_arc = []
                    for pt in arc_descent_v13:
                        row, col, flag, src_tag, base_col = pt
                        pt_row = float(row)
                                            
                        # 计算该点的 offset
                        pt_offset = (base_col - col) * pixel_spacing
                                            
                        # 检查该点属于哪个椎间盘区域
                        keep_point = True
                        for disc_idx, (row_min, row_max) in enumerate(disc_row_ranges):
                            if row_min - 1e-6 <= pt_row <= row_max + 1e-6:
                                # 点在椎间盘 disc_idx 区域内，用该椎间盘的基准线过滤
                                if disc_idx in disc_baselines:
                                    if pt_offset < disc_baselines[disc_idx] - 1e-6:
                                        keep_point = False
                                        print(f"   [下降沿过滤] 点 (row={pt_row:.1f}, offset={pt_offset:.1f}mm) "
                                              f"在椎间盘{disc_idx+1}内，offset<{disc_baselines[disc_idx]:.1f}mm → 过滤")
                                break
                                            
                        if keep_point:
                            filtered_arc.append(pt)
                                        
                    if len(filtered_arc) < len(arc_descent_v13):
                        print(f"   [椎间盘过滤] 下降沿 {len(arc_descent_v13) - len(filtered_arc)} 个点被过滤 "
                              f"({len(filtered_arc)}/{len(arc_descent_v13)} 保留)")
                                        
                    # 更新 arc_descent_v13
                    arc_descent_v13 = filtered_arc
                    v9_data['arc_descent_v13'] = filtered_arc
                                        
                    # ── V14_1: 过滤上升沿点 arc_refined1_v13 ──
                    if arc_refined1_v13:
                        filtered_rise = []
                        for pt in arc_refined1_v13:
                            # 格式：(row, col, val, base_col, flag)
                            if len(pt) >= 5:
                                pt_row = float(pt[0])
                                col = float(pt[1])
                                base_col = float(pt[3])
                                pt_offset = (base_col - col) * pixel_spacing
                                                    
                                # 检查该点属于哪个椎间盘区域
                                keep_point = True
                                for disc_idx, (row_min, row_max) in enumerate(disc_row_ranges):
                                    if row_min - 1e-6 <= pt_row <= row_max + 1e-6:
                                        # 点在椎间盘 disc_idx 区域内，用该椎间盘的基准线过滤
                                        if disc_idx in disc_baselines:
                                            if pt_offset < disc_baselines[disc_idx] - 1e-6:
                                                keep_point = False
                                        break
                                                    
                                if keep_point:
                                    filtered_rise.append(pt)
                                            
                        if len(filtered_rise) < len(arc_refined1_v13):
                            print(f"   [椎间盘过滤] 上升沿 {len(arc_refined1_v13) - len(filtered_rise)} 个点被过滤 "
                                  f"({len(filtered_rise)}/{len(arc_refined1_v13)} 保留)")
                                            
                        # 更新 arc_refined1_v13
                        arc_refined1_v13 = filtered_rise
                        v9_data['arc_refined1_v13'] = filtered_rise

                # Step5.1: 将下降沿 confirmed 点转换为统一格式，合并进密集窗口重跌
                # 下降沿点格式: (row, col, flag, src_tag, base_col)
                # 统一格式: (row, col, val, base_col, flag) -- 与 arc_refined1_v13 一致
                _descent_for_dense = []
                for _dp in arc_descent_v13:
                    if len(_dp) >= 5 and _dp[2] == 'confirmed':
                        _dp_row     = float(_dp[0])
                        _dp_col     = float(_dp[1])
                        _dp_basecol = float(_dp[4])
                        _descent_for_dense.append((_dp_row, _dp_col, 0.0, _dp_basecol, 'refined'))

                # 合并上升沿点（排除 kept_low）+ 下降沿 confirmed 点
                _combined_for_dense = [
                    p for p in arc_refined1_v13
                    if not (len(p) > 4 and p[4] in ('kept_low', 'kept'))
                ] + _descent_for_dense

                # ── 调试：打印合并点集 offset 分布 ──
                if _combined_for_dense:
                    _dbg_offs = sorted([(pt[3] - pt[1]) * pixel_spacing for pt in _combined_for_dense])
                    _dbg_rise = [p for p in arc_refined1_v13 if not (len(p) > 4 and p[4] in ('kept_low', 'kept'))]
                    print(f"   [调试-第二次密集] 合并点总数={len(_combined_for_dense)}"
                          f"（上升沿refined={len(_dbg_rise)}, 下降沿confirmed={len(_descent_for_dense)}）")
                    print(f"   [调试-第二次密集] offset范围: {_dbg_offs[0]:.1f}~{_dbg_offs[-1]:.1f}mm"
                          f"，中位数={_dbg_offs[len(_dbg_offs)//2]:.1f}mm")
                    # 打印offset直方图（每2mm一个桶）
                    import math
                    _lo_b = math.floor(_dbg_offs[0])
                    _hi_b = math.ceil(_dbg_offs[-1])
                    _buckets = {}
                    for _o in _dbg_offs:
                        _b = int((_o - _lo_b) // 2) * 2 + _lo_b
                        _buckets[_b] = _buckets.get(_b, 0) + 1
                    for _bk in sorted(_buckets):
                        print(f"     offset [{_bk:5.1f}~{_bk+2:.1f}mm]: {'█' * _buckets[_bk]} ({_buckets[_bk]})")

                # 用合并点集重新计算密集窗口
                if _combined_for_dense:
                    arc_filtered_v13, arc_best_range_v13 = filter_arc_roi_by_dense_offset(
                        _combined_for_dense, pixel_spacing, window_mm=6.0, step_mm=0.5,
                        expand_ratio=_dense_expand_v13)
                    # 额外统计：用动态扩展窗口分别统计两个位置的点数（对比用）
                    _r1_lo   = arc_best_range1_v13[0]
                    _r1_hi   = arc_best_range1_v13[1]
                    _r2_lo   = arc_best_range_v13[0]
                    _r2_hi   = arc_best_range_v13[1]
                    _dexp    = v9_data.get('dense_expand_v13', 3.0) if v9_data else 3.0
                    _c2r_st  = v9_data.get('c2_rows')
                    if _c2r_st is not None and len(_c2r_st) > 0:
                        _rmin_st = float(min(_c2r_st)); _rmax_st = float(max(_c2r_st))
                    else:
                        _rmin_st = float(min(p[0] for p in _combined_for_dense))
                        _rmax_st = float(max(p[0] for p in _combined_for_dense))
                    _rspan_st = max(_rmax_st - _rmin_st, 1.0)
                    _cnt_in_r1 = 0
                    _cnt_in_r2 = 0
                    for _p in _combined_for_dense:
                        _t_st  = (float(_p[0]) - _rmin_st) / _rspan_st
                        _off_p = (float(_p[3]) - float(_p[1])) * pixel_spacing
                        # 第一次窗口（动态扩展）
                        _hi1_dyn = _r1_lo + (_r1_hi - _r1_lo) * (1.0 + (_dexp - 1.0) * _t_st)
                        if _r1_lo - 1e-6 <= _off_p <= _hi1_dyn + 1e-6:
                            _cnt_in_r1 += 1
                        # 第二次窗口（动态扩展）
                        _hi2_dyn = _r2_lo + (_r2_hi - _r2_lo) * (1.0 + (_dexp - 1.0) * _t_st)
                        if _r2_lo - 1e-6 <= _off_p <= _hi2_dyn + 1e-6:
                            _cnt_in_r2 += 1
                    print(f"   [V13密集窗口-双模态] 上升沿:{len(arc_refined1_v13)}点 + "
                          f"下降沿 confirmed:{len(_descent_for_dense)}点 → "
                          f"第二次窗口:{_r2_lo:.1f}~{_r2_hi:.1f}mm（动态{_cnt_in_r2}点）"
                          f" | 第一次窗口:{_r1_lo:.1f}~{_r1_hi:.1f}mm（动态{_cnt_in_r1}点）")

                # 存入 v9_data 供可视化使用
                v9_data['arc_combined_v13']   = _combined_for_dense
                v9_data['arc_filtered_v13']   = arc_filtered_v13
                v9_data['arc_best_range_v13'] = arc_best_range_v13
            else:
                print("   [V15] 跳过V15聚类（c2_cols/scan_lines_v15 缺失）")
    
        else:
            print("   ⚠️ 未找到压水图，跳过终板检测")
    
    print("\n🎨 生成可视化...")
    vert_chain = visualize_results(slice_2d, traced, cord_mask, roi_points,
                     valid_rows, pixel_spacing, output_path, processor, v9_data,
                     f_img_2d=f_img_2d, f_data=f_data,
                     best_slice_idx=best_slice_idx, total_slices=data.shape[2])

    # ===== 导出掩模 + 坐标 CSV =====
    if vert_chain and f_img_2d is not None:
        print("\n📦 导出椎体掩模与几何数据...")
        orig_affine = nib.load(nifti_path).affine if nifti_path else None
        export_vertebra_data(vert_chain, f_img_2d.shape, output_path, pixel_spacing,
                             orig_affine=orig_affine)
        
    # 恢复标准输出
    sys.stdout = original_stdout
    
    # 提取关键信息
    log_content = log_capture.getvalue()
    # 将日志按行分割成列表
    log_lines = log_content.split('\n')
    
    json_filename = output_filename.replace('_TRACED.png', '_LOG.json')
    json_path = os.path.join(output_dir, json_filename)
    
    # 提取关键信息
    log_data = {
        'patient_dir': patient_dir,
        'seq_dir': seq_dir,
        'nifti_path': nifti_path,
        'best_slice_idx': int(best_slice_idx) if best_slice_idx is not None else None,
        'total_slices': data.shape[2] if 'data' in dir() else None,
        'pixel_spacing': pixel_spacing,
        'merged_regions': merged_regions,
        'status': 'success',
        'log_output': log_lines  # 使用列表形式
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    print(f"   [V14_2] 日志已保存：{os.path.basename(json_path)}")
    
    print(f"\n✅ 测试完成！")
    print(f"📁 输出文件：{output_path}")


# ============ 批量处理 ============
def process_batch(parent_dir, output_dir):
    """批量处理多个目录"""
    os.makedirs(output_dir, exist_ok=True)

    all_cases = []

    for root, dirs, files in os.walk(parent_dir):
        nifti_path    = os.path.join(root, 'scan.nii.gz')
        metadata_path = os.path.join(root, 'metadata.json')
        if not os.path.exists(nifti_path):
            continue
        # 从 metadata 校验必须是 T2 Dixon W 序列
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as fp:
                    m = json.load(fp)
                desc   = m.get('series_info', {}).get('series_description', '')
                s_type = _get_series_type(desc)
                if s_type != 'W':
                    continue  # 不是W(压脂)序列
                if 't2' not in desc.lower() or 'dixon' not in desc.lower():
                    continue  # 不是T2 Dixon序列
            except:
                continue  # metadata无法解析，跳过
        else:
            continue  # 没有metadata就无法校验，跳过
        patient_dir_name = os.path.basename(os.path.dirname(root))
        seq_dir_name     = os.path.basename(root)
        all_cases.append((patient_dir_name, seq_dir_name, nifti_path, metadata_path))

    print(f"📂 找到 {len(all_cases)} 个T2 Dixon W序列")

    for i, (patient_dir, seq_dir, nifti_path, metadata_path) in enumerate(all_cases):
        print(f"\n[{i+1}/{len(all_cases)}] 处理: {patient_dir}/{seq_dir}")
        test_single_image(nifti_path, metadata_path, output_dir, patient_dir, seq_dir)


# ============ 主函数 ============
def main():
    print("="*80)
    print("椎管分割与终板检测 - V14_2 版本（多掩模拼接增强 + 平行扫描线 + 终板暗线）")
    print("="*80)
    print("\n可选模式:")
    print("  1. 单张图像测试")
    print("  2. 批量处理")
    print()
    
    mode = input("请选择模式 (1/2): ").strip()
    
    if mode == '1':
        input_path = input("请输入图像文件或目录路径: ").strip()
        
        if os.path.isdir(input_path):
            nifti_path = os.path.join(input_path, "scan.nii.gz")
            metadata_path = os.path.join(input_path, "metadata.json")
            print(f"   📁 检测到目录，自动使用: {nifti_path}")
        else:
            nifti_path = input_path
            metadata_path = os.path.join(os.path.dirname(nifti_path), "metadata.json")
        
        if not os.path.exists(nifti_path):
            print(f"❌ 文件不存在: {nifti_path}")
            return
        
        if not os.path.exists(metadata_path):
            metadata_path = None
            print("   ⚠️ 未找到metadata.json，使用默认像素间距0.9375mm")
        
        output_dir = input("请输入输出目录 (直接回车默认为./test_output): ").strip()
        if not output_dir:
            output_dir = "./test_output"
        
        test_single_image(nifti_path, metadata_path, output_dir)
    
    elif mode == '2':
        parent_dir = input("请输入父目录路径: ").strip()
        if not parent_dir:
            parent_dir = "."
            
        output_dir = "/Users/mac/mri_lumbarpv/lumbar_roitest/batch_lsmatools_output"
        print(f"📁 输出目录：{output_dir}")
        process_batch(parent_dir, output_dir)
    
    else:
        print("❌ 无效选择")


if __name__ == "__main__":
    main()