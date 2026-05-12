#!/usr/bin/env python3
"""
LSMATools_WIFS – 主流程入口

功能：
  process_single(nifti_path, metadata_path, output_dir, ...) - 处理单例
  process_batch(input_dir, output_dir)                        - 批量处理
  main()                                                      - CLI 入口

流程（与 LSMATOOLS_CR mode4 完全对齐）：
  1. 加载 W 图 NIfTI + metadata
  2. 切片优选（椎管种子面积最大）
  3. 椎管追踪 → 皮质线1
  4. mode4（IN序列膜态分割）：
     皮质线1尾部延伸 → 皮质线2-2派生
     加载 IN 序列
     Step1: 信号参考值
     Step2+2b: 终板汇合点扫描 + 修补
     Step3+3.5: 椎间盘/椎体中心 + 最后两椎体汇合点校验
     Step4: 扇形扫描
     Step4c: 前缘二次校验（无条件执行）
     Step5: 聚类
     Step6: 椎体链路
  5. 四种输出：掩模/CSV/日志/可视化
"""

import os
import io
import sys
import json
import math
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

import numpy as np
import nibabel as nib
from pathlib import Path

# ── 内部模块导入 ──
from config.metadata_parser import load_metadata, parse_pixel_spacing
from config.params import SMOOTH_MM_C2, OFFSET_MM_SIGNAL
from preprocessing.series_utils import _get_series_type, _is_dixon_sequence
from preprocessing.slice_selector import select_best_slice
from preprocessing.image_loader import find_in_image
from segmentation.canal_processor import SpinalCanalProcessor
from segmentation.endplate_clusterer import cluster_endplates_v15
from detection.signal_ref import compute_signal_references
from detection.junction_detector import scan_endplate_junction_points, repair_junction_pts
from detection.disc_centers import compute_disc_and_vertebra_centers, verify_last_junction_point
from detection.fan_scanner import (
    fan_scan_vertebra, _sample_ant_local_signal, _calc_ant_angle_deg,
    _verify_ant_pts_forward, scan_disc_endplates,
    _fan_scan_direction, _scan_normal_descent_ant, _scan_normal_descent_ant_diag,
)
from detection.anterior_edge import cluster_all_vertebrae
from chain.vertebra_chain import build_vertebra_chain
from output.mask_export import export_masks
from output.csv_export import export_csv
from output.log_export import export_log
from output.visualization import visualize_wifs


# ─────────────────────────────────────────────────────────────────────────────
# 皮质线工具（mode4 内部辅助）
# ─────────────────────────────────────────────────────────────────────────────

def _extend_line_tail(rows, cols, pixel_spacing,
                      extend_mm=5.0, ref_mm=5.0):
    """取皮质线尾部做 polyfit，沿斜率延伸 extend_mm 路径距离。"""
    rows = list(rows); cols = list(cols)
    if len(rows) < 2:
        return rows, cols
    paired = sorted(zip(rows, cols), key=lambda p: p[0])
    rs = np.array([p[0] for p in paired], dtype=np.float64)
    cs = np.array([p[1] for p in paired], dtype=np.float64)

    cum_dist = np.zeros(len(rs))
    for k in range(1, len(rs)):
        cum_dist[k] = cum_dist[k-1] + math.sqrt(
            (rs[k]-rs[k-1])**2 + (cs[k]-cs[k-1])**2)
    total_dist = cum_dist[-1]

    ref_px = ref_mm / pixel_spacing
    mask = cum_dist >= (total_dist - ref_px)
    seg_r = rs[mask]; seg_c = cs[mask]
    if len(seg_r) < 2:
        seg_r = rs[-2:]; seg_c = cs[-2:]
    try:
        slp, _ = np.polyfit(seg_r, seg_c, 1)
    except Exception:
        slp = float(seg_c[-1] - seg_c[0]) / max(float(seg_r[-1] - seg_r[0]), 1e-9)

    t_len = math.sqrt(1.0 + slp**2)
    d_row = 1.0 / t_len
    d_col = slp / t_len

    ext_px = extend_mm / pixel_spacing
    cur_r = float(rs[-1]); cur_c = float(cs[-1])
    acc = 0.0
    ext_rows = []; ext_cols = []
    while acc < ext_px:
        cur_r += d_row; cur_c += d_col
        acc += math.sqrt(d_row**2 + d_col**2)
        ext_rows.append(cur_r); ext_cols.append(cur_c)

    return list(rs) + ext_rows, list(cs) + ext_cols


def _repair_slope(cols, slope_thr=0.5, neighbor_px=10, max_iter=3):
    """皮质线斜率连续性修复。"""
    cols = cols.astype(np.float64).copy()
    for _ in range(max_iter):
        slopes = np.diff(cols)
        bad = np.zeros(len(cols), dtype=bool)
        for i in range(len(slopes)):
            i0 = max(0, i - neighbor_px)
            i1 = min(len(slopes), i + neighbor_px + 1)
            local_med = float(np.median(slopes[i0:i1]))
            if abs(slopes[i] - local_med) > slope_thr:
                bad[i] = True; bad[i + 1] = True
        if not bad.any():
            break
        good_idx = np.where(~bad)[0]
        if len(good_idx) < 2:
            break
        for gi in range(len(good_idx) - 1):
            s = good_idx[gi]; e = good_idx[gi + 1]
            if e - s > 1:
                cols[s:e+1] = np.linspace(cols[s], cols[e], e - s + 1)
    return cols.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# process_single：单例处理主流程
# ─────────────────────────────────────────────────────────────────────────────

def process_single(nifti_path, metadata_path=None, output_dir=None,
                   patient_dir=None, seq_dir=None, fast_mode=False):
    """
    处理单例 W 图，输出掩模/CSV/日志/可视化。

    参数：
        nifti_path    - 压脂图 scan.nii.gz 路径
        metadata_path - metadata.json 路径（可为 None）
        output_dir    - 输出目录（默认 ./wifs_output）
        patient_dir   - 患者目录（可为 None，用于日志）
        seq_dir       - 序列目录（可为 None）
        fast_mode     - True：快速轻量模式（仅掩膜+CSV+左图可视化，不输出ROI和单例日志）

    返回：
        {'status': 'success'|'failed', 'n_vertebrae': int, ...}
    """
    # ── 捕获日志 ──
    log_capture = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = log_capture

    try:
        result = _run_single(
            nifti_path, metadata_path, output_dir,
            patient_dir, seq_dir, fast_mode=fast_mode)
    except Exception as exc:
        import traceback
        print(f"\n[FATAL] 处理异常: {exc}")
        traceback.print_exc()
        result = {'status': 'failed', 'error': str(exc), 'n_vertebrae': 0}
        if not result.get('stem'):
            try:
                p = Path(nifti_path)
                result['stem'] = f"{p.parent.parent.name}_{p.parent.name}"
            except Exception:
                result['stem'] = f"crash_{os.path.basename(nifti_path).replace('.nii.gz', '')}"
    finally:
        sys.stdout = original_stdout

    log_content = log_capture.getvalue()
    if output_dir:
        stem = result.get('stem')
        if not stem:
            stem = f"crash_{datetime.now().strftime('%H%M%S')}"
            result['stem'] = stem
        try:
            export_log(log_content, output_dir, stem)
        except Exception as log_exc:
            print(f"[log] 日志保存失败: {log_exc}", file=sys.stderr)

    # 终端仅输出保存记录和错误信息（其余分割日志只在文件中）
    _filtered_lines = []
    for _line in log_content.split('\n'):
        _s = _line.strip()
        if any(_s.startswith(f'[{t}]') for t in ['mask', 'roi', 'csv', 'log', 'vis']):
            _filtered_lines.append(_line)
        elif '[FATAL]' in _s or '❌' in _s:
            _filtered_lines.append(_line)
    if _filtered_lines:
        print('\n'.join(_filtered_lines))
    return result


def _run_single(nifti_path, metadata_path, output_dir,
                patient_dir, seq_dir, fast_mode=False):
    """实际的单例处理逻辑（由 process_single 包装）。"""
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'wifs_output')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("LSMATools_WIFS – 单例处理")
    print("=" * 70)
    print(f"\n📂 图像: {nifti_path}")

    # ── 构建输出文件名前缀 ──
    nii_path_obj = Path(nifti_path)
    seq_dir_name = nii_path_obj.parent.name
    patient_name = nii_path_obj.parent.parent.name
    stem = f"{patient_name}_{seq_dir_name}"

    # ── 加载图像 ──
    try:
        img = nib.load(nifti_path)
    except Exception as e:
        print(f"❌ 无法加载图像: {e}")
        return {'status': 'failed', 'stem': stem, 'n_vertebrae': 0}

    data       = img.get_fdata()
    orig_affine = img.affine

    # ── 读取 metadata ──
    pixel_spacing = 0.9375
    meta = {}
    if metadata_path and os.path.exists(metadata_path):
        try:
            meta = load_metadata(metadata_path)
            pixel_spacing = parse_pixel_spacing(meta)
            si = meta.get('series_info', {})
            w_desc = si.get('series_description', '')
            if w_desc:
                is_t2    = 't2' in w_desc.lower()
                is_dixon = _is_dixon_sequence(w_desc)
                is_w     = _get_series_type(w_desc) == 'W'
                if is_t2 and is_dixon and is_w:
                    print(f"   ✅ 序列校验通过: T2 Dixon W | '{w_desc}'")
                else:
                    print(f"   ⚠️ 序列校验: '{w_desc}' (t2={is_t2}, dixon={is_dixon}, W={is_w})")
        except Exception as e:
            print(f"   ⚠️ metadata 读取异常: {e}，使用默认 pixel_spacing={pixel_spacing}")
    print(f"   像素间距: {pixel_spacing:.4f} mm")

    # ── 切片优选 ──
    print("\n── 切片优选 ──")
    best_slice_idx, best_green_mask, _, merged_regions, best_canal_seed, csf_hints = \
        select_best_slice(data, pixel_spacing=pixel_spacing)
    slice_2d = data[:, :, best_slice_idx].astype(np.float32)
    print(f"   最佳切片第{best_slice_idx+1}张 / 共{data.shape[2]}张")

    # ── 椎管追踪 ──
    print("\n── 椎管追踪 ──")
    processor = SpinalCanalProcessor(pixel_spacing, meta=meta)
    if best_canal_seed is not None and np.any(best_canal_seed):
        traced, roi_points, valid_rows, status, v9_data = processor.process_with_mask(
            slice_2d, best_green_mask, best_canal_seed,
            merged_regions=merged_regions, csf_hints=csf_hints)
    else:
        traced, roi_points, valid_rows, status, v9_data = processor.process(slice_2d)

    if traced is None:
        print(f"   ❌ 椎管处理失败: {status}")
        return {'status': 'failed', 'stem': stem, 'n_vertebrae': 0}
    print(f"   ✅ 椎管追踪完成: {status}")

    # 从 v9_data 取皮质线1/2
    c1_rows = v9_data.get('all_rows')
    c1_cols = v9_data.get('smooth_cols')

    if c1_rows is None or c1_cols is None:
        print("   ❌ 皮质线1坐标缺失，跳过 mode4")
        return {'status': 'failed', 'stem': stem, 'n_vertebrae': 0}

    # ── 椎管掩模（canal_processor 返回的 traced 即椎管布尔掩模）──
    cord_mask_cut = traced

    # ── Mode4 开始 ──
    print("\n" + "=" * 60)
    print("Mode4: IN序列膜态分割")
    print("=" * 60)

    # 皮质线1尾部延伸
    c1_rows_ext, c1_cols_ext = _extend_line_tail(
        c1_rows, c1_cols, pixel_spacing, extend_mm=5.0, ref_mm=5.0)

    # 皮质线2-2（由延伸后皮质线1派生）
    _c1_cols_arr = np.array(c1_cols_ext, dtype=np.float32)
    _smooth_mm   = SMOOTH_MM_C2
    _k = max(3, int(_smooth_mm / pixel_spacing))
    if _k % 2 == 0: _k += 1
    _pad    = _k // 2
    _padded = np.pad(_c1_cols_arr.astype(np.float64), _pad, mode='edge')
    _kernel = np.ones(_k) / _k
    _c2_cols_smooth = np.convolve(_padded, _kernel, mode='valid').astype(np.float32)
    _c2_cols_fixed = _repair_slope(_c2_cols_smooth)
    c2_rows_mode4 = list(c1_rows_ext)
    c2_cols_mode4 = list(_c2_cols_fixed)

    # 加载 IN 序列
    print("\n── 加载 IN 序列 ──")
    in_img_2d, in_meta, in_nii_path = find_in_image(
        nifti_path, slice_idx=best_slice_idx)
    if in_img_2d is None:
        print("   ❌ 未找到IN序列，跳过 mode4")
        return {'status': 'failed', 'stem': stem, 'n_vertebrae': 0}

    H_in, W_in = in_img_2d.shape

    # Step1
    print("\n── Step1: 信号参考值 ──")
    low_mean, high_mean, profile_pts = compute_signal_references(
        in_img_2d, c2_rows_mode4, c2_cols_mode4, pixel_spacing, offset_mm=OFFSET_MM_SIGNAL)

    # Step2
    print("\n── Step2: 终板汇合点扫描 ──")
    junction_pts = []
    anchor_pts_list = []
    for pass_i, (off_start, off_end) in enumerate(
            [(1.0, 3.0), (3.0, 5.0), (5.0, 7.0)]):
        if pass_i > 0:
            print(f"   [Pass{pass_i+1}] offset{off_start:.0f}→{off_end:.0f}mm 重试...")
        junction_pts, anchor_pts_list = scan_endplate_junction_points(
            in_img_2d, c2_rows_mode4, c2_cols_mode4,
            pixel_spacing, low_mean,
            offset_start_mm=off_start, offset_end_mm=off_end)
        if len(junction_pts) >= 2:
            break
        print(f"   [Pass{pass_i+1}] 仅找到 {len(junction_pts)} 个汇合点，offset不足")

    if len(junction_pts) < 2:
        print("   ❌ 终板汇合点不足2个（三pass均失败），跳过mode4")
        return {'status': 'failed', 'stem': stem, 'n_vertebrae': 0}

    # Step2b
    print("\n── Step2b: 终板汇合点修补 ──")
    junction_pts, anchor_pts_list = repair_junction_pts(
        junction_pts, anchor_pts_list,
        c2_rows_mode4, c2_cols_mode4, pixel_spacing,
        in_img_2d=in_img_2d, c2_rows_scan=c2_rows_mode4, c2_cols_scan=c2_cols_mode4,
        low_mean=low_mean)

    # Step3
    print("\n── Step3: 椎间盘/椎体中心 ──")
    disc_centers, vert_centers = compute_disc_and_vertebra_centers(
        junction_pts, c2_rows_mode4, c2_cols_mode4, pixel_spacing,
        extend_start_mm=15.0, extend_end_mm=20.0, merge_dist_mm=5.0)
    if not vert_centers:
        print("   ❌ 未能生成椎体中心点，跳过mode4")
        return {'status': 'failed', 'stem': stem, 'n_vertebrae': 0}

    # Step3.5: 最后两椎体椎间盘中心角度自适应校验（无条件执行）
    print("\n── Step3.5: 最后两椎体汇合点校验 ──")
    angle_last = None   # 保留供 Step4/Step5 使用
    angle_2nd  = None   # 保留供 Step5 使用

    def _calc_junc_angle(jt, jb):
        d_r = float(jb[0]) - float(jt[0])
        d_c = float(jb[1]) - float(jt[1])
        return abs(math.degrees(math.atan2(abs(d_r), abs(d_c) + 1e-9)))

    if len(junction_pts) >= 2 and len(anchor_pts_list) >= 1 and len(disc_centers) >= 1:
        angle_last = _calc_junc_angle(junction_pts[-2], junction_pts[-1])
        print(f"   [Step3.5] 最后椎体连线夹角={angle_last:.1f}° → 执行校验")
        updated, new_disc, _ = verify_last_junction_point(
            in_img_2d, junction_pts[-1], anchor_pts_list[-1], disc_centers[-1],
            c2_rows_mode4, c2_cols_mode4, pixel_spacing, high_mean,
            angle_deg=angle_last)
        if updated:
            disc_centers[-1] = new_disc
            if len(disc_centers) >= 2:
                a = disc_centers[-2]; b = disc_centers[-1]
                vert_centers[-1] = ((a[0]+b[0])/2.0, (a[1]+b[1])/2.0)

    if len(junction_pts) >= 3 and len(anchor_pts_list) >= 2 and len(disc_centers) >= 2:
        angle_2nd = _calc_junc_angle(junction_pts[-3], junction_pts[-2])
        print(f"   [Step3.5] 倒数第二椎体连线夹角={angle_2nd:.1f}° → 执行校验")
        updated_sl, new_disc_sl, _ = verify_last_junction_point(
            in_img_2d, junction_pts[-2], anchor_pts_list[-2], disc_centers[-2],
            c2_rows_mode4, c2_cols_mode4, pixel_spacing, high_mean,
            angle_deg=angle_2nd)
        if updated_sl:
            disc_centers[-2] = new_disc_sl
            if len(disc_centers) >= 3:
                a = disc_centers[-3]; b = disc_centers[-2]
                vert_centers[-2] = ((a[0]+b[0])/2.0, (a[1]+b[1])/2.0)

    # Step4: 终板扫描（支持扇形/矩阵两种模式）
    print("\n── Step4: 终板扫描 ──")
    ep_scan_mode = 'disc'  # 'fan' = 扇形扫描（现有），'disc' = 椎间盘矩阵扫描（新）
    scan_results = []
    fan_params_list = []
    n_total_vert = len(vert_centers)
    for vi, vc in enumerate(vert_centers):
        print(f"   椎体{vi}...")
        _disc_top = disc_centers[vi]     if vi < len(disc_centers) else None
        _disc_bot = disc_centers[vi + 1] if vi + 1 < len(disc_centers) else None

        _is_last_vert = (vi == n_total_vert - 1)

        _ant_low_mean = None
        _ant_high_mean_local = None
        if vi >= n_total_vert - 2:
            _ant_ang = _calc_ant_angle_deg(vc, c2_rows_mode4, c2_cols_mode4,
                                           _disc_top, _disc_bot)
            _alm, _ahm = _sample_ant_local_signal(
                in_img_2d, float(vc[0]), float(vc[1]),
                _ant_ang, fan_step_deg=2.0, scan_mm=40.0,
                pixel_spacing=pixel_spacing)
            if _alm is not None:
                _ant_low_mean = _alm
                _ant_high_mean_local = _ahm
                _local_tag = 'last' if _is_last_vert else '2nd-last'
                print(f"      [局部信号] 椎体{vi}({_local_tag}) ant_ang={_ant_ang:.1f}°  "
                      f"local_low={_alm:.1f}  local_high={_ahm:.1f}  "
                      f"(全局 low={low_mean:.1f} high={high_mean:.1f})")

        _ant_diag = (_is_last_vert and angle_last is not None and angle_last < 65.0)

        if ep_scan_mode == 'disc':
            # 椎间盘矩阵扫描模式（仅终板，前缘仍用扇形）
            _sup_pts = []
            _inf_pts = []
            
            # 上终板用 disc_centers[vi] 和 junction_pts[vi]
            if _disc_top is not None and vi < len(junction_pts):
                _j_top = junction_pts[vi]
            else:
                _j_top = None
            
            # 下终板用 disc_centers[vi+1] 和 junction_pts[vi+1]
            if _disc_bot is not None and vi + 1 < len(junction_pts):
                _j_bot = junction_pts[vi + 1]
            else:
                _j_bot = None
            
            # 至少需要一个junction_pt才能构造起点连线
            if _j_top is not None and _j_bot is not None:
                _sr = scan_disc_endplates(
                    in_img_2d, vc, _j_top, _j_bot,
                    c2_rows_mode4, c2_cols_mode4, pixel_spacing,
                    low_mean, high_mean, scan_up_mm=30.0, scan_dn_mm=30.0,
                    drop_ratio=0.35)
                _sup_pts = _sr['sup_pts']
                _inf_pts = _sr['inf_pts']
            
            # 前缘仍用扇形扫描
            _ant_lm = _ant_low_mean if _ant_low_mean is not None else low_mean
            _ant_hm = _ant_high_mean_local if _ant_high_mean_local is not None else high_mean
            _ant_scan_fn = _scan_normal_descent_ant_diag if _ant_diag else _scan_normal_descent_ant
            _ant_angle_deg = _calc_ant_angle_deg(vc, c2_rows_mode4, c2_cols_mode4, _disc_top, _disc_bot)
            # S1 前缘信号对比度低，单独放宽下降沿阈值
            _ant_drop = 0.20 if _ant_diag else 0.35
            if _ant_diag:
                print(f"      [前缘S1] drop_ratio={_ant_drop:.2f} (全局0.35)")
            ant_pts, ant_dirs = _fan_scan_direction(
                float(vc[0]), float(vc[1]),
                _ant_angle_deg,
                50.0, 2.0, 40.0, pixel_spacing, in_img_2d,
                _ant_lm, _ant_hm,
                drop_ratio=_ant_drop, low_ratio=1.3, scan_fn=_ant_scan_fn)
            
            sr = {
                'sup': {'points': _sup_pts} if _sup_pts else {'points': []},
                'inf': {'points': _inf_pts} if _inf_pts else {'points': []},
                'ant': {'points': ant_pts, 'dirs': ant_dirs} if ant_pts else {'points': [], 'dirs': []},
                'fan_params': None,
                'ant_fan_params': {
                    'center': (float(vc[0]), float(vc[1])),
                    'angle':  _ant_angle_deg,
                    'half':   50.0,
                    'scan_mm': 40.0,
                },
            }
            sr['ant_high_mean_local'] = _ant_high_mean_local
        else:
            # 扇形扫描模式（原有逻辑）
            sr = fan_scan_vertebra(
                in_img_2d, vc, c2_rows_mode4, c2_cols_mode4, pixel_spacing, low_mean,
                fan_half_deg=50.0, fan_step_deg=2.0,
                scan_up_mm=30.0, scan_dn_mm=30.0, scan_ant_mm=40.0,
                low_ratio=1.3,
                high_mean=high_mean,
                drop_ratio=0.25,
                ant_drop_ratio=0.35,
                disc_top=_disc_top, disc_bot=_disc_bot,
                fan_half_ep_deg=60.0,
                ant_low_mean=_ant_low_mean,
                ant_high_mean=_ant_high_mean_local if _ant_high_mean_local is not None else high_mean,
                ant_diag_confirm=_ant_diag,
                ep_diag_confirm=_ant_diag)
            if _ant_diag:
                print(f"      [Step4前缘] 最后椎体后缘角={angle_last:.1f}°<65° → 启用对角信号确认（前缘\\+终板）")
            sr['ant_high_mean_local'] = _ant_high_mean_local
        scan_results.append(sr)
        fan_params_list.append(sr.get('fan_params'))

    # Step4c: 前缘二次校验（夹角 < 65° 才执行，高倾角骶椎区域跳过）
    last_junc_angle_deg        = None
    second_last_junc_angle_deg = None
    n_vert = len(scan_results)
    if len(junction_pts) >= 2 and n_vert >= 1:
        jt = junction_pts[-2]; jb = junction_pts[-1]
        last_junc_angle_deg = abs(math.degrees(
            math.atan2(abs(float(jb[0])-float(jt[0])),
                       abs(float(jb[1])-float(jt[1])) + 1e-9)))
        print(f"\n── Step4c: 最后椎体汇合点夹角={last_junc_angle_deg:.1f}°")
        if last_junc_angle_deg < 65.0:
            last_sr = scan_results[-1]
            _hm = last_sr.get('ant_high_mean_local') or high_mean
            _ant_ang_last = (last_sr.get('fan_params') or {}).get('ant', {}).get('angle')
            if _ant_ang_last is None:
                _ant_ang_last = (last_sr.get('ant_fan_params') or {}).get('angle')
            
            # 兼容两种格式：disc模式用 ant['points']/ant['dirs']，fan模式用 ant_pts/ant_dirs
            if 'ant' in last_sr:
                _ant_data = last_sr.get('ant')
                _pts = _ant_data.get('points', []) if _ant_data else []
                _dirs = _ant_data.get('dirs', []) if _ant_data else []
                _before = len(_pts)
                if _pts:
                    kept_pts, kept_dirs = _verify_ant_pts_forward(
                        _pts, _dirs, in_img_2d, pixel_spacing, _hm, forward_mm=5.0,
                        ant_angle_deg=_ant_ang_last)
                    last_sr['ant'] = {'points': kept_pts, 'dirs': kept_dirs}
            else:
                _pts = last_sr.get('ant_pts', [])
                _dirs = last_sr.get('ant_dirs', [])
                _before = len(_pts)
                if _pts:
                    kept_pts, kept_dirs = _verify_ant_pts_forward(
                        _pts, _dirs, in_img_2d, pixel_spacing, _hm, forward_mm=5.0,
                        ant_angle_deg=_ant_ang_last)
                    last_sr['ant_pts'] = kept_pts
                    last_sr['ant_dirs'] = kept_dirs
            
            _after = len((last_sr.get('ant') or {}).get('points', [])) if 'ant' in last_sr else len(last_sr.get('ant_pts', []))
            print(f"   [Step4c] 最后椎体前缘二次校验: {_before} → {_after} 点保留")
        else:
            print(f"   [Step4c] 最后椎体夹角={last_junc_angle_deg:.1f}° ≥ 65°，跳过前缘二次校验")

    if len(junction_pts) >= 3 and n_vert >= 2:
        jt2 = junction_pts[-3]; jb2 = junction_pts[-2]
        second_last_junc_angle_deg = abs(math.degrees(
            math.atan2(abs(float(jb2[0])-float(jt2[0])),
                       abs(float(jb2[1])-float(jt2[1])) + 1e-9)))
        print(f"── Step4c: 倒数第二椎体汇合点夹角={second_last_junc_angle_deg:.1f}°")
        if second_last_junc_angle_deg < 65.0:
            sl_sr = scan_results[-2]
            _hm2 = sl_sr.get('ant_high_mean_local') or high_mean
            _ant_ang_sl = (sl_sr.get('fan_params') or {}).get('ant', {}).get('angle')
            if _ant_ang_sl is None:
                _ant_ang_sl = (sl_sr.get('ant_fan_params') or {}).get('angle')
            
            # 兼容两种格式
            if 'ant' in sl_sr:
                _ant_data2 = sl_sr.get('ant')
                _pts2 = _ant_data2.get('points', []) if _ant_data2 else []
                _dirs2 = _ant_data2.get('dirs', []) if _ant_data2 else []
                _before2 = len(_pts2)
                if _pts2:
                    kp2, kd2 = _verify_ant_pts_forward(
                        _pts2, _dirs2, in_img_2d, pixel_spacing, _hm2, forward_mm=5.0,
                        ant_angle_deg=_ant_ang_sl)
                    sl_sr['ant'] = {'points': kp2, 'dirs': kd2}
            else:
                _pts2 = sl_sr.get('ant_pts', [])
                _dirs2 = sl_sr.get('ant_dirs', [])
                _before2 = len(_pts2)
                if _pts2:
                    kp2, kd2 = _verify_ant_pts_forward(
                        _pts2, _dirs2, in_img_2d, pixel_spacing, _hm2, forward_mm=5.0,
                        ant_angle_deg=_ant_ang_sl)
                    sl_sr['ant_pts'] = kp2
                    sl_sr['ant_dirs'] = kd2
            
            _after2 = len((sl_sr.get('ant') or {}).get('points', [])) if 'ant' in sl_sr else len(sl_sr.get('ant_pts', []))
            print(f"   [Step4c] 倒数第二椎体前缘二次校验: {_before2} → {_after2} 点保留")
        else:
            print(f"   [Step4c] 倒数第二椎体夹角={second_last_junc_angle_deg:.1f}° ≥ 65°，跳过前缘二次校验")

    # Step5: 聚类
    print("\n── Step5: 扇形聚类 ──")
    cluster_results = cluster_all_vertebrae(
        scan_results, disc_centers, pixel_spacing,
        junction_pts=junction_pts,
        c3_cols=v9_data.get('c3_cols'),
        c3_rows=v9_data.get('c3_rows'),
        arc_len_mm=v9_data.get('arc_len_mm'),
        last_ant_angle_deg=angle_last,
        second_last_ant_angle_deg=angle_2nd,
        c2_rows=c2_rows_mode4,
        c2_cols=c2_cols_mode4)

    # Step6: 椎体链路
    print("\n── Step6: 椎体链路 ──")
    vertebra_chain, ant_line = build_vertebra_chain(
        cluster_results, vert_centers, c2_rows_mode4, c2_cols_mode4,
        pixel_spacing, img_shape=(H_in, W_in),
        c2_rows=c2_rows_mode4, c2_cols=c2_cols_mode4,
        junction_pts=junction_pts, disc_centers=disc_centers)

    # 统计
    n_complete = sum(
        1 for e in vertebra_chain
        if all(e['quad'].get(k) is not None
               for k in ('sup_ant', 'sup_post', 'inf_ant', 'inf_post')))
    print(f"\n   完整识别={n_complete}")

    # ── 输出 ──
    print("\n── 输出 ──")

    # 1. 掩膜（S2 不保留）
    from output.mask_export import export_masks as _export_masks, export_roi_zip as _export_roi_zip
    # 快速模式：临时屏蔽 export_roi_zip，只输出 seg.nii.gz
    if fast_mode:
        import numpy as _np_roi
        from skimage.draw import polygon as _poly
        from skimage import measure as _measure_roi
        # 直接调用不含ROI的掩膜导出（复用 export_masks 但跳过 ROI 部分）
        # 通过在 mask_export 模块层面临时 patch
        import output.mask_export as _me
        _orig_roi = _me.export_roi_zip
        _me.export_roi_zip = lambda *a, **kw: None  # 临时禁用
        try:
            _export_masks(vertebra_chain, (H_in, W_in), output_dir, stem,
                         pixel_spacing, orig_affine=None, cord_mask_cut=cord_mask_cut,
                         ant_line=ant_line, c1_rows=c2_rows_mode4, c1_cols=c2_cols_mode4)
        finally:
            _me.export_roi_zip = _orig_roi  # 还原
    else:
        _export_masks(vertebra_chain, (H_in, W_in), output_dir, stem,
                     pixel_spacing, orig_affine=None, cord_mask_cut=cord_mask_cut,
                     ant_line=ant_line, c1_rows=c2_rows_mode4, c1_cols=c2_cols_mode4)

    # 2. CSV（全量和快速模式都输出）
    export_csv(vertebra_chain, (H_in, W_in), output_dir, stem,
               pixel_spacing, cord_mask_cut=cord_mask_cut,
               best_slice_idx=best_slice_idx)

    # 3. 可视化：快速模式只输出左图（W图链路）
    visualize_wifs(
        w_img_2d    = slice_2d,
        in_img_2d   = in_img_2d,
        vertebrae_chain = vertebra_chain,
        ant_line    = ant_line,
        cord_mask_cut   = cord_mask_cut,
        output_dir  = output_dir,
        stem        = stem,
        pixel_spacing   = pixel_spacing,
        c1_rows     = c1_rows_ext,
        c1_cols     = c1_cols_ext,
        c2_rows     = c2_rows_mode4,
        c2_cols     = c2_cols_mode4,
        junction_pts    = junction_pts,
        disc_centers    = disc_centers,
        vert_centers    = vert_centers,
        patient_label   = f"{patient_name}/{seq_dir_name}",
        in_label        = f"{patient_name}/{Path(in_nii_path).parent.name}" if in_nii_path else '',
        left_only       = fast_mode,
        scan_results    = scan_results,
        cluster_results = cluster_results,
        fan_params_list = fan_params_list,
        profile_pts     = profile_pts,
    )

    # 4. 日志在 process_single 中写出（此处 stem 返回供外层使用）

    return {
        'status': 'success',
        'stem': stem,
        'n_vertebrae': n_complete,
        'n_vertebrae_detected': len(vertebra_chain),
    }


# ─────────────────────────────────────────────────────────────────────────────
# _collect_w_series：预扫描匹配的 T2 Dixon W 序列
# ─────────────────────────────────────────────────────────────────────────────

def _collect_w_series(input_dir):
    """
    遍历 input_dir 下所有 scan.nii.gz，收集符合条件的 T2 Dixon W 序列。
    返回 [(nii_path, meta_path, patient_dir, seq_dir), ...]，按路径排序。
    """
    cases = []
    for nii_path in sorted(Path(input_dir).rglob('scan.nii.gz')):
        seq_dir_obj     = nii_path.parent
        patient_dir_obj = seq_dir_obj.parent
        meta_path = seq_dir_obj / 'metadata.json'
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as fp:
                m = json.load(fp)
            desc = m.get('series_info', {}).get('series_description', '')
            if not ('t2' in desc.lower() and _is_dixon_sequence(desc)
                    and _get_series_type(desc) == 'W'):
                continue
        except Exception:
            continue
        cases.append((nii_path, meta_path, patient_dir_obj, seq_dir_obj))
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# process_batch：批量处理（全量模式，模式2）
# ─────────────────────────────────────────────────────────────────────────────

def process_batch(input_dir, output_dir):
    """
    批量处理 input_dir 下所有病例（全量分割模式）。

    输出：
        output_dir/{patient}_{seq}/ 目录下各自的四种输出文件
        output_dir/batch_summary.json  批量处理摘要
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import time as _time

    # 预扫描并显示总数
    cases = _collect_w_series(input_dir)
    n_total = len(cases)
    print(f"\n🗂  批量处理: {input_dir}")
    print(f"   共找到 {n_total} 个匹配病例（T2 Dixon W 序列）")
    if n_total == 0:
        print("   ⚠️  未找到匹配病例，退出")
        return []

    summary = []
    success = 0; failed = 0
    total_n_vert = 0
    batch_start = _time.time()
    batch_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for idx, (nii_path, meta_path, patient_dir_obj, seq_dir_obj) in enumerate(cases, 1):
        case_output = output_dir / f"{patient_dir_obj.name}_{seq_dir_obj.name}"
        case_output.mkdir(parents=True, exist_ok=True)

        case_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{idx}/{n_total}] {patient_dir_obj.name}/{seq_dir_obj.name}")
        case_start = _time.time()
        res = process_single(
            str(nii_path), str(meta_path), str(case_output),
            patient_dir=str(patient_dir_obj),
            seq_dir=str(seq_dir_obj))
        case_elapsed = round(_time.time() - case_start, 2)

        n_vert = res.get('n_vertebrae', 0)
        total_n_vert += n_vert

        if res.get('status') == 'success':
            success += 1
        else:
            failed += 1

        summary.append({
            'patient':     patient_dir_obj.name,
            'seq':         seq_dir_obj.name,
            'start_time':  case_start_time,
            'status':      res.get('status'),
            'n_vertebrae': n_vert,
            'elapsed_s':   case_elapsed,
        })

    batch_elapsed = round(_time.time() - batch_start, 2)
    avg_elapsed   = round(batch_elapsed / n_total, 2) if n_total > 0 else 0.0

    header = {
        'batch_start_time': batch_start_time,
        'batch_end_time':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_cases':     n_total,
        'success':         success,
        'failed':          failed,
        'total_vertebrae': total_n_vert,
        'total_elapsed_s': batch_elapsed,
        'avg_elapsed_s':   avg_elapsed,
    }

    summary_path = output_dir / 'batch_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as fp:
        json.dump({**header, 'cases': summary}, fp, ensure_ascii=False, indent=2)

    print(f"\n✅ 批量处理完成: total={n_total}  success={success}  failed={failed}")
    print(f"   识别椎体总数={total_n_vert}  用时={batch_elapsed}s  平均={avg_elapsed}s/case")
    print(f"   摘要: {summary_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# process_batch_fast：批量快速轻量处理（模式3）
# ─────────────────────────────────────────────────────────────────────────────

def process_batch_fast(input_dir, output_dir):
    """
    批量快速轻量处理（模式3）：
      - 输出：掩膜（seg.nii.gz）+ CSV + 左图可视化（vis.png）
      - 跳过：ROI ZIP、单例日志
      - 保留：全局 batch_summary.json
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import time as _time

    # 预扫描并显示总数
    cases = _collect_w_series(input_dir)
    n_total = len(cases)
    print(f"\n🗂  批量快速处理 (模式3): {input_dir}")
    print(f"   共找到 {n_total} 个匹配病例（T2 Dixon W 序列）")
    if n_total == 0:
        print("   ⚠️  未找到匹配病例，退出")
        return []

    summary = []
    success = 0; failed = 0
    total_n_vert = 0
    batch_start = _time.time()
    batch_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for idx, (nii_path, meta_path, patient_dir_obj, seq_dir_obj) in enumerate(cases, 1):
        case_output = output_dir / f"{patient_dir_obj.name}_{seq_dir_obj.name}"
        case_output.mkdir(parents=True, exist_ok=True)

        case_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{idx}/{n_total}] {patient_dir_obj.name}/{seq_dir_obj.name}")
        case_start = _time.time()
        res = process_single(
            str(nii_path), str(meta_path), str(case_output),
            patient_dir=str(patient_dir_obj),
            seq_dir=str(seq_dir_obj),
            fast_mode=True)
        case_elapsed = round(_time.time() - case_start, 2)

        n_vert = res.get('n_vertebrae', 0)
        total_n_vert += n_vert

        if res.get('status') == 'success':
            success += 1
        else:
            failed += 1

        summary.append({
            'patient':     patient_dir_obj.name,
            'seq':         seq_dir_obj.name,
            'start_time':  case_start_time,
            'status':      res.get('status'),
            'n_vertebrae': n_vert,
            'elapsed_s':   case_elapsed,
        })

    batch_elapsed = round(_time.time() - batch_start, 2)
    avg_elapsed   = round(batch_elapsed / n_total, 2) if n_total > 0 else 0.0

    header = {
        'mode':             'fast',
        'batch_start_time': batch_start_time,
        'batch_end_time':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_cases':     n_total,
        'success':         success,
        'failed':          failed,
        'total_vertebrae': total_n_vert,
        'total_elapsed_s': batch_elapsed,
        'avg_elapsed_s':   avg_elapsed,
    }

    summary_path = output_dir / 'batch_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as fp:
        json.dump({**header, 'cases': summary}, fp, ensure_ascii=False, indent=2)

    print(f"\n✅ 批量快速处理完成: total={n_total}  success={success}  failed={failed}")
    print(f"   识别椎体总数={total_n_vert}  用时={batch_elapsed}s  平均={avg_elapsed}s/case")
    print(f"   摘要: {summary_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LSMATools_WIFS – 腰椎 WATER/IN 序列膜态分割工具")
    print("=" * 60)
    print("\n请选择处理模式:")
    print("  1. 单张图像处理")
    print("  2. 批量处理（全量分割模式：掩膜+CSV+日志+ROI+W/I双图可视化）")
    print("  3. 批量处理（快速轻量模式：掩膜+CSV+W图可视化，无ROI/单例日志）")
    print()

    mode = input("模式选择 (1/2/3): ").strip()

    if mode == '1':
        # ── 二级：单张 ──
        input_path = input("\n请输入图像路径（scan.nii.gz 或序列目录）: ").strip()

        if os.path.isdir(input_path):
            nifti_path    = os.path.join(input_path, 'scan.nii.gz')
            metadata_path = os.path.join(input_path, 'metadata.json')
            print(f"   检测到目录，自动使用: {nifti_path}")
        else:
            nifti_path    = input_path
            metadata_path = os.path.join(os.path.dirname(nifti_path), 'metadata.json')

        if not os.path.exists(nifti_path):
            print(f"❌ 文件不存在: {nifti_path}")
            return

        if not os.path.exists(metadata_path):
            metadata_path = None
            print("   ⚠️  未找到 metadata.json，使用默认像素间距 0.9375mm")

        output_dir = input("请输入输出目录 (直接回车默认 ./wifs_output): ").strip()
        if not output_dir:
            output_dir = "./wifs_output"

        result = process_single(nifti_path, metadata_path=metadata_path,
                                output_dir=output_dir)
        print(f"\n结果: {result}")

    elif mode == '2':
        # ── 二级：批量全量 ──
        input_dir = input("\n请输入输入根目录: ").strip()
        if not input_dir:
            print("❌ 输入目录不能为空")
            return

        if not os.path.isdir(input_dir):
            print(f"❌ 目录不存在: {input_dir}")
            return

        output_dir = input("请输入输出根目录 (直接回车默认 ./wifs_batch_output): ").strip()
        if not output_dir:
            output_dir = "./wifs_batch_output"

        process_batch(input_dir, output_dir)

    elif mode == '3':
        # ── 二级：批量快速轻量 ──
        input_dir = input("\n请输入输入根目录: ").strip()
        if not input_dir:
            print("❌ 输入目录不能为空")
            return

        if not os.path.isdir(input_dir):
            print(f"❌ 目录不存在: {input_dir}")
            return

        output_dir = input("请输入输出根目录 (直接回车默认 ./wifs_batch_fast_output): ").strip()
        if not output_dir:
            output_dir = "./wifs_batch_fast_output"

        process_batch_fast(input_dir, output_dir)

    else:
        print("❌ 无效选择，请输入 1、2 或 3")


if __name__ == '__main__':
    main()
