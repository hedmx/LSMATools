"""
可视化渲染模块
包含 visualize_results：双图拼接输出（左压脂 右压水）
依赖：postprocessing.vertebrae_chain.assign_vertebra_labels
     segmentation.canal_processor.SpinalCordLocator
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from skimage import measure
from skimage.morphology import skeletonize


def visualize_results(img_raw, traced, cord_mask, roi_points,
                     valid_rows, pixel_spacing, output_path,
                     processor=None, v9_data=None,
                     f_img_2d=None, f_data=None,
                     best_slice_idx=None, total_slices=None):
    """完整可视化 - V12双图拼接版本"""

    _export_chain = []  # 供函数末尾返回给 main.py 的椎体链路
    has_water = f_img_2d is not None
    n_cols    = 2 if has_water else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(14 * n_cols, 14))
    ax = axes[0] if has_water else axes

    # V14: 若已构建椎体链路，则根据 L1 和最下椎体行号做垂直裁剪（各保留约30mm）
    _crop_top = 0
    _crop_bot = img_raw.shape[0]
    try:
        # 在可视化阶段重算一次椎体链路，避免依赖上游局部变量
        if v9_data:
            _cv15 = v9_data.get('consensus_endplates_v15') or []
            _ep_curves = v9_data.get('endplate_curves_v15') or []
            _ax_ep_intersects = v9_data.get('ax_ep_intersects_v15') or []
            _ax_c2r = v9_data.get('c2_rows')
            if (_cv15 and _ep_curves and _ax_ep_intersects and _ax_c2r is not None):
                _epi_ix_map = {}
                for _ixp in _ax_ep_intersects:
                    _ixr, _ixc, _ixside, _ixepi = float(_ixp[0]), float(_ixp[1]), _ixp[2], _ixp[3]
                    if _ixepi not in _epi_ix_map:
                        _epi_ix_map[_ixepi] = {}
                    _epi_ix_map[_ixepi][_ixside] = (_ixr, _ixc)
                _ax_sorted_curves = sorted(
                    zip(_cv15, _ep_curves),
                    key=lambda x: x[0].get('row_center', x[1][1].mean() if len(x[1][1]) else 0)
                )
                _vertebrae_chain_vis = []
                _used_epi_vis = set()
                for _pi in range(len(_ax_sorted_curves) - 1, 0, -1):
                    _ep_top_meta, (_etype_top, _r_top, _c_top) = _ax_sorted_curves[_pi - 1]
                    _ep_bot_meta, (_etype_bot, _r_bot, _c_bot) = _ax_sorted_curves[_pi]
                    if _ep_top_meta.get('ep_type') != 'superior':
                        continue
                    if _ep_bot_meta.get('ep_type') != 'inferior':
                        continue
                    _row_top = float(np.mean(_r_top))
                    _row_bot = float(np.mean(_r_bot))
                    if _row_bot <= _row_top or (_row_bot - _row_top) < 3:
                        continue
                    _top_epi = None
                    _bot_epi = None
                    for _ci, (_epm, _epc) in enumerate(zip(_cv15, _ep_curves)):
                        if _epm is _ep_top_meta:
                            _top_epi = _ci
                        if _epm is _ep_bot_meta:
                            _bot_epi = _ci
                    if _top_epi is None or _bot_epi is None:
                        continue
                    if _top_epi in _used_epi_vis or _bot_epi in _used_epi_vis:
                        continue
                    _top_ix = _epi_ix_map.get(_top_epi, {})
                    _bot_ix = _epi_ix_map.get(_bot_epi, {})
                    if 'c1' not in _top_ix or 'front' not in _top_ix:
                        continue
                    if 'c1' not in _bot_ix or 'front' not in _bot_ix:
                        continue
                    _vertebrae_chain_vis.append({
                        'row_top': int(_row_top),
                        'row_bot': int(_row_bot)
                    })
                    _used_epi_vis.add(_top_epi)
                    _used_epi_vis.add(_bot_epi)
                if _vertebrae_chain_vis:
                    _vertebrae_chain_vis.sort(key=lambda x: x['row_top'])
                    _row_top_L1 = _vertebrae_chain_vis[0]['row_top']
                    _row_bot_lowest = _vertebrae_chain_vis[-1]['row_bot']
                    _pad_px = int(round(30.0 / max(pixel_spacing, 1e-6)))
                    _crop_top = max(0, _row_top_L1 - _pad_px)
                    _crop_bot = min(img_raw.shape[0], _row_bot_lowest + _pad_px)
    except Exception:
        _crop_top = 0
        _crop_bot = img_raw.shape[0]

    # 按裁剪范围截取图像
    img_raw = img_raw[_crop_top:_crop_bot, :]

    # ===== 左图：压脂图 =====
    ax.imshow(img_raw, cmap='gray')
    ax.set_title('压脂图 (W) - V14_2', fontsize=11)

    if v9_data:
        smooth_cols     = v9_data['smooth_cols']
        all_rows_       = v9_data['all_rows']
        s_dorsal_cols   = v9_data.get('smooth_dorsal_cols')
        dorsal_all_rows = v9_data.get('dorsal_all_rows')

        # 1. 皮质线1（白色实线）
        ax.plot(smooth_cols, all_rows_, color='white', linewidth=1.8,
                linestyle='-', alpha=0.9)

        # 2. 背部线（橙色实线）
        if s_dorsal_cols is not None and dorsal_all_rows is not None:
            ax.plot(s_dorsal_cols, dorsal_all_rows, color='orange',
                    linewidth=1.8, linestyle='-', alpha=0.9)

        # 3. 骨髓轮廓（黄色）
        if cord_mask is not None and np.any(cord_mask):
            contours_cord = measure.find_contours(cord_mask, 0.5)
            for c in contours_cord:
                ax.plot(c[:, 1], c[:, 0], color='yellow', linewidth=1.5, alpha=0.8)


    # 5. 椎管宽度标注
    if traced is not None and np.any(traced):
        area = np.sum(traced) * (pixel_spacing ** 2)
        skeleton = skeletonize(traced)
        sk_len = np.sum(skeleton) * pixel_spacing
        avg_width = area / sk_len if sk_len > 0 else 0
        width_color = 'orange' if processor and processor.width_warning else 'cyan'
        ax.text(0.02, 0.92, f'Avg Width: {avg_width:.1f}mm', transform=ax.transAxes,
                color=width_color, fontsize=10,
                weight='bold' if processor and processor.width_warning else 'normal',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

    # 切片索引标注（右上角）：选中切片 / 总切片数
    if best_slice_idx is not None and total_slices is not None:
        ax.text(0.98, 0.98,
                f'Slice: {best_slice_idx + 1}/{total_slices}',
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    # 6. 宽度警告
    if processor and processor.width_warning:
        ax.text(0.5, 0.98, f'⚠️ 椎管宽度异常: {processor.width_value:.1f}mm',
                transform=ax.transAxes, fontsize=12, color='red', weight='bold',
                ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.text(0.02, 0.98, f'Pixel Spacing: {pixel_spacing:.3f}mm',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ===== 左图叠加：最终椎体轮廓输出（前缘线 + 皮质线2 + 终板三角 + 聚类×标注）=====
    if v9_data and has_water:
        # ── 皮质线2（紫色虚线）──
        _ax_c2c = v9_data.get('c2_cols')
        _ax_c2r = v9_data.get('c2_rows')
        if _ax_c2c is not None and _ax_c2r is not None:
            ax.plot(_ax_c2c, _ax_c2r, color='magenta', linewidth=1.2,
                    linestyle='--', alpha=0.70, zorder=8)

        # ── 终板线（聚类结果连线+两端延伸至交点，inferior=red，superior=lawngreen）──
        _ax_cv15 = v9_data.get('consensus_endplates_v15', [])
        # 预建前缘线 col 查表（row→col），用于交点搜索（此时 _ax_red_pts 尚未构建，延后引用）
        # 皮质线2 行→列 查表
        _ax_c2_lut = {}
        if _ax_c2r is not None and _ax_c2c is not None:
            for _ri, _ci in zip(_ax_c2r, _ax_c2c):
                _ax_c2_lut[int(_ri)] = float(_ci)
        # 皮质线1 行→列 查表（左端截断用）
        _ax_c1_lut_pre = {}
        _ax_c1c_pre = v9_data.get('smooth_cols')
        _ax_c1r_pre = v9_data.get('all_rows')
        if _ax_c1c_pre is not None and _ax_c1r_pre is not None:
            for _ri, _ci in zip(_ax_c1r_pre, _ax_c1c_pre):
                _ax_c1_lut_pre[int(_ri)] = float(_ci)
        # 存储每条终板线的完整平滑曲线（供后续椎体分析用）
        _ax_ep_curves = []   # [(ep_type, r_sm_full, c_sm_full), ...]
        _ax_ep_ext_segs = [] # 延伸段暂存列表
        _ax_ep_intersects = []  # [(row, col, side, ep_idx), ...]  交点 side='c1'/'front'
        _ax_main_segs = []  # [(epi, ep_type, color, r_sm, c_sm), ...] 实线暂存，延后截断后绘制

        _EXT_MM   = 20.0   # 两端各延伸最大距离 mm
        _REF_MM   = 2.0    # 取末端斜率参考段 mm

        for _epi, _ax_ep in enumerate(_ax_cv15):
            _ax_ep_pts  = _ax_ep.get('points', [])  # [(row, col, depth, off_mm), ...]
            if len(_ax_ep_pts) < 2:
                continue
            _ax_ep_type = _ax_ep.get('ep_type', 'superior')
            _ax_ln_col  = 'red' if _ax_ep_type == 'inferior' else 'lawngreen'
            # 按 off_mm 排序后平滑连线
            _ax_ep_sorted = sorted(_ax_ep_pts, key=lambda p: p[3])
            _ax_epr = np.array([float(p[0]) for p in _ax_ep_sorted])
            _ax_epc = np.array([float(p[1]) for p in _ax_ep_sorted])
            _ax_epo = np.array([float(p[3]) for p in _ax_ep_sorted])
            _ax_kep = max(3, int(round(5.0 / pixel_spacing)))
            if _ax_kep % 2 == 0: _ax_kep += 1
            _ax_off_uni = np.linspace(_ax_epo[0], _ax_epo[-1],
                                      max(len(_ax_epo), int((_ax_epo[-1]-_ax_epo[0])/pixel_spacing)+1))
            _ax_r_uni = np.interp(_ax_off_uni, _ax_epo, _ax_epr)
            _ax_c_uni = np.interp(_ax_off_uni, _ax_epo, _ax_epc)
            _ax_r_sm  = np.convolve(np.pad(_ax_r_uni, _ax_kep//2, mode='edge'),
                                    np.ones(_ax_kep)/_ax_kep, mode='valid')
            _ax_c_sm  = np.convolve(np.pad(_ax_c_uni, _ax_kep//2, mode='edge'),
                                    np.ones(_ax_kep)/_ax_kep, mode='valid')
            # 实线暂存，延后按交叉点截断后绘制
            _ax_main_segs.append((_epi, _ax_ep_type, _ax_ln_col,
                                   _ax_r_sm.copy(), _ax_c_sm.copy()))
            _ax_ep_curves.append((_ax_ep_type, _ax_r_sm.copy(), _ax_c_sm.copy()))

            # ── 两端延伸（用整条终板线平均斜率）──
            _ext_px = int(round(_EXT_MM / pixel_spacing))

            # 整条终板线做一次 polyfit，得到平均斜率 row = _ek_all * col + _eb_all
            if len(_ax_c_sm) < 2:
                continue
            try:
                _ek_all, _eb_all = np.polyfit(_ax_c_sm, _ax_r_sm, 1)
            except Exception:
                continue

            for _side in ('left', 'right'):
                if _side == 'left':
                    # left 端 = _c_sm[0] = off_mm最小端 = col最大/最右端（靠皮质线2/皮质线1侧）
                    # 延伸朝右（col增大，朝皮质线1方向）
                    _base_r = float(_ax_r_sm[0])
                    _base_c = float(_ax_c_sm[0])
                    _dc_dir = 1   # col 增大（向右/皮质线1方向）
                else:
                    # right 端 = _c_sm[-1] = off_mm最大端 = col最小/最左端（靠前缘侧）
                    # 延伸朝左（col减小，朝前缘线方向）
                    _base_r = float(_ax_r_sm[-1])
                    _base_c = float(_ax_c_sm[-1])
                    _dc_dir = -1   # col 减小（向左/前缘线方向）

                # 用整条线平均斜率延伸
                _ek = _ek_all
                _dc = float(_dc_dir)
                _dr = _ek * _dc
                _t_len = np.sqrt(_dc**2 + _dr**2)
                if _t_len < 1e-6:
                    continue
                _dc /= _t_len; _dr /= _t_len

                # 生成延伸点序列（逐像素）
                _ext_rs = []
                _ext_cs = []
                for _step in range(1, _ext_px + 1):
                    _er = _base_r + _dr * _step
                    _ec = _base_c + _dc * _step
                    _ext_rs.append(_er)
                    _ext_cs.append(_ec)

                if not _ext_rs:
                    continue

                # ── 交叉点检测：实线 + 虚线合并扫描，保证鲁棒性 ──
                # 构建合并序列：实线段（从当前端点出发）+ 虚线延伸段
                # is_ext[i]=False 表示实线段点，True 表示虚线延伸点
                if _side == 'left':
                    # left端：实线从[0]正向遍历（col增大方向）
                    _main_rs = list(_ax_r_sm)
                    _main_cs = list(_ax_c_sm)
                else:
                    # right端：实线从[-1]反向遍历（col减小方向）
                    _main_rs = list(_ax_r_sm[::-1])
                    _main_cs = list(_ax_c_sm[::-1])
                _merged_rs = _main_rs + list(_ext_rs)
                _merged_cs = _main_cs + list(_ext_cs)
                _merged_is_ext = [False] * len(_main_rs) + [True] * len(_ext_rs)
                _n_main = len(_main_rs)

                _cut_idx = len(_ext_rs)  # 默认不截断虚线（right端延后处理）
                _left_ix_c = None  # left端交叉点列坐标（找到则记录，供实线裁剪用）
                if _side == 'left' and _ax_c1_lut_pre:
                    # left 端：实线+虚线合并，找与皮质线1的穿越点
                    _prev_diff = None
                    _found = False
                    for _si in range(len(_merged_rs)):
                        _er = _merged_rs[_si]; _ec = _merged_cs[_si]
                        _ri_int = int(round(_er))
                        if _ri_int in _ax_c1_lut_pre:
                            _ref_c = _ax_c1_lut_pre[_ri_int]
                            _diff = _ec - _ref_c
                            if _prev_diff is not None and _prev_diff * _diff <= 0:
                                _t_frac = abs(_prev_diff) / (abs(_prev_diff) + abs(_diff)) if (abs(_prev_diff) + abs(_diff)) > 1e-9 else 0.5
                                _ix_r = _merged_rs[_si-1] + _t_frac * (_er - _merged_rs[_si-1])
                                _ix_r_int = int(round(_ix_r))
                                _ix_c = _ax_c1_lut_pre[_ix_r_int] if _ix_r_int in _ax_c1_lut_pre else _ref_c
                                _ax_ep_intersects.append((_ix_r, _ix_c, 'c1', _epi))
                                _left_ix_c = _ix_c  # 记录交叉点列坐标
                                # 若交叉在实线段内，虚线全不画（cut_idx=0）
                                # 若交叉在虚线段内，截断到该点
                                if not _merged_is_ext[_si]:
                                    _cut_idx = 0
                                else:
                                    _cut_idx = _si - _n_main + 1
                                _found = True
                                break
                            _prev_diff = _diff
                # right端交点：等 _ax_red_pts 构建后处理，暂存全段+实线快照+left交叉点列
                _ax_ep_ext_segs.append((_ax_ep_type, _ax_ln_col, _side, _epi,
                                        np.array(_ext_rs[:_cut_idx]),
                                        np.array(_ext_cs[:_cut_idx]),
                                        np.array(_main_rs, dtype=np.float64),
                                        np.array(_main_cs, dtype=np.float64),
                                        _left_ix_c))

        # 右端延伸截断需要 _ax_red_pts，延后在红线生成后处理（标记占位）
        _ax_ep_ext_right_pending = True  # 标志：右端截断待处理
        
        # ── 综合红色线：密集窗口过滤（双模态合并点集）生成──
        _ax_combined   = v9_data.get('arc_combined_v13', [])
        _ax_best_range = v9_data.get('arc_best_range_v13', None)
        _ax_expand     = v9_data.get('dense_expand_v13', 3.0)
        _ax_c2r        = v9_data.get('c2_rows')
                
        # 重建密集窗口范围（与黄线数据源完全一致）
        _ax_red_pts = {}
        if _ax_combined and _ax_best_range is not None:
            _ax_best_lo  = _ax_best_range[0]
            _ax_best_hi  = _ax_best_range[1]
            _ax_win_mm   = _ax_best_hi - _ax_best_lo
            if _ax_c2r is not None and len(_ax_c2r) > 0:
                _ax_rmin = float(min(_ax_c2r)); _ax_rmax = float(max(_ax_c2r))
            else:
                _ax_rmin = float(min(p[0] for p in _ax_combined))
                _ax_rmax = float(max(p[0] for p in _ax_combined))
            _ax_rspan = max(_ax_rmax - _ax_rmin, 1.0)
                    
            # 1. 密集窗口内点（arc_combined_v13，已含上升沿 + 下降沿 confirmed）
            for _rp in _ax_combined:
                _t   = (float(_rp[0]) - _ax_rmin) / _ax_rspan
                _hi  = _ax_best_lo + _ax_win_mm * (1.0 + (_ax_expand - 1.0) * _t)
                _off = (float(_rp[3]) - float(_rp[1])) * pixel_spacing
                if _ax_best_lo - 1e-6 <= _off <= _hi + 1e-6:
                    _ax_red_pts[int(_rp[0])] = (float(_rp[0]), float(_rp[1]))
            # 2. 右侧补充点（offset < best_lo，僅上升沿点，排除 kept_low）
            if _ax_red_pts and _ax_c2r is not None:
                _ax_top    = min(_ax_red_pts.keys())
                _ax_c2half = (float(max(_ax_c2r)) - float(min(_ax_c2r))) / 2.0
                _ax_rlimit = _ax_top + int(round(_ax_c2half))
                _ax_ref1_only = v9_data.get('arc_refined1_v13', [])
                for _rp in _ax_ref1_only:
                    if (len(_rp) > 4 and _rp[4] == 'kept_low'):
                        continue
                    _rp_off = (float(_rp[3]) - float(_rp[1])) * pixel_spacing
                    _rp_row = int(_rp[0])
                    if (_rp_off < _ax_best_lo
                            and _rp_off >= _ax_best_lo - 5.0
                            and _rp_row not in _ax_red_pts
                            and _rp_row <= _ax_rlimit):
                        _ax_red_pts[_rp_row] = (float(_rp[0]), float(_rp[1]))
            # 3. 尾部谷底补充：从 arc_min_pts_v13 查表，补充到 c2_rows 底行
            if _ax_red_pts and _ax_c2r is not None and len(_ax_c2r) > 0:
                _ax_tgt_r1 = int(_ax_c2r[-1])
                _ax_cur_r1 = max(_ax_red_pts.keys())
                if _ax_cur_r1 < _ax_tgt_r1:
                    _ax_min_pts = v9_data.get('arc_min_pts_v13', [])
                    _ax_min_lut = {int(mp[0]): mp for mp in _ax_min_pts}
                    _ax_added = 0
                    for _er in range(_ax_cur_r1 + 1, _ax_tgt_r1 + 1):
                        if _er not in _ax_red_pts and _er in _ax_min_lut:
                            _mp = _ax_min_lut[_er]
                            _ax_red_pts[_er] = (float(_mp[0]), float(_mp[1]))
                            _ax_added += 1
                    if _ax_added > 0:
                        print(f"   [左图红线尾部补充] 补充 {_ax_added} 行 "
                              f"({_ax_cur_r1+1}→{_ax_tgt_r1})，来源=arc_min_pts_v13")
                
        # ========== 前缘红线平滑 + 插值（V14_1 几何一致性优化）==========
        # 目标：MAD 过滤 → 逐行插值 → 构建查找表 → 可视化 + 交叉点检测
        _ax_front_lut = None  # 行→列 查找表（全局共享）
        _axr_sm = None
        _axr_ri = None
                
        if len(_ax_red_pts) >= 2:
            # 1. 排序原始点
            _ax_rp_sorted = sorted(_ax_red_pts.values(), key=lambda x: x[0])
            _axr_r = np.array([p[0] for p in _ax_rp_sorted], dtype=np.float32)
            _axr_c = np.array([p[1] for p in _ax_rp_sorted], dtype=np.float32)
                    
            # 2. MAD 过滤（去除噪声点）
            _axr_mw = min(11, len(_axr_c))
            if _axr_mw % 2 == 0: _axr_mw -= 1
            _axr_mw = max(1, _axr_mw)
            _axr_v  = np.ones(len(_axr_c), dtype=bool)
            _axr_hf = _axr_mw // 2
            for _axr_i in range(len(_axr_c)):
                _s = max(0, _axr_i - _axr_hf); _e = min(len(_axr_c), _axr_i + _axr_hf + 1)
                _ww = _axr_c[_s:_e]
                _med = np.median(_ww); _mad = np.median(np.abs(_ww - _med))
                if _mad > 0 and np.abs(_axr_c[_axr_i] - _med) / (_mad * 1.4826) > 2.0:
                    _axr_v[_axr_i] = False
                    
            _axr_cf = _axr_c[_axr_v]; _axr_rf = _axr_r[_axr_v]
                    
            # 3. 回退机制：过滤后点太少则使用全部点
            if len(_axr_rf) < 4:
                print(f"   ⚠️ 前缘红线 MAD 过滤后仅剩{len(_axr_rf)}点，回退到原始{len(_axr_r)}点")
                _axr_cf, _axr_rf = _axr_c, _axr_r
                    
            # 4. 插值到每一行（关键！确保椎体分割时任意行都有定义）
            _axr_row_start = int(_axr_rf[0])
            _axr_row_end = int(_axr_rf[-1])
            _axr_ri = np.arange(_axr_row_start, _axr_row_end + 1)
            _axr_ci = np.interp(_axr_ri, _axr_rf, _axr_cf)
                    
            # 5. 5mm 移动均值平滑（手动实现滑动平均，确保长度一致）
            _axr_k = max(3, int(round(5.0 / pixel_spacing)))
            if _axr_k % 2 == 0: _axr_k += 1
            _half_k = _axr_k // 2
            
            # 使用滚动窗口实现移动均值，保持长度一致
            _axr_sm = np.zeros_like(_axr_ci)
            for _i in range(len(_axr_ci)):
                _s = max(0, _i - _half_k)
                _e = min(len(_axr_ci), _i + _half_k + 1)
                _axr_sm[_i] = np.mean(_axr_ci[_s:_e])
                        
            # 6. 构建查找表（行→列，用于交叉点检测）★V14_1 优化
            _ax_front_lut = dict(zip(_axr_ri.tolist(), _axr_sm.tolist()))
                        
            print(f"   [前缘红线] MAD 过滤：{len(_axr_r)}→{len(_axr_rf)}点，"
                  f"插值范围：{_axr_row_start}~{_axr_row_end}，查找表：{len(_ax_front_lut)}行")
                        
            # 7. 绘制红色线（长度一致，直接绘制）
            ax.plot(_axr_sm, _axr_ri, '-', color='red', linewidth=1.2,
                    alpha=0.85, zorder=12)
                
        # ── left 端延伸截断（已有 _ax_front_lut 前缘线查找表）+ 画延伸段 + 交点 ──
        if _ax_ep_ext_right_pending and _ax_front_lut is not None:
            # 收集每条终板线两端交叉点列坐标
            # {epi: {'left': ix_c_or_None, 'right': ix_c_or_None}}
            _ep_ix_cols = {}
            for _eseg in _ax_ep_ext_segs:
                _etype, _ecol, _eside, _eidx, _ers, _ecs, _emrs, _emcs, _lixc = _eseg
                if _eidx not in _ep_ix_cols:
                    _ep_ix_cols[_eidx] = {'left': None, 'right': None, 'col': _ecol}
                if _eside == 'left':
                    _ep_ix_cols[_eidx]['left'] = _lixc
        
            # 处理每个延伸段，同时收集 right 端交叉点
            for _eseg in _ax_ep_ext_segs:
                _etype, _ecol, _eside, _eidx, _ers, _ecs, _emrs, _emcs, _lixc = _eseg
                _right_ix_c = None
                if _eside == 'right':
                    # right 端：col 最小/最左端，实线 + 虚线合并，找与前缘红线的穿越点
                    _merged_r = list(_emrs) + list(_ers)
                    _merged_c = list(_emcs) + list(_ecs)
                    _n_main_r = len(_emrs)
                    _is_ext_r = [False] * _n_main_r + [True] * len(_ers)
                    _cut = len(_ers)  # 默认虚线全画
                    _ers_final = np.array(_ers, dtype=np.float64)
                    _ecs_final = np.array(_ecs, dtype=np.float64)
                    _prev_diff_l = None
                    for _sil in range(len(_merged_r)):
                        _erl = float(_merged_r[_sil]); _ecl = float(_merged_c[_sil])
                        _ril_int = int(round(_erl))
                        if _ril_int in _ax_front_lut:  # ✅ 使用 MAD 过滤后的查找表
                            _ref_cl = _ax_front_lut[_ril_int]
                            _diffl = _ecl - _ref_cl
                            if _prev_diff_l is not None and _prev_diff_l * _diffl <= 0:
                                _tl_frac = abs(_prev_diff_l) / (abs(_prev_diff_l) + abs(_diffl)) if (abs(_prev_diff_l) + abs(_diffl)) > 1e-9 else 0.5
                                _ixl_r = float(_merged_r[_sil-1]) + _tl_frac * (_erl - float(_merged_r[_sil-1]))
                                _ixl_r_int = int(round(_ixl_r))
                                _ixl_c = _ax_front_lut[_ixl_r_int] if _ixl_r_int in _ax_front_lut else _ref_cl
                                _ax_ep_intersects.append((_ixl_r, _ixl_c, 'front', _eidx))
                                _right_ix_c = _ixl_c  # 记录 right 端交叉点列坐标
                                # 若交叉在实线段内，虚线全不画
                                if not _is_ext_r[_sil]:
                                    _cut = 0
                                else:
                                    _cut = _sil - _n_main_r + 1
                                break
                            _prev_diff_l = _diffl
                    _ers_final = _ers_final[:_cut]
                    _ecs_final = _ecs_final[:_cut]
                    _ep_ix_cols[_eidx]['right'] = _right_ix_c
                else:
                    _ers_final = np.array(_ers, dtype=np.float64)
                    _ecs_final = np.array(_ecs, dtype=np.float64)
                # 画虚线延伸段
                if len(_ers_final) >= 2:
                    ax.plot(_ecs_final, _ers_final, '--', color=_ecol,
                            linewidth=1.2, alpha=0.75, zorder=10)

            # 绘制实线（按两端交叉点裁剪）
            for _mseg in _ax_main_segs:
                _mepi, _metype, _mecol, _mr_sm, _mc_sm = _mseg
                _mix = _ep_ix_cols.get(_mepi, {'left': None, 'right': None})
                _mc_arr = np.array(_mc_sm, dtype=np.float64)
                _mr_arr = np.array(_mr_sm, dtype=np.float64)
                # left端交叉点（皮质线1侧，col最大端）：col > _mix['left'] 的部分切掉
                _c_lo = _mix['right']  # right端交叉点列（前缘侧，col小）
                _c_hi = _mix['left']   # left端交叉点列（皮质线1侧，col大）
                _mask = np.ones(len(_mc_arr), dtype=bool)
                if _c_lo is not None:
                    _mask &= (_mc_arr >= _c_lo)  # 切掉前缘交叉点左侧
                if _c_hi is not None:
                    _mask &= (_mc_arr <= _c_hi)  # 切掉皮质线1交叉点右侧
                _mr_plot = _mr_arr[_mask]
                _mc_plot = _mc_arr[_mask]
                if len(_mr_plot) >= 2:
                    ax.plot(_mc_plot, _mr_plot, '-', color=_mecol,
                            linewidth=1.8, alpha=0.90, zorder=11)
                elif len(_mr_arr) >= 2:  # 未找到交叉点则画全段
                    ax.plot(_mc_arr, _mr_arr, '-', color=_mecol,
                            linewidth=1.8, alpha=0.90, zorder=11)

            # 画交点 +标记
            _drawn_ips = set()
            for _ip in _ax_ep_intersects:
                _ip_r, _ip_c = float(_ip[0]), float(_ip[1])
                _ip_key = (round(_ip_r), round(_ip_c))
                if _ip_key in _drawn_ips:
                    continue
                _drawn_ips.add(_ip_key)
                ax.plot(_ip_c, _ip_r, '+', color='cyan',
                        markersize=10, markeredgewidth=1.5, zorder=15)

        # ── 最下方完整椎体分析：面积 + 斜率标注（左下角）──
        _ax_c1c = v9_data.get('smooth_cols')
        _ax_c1r = v9_data.get('all_rows')
        _ax_analysis_text = ''
        if (_ax_ep_curves and _ax_red_pts
                and _ax_c1c is not None and _ax_c1r is not None
                and _ax_c2r is not None):
            # 构建皆质线1 行→列 查表
            _ax_c1_lut = {int(r): float(c)
                          for r, c in zip(_ax_c1r, _ax_c1c)}
            # 构建皆质线2 行→列 查表（已有 _ax_c2_lut）
            # 按 row_center 排序终板线（升序）
            _ax_sorted_curves = sorted(
                zip(_ax_cv15, _ax_ep_curves),
                key=lambda x: x[0].get('row_center', x[1][1].mean() if len(x[1][1]) else 0)
            )
            # 找最下方完整椎体：
            #   1. 上方终板必须是 superior（上终板），下方必须是 inferior（下终板）
            #   2. 两条终板线各自在 _ax_ep_intersects 中都有 c1 交点 + front 交点（四角点齐全）
            # 预建交叉点索引：{epi: {'c1': (r,c), 'front': (r,c)}}
            _epi_ix_map = {}
            for _ixp in _ax_ep_intersects:
                _ixr, _ixc, _ixside, _ixepi = float(_ixp[0]), float(_ixp[1]), _ixp[2], _ixp[3]
                if _ixepi not in _epi_ix_map:
                    _epi_ix_map[_ixepi] = {}
                _epi_ix_map[_ixepi][_ixside] = (_ixr, _ixc)

            # V14: 构建完整椎体链路（从下往上找所有 superior+inferior 配对）
            _vertebrae_chain = []  # [{'name': 'L5', 'top':..., 'bot':..., 'top_ix':..., 'bot_ix':...}, ...]
            _used_epi = set()
            for _pi in range(len(_ax_sorted_curves) - 1, 0, -1):
                _ep_top_meta, (_etype_top, _r_top, _c_top) = _ax_sorted_curves[_pi - 1]
                _ep_bot_meta, (_etype_bot, _r_bot, _c_bot) = _ax_sorted_curves[_pi]
                # 条件1：正确配对 superior（上终板）+ inferior（下终板）
                if _ep_top_meta.get('ep_type') != 'superior':
                    continue
                if _ep_bot_meta.get('ep_type') != 'inferior':
                    continue
                _row_top = float(np.mean(_r_top))
                _row_bot = float(np.mean(_r_bot))
                if _row_bot <= _row_top:
                    continue
                if (_row_bot - _row_top) < 3:
                    continue
                # 条件2：两条终板线各自都有 c1 + front 四个角点
                _top_epi = None
                _bot_epi = None
                for _ci, (_epm, _epc) in enumerate(zip(_ax_cv15, _ax_ep_curves)):
                    if _epm is _ep_top_meta:
                        _top_epi = _ci
                    if _epm is _ep_bot_meta:
                        _bot_epi = _ci
                if _top_epi is None or _bot_epi is None:
                    continue
                if _top_epi in _used_epi or _bot_epi in _used_epi:
                    continue
                _top_ix = _epi_ix_map.get(_top_epi, {})
                _bot_ix = _epi_ix_map.get(_bot_epi, {})
                if 'c1' not in _top_ix or 'front' not in _top_ix:
                    continue
                if 'c1' not in _bot_ix or 'front' not in _bot_ix:
                    continue
                # ── 裁剪四条曲线到该椎体行范围，并用交叉点截断终板线（供掩模生成用）──
                _vb_r0, _vb_r1 = int(_row_top), int(_row_bot)
                
                # 上终板线：用交叉点截断（c1侧col大，front侧col小）
                _top_c1_col = float(_top_ix['c1'][1])
                _top_front_col = float(_top_ix['front'][1])
                _vb_top_pts = {'rows': [], 'cols': []}
                for _tr, _tc in zip(_r_top, _c_top):
                    if (_top_front_col - 1e-6) <= float(_tc) <= (_top_c1_col + 1e-6):
                        _vb_top_pts['rows'].append(float(_tr))
                        _vb_top_pts['cols'].append(float(_tc))
                
                # 下终板线：用交叉点截断
                _bot_c1_col = float(_bot_ix['c1'][1])
                _bot_front_col = float(_bot_ix['front'][1])
                _vb_bot_pts = {'rows': [], 'cols': []}
                for _br, _bc in zip(_r_bot, _c_bot):
                    if (_bot_front_col - 1e-6) <= float(_bc) <= (_bot_c1_col + 1e-6):
                        _vb_bot_pts['rows'].append(float(_br))
                        _vb_bot_pts['cols'].append(float(_bc))
                
                # 皮质线1：裁剪到 row_top ~ row_bot
                _vb_c1_pts = {'rows': [], 'cols': []}
                for _c1r_v, _c1c_v in zip(_ax_c1r, _ax_c1c):
                    if _vb_r0 <= int(_c1r_v) <= _vb_r1:
                        _vb_c1_pts['rows'].append(float(_c1r_v))
                        _vb_c1_pts['cols'].append(float(_c1c_v))
                
                # 前缘红线：裁剪到 row_top ~ row_bot（使用平滑插值结果 _axr_ri/_axr_sm）
                _vb_front_pts = {'rows': [], 'cols': []}
                if _axr_ri is not None and _axr_sm is not None:
                    for _fri, _fci in zip(_axr_ri, _axr_sm):
                        if _vb_r0 <= int(_fri) <= _vb_r1:
                            _vb_front_pts['rows'].append(float(_fri))
                            _vb_front_pts['cols'].append(float(_fci))
                
                _vertebrae_chain.append({
                    'top_meta': _ep_top_meta, 'r_top': _r_top, 'c_top': _c_top,
                    'bot_meta': _ep_bot_meta, 'r_bot': _r_bot, 'c_bot': _c_bot,
                    'row_top': int(_row_top), 'row_bot': int(_row_bot),
                    'top_ix': _top_ix, 'bot_ix': _bot_ix,
                    'top_pts': _vb_top_pts, 'bot_pts': _vb_bot_pts,
                    'c1_pts': _vb_c1_pts, 'front_pts': _vb_front_pts,
                })
                _used_epi.add(_top_epi)
                _used_epi.add(_bot_epi)
            # 按 row_top 升序排列（从上往下：L1, L2, ... L5/S1）
            _vertebrae_chain.sort(key=lambda x: x['row_top'])
            # V14: S1/L5 判定 - 基于最下方椎体的 Ant.edge 角度
            _lowest_vertebra = _vertebrae_chain[-1] if _vertebrae_chain else None
            _ax_best_pair = None
            if _lowest_vertebra:
                _pt_top_c1 = _lowest_vertebra['top_ix'].get('c1')
                _pt_top_front = _lowest_vertebra['top_ix'].get('front')
                _pt_bot_c1 = _lowest_vertebra['bot_ix'].get('c1')
                _pt_bot_front = _lowest_vertebra['bot_ix'].get('front')
                if all(p is not None for p in [_pt_top_c1, _pt_top_front, _pt_bot_c1, _pt_bot_front]):
                    _dc_fr = _pt_top_front[1] - _pt_bot_front[1]
                    _dr_fr = _pt_top_front[0] - _pt_bot_front[0]
                    _ang_fr_lowest = float(np.degrees(np.arctan2(
                        abs(_dr_fr), abs(_dc_fr)))) if abs(_dc_fr) > 1e-6 else 90.0
                    _lowest_name = 'S1' if _ang_fr_lowest < 45.0 else 'L5'
                else:
                    _lowest_name = 'L5'  # 默认L5
                # 为所有椎体分配名称（从下往上：
                #   若最下面是 L5：L5→L4→L3→L2→L1→T12→T11→...
                #   若最下面是 S1：S1→L5→L4→L3→L2→L1→T12→T11→...）
                if _lowest_name == 'L5':
                    _names_from_bottom = ['L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11', 'T10', 'T9', 'T8']
                else:
                    _names_from_bottom = ['S1', 'L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11', 'T10', 'T9', 'T8']
                for _idx_from_bottom, _vb in enumerate(reversed(_vertebrae_chain)):
                    if _idx_from_bottom < len(_names_from_bottom):
                        _vb['name'] = _names_from_bottom[_idx_from_bottom]
                    else:
                        _vb['name'] = f'T{12-(_idx_from_bottom-5)}' if _idx_from_bottom >= 5 else f'V{_idx_from_bottom+1}'
                # 保持与原代码兼容：_ax_best_pair 仍指向最下方椎体
                _ax_best_pair = (_lowest_vertebra['top_meta'],
                                 _lowest_vertebra['r_top'], _lowest_vertebra['c_top'],
                                 _lowest_vertebra['bot_meta'],
                                 _lowest_vertebra['r_bot'], _lowest_vertebra['c_bot'],
                                 _lowest_vertebra['row_top'], _lowest_vertebra['row_bot'],
                                 _lowest_vertebra['top_ix'], _lowest_vertebra['bot_ix'],
                                 _lowest_vertebra['name'], _vertebrae_chain)

            # V14: 为链路上每个椎体都做完整的几何分析 + 数据标注
            _vertebra_analyses = []  # [{'name': 'L5', 'area_mm2': .., 'angles': {...}, 'text': '..'}, ...]
            _info_x = 5  # 左侧数据列固定 X 坐标（像素）
            for _vb in _vertebrae_chain:
                _vb_rng_rows = range(_vb['row_top'], _vb['row_bot'] + 1)
                # ① 面积
                _vb_area_px2 = 0.0
                for _var in _vb_rng_rows:
                    _vfc = _ax_red_pts[_var][1] if _var in _ax_red_pts else None
                    _vc2c = _ax_c2_lut.get(_var)
                    if _vfc is not None and _vc2c is not None:
                        _vb_area_px2 += abs(_vfc - _vc2c)
                _vb_area_mm2 = _vb_area_px2 * (pixel_spacing ** 2)
                # ② 四角点
                _vb_pt_top_c1    = _vb['top_ix'].get('c1')
                _vb_pt_top_front = _vb['top_ix'].get('front')
                _vb_pt_bot_c1    = _vb['bot_ix'].get('c1')
                _vb_pt_bot_front = _vb['bot_ix'].get('front')
                _vb_ang_top = _vb_ang_bot = _vb_ang_c1 = _vb_ang_fr = float('nan')
                if all(p is not None for p in [_vb_pt_top_c1, _vb_pt_top_front, _vb_pt_bot_c1, _vb_pt_bot_front]):
                    # 画青色虚线四边形
                    _vb_quad_cols = [_vb_pt_top_c1[1], _vb_pt_top_front[1],
                                     _vb_pt_bot_front[1], _vb_pt_bot_c1[1], _vb_pt_top_c1[1]]
                    _vb_quad_rows = [_vb_pt_top_c1[0], _vb_pt_top_front[0],
                                     _vb_pt_bot_front[0], _vb_pt_bot_c1[0], _vb_pt_top_c1[0]]
                    ax.plot(_vb_quad_cols, _vb_quad_rows, '--',
                            color='cyan', linewidth=1.2, alpha=0.75, zorder=15)
                    for _qc, _qr in zip(_vb_quad_cols[:-1], _vb_quad_rows[:-1]):
                        ax.plot(_qc, _qr, 'o', color='cyan', markersize=3.5,
                                alpha=0.9, zorder=16)
                    _BL = 20
                    _bbox_ang = dict(boxstyle='round,pad=0.1', fc='black', alpha=0.45, lw=0)
                    # Superior EP
                    _dc_top = _vb_pt_top_c1[1] - _vb_pt_top_front[1]
                    _dr_top = _vb_pt_top_c1[0] - _vb_pt_top_front[0]
                    _vb_ang_top = float(np.degrees(np.arctan2(abs(_dr_top), abs(_dc_top)))) if abs(_dc_top) > 1e-6 else 90.0
                    ax.plot([_vb_pt_top_front[1], _vb_pt_top_front[1] + _BL],
                            [_vb_pt_top_front[0], _vb_pt_top_front[0]], '--',
                            color='yellow', lw=0.8, alpha=0.7, zorder=15)
                    ax.text(_vb_pt_top_front[1] + _BL + 2, _vb_pt_top_front[0] - 2,
                            f"{_vb_ang_top:.1f}\u00b0",
                            fontsize=6.5, color='lawngreen', ha='left', va='bottom',
                            zorder=16, bbox=_bbox_ang)
                    # Inferior EP
                    _dc_bot = _vb_pt_bot_c1[1] - _vb_pt_bot_front[1]
                    _dr_bot = _vb_pt_bot_c1[0] - _vb_pt_bot_front[0]
                    _vb_ang_bot = float(np.degrees(np.arctan2(abs(_dr_bot), abs(_dc_bot)))) if abs(_dc_bot) > 1e-6 else 90.0
                    ax.plot([_vb_pt_bot_front[1], _vb_pt_bot_front[1] + _BL],
                            [_vb_pt_bot_front[0], _vb_pt_bot_front[0]], '--',
                            color='yellow', lw=0.8, alpha=0.7, zorder=15)
                    ax.text(_vb_pt_bot_front[1] + _BL + 2, _vb_pt_bot_front[0] + 2,
                            f"{_vb_ang_bot:.1f}\u00b0",
                            fontsize=6.5, color='red', ha='left', va='top',
                            zorder=16, bbox=_bbox_ang)
                    # Cortical1
                    _dc_c1 = _vb_pt_top_c1[1] - _vb_pt_bot_c1[1]
                    _dr_c1 = _vb_pt_top_c1[0] - _vb_pt_bot_c1[0]
                    _vb_ang_c1 = float(np.degrees(np.arctan2(abs(_dr_c1), abs(_dc_c1)))) if abs(_dc_c1) > 1e-6 else 90.0
                    ax.plot([_vb_pt_bot_c1[1], _vb_pt_bot_c1[1] - _BL],
                            [_vb_pt_bot_c1[0], _vb_pt_bot_c1[0]], '--',
                            color='yellow', lw=0.8, alpha=0.7, zorder=15)
                    ax.text(_vb_pt_bot_c1[1] - _BL - 2, _vb_pt_bot_c1[0] + 2,
                            f"{_vb_ang_c1:.1f}\u00b0",
                            fontsize=6.5, color='white', ha='right', va='top',
                            zorder=16, bbox=_bbox_ang)
                    # Ant. edge
                    _dc_fr = _vb_pt_top_front[1] - _vb_pt_bot_front[1]
                    _dr_fr = _vb_pt_top_front[0] - _vb_pt_bot_front[0]
                    _vb_ang_fr = float(np.degrees(np.arctan2(abs(_dr_fr), abs(_dc_fr)))) if abs(_dc_fr) > 1e-6 else 90.0
                    ax.plot([_vb_pt_bot_front[1], _vb_pt_bot_front[1] - _BL],
                            [_vb_pt_bot_front[0], _vb_pt_bot_front[0]], '--',
                            color='yellow', lw=0.8, alpha=0.7, zorder=15)
                    ax.text(_vb_pt_bot_front[1] - _BL - 2, _vb_pt_bot_front[0] + 2,
                            f"{_vb_ang_fr:.1f}\u00b0",
                            fontsize=6.5, color='tomato', ha='right', va='top',
                            zorder=16, bbox=_bbox_ang)
                # 数据标注文本（左侧固定列，Y 与该椎体几何中心平行）
                _vb_text = f"{_vb.get('name', '??')}: {_vb_area_mm2:.1f}mm\u00b2 | Sup:{_vb_ang_top:.1f}\u00b0 Inf:{_vb_ang_bot:.1f}\u00b0 C1:{_vb_ang_c1:.1f}\u00b0 Ant:{_vb_ang_fr:.1f}\u00b0"
                if all(p is not None for p in [_vb_pt_top_c1, _vb_pt_top_front, _vb_pt_bot_c1, _vb_pt_bot_front]):
                    # 计算几何中心
                    _vb_center_row = (_vb_pt_top_c1[0] + _vb_pt_top_front[0] + _vb_pt_bot_front[0] + _vb_pt_bot_c1[0]) / 4.0
                    _vb_center_col = (_vb_pt_top_c1[1] + _vb_pt_top_front[1] + _vb_pt_bot_front[1] + _vb_pt_bot_c1[1]) / 4.0
                    ax.text(_info_x, _vb_center_row,
                            _vb_text,
                            fontsize=7.0, color='white', ha='left', va='center',
                            zorder=18,
                            bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.5, lw=0))
                _vertebra_analyses.append({
                    'name': _vb.get('name', '??'),
                    'area_mm2': _vb_area_mm2,
                    'angles': {'top': _vb_ang_top, 'bot': _vb_ang_bot, 'c1': _vb_ang_c1, 'fr': _vb_ang_fr}
                })
                # 同步写入 _vertebrae_chain（供 export 模块使用）
                _vb['area_mm2'] = _vb_area_mm2
                _vb['angles']   = {'top': _vb_ang_top, 'bot': _vb_ang_bot,
                                   'c1': _vb_ang_c1, 'fr': _vb_ang_fr}

        _export_chain = _vertebrae_chain  # 保存引用，供函数末尾返回
        _vertebrae_chain = None  # 初始化，避免 UnboundLocalError
        _vertebrae_chain_vis = []  # 可视化用的链路

        # V14: 左下角简短汇总信息
        if _vertebrae_chain_vis:
            _lowest_vb = _vertebrae_chain_vis[-1]
            _vname = _lowest_vb.get('name', '??')
            _summary = f"Detected vertebrae: {' / '.join([vb.get('name','?') for vb in _vertebrae_chain_vis])}\nLowest: {_vname} (row {_lowest_vb['row_top']}→{_lowest_vb['row_bot']})"
            ax.text(0.02, 0.02, _summary,
                    transform=ax.transAxes, fontsize=7.0,
                    color='yellow', va='bottom', ha='left', zorder=16,
                    bbox=dict(boxstyle='round,pad=0.25', fc='black', alpha=0.55, lw=0))
    
    # ===== 右图：压水图 =====
    if has_water:
        ax2 = axes[1]
        ax2.imshow(f_img_2d, cmap='gray')
        ax2.set_title('压水图 (F) - V14_2 终板检测', fontsize=11)

        _f_data_ok = v9_data.get('consensus_endplates_v15') if v9_data else None
        if not _f_data_ok:
            ax2.text(0.5, 0.5, '未找到压水图或终板检测失败',
                     transform=ax2.transAxes, ha='center', va='center',
                     fontsize=13, color='yellow')
        # ── V15 可视化：皮质线2、切线扫描线、终板分割直线、前缘角 ──
        if v9_data:
            # 皮质线1：白色实线
            _c1_cols = v9_data.get('smooth_cols')
            _c1_rows = v9_data.get('all_rows')
            if _c1_cols is not None and _c1_rows is not None:
                ax2.plot(_c1_cols, _c1_rows, color='white', linewidth=1.2,
                         linestyle='-', alpha=0.75, zorder=6, label='cortical1')
            # 皮质线2：紫色虚线
            _c2_cols = v9_data.get('c2_cols')
            _c2_rows = v9_data.get('c2_rows')
            if _c2_cols is not None and _c2_rows is not None:
                ax2.plot(_c2_cols, _c2_rows, color='magenta', linewidth=1.2,
                         linestyle='--', alpha=0.70, zorder=6, label='cortical2')

            # V15 切线方向扫描线：深蓝色细线
            _sl_v15 = v9_data.get('scan_lines_v15', [])
            for off_mm, rows_a, cols_a, nx_a, ny_a in _sl_v15:
                ax2.plot(cols_a, rows_a, color='steelblue', linewidth=0.5,
                         alpha=0.55, zorder=5)

            # V13 原始候选终板点（全量，缩小显示）：下终板 inferior=番茄红▼，上终板 superior=绿⯅
            _raw_cands_v15 = v9_data.get('raw_candidates_v15', [])
            for r, c, t, d, _li in _raw_cands_v15:
                if t == 'inferior':
                    ax2.plot(c, r, 'v', color='tomato', markersize=2,
                             markeredgewidth=0, alpha=0.35, zorder=7)
                else:
                    ax2.plot(c, r, '^', color='lime', markersize=2,
                             markeredgewidth=0, alpha=0.35, zorder=7)
            # V13 前缘线可视化（对标V12_4，以皮质线2为基准）
            arc_refined_v13   = v9_data.get('arc_refined_v13',   []) if v9_data else []
            arc_filtered_v13  = v9_data.get('arc_filtered_v13',  []) if v9_data else []
            arc_best_range_v13= v9_data.get('arc_best_range_v13',None) if v9_data else None
            arc_r2_raw_v13    = v9_data.get('arc_refined2_raw_v13', []) if v9_data else []

            if arc_refined_v13:
                # 分段画黄色/红色连线（kept_low→红，其余→黄）
                seg_rows, seg_cols, seg_flag = [], [], None
                def _flush_seg():
                    if len(seg_rows) >= 2:
                        kw = dict(color='yellow', linewidth=1.4, alpha=0.92, zorder=8)
                        ax2.plot(seg_cols, seg_rows, '-', **kw)
                for p in arc_refined_v13:
                    f = p[4] if len(p) > 4 else 'kept'
                    if seg_flag is None:
                        seg_flag = f
                    if f != seg_flag:
                        _flush_seg()
                        seg_rows, seg_cols = [], []
                        seg_flag = f
                    seg_rows.append(p[0])
                    seg_cols.append(p[1])
                _flush_seg()
                r_rows_v13 = [p[0] for p in arc_refined_v13]

                # 单轮精修全量点（arc_refined1_v13，含 flag）：按flag分色标注
                _refined1_pts = v9_data.get('arc_refined1_v13', []) if v9_data else []
                if _refined1_pts:
                    _pts_by_flag = {'refined': [], 'kept_low': [], 'kept': []}
                    for p in _refined1_pts:
                        _f = p[4] if len(p) > 4 else 'kept'
                        _pts_by_flag.get(_f, _pts_by_flag['kept']).append(p)
                    _flag_style = {
                        'refined':  dict(c='orange',   s=8,  linewidths=0, alpha=0.90, zorder=9),
                        'kept_low': dict(c='red',      s=8,  linewidths=0, alpha=0.90, zorder=9),
                        'kept':     dict(c='#aaaaaa',  s=5,  linewidths=0, alpha=0.60, zorder=8),
                    }
                    for _fl, _style in _flag_style.items():
                        _grp = _pts_by_flag[_fl]
                        if _grp:
                            ax2.scatter([p[1] for p in _grp], [p[0] for p in _grp], **_style)

                # 新方案：腹侧下降沿前缘线（青色连线，替代原精修点平滑青线）
                _arc_descent = v9_data.get('arc_descent_v13', []) if v9_data else []
                _descent_confirmed = sorted(
                    [(p[0], p[1]) for p in _arc_descent if p[2] in ('confirmed', 'confirmed_rise')],
                    key=lambda x: x[0])
                if len(_descent_confirmed) >= 2:
                    _dr = np.array([p[0] for p in _descent_confirmed], dtype=np.float32)
                    _dc = np.array([p[1] for p in _descent_confirmed], dtype=np.float32)
                    # MAD过滤 + 插值 + 5mm移动均值平滑
                    _mwd = min(11, len(_dc))
                    if _mwd % 2 == 0: _mwd -= 1
                    _mwd = max(1, _mwd)
                    _vmd = np.ones(len(_dc), dtype=bool)
                    _hmd = _mwd // 2
                    for _id in range(len(_dc)):
                        _sd = max(0, _id - _hmd); _ed = min(len(_dc), _id + _hmd + 1)
                        _wd = _dc[_sd:_ed]
                        _med_d = np.median(_wd); _mad_d = np.median(np.abs(_wd - _med_d))
                        if _mad_d > 0 and np.abs(_dc[_id] - _med_d) / (_mad_d * 1.4826) > 2.0:
                            _vmd[_id] = False
                    _crd = _dc[_vmd]; _rrd = _dr[_vmd]
                    if len(_rrd) < 4: _crd, _rrd = _dc, _dr
                    _arid = np.arange(int(_rrd[0]), int(_rrd[-1]) + 1)
                    _icd  = np.interp(_arid, _rrd, _crd)
                    _kd   = max(3, int(round(5.0 / pixel_spacing)))
                    if _kd % 2 == 0: _kd += 1
                    _csmd = np.convolve(np.pad(_icd, _kd // 2, mode='edge'),
                                        np.ones(_kd) / _kd, mode='valid')
                    ax2.plot(_csmd, _arid, '-', color='cyan', linewidth=1.2,
                             alpha=0.85, zorder=10)

                # 起始扫描线（offset=25mm）可视化：仅显示椎体段内的行
                _arc_descent_all = v9_data.get('arc_descent_v13', []) if v9_data else []
                if _arc_descent_all:
                    _descent_main = [(p[0], p[1]) for p in _arc_descent_all
                                     if p[2] == 'confirmed' and (len(p) < 4 or p[3] == 'main')]
                    _descent_supp = [(p[0], p[1]) for p in _arc_descent_all
                                     if p[2] == 'confirmed' and len(p) >= 4 and p[3] != 'main']
                    _c2c_d = v9_data.get('c2_cols')
                    _c2r_d = v9_data.get('c2_rows')
                    if _c2c_d is not None and _c2r_d is not None:
                        _off25_px = int(round(20.0 / pixel_spacing))
                        # 收集所有在椎体段内的行（排除not_found中 col_start占位符）
                        _seg_rows_vis = sorted(set(p[0] for p in _arc_descent_all))
                        _scan_cols_vis = []
                        _scan_rows_vis = []
                        _rows_arr_d = np.array(_c2r_d, dtype=np.float32)
                        _cols_arr_d = np.array(_c2c_d, dtype=np.float32)
                        for _sr in _seg_rows_vis:
                            _idx_d = int(np.argmin(np.abs(_rows_arr_d - _sr)))
                            _base_d = float(_cols_arr_d[_idx_d])
                            _scan_cols_vis.append(_base_d - _off25_px)
                            _scan_rows_vis.append(_sr)
                        if _scan_rows_vis:
                            ax2.plot(_scan_cols_vis, _scan_rows_vis,
                                     '-', color='deepskyblue', linewidth=0.7,
                                     alpha=0.45, zorder=6)

                # 下降沿 confirmed 点：主扫描=青色三角，补充扫描=黄色倒三角
                if _arc_descent_all:
                    if _descent_main:
                        ax2.scatter([p[1] for p in _descent_main],
                                    [p[0] for p in _descent_main],
                                    marker='<', s=18, c='cyan',
                                    linewidths=0, alpha=0.75, zorder=11)
                    if _descent_supp:
                        ax2.scatter([p[1] for p in _descent_supp],
                                    [p[0] for p in _descent_supp],
                                    marker='v', s=20, c='yellow',
                                    linewidths=0, alpha=0.85, zorder=11)
                # 上升沿 confirmed_rise 点：向左三角标（红色区分）
                _descent_rise2 = [(p[0], p[1]) for p in _arc_descent_all if p[2] == 'confirmed_rise']
                if _descent_rise2:
                    ax2.scatter([p[1] for p in _descent_rise2],
                                [p[0] for p in _descent_rise2],
                                marker='<', s=18, c='red',
                                linewidths=0, alpha=0.75, zorder=11)

                # 综合红色线：密集窗口过滤（双模态合并点集）生成
                _r2_combined   = v9_data.get('arc_combined_v13', []) if v9_data else []
                _r2_best_range = arc_best_range_v13
                _r2_expand     = v9_data.get('dense_expand_v13', 3.0) if v9_data else 3.0
                _r2_c2r        = v9_data.get('c2_rows') if v9_data else None
                _r2_red_pts    = {}
                if _r2_combined and _r2_best_range is not None:
                    _r2_best_lo = _r2_best_range[0]
                    _r2_best_hi = _r2_best_range[1]
                    _r2_win_mm  = _r2_best_hi - _r2_best_lo
                    if _r2_c2r is not None and len(_r2_c2r) > 0:
                        _r2_rmin = float(min(_r2_c2r)); _r2_rmax = float(max(_r2_c2r))
                    else:
                        _r2_rmin = float(min(p[0] for p in _r2_combined))
                        _r2_rmax = float(max(p[0] for p in _r2_combined))
                    _r2_rspan = max(_r2_rmax - _r2_rmin, 1.0)
                    # 1. 密集窗口内点（arc_combined_v13，已含上升沿+下降沿 confirmed）
                    for _rp in _r2_combined:
                        _t2  = (float(_rp[0]) - _r2_rmin) / _r2_rspan
                        _hi2 = _r2_best_lo + _r2_win_mm * (1.0 + (_r2_expand - 1.0) * _t2)
                        _of2 = (float(_rp[3]) - float(_rp[1])) * pixel_spacing
                        if _r2_best_lo - 1e-6 <= _of2 <= _hi2 + 1e-6:
                            _r2_red_pts[int(_rp[0])] = (float(_rp[0]), float(_rp[1]))
                    # 2. 右侧补充点（offset < best_lo，仅上升沿点，排除 kept_low）
                    if _r2_red_pts and _r2_c2r is not None:
                        _r2_top    = min(_r2_red_pts.keys())
                        _r2_c2half = (float(max(_r2_c2r)) - float(min(_r2_c2r))) / 2.0
                        _r2_rlimit = _r2_top + int(round(_r2_c2half))
                        _r2_ref1_only = v9_data.get('arc_refined1_v13', []) if v9_data else []
                        for _rp in _r2_ref1_only:
                            if (len(_rp) > 4 and _rp[4] == 'kept_low'):
                                continue
                            _rp_off2 = (float(_rp[3]) - float(_rp[1])) * pixel_spacing
                            _rp_row2 = int(_rp[0])
                            if (_rp_off2 < _r2_best_lo
                                    and _rp_off2 >= _r2_best_lo - 5.0
                                    and _rp_row2 not in _r2_red_pts
                                    and _rp_row2 <= _r2_rlimit):
                                _r2_red_pts[_rp_row2] = (float(_rp[0]), float(_rp[1]))
                    # 3. 尾部谷底补充：从 arc_min_pts_v13 查表，补充到密集空间底行
                    if _r2_red_pts and _r2_c2r is not None and len(_r2_c2r) > 0:
                        _r2_tgt_r1  = int(_r2_c2r[-1])
                        _r2_cur_r1  = max(_r2_red_pts.keys())
                        if _r2_cur_r1 < _r2_tgt_r1:
                            _r2_min_pts = v9_data.get('arc_min_pts_v13', []) if v9_data else []
                            _r2_min_lut = {int(mp[0]): mp for mp in _r2_min_pts}
                            _r2_added = 0
                            for _er in range(_r2_cur_r1 + 1, _r2_tgt_r1 + 1):
                                if _er not in _r2_red_pts and _er in _r2_min_lut:
                                    _mp = _r2_min_lut[_er]
                                    _r2_red_pts[_er] = (float(_mp[0]), float(_mp[1]))
                                    _r2_added += 1
                            if _r2_added > 0:
                                print(f"   [红线尾部补充] 补充 {_r2_added} 行 "
                                      f"({_r2_cur_r1+1}→{_r2_tgt_r1})，来源=arc_min_pts_v13")
                # 平滑生成红色线
                if len(_r2_red_pts) >= 2:
                    _r2_sorted = sorted(_r2_red_pts.values(), key=lambda x: x[0])
                    _r2r = np.array([p[0] for p in _r2_sorted], dtype=np.float32)
                    _r2c = np.array([p[1] for p in _r2_sorted], dtype=np.float32)
                    _r2mw = min(11, len(_r2c))
                    if _r2mw % 2 == 0: _r2mw -= 1
                    _r2mw = max(1, _r2mw)
                    _r2v  = np.ones(len(_r2c), dtype=bool)
                    _r2hf = _r2mw // 2
                    for _r2i in range(len(_r2c)):
                        _s2 = max(0, _r2i - _r2hf); _e2 = min(len(_r2c), _r2i + _r2hf + 1)
                        _w2 = _r2c[_s2:_e2]
                        _m2 = np.median(_w2); _d2 = np.median(np.abs(_w2 - _m2))
                        if _d2 > 0 and np.abs(_r2c[_r2i] - _m2) / (_d2 * 1.4826) > 2.0:
                            _r2v[_r2i] = False
                    _r2cf = _r2c[_r2v]; _r2rf = _r2r[_r2v]
                    if len(_r2rf) < 4: _r2cf, _r2rf = _r2c, _r2r
                    _r2ri = np.arange(int(_r2rf[0]), int(_r2rf[-1]) + 1)
                    _r2ci = np.interp(_r2ri, _r2rf, _r2cf)
                    _r2k  = max(3, int(round(8.0 / pixel_spacing)))
                    if _r2k % 2 == 0: _r2k += 1
                    _r2sm = np.convolve(np.pad(_r2ci, _r2k // 2, mode='edge'),
                                        np.ones(_r2k) / _r2k, mode='valid')
                    ax2.plot(_r2sm, _r2ri, '-', color='red', linewidth=1.2,
                             alpha=0.85, zorder=12)

                # 密集窗口黄色虚线框（对标V12_4黄色实线框）
                if arc_best_range_v13 is not None:
                    _c2c = v9_data.get('c2_cols')
                    _c2r = v9_data.get('c2_rows')
                    if _c2c is not None and _c2r is not None and len(r_rows_v13) > 0:
                        _rows_arr = np.array(_c2r, dtype=np.float32)
                        _cols_arr = np.array(_c2c, dtype=np.float32)
                        hi_px       = arc_best_range_v13[1] / pixel_spacing
                        lo_px_base  = arc_best_range_v13[0] / pixel_spacing
                        _dense_expand = v9_data.get('dense_expand_v13', 3.0) if v9_data else 3.0
                        rr_min_d  = float(_rows_arr.min())
                        rr_max_d  = float(_rows_arr.max())
                        rr_span_d = max(rr_max_d - rr_min_d, 1.0)
                        row_range = range(int(rr_min_d), int(rr_max_d) + 1)
                        left_cols_d, right_cols_d, valid_rows_d = [], [], []
                        for rr in row_range:
                            idx  = int(np.argmin(np.abs(_rows_arr - rr)))
                            base = float(_cols_arr[idx])
                            t    = (float(rr) - rr_min_d) / rr_span_d
                            window_px   = hi_px - lo_px_base
                            hi_px_dyn   = lo_px_base + window_px * (1.0 + (_dense_expand - 1.0) * t)
                            left_cols_d.append(base - hi_px_dyn)
                            right_cols_d.append(base - lo_px_base)
                            valid_rows_d.append(rr)
                        ax2.plot(left_cols_d,  valid_rows_d, '--', color='yellow',
                                 linewidth=0.9, alpha=0.55, zorder=7)
                        ax2.plot(right_cols_d, valid_rows_d, '--', color='yellow',
                                 linewidth=0.9, alpha=0.55, zorder=7)
                        ax2.plot([left_cols_d[0],  right_cols_d[0]],
                                 [valid_rows_d[0],  valid_rows_d[0]],
                                 '--', color='yellow', linewidth=0.9, alpha=0.55, zorder=7)
                        ax2.plot([left_cols_d[-1], right_cols_d[-1]],
                                 [valid_rows_d[-1], valid_rows_d[-1]],
                                 '--', color='yellow', linewidth=0.9, alpha=0.55, zorder=7)

                        # ── 第一次密集窗口（纯上升沿）橙色虚线框 ──
                        _best_range1 = v9_data.get('arc_best_range1_v13', None)
                        if _best_range1 is not None and _best_range1 != arc_best_range_v13:
                            hi1_px      = _best_range1[1] / pixel_spacing
                            lo1_px_base = _best_range1[0] / pixel_spacing
                            left1_cols_d, right1_cols_d, valid1_rows_d = [], [], []
                            for rr in row_range:
                                idx1  = int(np.argmin(np.abs(_rows_arr - rr)))
                                base1 = float(_cols_arr[idx1])
                                t1    = (float(rr) - rr_min_d) / rr_span_d
                                win1_px   = hi1_px - lo1_px_base
                                hi1_dyn   = lo1_px_base + win1_px * (1.0 + (_dense_expand - 1.0) * t1)
                                left1_cols_d.append(base1 - hi1_dyn)
                                right1_cols_d.append(base1 - lo1_px_base)
                                valid1_rows_d.append(rr)
                            ax2.plot(left1_cols_d,  valid1_rows_d, '--', color='orange',
                                     linewidth=0.9, alpha=0.55, zorder=7)
                            ax2.plot(right1_cols_d, valid1_rows_d, '--', color='orange',
                                     linewidth=0.9, alpha=0.55, zorder=7)
                            ax2.plot([left1_cols_d[0],  right1_cols_d[0]],
                                     [valid1_rows_d[0],  valid1_rows_d[0]],
                                     '--', color='orange', linewidth=0.9, alpha=0.55, zorder=7)
                            ax2.plot([left1_cols_d[-1], right1_cols_d[-1]],
                                     [valid1_rows_d[-1], valid1_rows_d[-1]],
                                     '--', color='orange', linewidth=0.9, alpha=0.55, zorder=7)

                        # ── V14_1: 椎间盘过滤区域可视化（红色虚线框）──
                        disc_row_ranges_vis = v9_data.get('disc_row_ranges', [])
                        disc_baselines_vis = v9_data.get('disc_baselines', {})
                        
                        if disc_row_ranges_vis and disc_baselines_vis:
                            for disc_idx, (row_min, row_max) in enumerate(disc_row_ranges_vis):
                                if disc_idx in disc_baselines_vis:
                                    baseline_offset = disc_baselines_vis[disc_idx]
                                    baseline_px = baseline_offset / pixel_spacing
                                    
                                    # 计算该椎间盘区域的左右边界
                                    filter_left_cols = []
                                    filter_right_cols = []
                                    filter_rows = []
                                    
                                    for rr in range(int(row_min), int(row_max) + 1):
                                        idx = int(np.argmin(np.abs(_rows_arr - rr)))
                                        base_col = float(_cols_arr[idx])
                                        
                                        # 左边界：基准线位置（offset = baseline_offset）
                                        filter_left_cols.append(base_col - baseline_px)
                                        # 右边界：皮质线 2
                                        filter_right_cols.append(base_col)
                                        filter_rows.append(rr)
                                    
                                    if filter_left_cols:
                                        # 画左边界（基准线）
                                        ax2.plot(filter_left_cols, filter_rows, '--', 
                                                 color='red', linewidth=1.0, alpha=0.6, zorder=8,
                                                 label=f'椎间盘{disc_idx+1}基准线' if disc_idx == 0 else "")
                                        # 画右边界（皮质线 2）
                                        ax2.plot(filter_right_cols, filter_rows, '--', 
                                                 color='red', linewidth=1.0, alpha=0.6, zorder=8)
                                        # 画顶部和底部横线
                                        ax2.plot([filter_left_cols[0], filter_right_cols[0]],
                                                 [filter_rows[0], filter_rows[0]],
                                                 '--', color='red', linewidth=1.0, alpha=0.6, zorder=8)
                                        ax2.plot([filter_left_cols[-1], filter_right_cols[-1]],
                                                 [filter_rows[-1], filter_rows[-1]],
                                                 '--', color='red', linewidth=1.0, alpha=0.6, zorder=8)
                                    
                                    print(f"   [V14_1 可视化] 椎间盘{disc_idx+1}过滤区域："
                                          f"row={row_min:.0f}~{row_max:.0f}, offset≥{baseline_offset:.1f}mm")
                                                
                        # offset 搜索空间范围色点线框
                        _left_off_base  = v9_data.get('arc_left_off_mm_v13',  20.0)
                        _right_off_mm_v = v9_data.get('arc_right_off_mm_v13', 40.0)
                        _off_expand_v   = v9_data.get('arc_off_expand_v13',    1.5)
                        _right_px_fixed = _left_off_base / pixel_spacing
                        rr_span_off = rr_span_d
                        oc_left, oc_right, ov_rows = [], [], []
                        for rr in row_range:
                            idx  = int(np.argmin(np.abs(_rows_arr - rr)))
                            base = float(_cols_arr[idx])
                            t    = (float(rr) - rr_min_d) / rr_span_off
                            right_off_dyn = _right_off_mm_v * (1.0 + (_off_expand_v - 1.0) * t)
                            _right_px_dyn = right_off_dyn / pixel_spacing
                            oc_left.append(base - _right_px_dyn)
                            oc_right.append(base - _right_px_fixed)
                            ov_rows.append(rr)
                        ax2.plot(oc_left,  ov_rows, ':', color='cyan',
                                 linewidth=0.8, alpha=0.45, zorder=7)
                        ax2.plot(oc_right, ov_rows, ':', color='cyan',
                                 linewidth=0.8, alpha=0.45, zorder=7)
                        ax2.plot([oc_left[0],  oc_right[0]],  [ov_rows[0],  ov_rows[0]],
                                 ':', color='cyan', linewidth=0.8, alpha=0.45, zorder=7)
                        ax2.plot([oc_left[-1], oc_right[-1]], [ov_rows[-1], ov_rows[-1]],
                                 ':', color='cyan', linewidth=0.8, alpha=0.45, zorder=7)

                    n_ref_v13 = len(arc_refined_v13)
                    ax2.text(0.02, 0.10,
                             f'V13前缘ROI: {arc_best_range_v13[0]:.1f}~{arc_best_range_v13[1]:.1f}mm '
                             f'|共:{n_ref_v13}点',
                             transform=ax2.transAxes, fontsize=8,
                             ha='left', va='bottom', color='yellow',
                             bbox=dict(boxstyle='round', facecolor='black', alpha=0.55))

            # ── V15 终板线平滑连接 + 向左右延伸 + 四角点标注 ──
            _c1c = v9_data.get('smooth_cols')
            _c1r = v9_data.get('all_rows')
            _fin = v9_data.get('arc_refined_v13', [])
            _cv15          = v9_data.get('consensus_endplates_v15', [])
            _cv15_retry_keys = v9_data.get('retry_added_keys_v15', set())  # {(row, line_idx)}
            # 构建 off_mm → line_idx 映射
            _cv15_off2li = {float(_sl[0]): _sli
                            for _sli, _sl in enumerate(v9_data.get('scan_lines_v15', []))}
            if _cv15:
                for _ep in _cv15:
                    _ep_pts = _ep.get('points', [])  # [(row, col, depth, off_mm), ...]
                    if len(_ep_pts) < 2:
                        continue
                    _ep_type = _ep.get('ep_type', 'superior')
                    # 初始点连线色/回退点连线色
                    _ln_col       = 'red'         if _ep_type == 'inferior' else 'lawngreen'
                    _xc_init      = 'deepskyblue' if _ep_type == 'inferior' else 'gold'
                    _xc_retry     = 'lime'        if _ep_type == 'inferior' else 'hotpink'
            
                    # ── 按 off_mm（扫描线方向，小→大即右→左）排序 ──
                    _ep_sorted = sorted(_ep_pts, key=lambda p: p[3])  # p[3]=off_mm
                    _epr = np.array([float(p[0]) for p in _ep_sorted])
                    _epc = np.array([float(p[1]) for p in _ep_sorted])
                    _epo = np.array([float(p[3]) for p in _ep_sorted])  # off_mm作为参数轴

                    # 以 off_mm 为参数轴，对 row 和 col 分别做 5mm 移动均値平滑
                    _kep = max(3, int(round(5.0 / pixel_spacing)))
                    if _kep % 2 == 0: _kep += 1
                    _off_uni = np.linspace(_epo[0], _epo[-1],
                                           max(len(_epo), int((_epo[-1]-_epo[0])/pixel_spacing)+1))
                    _r_uni = np.interp(_off_uni, _epo, _epr)
                    _c_uni = np.interp(_off_uni, _epo, _epc)
                    _r_sm  = np.convolve(np.pad(_r_uni, _kep//2, mode='edge'),
                                         np.ones(_kep)/_kep, mode='valid')
                    _c_sm  = np.convolve(np.pad(_c_uni, _kep//2, mode='edge'),
                                         np.ones(_kep)/_kep, mode='valid')
                    ax2.plot(_c_sm, _r_sm, '-', color=_ln_col,
                             linewidth=1.8, alpha=0.90, zorder=11)

                    # ── 各点×标注（回退新增点用不同颜色）──
                    for _pp in _ep_pts:
                        _pp_row = int(round(float(_pp[0])))
                        _pp_off = float(_pp[3])
                        _pp_li  = min(_cv15_off2li, key=lambda o: abs(o - _pp_off), default=None)
                        _pp_li  = _cv15_off2li.get(_pp_li) if _pp_li is not None else None
                        _pp_is_retry = (_pp_row, _pp_li) in _cv15_retry_keys if _pp_li is not None else False
                        _pp_col = _xc_retry if _pp_is_retry else _xc_init
                        ax2.plot(float(_pp[1]), float(_pp[0]), 'x',
                                 color=_pp_col, markersize=3,
                                 markeredgewidth=0.8, alpha=0.85, zorder=12)

        ax2.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [V14_2] 可视化已保存：{os.path.basename(output_path)}")
    return _export_chain


# ============ 主测试函数 ============
