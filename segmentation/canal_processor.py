"""
椎管处理器模块（SpinalCanalProcessor）
椎管分割、边界追踪、终板检测一体化 - V9版本
"""
import numpy as np
from scipy import ndimage, signal
from scipy.ndimage import binary_closing, binary_erosion
from skimage.measure import label as sk_label
from skimage.morphology import binary_dilation as sk_binary_dilation
from skimage import measure
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, disk

from .cortical_line import build_cortical2, extend_cortical2_tail
from .scan_lines_v15 import build_scan_lines_v15, convert_to_arc_coord

class SpinalCanalProcessor:
    """椎管分割、边界追踪、终板检测一体化 - V9版本"""
    
    def __init__(self, pixel_spacing, meta=None):
        self.pixel_spacing = pixel_spacing
        self.width_warning = False
        self.width_value = 0
        self.meta = meta or {}
        self.params = self._build_param_route(pixel_spacing, self.meta)

    def _build_param_route(self, pixel_spacing, meta):
        """
        基于 metadata 的参数路由
        根据像素间距、场强、并行采集等决定各个算法参数
        返回 dict: params
        """
        acq = meta.get('acquisition_params', {})
        par = meta.get('parallel_imaging', {})
        field_t = float(acq.get('magnetic_field_strength', 1.5))
        fat_sup  = acq.get('fat_suppressed', True)
        parallel = par.get('used', False)
        slice_th = float(acq.get('slice_thickness_mm', 4.0))

        # 分辨率等级
        if pixel_spacing <= 0.50:
            res = 'HR'   # 高分辨率
        elif pixel_spacing <= 0.75:
            res = 'STD'  # 标准分辨率
        else:
            res = 'LR'   # 低分辨率

        # 基础参数表
        base = {
            'HR':  dict(tol_mm=2.0, tol_mm_fallback=3.0, min_pts=7, min_pts_fallback=5,
                        depth_thresh={1:0.25, 2:0.15, 3:0.08, 4:0.10},
                        min_gap_mm=5.0),
            'STD': dict(tol_mm=2.5, tol_mm_fallback=4.0, min_pts=7, min_pts_fallback=5,
                        depth_thresh={1:0.22, 2:0.13, 3:0.07, 4:0.09},
                        min_gap_mm=4.5),
            'LR':  dict(tol_mm=3.0, tol_mm_fallback=5.0, min_pts=6, min_pts_fallback=5,
                        depth_thresh={1:0.15, 2:0.10, 3:0.05, 4:0.07},
                        min_gap_mm=4.0),
        }[res].copy()

        # 局部微调：3T場强SNR高，可略微提高阈値
        if field_t >= 2.5:
            for k in base['depth_thresh']:
                base['depth_thresh'][k] = round(base['depth_thresh'][k] * 1.10, 3)

        # 局部微调：并行采集会降低SNR，适当降低阈値
        if parallel:
            for k in base['depth_thresh']:
                base['depth_thresh'][k] = round(base['depth_thresh'][k] * 0.92, 3)

        base['res_grade'] = res
        base['field_t']   = field_t
        base['fat_sup']   = fat_sup
        base['parallel']  = parallel
        base['slice_th']  = slice_th

        print(f"   📊 参数路由: {res} | 像素间距={pixel_spacing:.3f}mm "
              f"| 场强={field_t}T | 并行采集={parallel} "
              f"| tol={base['tol_mm']}mm(回退{base['tol_mm_fallback']}mm) "
              f"| min_pts={base['min_pts']}(回退{base['min_pts_fallback']})")
        return base
    
    def segment_initial(self, img_raw):
        """
        双区域椎管初始化（V14_2 增强版）：
        - 下区域（覆盖腰骶尾椎）：行 30%~95%
        - 上区域（覆盖胸腰段）  ：行  0%~65%
        各自 Otsu + 保留所有合理连通域（非仅最大），合并 OR 得到完整椎管掩模。
        """
        h, w = img_raw.shape
        col_start, col_end = int(w * 0.30), int(w * 0.70)
        canal_seed = np.zeros_like(img_raw, dtype=bool)
        
        # V14_2: 最小面积门槛（避免小噪声）
        min_area_px = int(300 / (self.pixel_spacing ** 2))  # ~300mm²

        def _extract_best_region(r0, r1):
            """在 [r0,r1) 行、[col_start,col_end) 列内做 Otsu + 保留所有合理连通域"""
            r0 = max(0, min(r0, h - 10))
            r1 = max(r0 + 10, min(r1, h))
            region_img = img_raw[r0:r1, col_start:col_end]
            if region_img.size == 0:
                return None
            otsu_thresh = threshold_otsu(region_img)
            mask = region_img > otsu_thresh
            # V14_2: 使用 disk(1) 圆形结构元素（与 test_slice_diagnosis.py 一致）
            mask = sk_binary_dilation(mask, footprint=disk(1))
            # skimage.measure.label 只返回 labeled 数组，不返回数量
            labeled = sk_label(mask)
            
            # V14_2: 保留所有面积>=门槛的连通域（非仅最大）
            out = np.zeros_like(img_raw, dtype=bool)
            kept_count = 0
            for reg in measure.regionprops(labeled):
                if reg.area >= min_area_px:
                    minr, minc, maxr, maxc = reg.bbox
                    out[r0 + minr : r0 + maxr,
                        col_start + minc : col_start + maxc] = reg.image
                    kept_count += 1
            
            # 调试输出
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
            return None
        return canal_seed
    
    def find_dorsal_edge(self, img_raw, start_col, row):
        """
        背部边缘检测（斜率比较版）
        
        核心逻辑：
        1. 回退取参考信号（确保在CSF或脊髓内部）
        2. 向右扫描，找第一个落差点（记录斜率）
        3. 继续向右扫描固定距离（5mm），比较斜率
        4. 如果找到更大斜率，更新最大斜率点
        5. 返回最大斜率点作为背部边缘
        
        这样处理：起点在CSF→找到脊髓边界→继续找→找到背部边缘
                  起点在脊髓→直接找到背部边缘
        """
        h, w = img_raw.shape
        
        # 回退取参考信号（向左2mm，确保在CSF或脊髓内部）
        ref_distance = max(2, int(5 / self.pixel_spacing))
        ref_col = max(0, start_col - ref_distance)
        ref_signal = img_raw[row, ref_col]
        
        # 向右扫描范围（最大30mm）
        max_scan_distance = max(30, int(30 / self.pixel_spacing))
        scan_end = min(start_col + max_scan_distance, w)
        scan_cols = list(range(start_col, scan_end))
        
        if len(scan_cols) < 10:
            return start_col
        
        # 提取信号并平滑
        signals = np.array([img_raw[row, col] for col in scan_cols], dtype=np.float32)
        if len(signals) >= 5:
            smoothed = signal.savgol_filter(signals, window_length=5, polyorder=2)
        else:
            smoothed = signals
        
        # 计算梯度（斜率）
        gradient = np.gradient(smoothed)
        
        # 阈值：从参考信号下降40%认为是潜在边界
        drop_threshold = 0.40
        
        # 找第一个落差点（潜在边界）
        first_drop_idx = None
        first_drop_gradient = 0
        
        for i in range(1, len(smoothed)):
            # 计算相对于参考信号的下降比例
            drop_ratio = (ref_signal - smoothed[i]) / (ref_signal + 1e-6)
            
            if drop_ratio >= drop_threshold:
                # 找到第一个落差点，记录位置和斜率
                first_drop_idx = i
                first_drop_gradient = abs(gradient[i])
                break
        
        if first_drop_idx is None:
            # 没找到落差点，返回起始点
            return start_col
        
        # 继续向右扫描固定距离（7mm），比较斜率
        compare_distance = max(7, int(5 / self.pixel_spacing))
        search_end = min(first_drop_idx + compare_distance, len(smoothed) - 1)
        
        best_idx = first_drop_idx
        best_gradient = first_drop_gradient
        
        for i in range(first_drop_idx + 1, search_end):
            # 当前点的下降比例
            drop_ratio = (ref_signal - smoothed[i]) / (ref_signal + 1e-6)
            
            # 必须满足基本下降条件
            if drop_ratio >= drop_threshold * 0.8:  # 允许20%容差
                current_gradient = abs(gradient[i])
                
                # 如果找到更陡峭的下降沿，更新
                if current_gradient > best_gradient * 1.2:  # 必须明显更大（20%）
                    best_gradient = current_gradient
                    best_idx = i
        
        # 验证：最佳点信号应该显著低于参考信号
        final_drop_ratio = (ref_signal - smoothed[best_idx]) / (ref_signal + 1e-6)
        if final_drop_ratio < drop_threshold * 0.5:
            # 下降不够，可能不是真实边缘，返回第一个落差点
            return scan_cols[first_drop_idx]
        
        return scan_cols[best_idx]
    
    def find_highest_csf_segment(self, left_segment, start_col_in_original):
        """
        CSF中心定位（优化版）
        
        核心逻辑：
        1. 滑动窗口找最高平均信号
        2. 排除标准差过大的噪声区域
        3. 确保距离背部边缘合理
        """
        if len(left_segment) < 10:
            return None
        
        # 平滑
        if len(left_segment) > 5:
            smoothed = signal.savgol_filter(left_segment, window_length=5, polyorder=2)
        else:
            smoothed = np.array(left_segment, dtype=np.float32)
        
        # 滑动窗口参数
        window_size = max(3, int(5 / self.pixel_spacing))
        step = 1
        
        # 搜索范围：前50像素或根据像素间距调整
        max_search = max(50, int(50 / self.pixel_spacing))
        search_limit = min(max_search, len(smoothed) - window_size)
        
        if search_limit < window_size:
            return None
        
        # 计算全局统计
        global_mean = np.mean(smoothed[:search_limit])
        global_std = np.std(smoothed[:search_limit])
        
        best_avg = -1
        best_center = None
        
        # 滑动窗口找最高平均信号
        for start in range(0, search_limit, step):
            end = start + window_size
            window_data = smoothed[start:end]
            
            window_avg = np.mean(window_data)
            window_std = np.std(window_data)
            
            # 排除噪声区域（标准差过大）
            if window_std > global_std * 1.5:
                continue
            
            # 找最高平均
            if window_avg > best_avg:
                best_avg = window_avg
                best_center = start + window_size // 2
        
        if best_center is None:
            return None
        
        # 转回原图坐标
        mid_col = start_col_in_original - best_center
        
        return mid_col
    
    def find_ventral_edge(self, img_raw, row, csf_mid_col, csf_stack=None):
        """
        皮质线检测算法 V12.3

        算法逻辑：
        1. high_mean = csf_mid_col ±2mm 均值（不使用堆栈）
        2. 从 csf_mid_col 向腹侧（左）扫描最多30mm
        3. 找第一个相对 high_mean 下降 >= 60% 的候选点
        4. 垂直验证：候选点上下 ±2行同列信号（原图）相对 high_mean
           均值下降 >= 50%（过滤横向孤立高信号造成的假下降沿）
        5. 验证通过 → 返回该列作为皮质线
        6. 无候选通过 → 兜底返回距 csf_mid 约6mm处
        """
        if csf_mid_col is None:
            return None

        h, w = img_raw.shape

        # ── 1. high_mean = csf_mid_col ±2mm 均值 ──
        csf_half_px = max(2, int(2.0 / self.pixel_spacing))   # ±2mm
        csf_left  = max(0, csf_mid_col - csf_half_px)
        csf_right = min(w - 1, csf_mid_col + csf_half_px)
        high_mean = float(np.mean(img_raw[row, csf_left:csf_right + 1].astype(np.float32)))
        high_mean = max(high_mean, 1.0)  # 防除零

        # ── 2. 扫描范围：csf_mid_col 向左最多30mm ──
        scan_distance_px = max(20, int(30.0 / self.pixel_spacing))
        left_limit = max(0, csf_mid_col - scan_distance_px)

        if csf_mid_col - left_limit < 5:
            return max(5, csf_mid_col - 8)

        # ── 提取并平滑扫描剖面 ──
        profile = img_raw[row, left_limit:csf_mid_col + 1].astype(np.float32)
        if len(profile) < 10:
            return max(5, (csf_mid_col + left_limit) // 2)

        wl = min(9, len(profile)) if len(profile) >= 5 else len(profile)
        if wl % 2 == 0:
            wl -= 1
        wl = max(3, wl)
        smoothed = signal.savgol_filter(profile, window_length=wl, polyorder=2)
        smoothed = np.clip(smoothed, 0, None)

        # ── 3 & 4. 从右向左扫描，60%下降 + 垂直±1行验证 ──
        drop_thresh  = 0.50   # 主下降阈值
        vert_thresh  = 0.50   # 垂直验证阈值
        vert_px = 1           # 垂直验证行数 ±1行

        # 跳过最右4个点（紧贴csf_mid的CSF内部波动）
        # start_idx = len(smoothed) - 4
        start_idx = len(smoothed) - 1
        for i in range(start_idx, 1, -1):
            col = left_limit + i

            # 主条件：当前点相对high_mean下降>=60%
            if (high_mean - smoothed[i]) / high_mean < drop_thresh:
                continue

            # 条件B：垂直方向 ±1行同列原图信号，均值相对 high_mean 也下降 >= 50%
            # vert_rows = [r for r in range(row - vert_px, row + vert_px + 1)
            #              if r != row and 0 <= r < h]
            # if vert_rows:
            #     vert_vals = [float(img_raw[r, col]) for r in vert_rows]
            #     vert_mean = float(np.mean(vert_vals))
            #     if (high_mean - vert_mean) / high_mean < vert_thresh:
            #         continue

            # 验证通过 → 返回
            return max(left_limit + 2, min(col, csf_mid_col - 1))

        # ── 兜底：距 csf_mid 约6mm ──
        fallback_px = max(3, int(6.0 / self.pixel_spacing))
        ventral_edge = csf_mid_col - fallback_px
        return max(left_limit + 2, min(ventral_edge, csf_mid_col - 1))

    def _trace_single_row(self, img_raw, row, ref_dorsal_col, csf_stack=None, mask_bounds=None):
        """
        追踪单行椒管边界，以 ref_dorsal_col 为背侧起点锚点。
        返回 (dorsal_edge, ventral_edge, csf_mid) 或 None（失败）。
        锚点列允许偏移 ±col_tol_px 范围内搜索背侧边缘。
        csf_stack: 外部传入的high_mean历史堆栈，用于双源平均计算。
        mask_bounds: 可选元组 (mask_left, mask_right)，应用掩模边界约束。
        """
        h, w = img_raw.shape
        col_tol_px = max(4, int(6 / self.pixel_spacing))  # ±6mm 搜索容差

        # 在锚点附近找信号最高列（即背侧CSF峰值）
        search_left  = max(0, ref_dorsal_col - col_tol_px)
        search_right = min(w - 1, ref_dorsal_col + col_tol_px)
        row_signal = img_raw[row, search_left:search_right + 1]
        local_peak = int(np.argmax(row_signal)) + search_left

        # 背部线：直接用掩模右边界（线0），跳过灰度扫描
        if mask_bounds is not None:
            mask_left, mask_right = mask_bounds
            dorsal_edge = mask_right
        else:
            dorsal_edge = self.find_dorsal_edge(img_raw, local_peak, row)

        # 椎管宽度合理性检验：CSF区宽度至少3mm
        min_canal_px = max(3, int(3 / self.pixel_spacing))
        if dorsal_edge < min_canal_px:
            return None

        left_profile = img_raw[row, :dorsal_edge + 1][::-1]
        csf_mid = self.find_highest_csf_segment(left_profile, dorsal_edge)

        if csf_mid is None:
            ventral_edge = dorsal_edge - 8
        else:
            ve = self.find_ventral_edge(img_raw, row, csf_mid, csf_stack=csf_stack)
            ventral_edge = ve if ve is not None else dorsal_edge - 8

        ventral_edge = max(5, min(ventral_edge, dorsal_edge - 3))

        # ── 皮质线1约束：不能比掩模左边界更靠左 ──
        if mask_bounds is not None:
            mask_left, mask_right = mask_bounds
            ventral_edge = max(ventral_edge, mask_left)

        # 椎管宽度合理性检验（3-30mm）
        width_mm = (dorsal_edge - ventral_edge) * self.pixel_spacing
        if width_mm < 3.0 or width_mm > 30.0:
            return None

        return dorsal_edge, ventral_edge, csf_mid

    def trace_by_profile(self, canal_seed, img_raw, skip_upward=False):
        """
        逐行扫描追踪椎管边界 - 滑动窗口法
        先追踪种子区域（向下），再从种子顶端向上动态扩展。
        skip_upward=True 时跳过阶段2向上扩展（种子已覆盖完整范围时使用）。
        """
        if not np.any(canal_seed):
            return None, None, None, None, None
        
        h, w = img_raw.shape
        traced = np.zeros_like(img_raw, dtype=bool)
        
        rows, cols = np.where(canal_seed)
        if len(rows) == 0:
            return None, None, None, None, None
        
        row_min, row_max = np.min(rows), np.max(rows)

        # ── 阶段1：种子区域正向追踪（row_min → row_max）──
        dorsal_edges_down  = []
        ventral_edges_down = []
        valid_rows_down    = []
        csf_mid_down       = []
        csf_stack_down     = []   # 阶段1独立堆栈

        for row in range(row_min, row_max + 1):
            row_cols = cols[rows == row]
            if len(row_cols) == 0:
                continue
            
            # 提取该行掩模左右边界（作为线0）
            mask_left = np.min(row_cols)
            mask_right = np.max(row_cols)
            
            # ── 背部线：直接用掩模右边界（线0），跳过灰度扫描 ──
            dorsal_edge = mask_right
            
            left_profile = img_raw[row, :dorsal_edge+1][::-1]
            csf_mid = self.find_highest_csf_segment(left_profile, dorsal_edge)
            
            if csf_mid is None:
                ventral_edge = dorsal_edge - 8
            else:
                ventral_edge = self.find_ventral_edge(img_raw, row, csf_mid,
                                                      csf_stack=csf_stack_down)
                if ventral_edge is not None:
                    csf_mid_down.append((row, csf_mid))
                else:
                    ventral_edge = dorsal_edge - 8
            
            # ── 皮质线1约束：不能比掩模左边界更靠左 ──
            ventral_edge = max(ventral_edge, mask_left)
            
            ventral_edge = max(5, min(ventral_edge, dorsal_edge - 3))
            dorsal_edges_down.append(dorsal_edge)
            ventral_edges_down.append(ventral_edge)
            valid_rows_down.append(row)
            traced[row, ventral_edge:dorsal_edge+1] = True

        if len(valid_rows_down) == 0:
            return None, None, None, None, None

        # ── 阶段2：从种子顶端向上动态扩展 ──
        # 用种子顶部前5行的背侧边缘均值作为初始锚点列
        anchor_window = min(5, len(dorsal_edges_down))
        ref_dorsal = int(np.mean(dorsal_edges_down[:anchor_window]))

        dorsal_edges_up  = []
        ventral_edges_up = []
        valid_rows_up    = []
        csf_mid_up       = []
        csf_stack_up     = []   # 阶段2独立堆栈

        if not skip_upward:
            max_fail = 3       # 连续失败超过3行停止向上扩展
            fail_count = 0
            # 最高只向上追踪到图像顶部 5% 处，避免进入颈椎
            top_limit = max(int(h * 0.05), 0)

            for row in range(row_min - 1, top_limit - 1, -1):
                # 检查该行是否有掩模像素，用于边界约束
                row_mask_cols = cols[rows == row]
                mb = (np.min(row_mask_cols), np.max(row_mask_cols)) if len(row_mask_cols) > 0 else None
                result = self._trace_single_row(img_raw, row, ref_dorsal,
                                                 csf_stack=csf_stack_up,
                                                 mask_bounds=mb)
                if result is None:
                    fail_count += 1
                    if fail_count >= max_fail:
                        break
                    continue
                fail_count = 0
                dorsal_edge, ventral_edge, csf_mid = result
                ref_dorsal = dorsal_edge  # 更新锚点
                dorsal_edges_up.append(dorsal_edge)
                ventral_edges_up.append(ventral_edge)
                valid_rows_up.append(row)
                if csf_mid is not None:
                    csf_mid_up.append((row, csf_mid))
                traced[row, ventral_edge:dorsal_edge+1] = True

        # 向上的结果按行号升序排列后拼接到前面
        if valid_rows_up:
            order = np.argsort(valid_rows_up)
            dorsal_edges_up  = [dorsal_edges_up[i]  for i in order]
            ventral_edges_up = [ventral_edges_up[i] for i in order]
            valid_rows_up    = [valid_rows_up[i]    for i in order]

        # 合并：向上部分 + 向下部分
        dorsal_edges  = dorsal_edges_up  + dorsal_edges_down
        ventral_edges = ventral_edges_up + ventral_edges_down
        valid_rows    = valid_rows_up    + valid_rows_down
        csf_mid_points = csf_mid_up + csf_mid_down

        if len(valid_rows) == 0:
            return None, None, None, None, None

        print(f"   椎管追踪: 种子行 {row_min}-{row_max}, "
              f"向上扩展至行 {min(valid_rows_up) if valid_rows_up else row_min}, "
              f"共 {len(valid_rows)} 行")
        
        return traced, dorsal_edges, ventral_edges, valid_rows, csf_mid_points
    
    def calculate_roi_points(self, traced, dorsal_edges, ventral_edges, 
                            valid_rows, csf_mid_points, img_raw):
        """计算各ROI点"""
        h, w = img_raw.shape
        
        roi_points = {
            'csf': csf_mid_points if csf_mid_points else [],
            'vertebra': [],
            'disc': [],
            'spinal_cord': None,
            'vertebra_boundary': [],
        }
        
        # 椎体ROI点应该在腹侧边缘的左侧（向背部方向）
        for i, row in enumerate(valid_rows):
            # 腹侧边缘向左12像素作为椎体ROI
            col = max(0, ventral_edges[i] - 12)
            roi_points['vertebra'].append((row, col))
            roi_points['vertebra_boundary'].append((row, ventral_edges[i]))
        
        return roi_points
    
    def validate_tubular(self, traced, pixel_spacing):
        """管状特征验证 - 只警告，不拒绝"""
        if not np.any(traced):
            return True, "空掩膜"  # 让上层处理
        
        skeleton = skeletonize(traced)
        skeleton_length = np.sum(skeleton) * pixel_spacing
        
        if skeleton_length < 30:
            return True, f"⚠️ 椎管偏短({skeleton_length:.0f}mm)"  # 警告但不拒绝
        
        area = np.sum(traced) * (pixel_spacing ** 2)
        avg_width = area / skeleton_length if skeleton_length > 0 else 0
        self.width_value = avg_width
        
        # 解剖范围：正常椎管宽度 8-22mm，放宽到5-24mm
        if avg_width < 5:
            self.width_warning = True
            return True, f"⚠️ 椎管过窄({avg_width:.1f}mm)"
        elif avg_width > 24:
            self.width_warning = True
            return True, f"⚠️ 椎管过宽({avg_width:.1f}mm)"
        else:
            self.width_warning = False
            return True, f"椎管形态正常({avg_width:.1f}mm)"
    
    def process(self, img_raw):
        """完整椎管处理流程 - V12 版本"""
        self.width_warning = False
        self.width_value = 0
    
        canal_seed = self.segment_initial(img_raw)
        if canal_seed is None or not np.any(canal_seed):
            return None, None, None, "椎管分割失败", None
    
        traced, dorsal_edges, ventral_edges, valid_rows, csf_mid = \
            self.trace_by_profile(canal_seed, img_raw)
    
        if traced is None or not np.any(traced):
            return None, None, None, "边界追踪失败", None
    
        valid, status = self.validate_tubular(traced, self.pixel_spacing)
    
        roi_points = self.calculate_roi_points(
            traced, dorsal_edges, ventral_edges,
            valid_rows, csf_mid, img_raw
        )
    
        # ===== V12：背部线平滑 + 皮质线梳理 + 扫描线分析 =====
        # 背部线平滑
        smooth_dorsal_cols, dorsal_all_rows = self.smooth_dorsal_line(dorsal_edges, valid_rows)
    
        v9_data = self.analyze_endplates(ventral_edges, valid_rows, img_raw)
        if v9_data is not None:
            v9_data['smooth_dorsal_cols'] = smooth_dorsal_cols
            v9_data['dorsal_all_rows']    = dorsal_all_rows
            # 调试数据：MAD 平滑前原始皮质线点 + csf_mid 点
            v9_data['raw_ventral_edges']  = list(ventral_edges)
            v9_data['raw_ventral_rows']   = list(valid_rows)
            v9_data['csf_mid_points']     = list(csf_mid)  # [(row, col), ...]
    
        return traced, roi_points, valid_rows, status, v9_data
        
    def process_with_mask(self, img_raw, green_mask, canal_seed, merged_regions=None):
        """
        使用预设绿色掩模进行椎管处理（V14_2 增强流程）。
        
        green_mask = select_best_slice 阶段经 Step1/2/3 + 宽度检测精选的椎管掩模
                     作为追踪种子，排除盆腔等无关区域干扰
        canal_seed = 完整 segment_initial 掩模（保留备用，当前不参与追踪）
            
        参数：
            img_raw        - 原始图像
            green_mask     - 精选椎管掩模（追踪种子，排除盆腔干扰）
            canal_seed     - 完整 segment_initial 掩模（备用，暂不参与追踪）
            merged_regions - Step3 合并的区域列表，None 表示未合并
            
        返回：
            traced, roi_points, valid_rows, status, v9_data
        """
        self.width_warning = False
        self.width_value = 0
        
        # 使用精选的 green_mask 作为追踪种子（Step1+Step2+Step3 合并后，已排除盆腔）
        print(f"\n[V14_2] Using merged green_mask (area={np.sum(green_mask)}px) for processing...")
        
        traced, dorsal_edges, ventral_edges, valid_rows, csf_mid = \
            self.trace_by_profile(green_mask, img_raw, skip_upward=True)
        
        if traced is None or not np.any(traced):
            return None, None, None, "边界追踪失败", None
    
        valid, status = self.validate_tubular(traced, self.pixel_spacing)
    
        roi_points = self.calculate_roi_points(
            traced, dorsal_edges, ventral_edges,
            valid_rows, csf_mid, img_raw
        )
    
        # ===== V12：背部线平滑 + 皮质线梳理 + 扫描线分析 =====
        # 背部线平滑
        smooth_dorsal_cols, dorsal_all_rows = self.smooth_dorsal_line(dorsal_edges, valid_rows)
    
        v9_data = self.analyze_endplates(ventral_edges, valid_rows, img_raw)
        if v9_data is not None:
            v9_data['smooth_dorsal_cols'] = smooth_dorsal_cols
            v9_data['dorsal_all_rows']    = dorsal_all_rows
            v9_data['raw_ventral_edges']  = list(ventral_edges)
            v9_data['raw_ventral_rows']   = list(valid_rows)
            v9_data['csf_mid_points']     = list(csf_mid)
    
        return traced, roi_points, valid_rows, status, v9_data
    
    # ================================================================
    # V9新增：皮质线梳理 + 平行线终板分析
    # ================================================================
    
    def smooth_ventral_line(self, ventral_edges, valid_rows):
        """
        对皮质线进行MAD过滤 + 线性插值，使其连续平滑
        返回：(smoothed_cols, all_rows)，覆盖全行范围
        """
        edges = np.array(ventral_edges, dtype=np.float32)
        rows  = np.array(valid_rows)
        
        # --- MAD过滤（窗口11行，阈值2.0σ）---
        window = min(11, len(edges))
        if window % 2 == 0:
            window -= 1
        valid_mask = np.ones(len(edges), dtype=bool)
        half = window // 2
        for i in range(len(edges)):
            s = max(0, i - half)
            e = min(len(edges), i + half + 1)
            w = edges[s:e]
            med = np.median(w)
            mad = np.median(np.abs(w - med))
            if mad > 0 and np.abs(edges[i] - med) / (mad * 1.4826) > 2.0:
                valid_mask[i] = False
        
        clean_edges = edges[valid_mask]
        clean_rows  = rows[valid_mask]
        
        if len(clean_rows) < 4:
            return edges, rows   # 清洗后点太少，原样返回

        # --- 头尾大偏移裁切（MAD后、插值前）---
        # 检查头尾各 10mm 内，若某点偏离相邻稳定段中位数 > 50mm 则剔除
        _check_px = max(3, int(10.0 / self.pixel_spacing))  # 检查范围 10mm
        _ref_px   = max(3, int(10.0 / self.pixel_spacing))  # 参照段长度 10mm
        _dev_px   = int(20.0 / self.pixel_spacing)           # 偏差门槛 20mm

        # 头部：参照段为 check_px ~ check_px+ref_px 的中位数
        if len(clean_edges) > _check_px + _ref_px:
            _ref_top = np.median(clean_edges[_check_px : _check_px + _ref_px])
            _head_ok = np.ones(len(clean_edges), dtype=bool)
            _max_dev_h = 0.0
            for _i in range(min(_check_px, len(clean_edges))):
                _d = abs(clean_edges[_i] - _ref_top)
                _max_dev_h = max(_max_dev_h, _d)
                if _d > _dev_px:
                    _head_ok[_i] = False
            _removed_h = int(np.sum(~_head_ok))
            print(f"   [皮质线1] 头部最大偏差={_max_dev_h*self.pixel_spacing:.1f}mm, 裁切门槛=20mm, 裁切点数={_removed_h}")
            clean_edges = clean_edges[_head_ok]
            clean_rows  = clean_rows[_head_ok]

        # 尾部：参照段为 -(check_px+ref_px) ~ -check_px 的中位数
        if len(clean_edges) > _check_px + _ref_px:
            _ref_bot = np.median(clean_edges[-(  _check_px + _ref_px) : -_check_px])
            _tail_ok = np.ones(len(clean_edges), dtype=bool)
            _max_dev_t = 0.0
            for _i in range(max(0, len(clean_edges) - _check_px), len(clean_edges)):
                _d = abs(clean_edges[_i] - _ref_bot)
                _max_dev_t = max(_max_dev_t, _d)
                if _d > _dev_px:
                    _tail_ok[_i] = False
            _removed_t = int(np.sum(~_tail_ok))
            print(f"   [皮质线1] 尾部最大偏差={_max_dev_t*self.pixel_spacing:.1f}mm, 裁切门槛=20mm, 裁切点数={_removed_t}")
            clean_edges = clean_edges[_tail_ok]
            clean_rows  = clean_rows[_tail_ok]

        if len(clean_rows) < 4:
            return edges, rows   # 裁切后点太少，原样返回
        
        # --- 线性插值到所有封闭行 ---
        all_rows = np.arange(rows[0], rows[-1] + 1)
        interp_cols = np.interp(all_rows, clean_rows, clean_edges)
        
        # --- 轻度平滑（移动均値）---
        k = max(3, int(round(5.0 / self.pixel_spacing)))   # ~5mm平滑宽度
        if k % 2 == 0:
            k += 1
        pad = k // 2
        padded = np.pad(interp_cols, pad, mode='edge')
        kernel = np.ones(k) / k
        smooth_cols = np.convolve(padded, kernel, mode='valid')
        
        return smooth_cols.astype(np.float32), all_rows
    
    def smooth_dorsal_line(self, dorsal_edges, valid_rows):
        """
        对背部线进行MAD过滤 + 线性插值，使其连续平滑
        与smooth_ventral_line逻辑完全一致
        返回：(smoothed_cols, all_rows)
        """
        edges = np.array(dorsal_edges, dtype=np.float32)
        rows  = np.array(valid_rows)

        # --- MAD过滤 ---
        window = min(21, len(edges))
        if window % 2 == 0:
            window -= 1
        valid_mask = np.ones(len(edges), dtype=bool)
        half = window // 2
        for i in range(len(edges)):
            s = max(0, i - half)
            e = min(len(edges), i + half + 1)
            w = edges[s:e]
            med = np.median(w)
            mad = np.median(np.abs(w - med))
            if mad > 0 and np.abs(edges[i] - med) / (mad * 1.4826) > 3.5:
                valid_mask[i] = False

        clean_edges = edges[valid_mask]
        clean_rows  = rows[valid_mask]

        if len(clean_rows) < 4:
            return edges, rows   # 清洗后点太少，原样返回

        # --- 线性插值到所有封闭行 ---
        all_rows = np.arange(rows[0], rows[-1] + 1)
        interp_cols = np.interp(all_rows, clean_rows, clean_edges)

        # --- 轻度平滑（移动均往）---
        k = max(3, int(5 / self.pixel_spacing))   # ~5mm平滑宽度
        if k % 2 == 0:
            k += 1
        pad = k // 2
        padded = np.pad(interp_cols, pad, mode='edge')
        kernel = np.ones(k) / k
        smooth_cols = np.convolve(padded, kernel, mode='valid')

        return smooth_cols.astype(np.float32), all_rows

    def build_scan_lines(self, smooth_cols, all_rows, img_raw):
        """
        V11: 以平滑皮质线为基准，向内侧展开10条平行扫描线
        间距2mm，共覆盖2-20mm深度
        返回：列表[(offset_mm, cols_array, rows_array)]
        """
        h, w = img_raw.shape
        offsets_mm = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]  # V12.1: 13条线，多3条向腹侧延伸
        scan_lines = []
        for off_mm in offsets_mm:
            off_px = int(round(off_mm / self.pixel_spacing))
            cols = (smooth_cols - off_px).clip(0, w - 1).astype(np.int32)
            scan_lines.append((off_mm, cols, all_rows))
        return scan_lines
    
    def find_endplates_unified(self, scan_lines, img_raw):
        """
        统一分析所有扫描线的信号，找终板暗线
        步骤：1)采集所有线的信号 2)综合找谷底 3)验证终板特征
        返回：[(row, col), ...] 所有确认的终板暗线位置
        """
        from scipy.signal import savgol_filter
        
        if not scan_lines:
            return []
        
        # 1. 采集所有扫描线的信号
        all_signals = []
        all_rows = scan_lines[0][2]  # 所有线的行号相同
        
        print(f"\n   采集 {len(scan_lines)} 条扫描线信号...")
        
        for off_mm, cols, rows in scan_lines:
            sig = np.array([img_raw[r, c] for r, c in zip(rows, cols)], dtype=np.float32)
            all_signals.append(sig)
            print(f"     线+{off_mm}mm: 信号范围 [{sig.min():.1f}, {sig.max():.1f}], 均值={sig.mean():.1f}")
        
        # 2. 计算平均信号曲线（综合所有线）
        avg_signal = np.mean(all_signals, axis=0)
        
        # 3. 轻度平滑（可选）- 如果暗线不明显，可以注释掉此步骤
        use_filter = False  # 设置为False使用原始信号
        if use_filter:
            wl = min(11, len(avg_signal) if len(avg_signal) % 2 == 1 else len(avg_signal) - 1)
            wl = max(wl, 5)
            try:
                smooth = savgol_filter(avg_signal, window_length=wl, polyorder=2)
                print(f"   使用Savitzky-Golay滤波（窗口={wl}）")
            except:
                smooth = avg_signal
        else:
            smooth = avg_signal
            print(f"   使用原始信号（无滤波）")
        
        # 4. 找信号谷底（终板暗线候选）
        # 短探查距离：只找局部谷底，不跨越太远
        probe_distance = max(5, int(8 / self.pixel_spacing))  # 减小到8mm
        valley_threshold = 0.20  # 降低到20%，更灵敏
        
        valleys = []
        i = 3  # 从第3个点开始
        
        while i < len(smooth) - 3:
            # 检查当前点是否是局部谷底
            # 只看附近几个点，不是全局搜索
            local_window = smooth[max(0, i-2):min(len(smooth), i+3)]
            
            if smooth[i] == np.min(local_window):  # 是局部最低
                # 验证：比周围±5mm低8%以上
                ctx_window = max(3, int(5 / self.pixel_spacing))
                s = max(0, i - ctx_window)
                e = min(len(smooth), i + ctx_window)
                ctx_mean = np.mean(smooth[s:e])
                
                depth = (ctx_mean - smooth[i]) / ctx_mean
                
                if depth > valley_threshold:  # 低于周围均值8%以上
                    valleys.append((i, depth))  # 记录位置和深度
                    print(f"     发现谷底 @ idx {i}: 信号={smooth[i]:.1f}, 深度={depth*100:.1f}%")
                    i += probe_distance  # 跳过一段，避免重复
                    continue
            i += 1
        
        print(f"\n   找到 {len(valleys)} 个信号谷底候选")
        
        # 5. 验证每个谷底并计算位置
        valid_endplates = []
        min_ep_dist = max(5, int(8 / self.pixel_spacing))  # 最小8mm
        
        for v_idx, depth in sorted(valleys, key=lambda x: x[0]):
            # 计算该行在各线上的平均列位置
            row = int(all_rows[v_idx])
            
            # 检查与已有终板的距离
            if valid_endplates and row - valid_endplates[-1][0] < min_ep_dist:
                # 相距太近，保留更深的那个
                if depth > valid_endplates[-1][2]:  # 当前更深
                    valid_endplates[-1] = (row, 0, depth, v_idx)  # 暂时列=0，后面计算
                continue
            
            valid_endplates.append((row, 0, depth, v_idx))
        
        # 计算每个有效终板的平均列位置
        final_endplates = []
        for row, _, depth, v_idx in valid_endplates:
            avg_col = 0
            valid_count = 0
            for off_mm, cols, rows in scan_lines:
                if row in rows:
                    idx = list(rows).index(row)
                    avg_col += cols[idx]
                    valid_count += 1
            
            if valid_count > 0:
                avg_col = int(avg_col / valid_count)
                final_endplates.append((row, avg_col))
        
        print(f"   终板验证: {len(final_endplates)} 个有效终板")
        for ep in final_endplates[:8]:
            print(f"     row {ep[0]}, col {ep[1]}")
        
        # 6. 区域类型识别与交替验证
        validated_endplates = self.validate_alternating_pattern(
            final_endplates, smooth, all_rows, scan_lines, img_raw
        )
        
        return validated_endplates
    
    def validate_alternating_pattern(self, endplates, smooth_signal, all_rows, scan_lines, img_raw):
        """
        验证终板是否呈现椎间盘-椎体交替规律
        返回: [(row, col, ep_type), ...]  ep_type: 'inferior'/红色(下终板), 'superior'/绿色(上终板)
        """
        if len(endplates) < 2:
            return []
        
        print(f"\n   区域类型识别与交替验证...")
        
        # 计算信号基准
        median_sig = np.median(smooth_signal)
        
        # 识别每个区间的类型
        regions = []  # [(start_idx, end_idx, region_type, confidence), ...]
        
        for i in range(len(endplates) - 1):
            row1, col1 = endplates[i]
            row2, col2 = endplates[i + 1]
            
            # 找到在signal中的索引
            idx1 = np.where(all_rows == row1)[0]
            idx2 = np.where(all_rows == row2)[0]
            if len(idx1) == 0 or len(idx2) == 0:
                continue
            
            s_idx, e_idx = idx1[0], idx2[0]
            if e_idx <= s_idx:
                continue
            
            # 计算区间平均信号
            region_mean = np.mean(smooth_signal[s_idx:e_idx])
            region_len_mm = (e_idx - s_idx) * self.pixel_spacing
            
            # 优先识别椎间盘（高信号）- 尺寸范围：5-15mm
            if region_mean > median_sig * 1.10 and 5 <= region_len_mm <= 15:
                region_type = 'disc'
                confidence = 2  # 椎间盘置信度高
            elif region_mean > median_sig * 1.05 and 5 <= region_len_mm <= 15:
                region_type = 'disc'
                confidence = 1  # 椎间盘置信度中
            # 再识别椎体（中等信号）- 尺寸范围：20-55mm（腰椎椎体高度），与椎间盘不重叠
            elif 0.85 <= region_mean / median_sig <= 1.15 and 20 <= region_len_mm <= 55:
                region_type = 'vertebra'
                confidence = 1
            else:
                region_type = 'unknown'
                confidence = 0
            
            regions.append({
                'start_idx': s_idx,
                'end_idx': e_idx,
                'start_row': row1,
                'end_row': row2,
                'type': region_type,
                'confidence': confidence,
                'mean': region_mean,
                'length_mm': region_len_mm
            })
            
            print(f"     区间 {row1}-{row2}: {region_type} (置信度={confidence}, 长度={region_len_mm:.1f}mm)")
        
        # 交替验证：必须是 椎间盘-椎体-椎间盘-椎体...
        validated = []
        
        for i, region in enumerate(regions):
            if region['type'] == 'unknown':
                print(f"     ⚠️ 区间 {region['start_row']}-{region['end_row']} 类型不确定，检查上下文")
                
                # 检查上下区间确定类型
                prev_type = regions[i-1]['type'] if i > 0 else None
                next_type = regions[i+1]['type'] if i < len(regions)-1 else None
                
                # 如果上下都是椎间盘，则这个应该是椎体
                if prev_type == 'disc' and next_type == 'disc':
                    region['type'] = 'vertebra'
                    print(f"        → 推断为椎体（夹在两个椎间盘之间）")
                # 如果上下都是椎体，则这个应该是椎间盘
                elif prev_type == 'vertebra' and next_type == 'vertebra':
                    region['type'] = 'disc'
                    print(f"        → 推断为椎间盘（夹在两个椎体之间）")
                else:
                    print(f"        → 无法推断，跳过")
                    continue
            
            # 确定终板类型
            if region['type'] == 'disc':
                # 椎间盘的上边界是上终板 Superior EP（绿色），下边界是下终板 Inferior EP（红色）
                # 强制交替：如果上一个是inferior，这个必须是superior，反之亦然
                if not validated:
                    # 第一个终板，默认为上终板 Superior EP（绿色）
                    validated.append((region['start_row'], 
                                    self._get_col_at_row(region['start_row'], scan_lines), 
                                    'superior'))
                    validated.append((region['end_row'], 
                                    self._get_col_at_row(region['end_row'], scan_lines), 
                                    'inferior'))
                else:
                    last_type = validated[-1][2]
                    if last_type == 'superior':
                        # 上一个是上终板，这个应该是下终板 Inferior EP（红色）
                        validated.append((region['end_row'], 
                                        self._get_col_at_row(region['end_row'], scan_lines), 
                                        'inferior'))
                    else:
                        # 上一个是下终板，这个应该是上终板 Superior EP（绿色）
                        validated.append((region['start_row'], 
                                        self._get_col_at_row(region['start_row'], scan_lines), 
                                        'superior'))
                        validated.append((region['end_row'], 
                                        self._get_col_at_row(region['end_row'], scan_lines), 
                                        'inferior'))
        
        # 去重
        final_validated = []
        for ep in validated:
            if not final_validated or abs(ep[0] - final_validated[-1][0]) > 3:
                final_validated.append(ep)
        
        # 去重后进行全局交替复核
        final_validated = self.strict_alternation_check(final_validated)
        
        print(f"\n   ✅ 交替验证后: {len(final_validated)} 个终板")
        for ep in final_validated[:10]:
            color = '红' if ep[2] == 'inferior' else '绿'
            print(f"      {color}色 {ep[2]} endplate @ row {ep[0]}")
        
        return final_validated
    
    def strict_alternation_check(self, endplates):
        """
        严格交替复核：确保 inferior→superior→inferior→superior 交替，抛弃破坏规律的点
        空间约束：
          inferior→superior = 椎间盘 (5-15mm)
          superior→inferior = 椎体 (20-60mm)
        """
        if len(endplates) < 2:
            return endplates
        
        print(f"\n   严格交替复核（带空间约束）...")
        
        # 按行号排序（从上到下，row从小到大）
        sorted_eps = sorted(endplates, key=lambda x: x[0])
        
        result = [sorted_eps[0]]  # 保留第一个
        
        for i in range(1, len(sorted_eps)):
            current = sorted_eps[i]
            last = result[-1]
            
            if current[2] == last[2]:
                # 连续同色，需要根据空间约束判断保留哪个
                print(f"     ⚠️ 连续同色: row {last[0]}({last[2]}) vs row {current[0]}({current[2]})")
                
                # 计算欧氏距离（考虑水平偏移）
                if len(result) > 1:
                    prev_row, prev_col, _ = result[-2]
                    row_diff_last = abs(last[0] - prev_row) * self.pixel_spacing
                    col_diff_last = abs(last[1] - prev_col) * self.pixel_spacing
                    dist_last_to_prev = np.sqrt(row_diff_last**2 + col_diff_last**2)
                    
                    row_diff_curr = abs(current[0] - prev_row) * self.pixel_spacing
                    col_diff_curr = abs(current[1] - prev_col) * self.pixel_spacing
                    dist_current_to_prev = np.sqrt(row_diff_curr**2 + col_diff_curr**2)
                else:
                    # 没有前一个点，使用默认值
                    dist_last_to_prev = 30
                    dist_current_to_prev = 30
                
                # 判断上一个合法终板的类型
                prev_type = result[-2][2] if len(result) > 1 else ('superior' if last[2] == 'inferior' else 'inferior')
                
                # 预期距离范围
                if prev_type == 'superior' and last[2] == 'inferior':
                    # 上一个是上终板 Superior EP，当前应该是下终板 Inferior EP（红）
                    # 下终板应该在椎间盘下缘，距离上终板 5-15mm
                    expected_min, expected_max = 5, 15
                else:
                    # 上一个是下终板 Inferior EP，当前应该是上终板 Superior EP（绿）
                    # 上终板应该在椎体上缘，距离下终板 20-55mm（腰椎椎体高度）
                    expected_min, expected_max = 20, 55
                
                # 计算哪个点更符合预期距离
                score_last = self._distance_score(dist_last_to_prev, expected_min, expected_max)
                score_current = self._distance_score(dist_current_to_prev, expected_min, expected_max)
                
                print(f"        上一个终板类型: {prev_type}, 预期距离: {expected_min}-{expected_max}mm")
                print(f"        row {last[0]}: 距离={dist_last_to_prev:.1f}mm, 得分={score_last:.2f}")
                print(f"        row {current[0]}: 距离={dist_current_to_prev:.1f}mm, 得分={score_current:.2f}")
                
                if score_current > score_last:
                    print(f"        → 保留当前点 row {current[0]}")
                    result[-1] = current
                else:
                    print(f"        → 保留上一个点 row {last[0]}")
            else:
                # 正常交替，仍需要检查距离是否符合解剖约束（欧氏距离）
                row_diff = abs(current[0] - last[0]) * self.pixel_spacing
                col_diff = abs(current[1] - last[1]) * self.pixel_spacing
                dist = np.sqrt(row_diff**2 + col_diff**2)
                
                # 判断距离类型
                if last[2] == 'superior' and current[2] == 'inferior':
                    # superior→inferior = 椎间盘，应该是 5-15mm
                    if not (5 <= dist <= 15):
                        print(f"     ⚠️ 距离异常: 红→绿 距离={dist:.1f}mm (预期5-15mm)")
                        # 距离不对，保留更接近预期中值的点
                        if dist < 5:
                            # 太近，抛弃当前点
                            print(f"        → 太近，抛弃 row {current[0]}")
                            continue
                        # 太远也警告，但仍添加
                else:
                    # 绿→红 = 椎体，应该是 20-55mm
                    if not (20 <= dist <= 55):
                        print(f"     ⚠️ 距离异常: 绿→红 距离={dist:.1f}mm (预期20-55mm)")
                        if dist < 20:
                            # 太近，可能是假点（如图中的情况）
                            print(f"        → 太近，抛弃 row {current[0]}")
                            continue
                
                # 正常交替且距离合理，添加
                result.append(current)
        
        return result
    
    def _distance_score(self, distance, expected_min, expected_max):
        """
        计算距离得分：越接近预期范围，得分越高
        """
        if expected_min <= distance <= expected_max:
            # 在预期范围内，得分最高
            return 1.0
        elif distance < expected_min:
            # 太近，线性减分
            return max(0, 1.0 - (expected_min - distance) / expected_min)
        else:
            # 太远，线性减分
            return max(0, 1.0 - (distance - expected_max) / expected_max)
    
    def _get_col_at_row(self, row, scan_lines):
        """获取指定行在各扫描线上的平均列位置"""
        cols = []
        for off_mm, line_cols, line_rows in scan_lines:
            if row in line_rows:
                idx = list(line_rows).index(row)
                cols.append(line_cols[idx])
        return int(np.mean(cols)) if cols else 0
    
    def _run_clustering(self, all_candidates, scan_lines, min_pts, min_cover, tol_mm=None, texture_case=1):
        """
        辅助：以指定参数运行聚类 + 解剖复核 + 交替过滤 + 颜色标注
        tol_mm: 行号允差 mm，None表示自动根据像素间距确定
        返回: (candidate_lines, valid_lines, colored_lines, has_pair)
        """
        candidate_lines = self.cluster_to_endplate_lines(
            all_candidates, scan_lines,
            min_pts=min_pts, min_cover=min_cover, tol_mm=tol_mm, texture_case=texture_case
        )
        valid_lines = self.anatomical_distance_recheck(list(candidate_lines))
        valid_lines = self.filter_by_alternating(valid_lines)  # 交替过滤剪裁违规线
        colored_lines = self.assign_endplate_colors(valid_lines)
        has_pair = (
            any(line.get('region_type') == 'disc'     for line in valid_lines) and
            any(line.get('region_type') == 'vertebra' for line in valid_lines)
        )
        return candidate_lines, valid_lines, colored_lines, has_pair

    def filter_by_alternating(self, valid_lines):
        """
        交替性过滤：剪除违反 disc→vertebra→disc 交替规律的终板线
        规则：
          1. 跳过 unknown 区间，寻找下一个确定类型做比较
          2. 如果上一个确定类型 == 下一个确定类型，剪除两者之间点数最少的那一条
          3. 同时强制过滤解剖间距 < 4mm 的终板线（过于接近必是噪声点）
        返回: 过滤后的终板线列表
        """
        if len(valid_lines) < 2:
            return valid_lines

        # 第一步：剪除与任意相邻线间距 < 4mm 的终板线（解剖无意义的贴近线）
        min_valid_dist = 4.0
        lines = list(valid_lines)
        i = 0
        while i < len(lines) - 1:
            dist = lines[i].get('region_dist_mm', 0.0)
            if 0 < dist < min_valid_dist:
                pts_a = len(lines[i].get('points', []))
                pts_b = len(lines[i + 1].get('points', []))
                if pts_a >= pts_b:
                    print(f"   ✂️ 间距过小({dist:.1f}mm): 剪除第{i+1}条")
                    lines.pop(i + 1)
                else:
                    print(f"   ✂️ 间距过小({dist:.1f}mm): 剪除第{i}条")
                    lines.pop(i)
                lines = self.anatomical_distance_recheck(lines)
            else:
                i += 1

        # 第二步：跳过unknown寻找连续同类型区间，迍代剪除
        changed = True
        while changed:
            changed = False
            # 收集确定类型的线的索引序列
            confirmed_indices = [
                j for j, ln in enumerate(lines)
                if ln.get('region_type', 'unknown') in ('disc', 'vertebra')
            ]
            # 对相邻两个确定类型线做比较
            for k in range(len(confirmed_indices) - 1):
                ia = confirmed_indices[k]
                ib = confirmed_indices[k + 1]
                rt_a = lines[ia].get('region_type')
                rt_b = lines[ib].get('region_type')
                if rt_a == rt_b:  # 连续同类型
                    pts_a = len(lines[ia].get('points', []))
                    pts_b = len(lines[ib].get('points', []))
                    if pts_a >= pts_b:
                        print(f"   ✂️ 交替过滤: 剪除第{ib}条(row={lines[ib]['row_center']:.0f}, 点数={pts_b}, 连续{rt_b})")
                        lines.pop(ib)
                    else:
                        print(f"   ✂️ 交替过滤: 剪除第{ia}条(row={lines[ia]['row_center']:.0f}, 点数={pts_a}, 连续{rt_a})")
                        lines.pop(ia)
                    changed = True
                    lines = self.anatomical_distance_recheck(lines)
                    break  # 重新扫描

        print(f"   交替过滤后: {len(lines)} 条终板线")
        return lines
    
    def analyze_endplates(self, ventral_edges, valid_rows, img_raw):
        """
        V15主入口：皮质线梳理 + 扫描线建立
        - 皮质线1（smooth_cols）：小窗口平滑，保留细节，用于平移扫描线
        - 皮质线2（c2_cols）：50mm 强平滑，跟随整体弧度，用于 V15 切线方向扫描线和聚类坐标系
        """
        # 1. 皮质线1：小窗口平滑
        smooth_cols, all_rows = self.smooth_ventral_line(ventral_edges, valid_rows)

        # 2. 皮质线2：50mm 强平滑
        c2_cols, c2_rows = build_cortical2(
            smooth_cols, all_rows, self.pixel_spacing, smooth_mm=50.0)

        # 2.5 皮质线2下端延伸（切线方向基于皮质线1末端20mm回归）
        c2_cols, c2_rows = extend_cortical2_tail(
            c2_cols, c2_rows, self.pixel_spacing,
            img_shape=img_raw.shape,
            extend_mm=10.0, tail_mm=20.0,
            ref_cols=smooth_cols, ref_rows=all_rows)

        # 3. 建立平行扫描线（V12 兰山 供压水图坐标对齐使用）
        scan_lines = self.build_scan_lines(smooth_cols, all_rows, img_raw)

        # 4. V15 切线方向扫描线（33条，基于皮质线2）
        scan_lines_v15 = build_scan_lines_v15(
            c2_cols, c2_rows, img_raw.shape, self.pixel_spacing,
            n_lines=40, step_mm=1.0)
        # 计算皮质线2弧长数组（供聚类坐标系使用）
        arc_len_mm = np.zeros(len(c2_rows), dtype=np.float64)
        for i in range(1, len(c2_rows)):
            dr = float(c2_rows[i] - c2_rows[i - 1])
            dc = float(c2_cols[i] - c2_cols[i - 1])
            arc_len_mm[i] = arc_len_mm[i - 1] + np.sqrt(dr * dr + dc * dc) * self.pixel_spacing

        return {
            'smooth_cols':         smooth_cols,
            'all_rows':            all_rows,
            'scan_lines':          scan_lines,
            'c2_cols':             c2_cols,        # V15 皮质线2
            'c2_rows':             c2_rows,
            'arc_len_mm':          arc_len_mm,     # V15 弧长坐标
            'scan_lines_v15':      scan_lines_v15, # V15 切线方向扫描线
            'all_candidates':      {},
            'consensus_endplates': [],
            'texture_case':        1,
            'endplate_marks':      {},
            'all_line_valleys':    {},
        }

    # =====================================================================
    # V11 核心模块
    # =====================================================================

    def find_candidate_points(self, scan_lines, all_sigs, texture_case):
        """
        V11模块A：每条扫描线独立找候选终板点
        Case 1/3/4：找信号谷底（终板暗线）
        Case 2：椎间盘亮信号，找椎间盘亮区两侧的下降沿（终板暗线位于亮/暗交界处）
        返回：{offset_mm: [(row, col, depth), ...], ...}
        """
        from scipy.ndimage import gaussian_filter1d

        dt_map = self.params.get('depth_thresh', {1:0.25, 2:0.15, 3:0.08, 4:0.10})
        depth_thresh = dt_map.get(texture_case, dt_map.get(1, 0.20))
        sigma = 0.0 if texture_case == 3 else 1.0

        ctx = max(4, int(8 / self.pixel_spacing))
        min_gap_px = int(self.params.get('min_gap_mm', 5.0) / self.pixel_spacing)
        all_candidates = {}

        for k, (off_mm, cols, rows) in enumerate(scan_lines):
            sig_raw = all_sigs[k]
            sig = gaussian_filter1d(sig_raw, sigma=sigma) if sigma > 0 else sig_raw.copy()

            if texture_case == 2:
                # ── Case 2 专用：找椎间盘亮区局部峰，在每个峰两侧找下降沿 ──
                # 终板暗线 = 亮椎间盘信号急剧下降到暗椎体的边缘位置
                # 策略：
                #   1. 找信号局部峰（椎间盘中心）
                #   2. 在每个峰左右各找第一个下降沿：
                #      下降沿定义：该点信号 <= 峰值 * (1 - drop_ratio)
                #      且该点是局部极小值
                drop_ratio = 0.20   # 从峰值下降 20% 以上才算边缘
                min_peak_height_ratio = 0.30  # 峰值必须高于全局均值的30%才算有效峰
                sig_mean = np.mean(sig)
                sig_max  = np.max(sig)
                peak_thresh = sig_mean + (sig_max - sig_mean) * min_peak_height_ratio

                # 找局部峰
                peaks = []
                for i in range(ctx, len(sig) - ctx):
                    local_w = sig[i-2:i+3]
                    if sig[i] == np.max(local_w) and sig[i] >= peak_thresh:
                        peaks.append(i)

                # 去重峰（相邻 < min_gap_px 保留最高）
                merged_peaks = []
                for pk in peaks:
                    if merged_peaks and pk - merged_peaks[-1] < min_gap_px:
                        if sig[pk] > sig[merged_peaks[-1]]:
                            merged_peaks[-1] = pk
                    else:
                        merged_peaks.append(pk)

                # 在每个峰两侧找下降沿
                edge_pts = []
                search_half = max(ctx, int(15 / self.pixel_spacing))  # 向两侧最多搜15mm
                for pk in merged_peaks:
                    pk_val = sig[pk]
                    edge_thresh = pk_val * (1.0 - drop_ratio)

                    # 向上（减小行索引）找左侧下降沿
                    for j in range(pk - 1, max(0, pk - search_half), -1):
                        if sig[j] <= edge_thresh:
                            # 找该下降区间的局部最小值
                            lo = max(0, j - 3)
                            hi = min(len(sig)-1, j + 3)
                            local_min_idx = lo + int(np.argmin(sig[lo:hi+1]))
                            depth_val = (pk_val - sig[local_min_idx]) / (pk_val + 1e-6)
                            edge_pts.append((local_min_idx, depth_val))
                            break

                    # 向下（增大行索引）找右侧下降沿
                    for j in range(pk + 1, min(len(sig), pk + search_half)):
                        if sig[j] <= edge_thresh:
                            lo = max(0, j - 3)
                            hi = min(len(sig)-1, j + 3)
                            local_min_idx = lo + int(np.argmin(sig[lo:hi+1]))
                            depth_val = (pk_val - sig[local_min_idx]) / (pk_val + 1e-6)
                            edge_pts.append((local_min_idx, depth_val))
                            break

                # 去重：相邻过近保留 depth 更大的
                edge_pts.sort(key=lambda x: x[0])
                filtered = [edge_pts[0]] if edge_pts else []
                for v in edge_pts[1:]:
                    if v[0] - filtered[-1][0] >= min_gap_px:
                        filtered.append(v)
                    elif v[1] > filtered[-1][1]:
                        filtered[-1] = v

            else:
                # ── Case 1/3/4：原有谷底检测逻辑 ──
                valleys = []
                for i in range(ctx, len(sig) - ctx):
                    local_w = sig[i-2:i+3]
                    if sig[i] == np.min(local_w):
                        ctx_mean = np.mean(sig[max(0, i-ctx):min(len(sig), i+ctx)])
                        depth = (ctx_mean - sig[i]) / (ctx_mean + 1e-6)
                        if depth > depth_thresh:
                            valleys.append((i, depth))

                # 去重：相邻过近保留较深的
                filtered = [valleys[0]] if valleys else []
                for v in valleys[1:]:
                    if v[0] - filtered[-1][0] >= min_gap_px:
                        filtered.append(v)
                    elif v[1] > filtered[-1][1]:
                        filtered[-1] = v

                # 噪声抑制：数量上限
                max_per_line = 12
                if len(filtered) > max_per_line:
                    filtered.sort(key=lambda x: x[1], reverse=True)
                    filtered = filtered[:max_per_line]
                    filtered.sort(key=lambda x: x[0])

                # 相对深度门限：只保留深度 >= 本线最大深度 50% 的谷底
                if filtered:
                    max_depth = max(v[1] for v in filtered)
                    rel_thresh = max_depth * 0.50
                    filtered = [v for v in filtered if v[1] >= rel_thresh]

            # 转换为 (row, col, depth)
            pts = [(int(rows[i]), int(cols[i]), d) for i, d in filtered]
            all_candidates[off_mm] = pts
            print(f"     线+{off_mm}mm: {len(pts)} 个候选点")

        return all_candidates

    def cluster_to_endplate_lines(self, all_candidates, scan_lines, min_pts=7, min_cover=5, tol_mm=None, texture_case=1):
        """
        V12.4 新聚类方案：滑动窗口法
        流程：
          1. 把所有候选点按类型（upper/lower）分组，格式 {off_mm: [(row,col,depth), ...]}
          2. 确定窗口高度=5mm（图像行像素）
          3. 窗口左边界 = 最左扫描线列坐标 - 11mm（安全余量）
          4. 沿图像行号从上到下滑动窗口（步长1px）
          5. 统计窗口内（行号在[r, r+win_px]范围）有多少条扫描线有 upper/lower 候选点
          6. 有效扫描线数 >= min_pts → 确认为一条终板线
          7. 同一个峰区域只输出一条线（取窗口内点最多的行）
        """
        win_mm   = 5.0                                          # 窗口高度 5mm
        win_px   = max(3, int(win_mm / self.pixel_spacing))    # 转像素
        extra_px = max(1, int(11.0 / self.pixel_spacing))      # 左侧安全余量 11mm

        # 展开所有候选点：[(row, col, depth, off_mm, ep_type), ...]
        # all_candidates 格式：{off_mm: [(row, col, depth), ...]}
        # 但 all_candidates 里没有 ep_type，需要从 find_candidate_points 的返回值来
        # 这里 all_candidates 的结构是从 find_candidate_points 来的，包含 (row, col, depth)
        # ep_type 是在每条线上独立标注的，我们从 scan_lines 的信号方向推断
        # ── 重组为按 off_mm 索引的点列表 ──
        pts_by_off = {}   # {off_mm: [(row, col, depth), ...]}
        all_flat   = []   # [(row, col, depth, off_mm)]
        for off_mm, pts in all_candidates.items():
            pts_by_off[off_mm] = list(pts)
            for (row, col, depth) in pts:
                all_flat.append((row, col, depth, off_mm))

        if not all_flat:
            return []

        offsets_sorted = sorted(pts_by_off.keys())
        n_lines = len(offsets_sorted)

        # 确定行号范围
        all_rows_arr = np.array([p[0] for p in all_flat])
        r_min = int(np.min(all_rows_arr))
        r_max = int(np.max(all_rows_arr))

        # ── 滑动窗口扫描 ──
        # 对每个起始行 r，统计 [r, r+win_px] 内各 off_mm 有多少个点
        # 为加速，先建立 {off_mm: sorted_rows_array} 索引
        rows_index = {}
        for off_mm, pts in pts_by_off.items():
            rows_index[off_mm] = np.array(sorted([p[0] for p in pts]), dtype=np.int32)

        # 滑动窗口，记录每个起始行的「覆盖线数」
        cover_arr = np.zeros(r_max + win_px + 2, dtype=np.int32)
        for r in range(r_min, r_max + 1):
            cnt = 0
            for off_mm in offsets_sorted:
                rows_of_line = rows_index[off_mm]
                # 该扫描线在窗口 [r, r+win_px] 内有候选点？
                mask = (rows_of_line >= r) & (rows_of_line <= r + win_px)
                if np.any(mask):
                    cnt += 1
            cover_arr[r] = cnt

        # ── 找覆盖线数 >= min_pts 的峰区段 ──
        candidate_lines = []
        used_rows = set()   # 已归属的候选点行（防止重复使用）
        r = r_min
        min_peak_sep_px = max(4, int(8.0 / self.pixel_spacing))  # 峰间最小间距 8mm

        # 找所有满足条件的连续区段，每段取覆盖线数最大的行作为峰
        peaks = []
        in_peak = False
        seg_start = 0
        for r in range(r_min, r_max + 1):
            if cover_arr[r] >= min_pts:
                if not in_peak:
                    seg_start = r
                    in_peak = True
            else:
                if in_peak:
                    # 找该区段峰值行
                    seg = range(seg_start, r)
                    peak_r = int(seg_start + np.argmax(cover_arr[seg_start:r]))
                    peaks.append(peak_r)
                    in_peak = False
        if in_peak:
            peak_r = int(seg_start + np.argmax(cover_arr[seg_start:r_max+1]))
            peaks.append(peak_r)

        # 合并过近的峰
        merged_peaks = []
        for pk in sorted(peaks):
            if merged_peaks and abs(pk - merged_peaks[-1]) < min_peak_sep_px:
                # 保留覆盖线数更多的
                if cover_arr[pk] > cover_arr[merged_peaks[-1]]:
                    merged_peaks[-1] = pk
            else:
                merged_peaks.append(pk)

        print(f"   [V12.4滑动窗口] 峰: {len(merged_peaks)} 个，行号={merged_peaks}")

        # ── 对每个峰收集窗口内的点，构建终板线 ──
        used_pt_ids = set()
        for peak_row in merged_peaks:
            win_pts = []
            for p in all_flat:
                if (p[0] >= peak_row - win_px // 2
                        and p[0] <= peak_row + win_px // 2
                        and id(p) not in used_pt_ids):
                    win_pts.append(p)

            covered_offsets = set(p[3] for p in win_pts)
            if len(win_pts) < min_pts or len(covered_offsets) < min_cover:
                continue

            row_center = float(np.mean([p[0] for p in win_pts]))
            candidate_lines.append({
                'points':     win_pts,
                'row_center': row_center,
                'covered':    len(covered_offsets),
                'fit_coeffs': [],
            })
            for p in win_pts:
                used_pt_ids.add(id(p))

        candidate_lines.sort(key=lambda x: x['row_center'])
        print(f"   [V12.4] 聚类候选终板线: {len(candidate_lines)} 条 (滑动窗口法 win={win_mm}mm min_pts={min_pts})")
        return candidate_lines

    def validate_arc_shape(self, candidate_lines):
        """
        V11模块C：二次曲线拟合验证弧度合理性
        上终板(椎间盘上缘) = 凸弧（a < 0）
        下终板(椎间盘下缘) = 凹弧（a > 0）
        接近直线（|a|小）也接受
        返回：通过验证的线，每条线新增 'curvature', 'arc_type'
        """
        valid = []
        for line in candidate_lines:
            pts = line['points']
            # 用偏移量（mm）为x，行号为y，拟合 y = a*x^2 + b*x + c
            xs = np.array([p[3] for p in pts], dtype=np.float32)  # offset_mm
            ys = np.array([p[0] for p in pts], dtype=np.float32)  # row

            if len(xs) < 3:
                # 点数不足，直接保留为直线
                line['curvature'] = 0.0
                line['arc_type']  = 'straight'
                line['fit_coeffs'] = None
                valid.append(line)
                continue

            try:
                coeffs = np.polyfit(xs, ys, 2)  # [a, b, c]
                a = coeffs[0]
                line['curvature']  = float(a)
                line['fit_coeffs'] = coeffs

                # 弧度大小判断：|a| < 0.05 为直线
                if abs(a) < 0.05:
                    line['arc_type'] = 'straight'
                elif a < 0:
                    line['arc_type'] = 'convex'   # 凸弧，候选upper
                else:
                    line['arc_type'] = 'concave'  # 凹弧，候选lower

                # 弧度过大的拒绝（异常点）
                if abs(a) > 2.0:
                    print(f"     ✖ 弧度过大({a:.3f})，跳过")
                    continue

                valid.append(line)
            except Exception:
                line['curvature']  = 0.0
                line['arc_type']   = 'straight'
                line['fit_coeffs'] = None
                valid.append(line)

        print(f"   弧度验证后: {len(valid)} 条有效终板线")
        return valid

    def anatomical_distance_recheck(self, valid_lines, vert_range=(20, 45)):
        """
        V11模块D：解剖复核（纯间距驱动版）
        对相邻两条终板线，在每个共同 offset_mm 上计算两点欧氏距离，
        取最大值判断区间类型：
          5–18mm  → disc（椎间盘）
          vert_range[0]–vert_range[1]mm → vertebra（椎体）
          其他    → unknown
        新增每条线的 'region_type' 和 'region_dist_mm'（属于该线下方的区间）
        """
        if len(valid_lines) < 2:
            for line in valid_lines:
                line.setdefault('region_type', 'unknown')
                line.setdefault('region_dist_mm', 0.0)
            return valid_lines

        for i in range(len(valid_lines) - 1):
            line_a = valid_lines[i]
            line_b = valid_lines[i + 1]

            # 找两条线共同有的 offset_mm
            offsets_a = {p[3]: p for p in line_a['points']}
            offsets_b = {p[3]: p for p in line_b['points']}
            common_offsets = set(offsets_a.keys()) & set(offsets_b.keys())

            if not common_offsets:
                line_a['region_type']    = 'unknown'   # 强制覆盖，不用 setdefault
                line_a['region_dist_mm'] = 0.0
                continue

            # 在每个共同展开量上计算欧氏距离（单位 mm）
            dists = []
            for off in common_offsets:
                pa = offsets_a[off]
                pb = offsets_b[off]
                row_diff = (pa[0] - pb[0]) * self.pixel_spacing
                col_diff = (pa[1] - pb[1]) * self.pixel_spacing
                dist = np.sqrt(row_diff**2 + col_diff**2)
                dists.append(dist)

            max_dist = max(dists)

            if 5 <= max_dist <= 18:
                region_type = 'disc'
            elif vert_range[0] <= max_dist <= vert_range[1]:
                region_type = 'vertebra'
            else:
                region_type = 'unknown'

            line_a['region_type']    = region_type   # 强制覆盖，不用 setdefault
            line_a['region_dist_mm'] = max_dist
            print(f"     区间[{i}→{i+1}]: max_dist={max_dist:.1f}mm → {region_type}")

        # 最后一条线没有下方区间，强制覆盖（避免残留旧值）
        valid_lines[-1]['region_type']    = 'unknown'
        valid_lines[-1]['region_dist_mm'] = 0.0
        return valid_lines

    def assign_endplate_colors(self, valid_lines):
        """
        V11模块E：纯间距驱动的标注 + 交替验证自我纠错
        流程：
          1. 找第一个确定区间(类型已知的线)向上推断第一条线颜色
          2. 从上到下递推，每步验证当前颜色与下方区间的解剖合理性：
             upper(红)下方必须是 disc，否则抛弃该线，重新从下一个确定区间锁定
             lower(绿)下方必须是 vertebra，否则同上
             unknown 区间容忍，保留当前点但不更新锁定
          disc区间：上方线=upper(红)，下方线=lower(绿)
          vertebra区间：上方线=lower(绿)，下方线=upper(红)
        """
        if not valid_lines:
            return []

        def _infer_first_type(lines, start_idx):
            """从 lines[start_idx] 开始向前找第一个确定区间，推断 lines[0] 的颜色。"""
            for i in range(start_idx, len(lines)):
                rtype = lines[i].get('region_type', 'unknown')
                if rtype == 'disc':
                    # i条线下方是 disc → i条线是 superior（上终板）
                    # 如果 i=start_idx，那 start_idx 条线就是 superior
                    steps = i - start_idx
                    return 'superior' if steps % 2 == 0 else 'inferior', i
                elif rtype == 'vertebra':
                    steps = i - start_idx
                    return 'inferior' if steps % 2 == 0 else 'superior', i
            return None, -1

        # 第一步：向上推断第一条线颜色
        first_type, anchor_idx = _infer_first_type(valid_lines, 0)
        if first_type is None:
            first_type = 'superior'

        # 第二步：从上到下递推 + 交替验证自我纠错
        current_type = first_type
        result = []
        discard_indices = set()

        i = 0
        while i < len(valid_lines):
            line = valid_lines[i]
            rtype = line.get('region_type', 'unknown')

            # 验证当前线的颜色与下方区间是否吃合
            mismatch = False
            if rtype == 'disc':
                # superior 下面应该是 disc；inferior 下面不应该是 disc
                if current_type == 'inferior':
                    mismatch = True
            elif rtype == 'vertebra':
                # inferior 下面应该是 vertebra；superior 下面不应该是 vertebra
                if current_type == 'superior':
                    mismatch = True
            # unknown区间不验证，容忍

            if mismatch:
                print(f"   ⚠️ 交替验证失败: 第{i}条(row={line['row_center']:.0f}) "
                      f"颜色预期={current_type}, 但下方区间={rtype}({line.get('region_dist_mm',0):.1f}mm) → 抛弃")
                discard_indices.add(i)
                # 抛弃该线，从下一个确定区间重新锁定颜色
                new_type, _ = _infer_first_type(valid_lines, i + 1)
                if new_type is not None:
                    current_type = new_type
                # 不把该线加入 result
                i += 1
                continue

            # 通过验证，打颜色
            line['ep_type'] = current_type

            # 取代表点（展开量最接近 10mm 的点）
            best_pt = min(line['points'], key=lambda p: abs(p[3] - 10), default=None)
            if best_pt:
                line['center_row'] = best_pt[0]
                line['center_col'] = best_pt[1]
            else:
                line['center_row'] = int(line['row_center'])
                line['center_col'] = 0

            result.append(line)

            # 递推下一条线的颜色
            if rtype == 'disc':
                current_type = 'inferior' if current_type == 'superior' else 'superior'
            elif rtype == 'vertebra':
                current_type = 'superior' if current_type == 'inferior' else 'inferior'
            # unknown: 保持交替
            else:
                current_type = 'inferior' if current_type == 'superior' else 'superior'

            i += 1

        if discard_indices:
            print(f"   交替验证共抛弃 {len(discard_indices)} 条线, 保留 {len(result)} 条")

        return result

    
    def classify_signal_texture(self, sig):
        """
        V10.1: 信号曲线纹理分类
        返回: (case, info)
          case=1: 暗线清晰，高对比
          case=2: 暗线明显，中对比
          case=3: 暗线模糊，整体对比低
          case=4: 异常，椎体信号高于椎间盘（反转）
        """
        from scipy.ndimage import gaussian_filter1d
        
        # 轻度滤波用于分类分析（不影响后续谷底检测）
        sig_smooth = gaussian_filter1d(sig, sigma=2.0)
        sig_mean = np.mean(sig_smooth)
        
        # 指标A：整体对比度
        sig_range = np.max(sig_smooth) - np.min(sig_smooth)
        contrast = sig_range / (sig_mean + 1e-6)
        
        # 指标B：谷底平均深度（直接衡量暗线深度）
        ctx = max(5, int(10 / self.pixel_spacing))
        valleys_depth = []
        for i in range(3, len(sig_smooth) - 3):
            local_window = sig_smooth[i-2:i+3]
            if sig_smooth[i] == np.min(local_window):
                ctx_mean = np.mean(sig_smooth[max(0,i-ctx):min(len(sig_smooth),i+ctx)])
                depth = (ctx_mean - sig_smooth[i]) / (ctx_mean + 1e-6)
                if depth > 0.05:
                    valleys_depth.append(depth)
        avg_valley_depth = np.mean(valleys_depth) if valleys_depth else 0
        
        # 指标C：用已知尺寸范围将波峰分类为椎间盘/椎体
        # 找波峰
        peaks = []
        for i in range(ctx, len(sig_smooth) - ctx):
            if sig_smooth[i] == np.max(sig_smooth[i-3:i+4]):
                ctx_min = np.min(sig_smooth[max(0,i-ctx):min(len(sig_smooth),i+ctx)])
                prominence = (sig_smooth[i] - ctx_min) / (sig_mean + 1e-6)
                if prominence > 0.05:
                    peaks.append((i, float(sig_smooth[i])))
        
        # 波峰去重（间距太近的只保留较高的）
        min_peak_dist = int(5 / self.pixel_spacing)
        filtered_peaks = []
        for p in peaks:
            if not filtered_peaks or p[0] - filtered_peaks[-1][0] >= min_peak_dist:
                filtered_peaks.append(p)
            elif p[1] > filtered_peaks[-1][1]:
                filtered_peaks[-1] = p
        
        # 用相邻波峰间距判断类型：
        # 相邻两峰间距 5-18mm → 其间是椎间盘，这两个峰是椎体
        # 相邻两峰间距 20-60mm → 其间是椎体，这两个峰是椎间盘
        disc_signals = []    # 椎间盘峰信号
        vert_signals = []    # 椎体峰信号
        
        for i in range(len(filtered_peaks) - 1):
            idx_a, sig_a = filtered_peaks[i]
            idx_b, sig_b = filtered_peaks[i + 1]
            dist_mm = (idx_b - idx_a) * self.pixel_spacing
            
            if 5 <= dist_mm <= 18:
                # 两峰间距符合椎间盘尺寸 → 这两个峰是椎体峰
                vert_signals.extend([sig_a, sig_b])
            elif 20 <= dist_mm <= 60:
                # 两峰间距符合椎体尺寸 → 这两个峰是椎间盘峰
                disc_signals.extend([sig_a, sig_b])
        
        disc_mean = np.mean(disc_signals) if disc_signals else sig_mean
        vert_mean = np.mean(vert_signals) if vert_signals else sig_mean
        
        # 分类决策
        # 情况4必须满足：椎体信号高于椎间盘 AND 有明确渠动（两种峰都能识别到）
        if disc_signals and vert_signals and vert_mean > disc_mean * 1.20:
            case = 4
        elif contrast > 0.40 and avg_valley_depth > 0.25:
            case = 1
        elif avg_valley_depth > 0.12:
            case = 2
        else:
            case = 3
        
        info = {
            'contrast': contrast,
            'avg_valley_depth': avg_valley_depth,
            'disc_mean': disc_mean,
            'vert_mean': vert_mean,
            'n_peaks': len(filtered_peaks)
        }
        
        print(f"     纹理分类: 情况{case} | 对比度={contrast:.2f} 谷底深度={avg_valley_depth:.2f} 椎间盘信号={disc_mean:.0f} 椎体信号={vert_mean:.0f}")
        return case, info
    
    def adaptive_filter(self, sig, case):
        """
        V10.1: 根据纹理情况自适应滤波
        情况1/2/4: 轻度高斯 (sigma=1.0)
        情况3: 不滤波（暗线本来就浅，不能平滑）
        """
        from scipy.ndimage import gaussian_filter1d
        if case in (1, 2, 4):
            return gaussian_filter1d(sig.astype(np.float32), sigma=1.0)
        else:  # case 3
            return sig.astype(np.float32)  # 不滤波
    
    def find_valleys_adaptive(self, sig_raw, case):
        """
        V10.1: 根据纹理情况自适应谷底检测
        返回: [(idx, depth, confidence), ...]
          confidence: 'high'=高置信度, 'low'=低置信度, 'inferred'=推断
        """
        sig = self.adaptive_filter(sig_raw, case)
        ctx = max(5, int(10 / self.pixel_spacing))
        valleys = []
        
        if case == 1:
            # 标准閘唃：谷底必须低于周围25%以上
            depth_thresh = 0.25
            confidence = 'high'
        elif case == 2:
            # 降低閘唃：15%以上
            depth_thresh = 0.15
            confidence = 'high'
        elif case == 3:
            # 最低閘唃：8%以上，标为低置信度
            depth_thresh = 0.08
            confidence = 'low'
        else:  # case 4
            # 反转情况：找峰値两端的边界
            depth_thresh = 0.10
            confidence = 'inferred'
        
        for i in range(ctx, len(sig) - ctx):
            local_window = sig[i-2:i+3]
            if sig[i] == np.min(local_window):
                ctx_mean = np.mean(sig[max(0,i-ctx):min(len(sig),i+ctx)])
                depth = (ctx_mean - sig[i]) / (ctx_mean + 1e-6)
                if depth > depth_thresh:
                    valleys.append((i, depth, confidence))
        
        return valleys
    
    def find_valleys_per_line(self, scan_lines, img_raw):
        """
        V10.1: 每条扫描线独立找谷底（自适应纹理分类+滤波+检测）
        返回: {offset_mm: [(row, col, depth, confidence), ...], ...}
        """
        # 先用所有线平均信号做纹理分类
        all_sigs = []
        for off_mm, cols, rows in scan_lines:
            sig = np.array([img_raw[r, c] for r, c in zip(rows, cols)], dtype=np.float32)
            all_sigs.append(sig)
        avg_sig = np.mean(all_sigs, axis=0)
        
        print(f"\n   V10.1 纹理分类...")
        case, info = self.classify_signal_texture(avg_sig)
        
        # 每条线独立找谷底
        all_valleys = {}
        for k, (off_mm, cols, rows) in enumerate(scan_lines):
            sig = all_sigs[k]
            raw_valleys = self.find_valleys_adaptive(sig, case)
            
            # 过滤过远过近的谷底
            min_dist_px = int(5 / self.pixel_spacing)
            filtered = [raw_valleys[0]] if raw_valleys else []
            for v in raw_valleys[1:]:
                if v[0] - filtered[-1][0] >= min_dist_px:
                    filtered.append(v)
                elif v[1] > filtered[-1][1]:
                    filtered[-1] = v
            
            # 转换为 (row, col, depth, confidence)
            valleys = []
            for (idx, depth, conf) in filtered:
                valleys.append((rows[idx], cols[idx], depth, conf))
            
            all_valleys[off_mm] = valleys
            print(f"     线+{off_mm}mm: {len(valleys)} 个谷底 (情况{case}")
        
        # 将情况信息并入返回值
        all_valleys['_case'] = case
        all_valleys['_info'] = info
        return all_valleys
    
    def fallback_endplate_detection(self, scan_lines, img_raw, all_rows):
        """
        V10回退模式：当正常算法失败时，直接根据谷底间距判断类型
        短间距(5-15mm) = 椎间盘 → 上红下绿
        长间距(20-55mm) = 椎体 → 上绿下红
        """
        from scipy.signal import savgol_filter
        
        print("\n   V10回退模式：直接根据谷底间距判断...")
        
        # 1. 采集所有线的信号并平均
        all_signals = []
        for off_mm, cols, rows in scan_lines:
            sig = np.array([img_raw[r, c] for r, c in zip(rows, cols)], dtype=np.float32)
            all_signals.append(sig)
        avg_signal = np.mean(all_signals, axis=0)
        
        # 2. 找所有谷底（低于周围20%以上）
        valleys = []
        ctx_window = max(5, int(10 / self.pixel_spacing))
        
        for i in range(ctx_window, len(avg_signal) - ctx_window):
            local_window = avg_signal[i-2:i+3]
            if avg_signal[i] == np.min(local_window):
                # 验证深度
                ctx_mean = np.mean(avg_signal[max(0,i-ctx_window):min(len(avg_signal),i+ctx_window)])
                depth = (ctx_mean - avg_signal[i]) / ctx_mean
                if depth > 0.20:  # 低于周围20%以上
                    valleys.append((i, depth, avg_signal[i]))
        
        print(f"   找到 {len(valleys)} 个谷底候选")
        
        if len(valleys) < 2:
            return []
        
        # 3. 去重（距离太近的只保留一个）
        min_dist_px = int(8 / self.pixel_spacing)
        filtered_valleys = [valleys[0]]
        for v in valleys[1:]:
            if v[0] - filtered_valleys[-1][0] >= min_dist_px:
                filtered_valleys.append(v)
        
        print(f"   去重后: {len(filtered_valleys)} 个谷底")
        
        # 4. 根据间距判断类型并标记（使用双向验证确定第一个点的颜色）
        endplates = self.mark_endplate_colors(filtered_valleys, all_rows, scan_lines)
        
        # 5. 回退模式后复核：根据间距特征过滤
        endplates = self.fallback_validation(endplates)
        
        return endplates
    
    def mark_endplate_colors(self, valleys, all_rows, scan_lines):
        """
        根据间距标记终板颜色
        简单逻辑：看第一个间距确定第一个点颜色，然后交替
        椎间盘间距（短）：上红下绿
        椎体间距（长）：上绿下红
        """
        if len(valleys) == 0:
            return []
        
        # 提取行号
        rows = [int(all_rows[v[0]]) for v in valleys]
        
        # 计算所有间距（欧氏距离，考虑水平偏移）
        distances = []
        for i in range(len(rows) - 1):
            row_diff = abs(rows[i+1] - rows[i]) * self.pixel_spacing
            # 从valleys中获取列坐标（如果有的话）
            col_diff = 0  # 默认没有水平偏移
            dist_mm = np.sqrt(row_diff**2 + col_diff**2)
            distances.append(dist_mm)
        
        # 确定第一个点的颜色
        if len(distances) > 0:
            first_dist = distances[0]
            if 5 <= first_dist <= 15:
                # 第一个间距是椎间盘（短），第一个点是上终板 Superior EP（绿）
                first_type = 'superior'
                print(f"   第一个间距{first_dist:.1f}mm=椎间盘，第一个点为 Superior EP")
            elif 20 <= first_dist <= 55:
                # 第一个间距是椎体（长），第一个点是下终板 Inferior EP（红）
                first_type = 'inferior'
                print(f"   第一个间距{first_dist:.1f}mm=椎体，第一个点为 Inferior EP")
            else:
                # 不确定，默认 Superior EP（绿）
                first_type = 'superior'
                print(f"   第一个间距{first_dist:.1f}mm不确定，默认 Superior EP")
        else:
            first_type = 'superior'
        
        # 生成终板列表（交替颜色）
        endplates = []
        for i, (v_idx, depth, sig_val) in enumerate(valleys):
            row = rows[i]
            
            # 交替颜色
            if i % 2 == 0:
                ep_type = first_type
            else:
                ep_type = 'inferior' if first_type == 'superior' else 'superior'
            
            # 计算列位置
            avg_col = 0
            valid_count = 0
            for off_mm, cols, line_rows in scan_lines:
                if row in line_rows:
                    idx = list(line_rows).index(row)
                    avg_col += cols[idx]
                    valid_count += 1
            
            if valid_count > 0:
                avg_col = int(avg_col / valid_count)
                endplates.append((row, avg_col, ep_type))
                color = '红' if ep_type == 'inferior' else '绿'
                print(f"     {color}色 {ep_type} @ row {row}")
        
        return endplates
    
    def fallback_validation(self, endplates):
        """
        V10回退模式后复核：纯距离驱动，不预设颜色
        步骤：1)计算所有间距 2)根据距离识别类型 3)验证交替规律 4)抛弃异常点 5)标记颜色
        """
        if len(endplates) < 2:
            return endplates
        
        print(f"\n   回退模式后复核（纯距离驱动）...")
        
        # 按行号排序（从上到下）
        sorted_eps = sorted(endplates, key=lambda x: x[0])
        
        # 计算所有间距并识别类型
        intervals = []  # [(start_idx, end_idx, distance, type, score), ...]
        
        for i in range(len(sorted_eps) - 1):
            start_idx = i
            end_idx = i + 1
            # 计算欧氏距离（考虑水平偏移）
            row_diff = abs(sorted_eps[end_idx][0] - sorted_eps[start_idx][0]) * self.pixel_spacing
            col_diff = abs(sorted_eps[end_idx][1] - sorted_eps[start_idx][1]) * self.pixel_spacing
            dist_mm = np.sqrt(row_diff**2 + col_diff**2)
            
            # 根据距离识别类型并计算得分
            if 5 <= dist_mm <= 15:
                # 椎间盘，预期中值10mm
                region_type = 'disc'
                score = 1.0 - abs(dist_mm - 10) / 5  # 越接近10mm得分越高
            elif 20 <= dist_mm <= 55:
                # 椎体，预期中值40mm
                region_type = 'vertebra'
                score = 1.0 - abs(dist_mm - 40) / 15  # 越接近40mm得分越高
            else:
                # 不确定
                region_type = 'unknown'
                score = 0.0
            
            intervals.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'dist': dist_mm,
                'type': region_type,
                'score': max(0, score)
            })
            
            print(f"     间距 {sorted_eps[start_idx][0]} → {sorted_eps[end_idx][0]}: {dist_mm:.1f}mm, 类型={region_type}, 得分={score:.2f}")
        
        # 验证交替规律，抛弃异常点
        keep = [True] * len(sorted_eps)  # 标记哪些点保留
        
        for i in range(len(intervals) - 1):
            curr = intervals[i]
            next_int = intervals[i + 1]
            
            if curr['type'] == 'unknown' or next_int['type'] == 'unknown':
                continue  # 有不确定的，跳过
            
            if curr['type'] == next_int['type']:
                # 连续同类型，需要抛弃一个
                print(f"     ⚠️ 连续{curr['type']}: 间距{i}({curr['dist']:.1f}mm,得分{curr['score']:.2f}) vs 间距{i+1}({next_int['dist']:.1f}mm,得分{next_int['score']:.2f})")
                
                if curr['score'] >= next_int['score']:
                    # 当前得分高，抛弃下一个间距的终点（即i+1点）
                    keep[next_int['end_idx']] = False
                    print(f"        → 抛弃点 row {sorted_eps[next_int['end_idx']][0]}")
                else:
                    # 下一个得分高，抛弃当前间距的终点（即i+1点）
                    keep[curr['end_idx']] = False
                    print(f"        → 抛弃点 row {sorted_eps[curr['end_idx']][0]}")
        
        # 收集保留的点
        filtered = [sorted_eps[i] for i in range(len(sorted_eps)) if keep[i]]
        print(f"   复核完成: {len(filtered)}/{len(sorted_eps)} 个点保留")
        
        # 根据交替序列标记颜色
        # 简单逻辑：看第一个间距确定第一个点颜色，然后交替
        # 椎间盘间距（短5-15mm）：上红下绿 → 第一个点是红
        # 椎体间距（长20-55mm）：上绿下红 → 第一个点是绿
        result = []
        
        for i in range(len(filtered)):
            row, col, _ = filtered[i]
            
            if i == 0:
                # 第一个点，查看与下一个的间距（欧氏距离）
                if len(filtered) > 1:
                    next_row, next_col, _ = filtered[i+1]
                    row_diff = abs(next_row - row) * self.pixel_spacing
                    col_diff = abs(next_col - col) * self.pixel_spacing
                    dist_to_next = np.sqrt(row_diff**2 + col_diff**2)
                    if 5 <= dist_to_next <= 15:
                        # 第一个间距是椎间盘（短），第一个点是上终板 Superior EP（绿）
                        first_type = 'superior'
                        print(f"   复核后：第一个间距{dist_to_next:.1f}mm=椎间盘，第一个点为 Superior EP")
                    elif 20 <= dist_to_next <= 55:
                        # 第一个间距是椎体（长），第一个点是下终板 Inferior EP（红）
                        first_type = 'inferior'
                        print(f"   复核后：第一个间距{dist_to_next:.1f}mm=椎体，第一个点为 Inferior EP")
                    else:
                        first_type = 'superior'  # 默认
                        print(f"   复核后：第一个间距{dist_to_next:.1f}mm不确定，默认 Superior EP")
                else:
                    first_type = 'superior'
                ep_type = first_type
            else:
                # 后续点，与上一个交替
                last_type = result[-1][2]
                ep_type = 'inferior' if last_type == 'superior' else 'superior'
            
            result.append((row, col, ep_type))
            color = '红' if ep_type == 'inferior' else '绿'
            print(f"     {color}色 {ep_type} @ row {row}")
        
        return result


# ============ V12新增：压水图定位、坐标对齐 ============


class SpinalCordLocator:
    """在椎管内定位骨髓"""
    
    def __init__(self, pixel_spacing):
        self.pixel_spacing = pixel_spacing
    
    def locate(self, img_raw, traced, roi_points):
        """在椎管内找骨髓"""
        h, w = img_raw.shape
        
        if roi_points is None:
            roi_points = {
                'csf': [],
                'vertebra': [],
                'disc': [],
                'spinal_cord': None,
                'vertebra_boundary': [],
            }
        
        # 只在椎管内找低信号
        canal_pixels = img_raw[traced]
        if len(canal_pixels) < 50:
            return False, None, roi_points
        
        cord_otsu = threshold_otsu(canal_pixels)
        candidates = (img_raw < cord_otsu) & traced
        
        # 形态学清理
        struct = ndimage.generate_binary_structure(2, 1)
        candidates = binary_closing(candidates, structure=struct, iterations=1)
        candidates = binary_erosion(candidates, iterations=1)
        
        # skimage.measure.label 只返回 labeled 数组
        labeled = sk_label(candidates)
        max_area = 0
        best_region = None
        for region in measure.regionprops(labeled):
            if region.area > max_area:
                max_area = region.area
                best_region = region
        
        if best_region is None or max_area < 30:
            return False, None, roi_points
        
        centroid = best_region.centroid
        cord_mask = np.zeros_like(img_raw, dtype=bool)
        minr, minc, maxr, maxc = best_region.bbox
        cord_mask[minr:maxr, minc:maxc] = best_region.image
        
        roi_points['spinal_cord'] = (int(centroid[0]), int(centroid[1]))
        
        return True, cord_mask, roi_points


# ============ 椎体编号分配 ============
