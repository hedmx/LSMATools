"""
参数路由模块
根据像素间距、场强、并行采集等构建自适应参数字典
（从 SpinalCanalProcessor._build_param_route 提取为独立函数）
"""


def build_param_route(pixel_spacing: float, meta: dict) -> dict:
    """
    基于 metadata 的参数路由。
    根据像素间距、场强、并行采集等决定各个算法参数。

    参数：
        pixel_spacing - 像素间距（mm）
        meta          - metadata dict（acquisition_params / parallel_imaging）

    返回：
        params dict，包含 tol_mm / min_pts / depth_thresh / res_grade 等
    """
    acq      = meta.get('acquisition_params', {})
    par      = meta.get('parallel_imaging', {})
    field_t  = float(acq.get('magnetic_field_strength', 1.5))
    fat_sup  = acq.get('fat_suppressed', True)
    parallel = par.get('used', False)
    slice_th = float(acq.get('slice_thickness_mm', 4.0))

    # 分辨率等级
    if pixel_spacing <= 0.50:
        res = 'HR'    # 高分辨率
    elif pixel_spacing <= 0.75:
        res = 'STD'   # 标准分辨率
    else:
        res = 'LR'    # 低分辨率

    # 基础参数表
    base = {
        'HR':  dict(tol_mm=2.0, tol_mm_fallback=3.0, min_pts=7, min_pts_fallback=5,
                    depth_thresh={1: 0.25, 2: 0.15, 3: 0.08, 4: 0.10},
                    min_gap_mm=5.0),
        'STD': dict(tol_mm=2.5, tol_mm_fallback=4.0, min_pts=7, min_pts_fallback=5,
                    depth_thresh={1: 0.22, 2: 0.13, 3: 0.07, 4: 0.09},
                    min_gap_mm=4.5),
        'LR':  dict(tol_mm=3.0, tol_mm_fallback=5.0, min_pts=6, min_pts_fallback=5,
                    depth_thresh={1: 0.15, 2: 0.10, 3: 0.05, 4: 0.07},
                    min_gap_mm=4.0),
    }[res].copy()

    # 局部微调：3T 场强 SNR 高，可略微提高阈值
    if field_t >= 2.5:
        for k in base['depth_thresh']:
            base['depth_thresh'][k] = round(base['depth_thresh'][k] * 1.10, 3)

    # 局部微调：并行采集会降低 SNR，适当降低阈值
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
