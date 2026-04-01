"""
皮质线构建模块
包含 build_cortical2、_repair_cortical2_slope、extend_cortical2_tail
"""
import numpy as np
from scipy import signal


def build_cortical2(smooth_cols, all_rows, pixel_spacing, smooth_mm=50.0):
    """
    皮质线2：基于皮质线1（smooth_cols/all_rows）做大窗口移动均値强平滑，
    去除单个椎体倾斜引起的局部波动，保留整体脊柱弧度走势。
    用于指导扫描线生成方向和聚类坐标系。

    参数：
        smooth_cols   - 皮质线1列坐标数组 (float32)
        all_rows      - 皮质线1行坐标数组
        pixel_spacing - 像素间距 mm
        smooth_mm     - 平滑窗口，默认 50mm

    返回：
        c2_cols - 皮质线2列坐标数组 (float32, 与 all_rows 等长)
        c2_rows - 与 all_rows 相同
    """
    k = max(3, int(smooth_mm / pixel_spacing))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(smooth_cols.astype(np.float64), pad, mode='edge')
    kernel = np.ones(k) / k
    c2_cols = np.convolve(padded, kernel, mode='valid').astype(np.float32)
    print(f"   [V15] 皮质线2 平滑窗口={smooth_mm}mm ({k}px), 行范围=[{all_rows[0]},{all_rows[-1]}]")

    # ── 方案C：曲率连续性校验与修复 ──
    # 计算逐点斜率（dc/dr，像素单位），用邻域中位数检测突变
    # 突变定义：某点斜率与其邻域（±neighbor_px）的中位数偏差 > slope_thr
    # 修复方式：将突变段两端的良好点做线性插值，使切线方向渐变
    c2_cols = _repair_cortical2_slope(c2_cols, pixel_spacing)

    return c2_cols, np.array(all_rows)


def _repair_cortical2_slope(c2_cols, pixel_spacing,
                            slope_thr=0.5, neighbor_px=10, max_iter=3):
    """
    对皮质线2做斜率连续性修复：
    - 逐点计算斜率 dc/dr
    - 与邻域中位数斜率偏差超过 slope_thr（像素/像素）的点标记为突变
    - 将突变段的 c2_cols 用两端锚点线性插值替换
    - 迭代最多 max_iter 次直到无突变为止

    slope_thr：斜率突变阈值，默认 0.5（约对应弧度 ~26°的切线跳变）
    neighbor_px：邻域窗口半径（像素），用于估计局部中位数斜率
    """
    cols = c2_cols.astype(np.float64).copy()
    N = len(cols)
    if N < 4:
        return c2_cols

    for _iter in range(max_iter):
        # 1. 计算逐点斜率（中心差分，边界单侧）
        slopes = np.zeros(N, dtype=np.float64)
        for i in range(N):
            i0 = max(0, i - 1)
            i1 = min(N - 1, i + 1)
            slopes[i] = (cols[i1] - cols[i0]) / max(1, i1 - i0)

        # 2. 逐点与邻域中位斜率对比，标记突变点
        bad = np.zeros(N, dtype=bool)
        for i in range(N):
            s0 = max(0, i - neighbor_px)
            s1 = min(N, i + neighbor_px + 1)
            nbr = np.concatenate([slopes[s0:i], slopes[i+1:s1]])
            if len(nbr) == 0:
                continue
            med = float(np.median(nbr))
            if abs(slopes[i] - med) > slope_thr:
                bad[i] = True

        n_bad = int(bad.sum())
        if n_bad == 0:
            break  # 无突变，退出

        # 3. 将连续突变段用两端锚点线性插值修复
        i = 0
        while i < N:
            if bad[i]:
                seg_start = i
                while i < N and bad[i]:
                    i += 1
                seg_end = i - 1  # 突变段 [seg_start, seg_end]
                # 找左锚点（seg_start 左侧第一个好点）
                left_anchor  = seg_start - 1
                right_anchor = seg_end + 1
                if left_anchor < 0:
                    left_anchor = right_anchor  # 段在最左，用右锚点水平延伸
                if right_anchor >= N:
                    right_anchor = left_anchor  # 段在最右，用左锚点水平延伸
                c_left  = cols[left_anchor]
                c_right = cols[right_anchor]
                # 线性插值替换突变段
                for j in range(seg_start, seg_end + 1):
                    t = ((j - left_anchor) / max(1, right_anchor - left_anchor)
                         if right_anchor != left_anchor else 0.0)
                    cols[j] = c_left + t * (c_right - c_left)
            else:
                i += 1

        print(f"   [V15] 皮质线2曲率修复 iter={_iter+1}: 修复突变点 {n_bad} 个")

    return cols.astype(np.float32)


def extend_cortical2_tail(c2_cols, c2_rows, pixel_spacing,
                          img_shape, extend_mm=40.0, tail_mm=20.0,
                          ref_cols=None, ref_rows=None):
    """
    皮质线2下端延伸：基于末端切线方向外推。

    算法：
      1. 优先用 ref_cols/ref_rows（皮质线1）末端 tail_mm 段做线性回归，
         获得保真的局部弧度方向 dc/dr；若未提供则退回到皮质线2自身
      2. 从皮质线2末端点逐行外推 extend_mm，越界裁剪
      3. 将新行追加到 c2_rows/c2_cols

    参数：
        c2_cols/c2_rows - 皮质线2（已完成平滑）
        pixel_spacing   - mm/pixel
        img_shape       - (H, W) 图像尺寸，用于边界裁剪
        extend_mm       - 延伸长度（mm），默认 40mm
        tail_mm         - 估算末端切线的参考段长度（mm），默认 20mm
        ref_cols/ref_rows - 参考线（皮质线1），用其末端弧度替代皮质线2
    """
    H, W = img_shape
    N = len(c2_rows)
    if N < 4:
        return c2_cols, c2_rows

    # 选择斜率来源：优先皮质线1，否则用皮质线2自身
    if ref_cols is not None and ref_rows is not None and len(ref_rows) >= 4:
        src_cols = np.array(ref_cols, dtype=np.float64)
        src_rows = np.array(ref_rows, dtype=np.float64)
        src_label = "皮质线1"
    else:
        src_cols = c2_cols.astype(np.float64)
        src_rows = c2_rows.astype(np.float64)
        src_label = "皮质线2"

    # 参考段像素数：末端 tail_mm 对应的像素数
    tail_px = max(4, int(round(tail_mm / pixel_spacing)))
    tail_px = min(tail_px, len(src_rows))

    tail_rows = src_rows[-tail_px:]
    tail_cols = src_cols[-tail_px:]

    # 线性回归： col = slope * row + intercept，row 单调递增无多对一问题
    try:
        slope, _ = np.polyfit(tail_rows, tail_cols, 1)  # dc/dr
    except Exception:
        slope = float(tail_cols[-1] - tail_cols[0]) / max(float(tail_rows[-1] - tail_rows[0]), 1.0)

    # 外推：逐行 row += 1，col += slope
    extend_px = int(round(extend_mm / pixel_spacing))
    last_row  = int(c2_rows[-1])
    last_col  = float(c2_cols[-1])

    new_rows = []
    new_cols = []
    for step in range(1, extend_px + 1):
        nr = last_row + step
        nc = last_col + slope * step
        if nr >= H:          # 超出图像下边界停止
            break
        nc = float(np.clip(nc, 0, W - 1))
        new_rows.append(nr)
        new_cols.append(nc)

    if not new_rows:
        return c2_cols, c2_rows

    ext_rows = np.array(new_rows, dtype=c2_rows.dtype)
    ext_cols = np.array(new_cols, dtype=np.float32)

    c2_rows_ext = np.concatenate([c2_rows, ext_rows])
    c2_cols_ext = np.concatenate([c2_cols, ext_cols])

    _actual_mm = len(new_rows) * pixel_spacing
    print(f"   [皮质线2延伸] 斜率来源={src_label}(tail={tail_mm}mm), dc/dr={slope:.4f}, "
          f"延伸行数={len(new_rows)}px ({_actual_mm:.1f}mm), "
          f"行范围 [{c2_rows[0]}, {c2_rows_ext[-1]}]")

    return c2_cols_ext, c2_rows_ext


