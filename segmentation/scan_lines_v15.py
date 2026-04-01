"""
V15 扫描线生成与弧长坐标转换
包含 build_scan_lines_v15、convert_to_arc_coord
"""
import numpy as np


def build_scan_lines_v15(c2_cols, c2_rows, img_shape, pixel_spacing,
                         n_lines=40, step_mm=1.0):
    """
    V13 扫描线生成：沿皮质线2走向生成 n_lines 条平行扫描线，
    平移方式由 V12.5 水平平移改为沿局部法线方向平移，间距 step_mm。

    逻辑与 V12.5 build_scan_lines 对应：
      V12.5: cols = smooth_cols - off_px  (水平平移，2mm间距，13条)
      V13:   每个点沿局部法线方向偏移 i*step_mm，共 n_lines 条

    每条扫描线的形状与皮质线2一致（逐点沿该点法线方向偏移），
    不是从单一起点延伸的射线。

    参数：
        c2_cols/c2_rows - 皮质线2坐标数组（等长）
        img_shape       - (H, W)
        pixel_spacing   - mm/pixel
        n_lines         - 扫描线数量，默认 26（覆盖 1~26mm 深度）
        step_mm         - 法线方向平移间距，默认 1mm

    返回：
        scan_lines_v15: [(off_mm, rows_arr, cols_arr, nx_arr, ny_arr), ...]
          off_mm   - 该扫描线距皮质线2的法线偏移量(mm)
          rows_arr - 扫描线上每个像素行坐标（与 c2_rows 等长，越界点裁剪到边界）
          cols_arr - 扫描线上每个像素列坐标
          nx_arr   - 各点行方向法线单位向量
          ny_arr   - 各点列方向法线单位向量
    """
    H, W = img_shape
    N = len(c2_rows)

    # ---- 逐点计算局部切线和法线方向 ----
    # 使用前后差分（边界点用单侧差分）
    nx_arr = np.zeros(N, dtype=np.float64)  # 行方向法线
    ny_arr = np.zeros(N, dtype=np.float64)  # 列方向法线
    for i in range(N):
        i_prev = max(0, i - 1)
        i_next = min(N - 1, i + 1)
        t_dr = float(c2_rows[i_next] - c2_rows[i_prev])
        t_dc = float(c2_cols[i_next] - c2_cols[i_prev])
        t_len = np.sqrt(t_dr * t_dr + t_dc * t_dc)
        if t_len < 1e-6:
            t_dr, t_dc = 1.0, 0.0
            t_len = 1.0
        t_dr /= t_len
        t_dc /= t_len
        # 切线顺时针旋转 90° 得法线：(t_dr, t_dc) → (t_dc, -t_dr)
        nx = t_dc
        ny = -t_dr
        # 确保法线朝向腹侧（列减小方向，即椎体内部）
        if ny > 0:
            nx, ny = -nx, -ny
        nx_arr[i] = nx
        ny_arr[i] = ny

    # ---- 生成 n_lines 条扫描线 ----
    # 第 k 条线（k=1..n_lines）距皮质线2的法线偏移 = k * step_mm
    offsets_mm = [round((k + 1) * step_mm, 3) for k in range(n_lines)]
    scan_lines_v15 = []

    for off_mm in offsets_mm:
        off_px = off_mm / pixel_spacing  # 偏移像素数（浮点）
        rows_arr = []
        cols_arr = []
        for i in range(N):
            r_f = float(c2_rows[i]) + nx_arr[i] * off_px
            c_f = float(c2_cols[i]) + ny_arr[i] * off_px
            ri = int(round(r_f))
            ci = int(round(c_f))
            # 越界裁剪到图像边界
            ri = int(np.clip(ri, 0, H - 1))
            ci = int(np.clip(ci, 0, W - 1))
            rows_arr.append(ri)
            cols_arr.append(ci)

        scan_lines_v15.append((
            float(off_mm),
            np.array(rows_arr, dtype=np.int32),
            np.array(cols_arr, dtype=np.int32),
            nx_arr.copy(),
            ny_arr.copy(),
        ))

    print(f"   [V13] 生成扫描线 {len(scan_lines_v15)} 条 "
          f"(法线平移 {offsets_mm[0]}~{offsets_mm[-1]}mm, 间距={step_mm}mm)")
    return scan_lines_v15



def convert_to_arc_coord(row, col, c2_cols, c2_rows, arc_len_mm):
    """
    将图像坐标 (row, col) 转换为皮质线2弧长坐标系。

    返回：
        s_mm  - 在皮质线2上最近投影点的弧长坐标 (mm)
        d_px  - 到皮质线2的垂直距离 (像素)
    """
    # 找最近投影点：最小欧氏距离
    dists = np.sqrt((c2_rows.astype(np.float64) - row) ** 2 +
                    (c2_cols.astype(np.float64) - col) ** 2)
    best_i = int(np.argmin(dists))
    s_mm   = float(arc_len_mm[best_i])
    d_px   = float(dists[best_i])
    return s_mm, d_px

