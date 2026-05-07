"""
disc_centers.py - Step3 椎间盘中心点 + 椎体中心点 + Step3.5 末端校验
"""
import math
import numpy as np
from ._scan_utils import _project_to_c2


def compute_disc_and_vertebra_centers(junction_pts, c2_rows, c2_cols,
                                       pixel_spacing,
                                       extend_start_mm=15.0, extend_end_mm=20.0,
                                       merge_dist_mm=5.0):
    """
    1. 对每个终板汇合点，沿皮质线2法线方向延伸（渐变 15→20mm）→ 椎间盘中心点
    2. 过近点合并
    3. 相邻两椎间盘中心点中点 → 椎体中心点

    返回：
        disc_centers  [(r, c), ...]
        vert_centers  [(r, c), ...]
    """
    c2r = np.array(c2_rows, dtype=np.float64)
    c2c = np.array(c2_cols, dtype=np.float64)
    merge_px = merge_dist_mm / pixel_spacing

    r_min = float(c2r[0]); r_max = float(c2r[-1])
    r_span = max(r_max - r_min, 1e-9)

    raw_disc = []
    for (jr, jc, jval, jidx) in junction_pts:
        t = float(np.clip((jr - r_min) / r_span, 0.0, 1.0))
        extend_mm = extend_start_mm + t * (extend_end_mm - extend_start_mm)
        ext_px = extend_mm / pixel_spacing

        r_c2, c_c2, n_row, n_col = _project_to_c2(jr, jc, c2r, c2c)
        dr = r_c2 + n_row * ext_px
        dc = c_c2 + n_col * ext_px
        raw_disc.append((dr, dc))

    if not raw_disc:
        return [], []

    raw_disc.sort(key=lambda p: p[0])

    disc_centers = [raw_disc[0]]
    for p in raw_disc[1:]:
        last = disc_centers[-1]
        dist = math.sqrt((p[0] - last[0]) ** 2 + (p[1] - last[1]) ** 2)
        if dist < merge_px:
            disc_centers[-1] = ((last[0] + p[0]) / 2.0, (last[1] + p[1]) / 2.0)
        else:
            disc_centers.append(p)

    vert_centers = []
    for i in range(len(disc_centers) - 1):
        a = disc_centers[i]
        b = disc_centers[i + 1]
        vert_centers.append(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0))

    print(f"   [Step3] 椎间盘中心点={len(disc_centers)}  椎体中心点={len(vert_centers)}")
    return disc_centers, vert_centers


def verify_last_junction_point(in_img_2d, last_junction_pt, last_anchor_pt,
                                last_disc_center, c2_rows, c2_cols,
                                pixel_spacing, high_mean,
                                angle_deg=None):
    """
    校验末尾椎体终板汇合点，基于后缘线水平夹角动态法线偏移重定位椎间盘中心点。

    流程（与主扫描 compute_disc_and_vertebra_centers 对齐）：
      1. 从 last_junction_pt（皮质线2上的汇合点）投影到皮质线2，取法线方向
      2. 按 angle_deg 动态计算偏移量（夹角越大椎体越直，偏移量越大）：
           offset_mm = 8 + (clamp(angle_deg, 40, 80) - 40) / 40 × 12
         angle_deg=None 时使用固定默认值 15mm
      3. 从皮质线2投影点沿法线延伸 offset_mm → 新的椎间盘中心候选点
      4. 若候选点 col > 当前椎间盘中心 col → 更新

    参数：
        last_anchor_pt  - 保留但不再使用（兼容旧调用签名）
        angle_deg       - 后缘连线与水平线的夹角（度），None 使用默认 15mm

    返回：
        (updated, new_disc_center, None)
    """
    if last_junction_pt is None or last_disc_center is None:
        return False, last_disc_center, None

    c2r = np.array(c2_rows, dtype=np.float64)
    c2c = np.array(c2_cols, dtype=np.float64)

    jr = float(last_junction_pt[0])
    jc = float(last_junction_pt[1])

    r_c2, c_c2, n_row, n_col = _project_to_c2(jr, jc, c2r, c2c)

    # 确保法线朝腹侧（n_col < 0）
    if n_col > 0:
        n_row, n_col = -n_row, -n_col

    # 动态偏移量：夹角小（椎体斜）→ 偏移小，夹角大（椎体直）→ 偏移大
    if angle_deg is not None:
        a = float(np.clip(angle_deg, 40.0, 80.0))
        offset_mm = 8.0 + (a - 40.0) / 40.0 * 12.0
    else:
        offset_mm = 15.0
    offset_px = offset_mm / pixel_spacing

    cal_r = r_c2 + n_row * offset_px
    cal_c = c_c2 + n_col * offset_px

    cur_r, cur_c = last_disc_center
    if cal_c > cur_c:
        print(f"   [Step3.5校验] offset={offset_mm:.1f}mm "
              f"更新椎间盘中心: ({cur_r:.1f},{cur_c:.1f}) → ({cal_r:.1f},{cal_c:.1f})")
        return True, (cal_r, cal_c), None
    else:
        print(f"   [Step3.5校验] offset={offset_mm:.1f}mm "
              f"候选col={cal_c:.1f} ≤ 当前col={cur_c:.1f}，不更新")
        return False, last_disc_center, None
