"""
椎体几何中心计算模块（V12.5）
包含 compute_vertebra_geometry：
  前缘角点 AP_top/AP_bot（arc_final 黄红/红黄切换行）
  后缘角点 BP_top/BP_bot（双线共识结果）
  椎体各几何中心列坐标从皮质线插值
"""
import numpy as np


def compute_vertebra_geometry(endplates_f, arc_final, smooth_cols, all_rows, pixel_spacing,
                               vert_edge_data=None):
    """
    V12.5 椎体几何建模：
      前缘角点 AP_top/AP_bot：arc_final 黄红/红黄切换行
      后缘角点 BP_top/BP_bot：V12.4 双线共识结果（vert_edge_data）的皮质线列坐标
        BP_top = confirmed_superior（绿色，行号小，上终板 Superior EP）→ 距 top_row 最近的那个
        BP_bot = confirmed_inferior（红色，行号大，下终板 Inferior EP）→ 距 bot_row 最近的那个
      各几何中心列坐标均从 smooth_cols 皮质线插值，不取两点列均值
    """
    if not endplates_f or not arc_final:
        return []

    # 建立 arc_final 行号 → (col, flag) 映射
    arc_row_map = {}
    for p in arc_final:
        arc_row_map[int(p[0])] = (float(p[1]), p[4] if len(p) > 4 else 'kept')

    # 建立皮质线行号 → 列号映射（用于精确插值）
    cortical_row_map = {int(r): float(c) for r, c in zip(all_rows, smooth_cols)}
    cortical_rows_arr = np.array(sorted(cortical_row_map.keys()))

    def get_cortical_col(row):
        idx = int(np.argmin(np.abs(cortical_rows_arr - row)))
        return cortical_row_map[cortical_rows_arr[idx]]

    def get_arc_col(row):
        rows_sorted = sorted(arc_row_map.keys())
        if not rows_sorted:
            return None
        cols = [arc_row_map[r][0] for r in rows_sorted]
        return float(np.interp(row, rows_sorted, cols))

    # 解析双线共识后缘角点列表
    # confirmed_inferior（红色 Inferior EP）= BP_bot 候选，行号大，椎体下终板
    # confirmed_superior（绿色 Superior EP）= BP_top 候选，行号小，椎体上终板
    bp_bot_cands = []  # [(row, col), ...] 红色，下角点
    bp_top_cands = []  # [(row, col), ...] 绿色，上角点
    if vert_edge_data:
        bp_bot_cands = list(vert_edge_data.get('inferior_edges', []))  # 红色
        bp_top_cands = list(vert_edge_data.get('superior_edges', []))  # 绿色

    def find_nearest_bp(target_row, cand_list, max_gap_px=None):
        """在候选列表中找距 target_row 最近的点，返回 (row, cortical_col)"""
        if not cand_list:
            return None
        dists = [abs(r - target_row) for r, c in cand_list]
        idx = int(np.argmin(dists))
        if max_gap_px is not None and dists[idx] > max_gap_px:
            return None
        r_near, _ = cand_list[idx]
        # 列坐标从 smooth_cols 精确插值
        return (float(r_near), float(get_cortical_col(r_near)))

    # 建立椎体对
    eps_sorted = sorted(endplates_f, key=lambda e: e[0])
    superior_eps = [e for e in eps_sorted if e[3] == 'superior']  # 上升沿，上终板 Superior EP，行号小
    inferior_eps = [e for e in eps_sorted if e[3] == 'inferior']  # 下降沿，下终板 Inferior EP，行号大

    vertebrae_pairs = []
    used_inferior = set()
    for li, lep in enumerate(superior_eps):
        best_ui, best_uep = None, None
        for ui, uep in enumerate(inferior_eps):
            if uep[0] > lep[0] and ui not in used_inferior:
                best_ui, best_uep = ui, uep
                break
        if best_uep is not None:
            used_inferior.add(best_ui)
            vertebrae_pairs.append((lep, best_uep))

    # 计算 arc_final 黄红/红黄切换点（前缘角点候选）
    arc_rows_sorted = sorted(arc_row_map.keys())
    yellow_to_red = []  # 黄→红切换行（AP_bot候选）
    red_to_yellow = []  # 红→黄切换行（AP_top候选）
    for i in range(1, len(arc_rows_sorted)):
        r_prev = arc_rows_sorted[i - 1]
        r_cur  = arc_rows_sorted[i]
        is_low_prev = (arc_row_map[r_prev][1] == 'kept_low')
        is_low_cur  = (arc_row_map[r_cur][1]  == 'kept_low')
        if not is_low_prev and is_low_cur:
            yellow_to_red.append(r_prev)
        elif is_low_prev and not is_low_cur:
            red_to_yellow.append(r_cur)

    def find_nearest_switch(target_row, switch_list, max_gap_px):
        if not switch_list:
            return None
        dists = [abs(r - target_row) for r in switch_list]
        idx = int(np.argmin(dists))
        return switch_list[idx] if dists[idx] <= max_gap_px else None

    max_sw_px = int(20.0 / pixel_spacing)
    max_bp_px = int(15.0 / pixel_spacing)  # 后缘角点匹配容差 ±15mm

    results = []
    for lep, uep in vertebrae_pairs:
        top_row = int(lep[0])  # 上终板行（行号小）
        bot_row = int(uep[0])  # 下终板行（行号大）

        # ── 前缘角点（arc_final 黄红切换行）──
        ap_top_row = find_nearest_switch(top_row, red_to_yellow, max_sw_px) or top_row
        ap_bot_row = find_nearest_switch(bot_row, yellow_to_red, max_sw_px) or bot_row
        ap_top_col = get_arc_col(ap_top_row)
        ap_bot_col = get_arc_col(ap_bot_row)
        if ap_top_col is None or ap_bot_col is None:
            continue
        AP_top = (float(ap_top_row), ap_top_col)
        AP_bot = (float(ap_bot_row), ap_bot_col)

        # ── 后缘角点（双线共识结果，皮质线精确插值列）──
        bp_top_result = find_nearest_bp(top_row, bp_top_cands, max_bp_px)
        bp_bot_result = find_nearest_bp(bot_row, bp_bot_cands, max_bp_px)

        # fallback：无共识点时直接取终板行的皮质线列
        if bp_top_result is None:
            bp_top_result = (float(top_row), float(get_cortical_col(top_row)))
            print(f"   [几何建模 fallback] BP_top 无共识，用终板行皮质线 row={top_row}")
        if bp_bot_result is None:
            bp_bot_result = (float(bot_row), float(get_cortical_col(bot_row)))
            print(f"   [几何建模 fallback] BP_bot 无共识，用终板行皮质线 row={bot_row}")

        BP_top = bp_top_result
        BP_bot = bp_bot_result

        # ── 几何中心（列坐标从皮质线/arc插值，不取两点列均值）──
        mid_top_row  = (AP_top[0] + BP_top[0]) / 2
        mid_bot_row  = (AP_bot[0] + BP_bot[0]) / 2
        mid_ant_row  = (AP_top[0] + AP_bot[0]) / 2
        mid_pos_row  = (BP_top[0] + BP_bot[0]) / 2
        cent_row     = (AP_top[0] + AP_bot[0] + BP_top[0] + BP_bot[0]) / 4

        mid_top  = (mid_top_row,  (AP_top[1] + get_cortical_col(mid_top_row))  / 2)
        mid_bot  = (mid_bot_row,  (AP_bot[1] + get_cortical_col(mid_bot_row))  / 2)
        mid_ant  = (mid_ant_row,  float(get_arc_col(mid_ant_row) or AP_top[1]))
        mid_pos  = (mid_pos_row,  float(get_cortical_col(mid_pos_row)))
        centroid = (cent_row,     (AP_top[1] + AP_bot[1] + BP_top[1] + BP_bot[1]) / 4)

        results.append({
            'AP_top':   AP_top,
            'AP_bot':   AP_bot,
            'BP_top':   BP_top,
            'BP_bot':   BP_bot,
            'mid_top':  mid_top,
            'mid_bot':  mid_bot,
            'mid_ant':  mid_ant,
            'mid_pos':  mid_pos,
            'centroid': centroid,
        })
        print(f"   [椎体几何] AP_top=({AP_top[0]:.0f},{AP_top[1]:.0f}) "
              f"AP_bot=({AP_bot[0]:.0f},{AP_bot[1]:.0f}) "
              f"BP_top=({BP_top[0]:.0f},{BP_top[1]:.0f}) "
              f"BP_bot=({BP_bot[0]:.0f},{BP_bot[1]:.0f})")

    print(f"   [椎体几何] 共建模 {len(results)} 个椎体")
    return results


