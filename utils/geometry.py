"""
公共几何工具

提供常用几何计算函数，供各模块复用。
"""

import math
import numpy as np


def dist2d(pt1, pt2) -> float:
    """计算两点欧氏距离。"""
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def angle_between_vectors(v1, v2, degrees=True) -> float:
    """
    计算两个二维向量之间的夹角。

    参数：
        v1, v2   - (dr, dc) 形式的向量
        degrees  - True 返回角度，False 返回弧度

    返回：
        夹角（0~180°）
    """
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    len1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2) + 1e-9
    len2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2) + 1e-9
    cos_val = float(np.clip(dot / (len1 * len2), -1.0, 1.0))
    angle_rad = math.acos(cos_val)
    return math.degrees(angle_rad) if degrees else angle_rad


def polyfit_slope_intercept(xs, ys):
    """
    对点集做一次线性回归，返回 (slope, intercept)。
    xs, ys 可为列表或 numpy 数组。
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if len(xs) < 2:
        return 0.0, float(ys[0]) if len(ys) > 0 else 0.0
    try:
        slope, intercept = np.polyfit(xs, ys, 1)
        return float(slope), float(intercept)
    except Exception:
        dx = float(xs[-1] - xs[0])
        dy = float(ys[-1] - ys[0])
        slope = dy / max(abs(dx), 1e-9)
        intercept = float(ys[0]) - slope * float(xs[0])
        return slope, intercept
