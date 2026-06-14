"""ICC(2,1) 双向随机效应，绝对一致性，单一评分者"""

import numpy as np
from scipy import stats


def icc_2way(data: np.ndarray) -> float:
    """
    ICC(2,1) — 双向随机效应、绝对一致性模型。

    参数:
      data : shape (n_subjects, k_raters)

    返回:
      float: ICC 值 [0, 1]
    """
    n, k = data.shape
    gm = data.mean()
    ms_rows = k * np.sum((data.mean(axis=1) - gm) ** 2) / (n - 1)
    ms_err = (np.sum(data ** 2) - n * sum(data.mean(axis=0) ** 2)
              - k * sum(data.mean(axis=1) ** 2) + n * k * gm ** 2) / ((n - 1) * (k - 1))
    ms_cols = n * np.sum((data.mean(axis=0) - gm) ** 2) / (k - 1)
    icc = (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n)
    return float(max(0.0, min(1.0, icc)))


def icc_ci(icc: float, n_subjects: int, k_raters: int, alpha: float = 0.05) -> tuple:
    """ICC 95% 置信区间 (F分布法)"""
    if icc >= 1.0:
        return 0.999, 1.0
    df1 = n_subjects - 1
    df2 = (n_subjects - 1) * (k_raters - 1)
    F = (1 + k_raters * icc / (1 - icc))
    FL = F / stats.f.ppf(1 - alpha / 2, df1, df2)
    FU = F * stats.f.ppf(1 - alpha / 2, df2, df1)
    return max(0.0, (FL - 1) / (FL + k_raters)), min(1.0, (FU - 1) / (FU + k_raters))


def rating(icc: float) -> str:
    """ICC → 评级"""
    if icc >= 0.90: return '\u4f18'
    if icc >= 0.75: return '\u826f'
    if icc >= 0.50: return '\u4e2d'
    return '\u5dee'
