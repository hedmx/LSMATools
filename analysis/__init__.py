"""
analysis — LSMATools_WIFS 独立数据分析模块

子模块:
  icc_utils       — ICC(2,1) 计算工具
  dice_evaluator  — DICE / IoU / HD95 / ASD 掩膜指标
  geom_evaluator  — 形态学对比 (高度/面积 ICC)
  report_generator — CSV → Markdown/HTML 报告
  run_analysis    — CLI 一键入口
"""

from .icc_utils import icc_2way, icc_ci
from .dice_evaluator import DiceEvaluator
from .geom_evaluator import GeomEvaluator
from .report_generator import generate_report
