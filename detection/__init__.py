"""detection 模块：IN序列 mode4 分割检测算法"""

from .signal_ref import compute_signal_references
from .junction_detector import scan_endplate_junction_points, repair_junction_pts
from .disc_centers import compute_disc_and_vertebra_centers, verify_last_junction_point
from .fan_scanner import (
    fan_scan_vertebra,
    _sample_ant_local_signal,
    _fan_scan_direction,
    _calc_ant_angle_deg,
    _verify_ant_pts_forward,
    scan_disc_endplates,
)
from .anterior_edge import (
    cluster_anterior_edge,
    cluster_all_vertebrae,
    _mad_smooth_line,
    _smooth_ant_line,
)

__all__ = [
    'compute_signal_references',
    'scan_endplate_junction_points',
    'repair_junction_pts',
    'compute_disc_and_vertebra_centers',
    'verify_last_junction_point',
    'fan_scan_vertebra',
    '_sample_ant_local_signal',
    '_fan_scan_direction',
    '_calc_ant_angle_deg',
    '_verify_ant_pts_forward',
    'scan_disc_endplates',
    'cluster_anterior_edge',
    'cluster_all_vertebrae',
    '_mad_smooth_line',
    '_smooth_ant_line',
]
