"""
V15 核心参数定义
包含全局回退因子、V15 扫描线参数、点数回退序列等
"""

# ─── V15 终板检测参数 ───────────────────────────────────────────────
class V15Params:
    """V15 算法全局参数常量"""

    # ── 扫描线 ──
    N_SCAN_LINES        = 40       # 法线扫描线数量
    SCAN_LINE_OFFSET_MM = 2.0      # 每条线间距（mm）

    # ── 聚类 ──
    CLUSTER_WIN_MM      = 5.0      # 弧长聚类窗口（mm）
    CLUSTER_MIN_LINES   = 15       # 默认最小支持线数

    # ── 终板候选点 ──
    MIN_PTS_EP_BASE     = 28       # 点数阈值基准
    MIN_EP_COUNT        = 5        # 全局回退触发门槛（上/下终板线不足此数时触发）

    # ── 全局回退因子 ──
    GLOBAL_MED_FACTOR   = 0.5      # global_med 降低因子
    DROP_RATIO_FACTOR   = 0.7      # drop_ratio 降低因子
    RISE_RATIO_FACTOR   = 0.7      # rise_ratio 降低因子
    MIN_LINES_FACTOR    = 0.7      # min_lines 降低因子（→ 10）

    # ── 点数回退序列 ──
    RETRY_FACTORS       = [0.80, 0.65]  # 每轮点数回退阈值比例

    # ── 椎体间距校验 ──
    VERT_GAP_MIN_MM     = 20.0     # 椎体最小间距（mm）
    VERT_GAP_MAX_MM     = 45.0     # 椎体最大间距（mm）

    # ── 前缘检测 ──
    ANT_SCAN_OFFSET_MM  = 3.0      # 前缘扫描线偏移（mm）
    ANT_SEARCH_MM       = 30.0     # 前缘角最大延伸距离（mm）
    ANT_PROBE_MM        = 3.0      # 前缘角双侧探测长度（mm）

    # ── 切片优选 ──
    SLICE_CANDIDATES    = 5        # 优选候选切片数（中间 ±2）
    CANAL_MIN_AREA_MM2  = 300.0    # 椎管连通域最小面积（mm²）
    MAX_CANAL_WIDTH_MM  = 60.0     # 椎管宽度异常阈值（mm）

    # ── 状态机基础参数（常规时，低分辨率取 LR 档位） ──
    GLOBAL_MED_DEFAULT  = 226      # 状态机 global_med 初始值
    DROP_RATIO_DEFAULT  = 0.29     # 状态机下降沿阈值
    RISE_RATIO_DEFAULT  = 0.26     # 状态机上升沿阈值
