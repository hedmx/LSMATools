"""
序列类型工具函数
从 series_description 解析序列类型、前缀和序号
"""
import re


def _get_series_type(series_desc: str):
    """
    从 series_description 解析序列类型。
    支持两种格式：
      - 末尾格式: T2_TSE_DIXON_SAG_W → W / T2_fse_wfi_sag_Water → W
      - 开头格式 (GE): WATER: Sag T2 FSE Flex → W / FAT: Sag T2 FSE Flex → F
    返回 'W' / 'F' / 'IN' / None
    """
    if not series_desc:
        return None
    desc_upper = series_desc.upper()
    
    # GE 开头格式: "WATER: ..." 或 "FAT: ..."
    if ':' in desc_upper:
        prefix = desc_upper.split(':', 1)[0].strip()
        if prefix == 'WATER':
            return 'W'
        if prefix == 'FAT':
            return 'F'
    
    # 标准末尾格式: ..._W 或 ..._WATER
    if '_' in desc_upper:
        suffix = desc_upper.rsplit('_', 1)[-1]
        if suffix in ('W', 'WATER'):
            return 'W'
        if suffix in ('F', 'FAT'):
            return 'F'
        if suffix in ('IN', 'INPHASE', 'IN_PHASE', 'IP'):
            return 'IN'
    return None


def _get_series_prefix(series_desc: str) -> str:
    """去掉最后一个'_X'后的前缀（用于同前缀配对）"""
    if not series_desc or '_' not in series_desc:
        return series_desc
    return series_desc.rsplit('_', 1)[0].lower()


def _get_series_number(folder_name: str) -> int:
    """从文件夹名末尾的数字提取 series number，如 T2_TSE_DIXON_SAG_W_0008 → 8"""
    m = re.search(r'_(\d+)$', folder_name)
    return int(m.group(1)) if m else 0


def _is_dixon_sequence(series_desc: str) -> bool:
    """
    判断是否为 Dixon/WFI 序列（不区分大小写）。
    兼容三种命名规范：
      - 'dixon'：如 T2_TSE_DIXON_SAG_W
      - 'wfi'  ：如 T2_TSE_WFI_SAG_W
      - 'flex' ：如 WATER: Sag T2 FSE Flex (GE)
    """
    d = series_desc.lower()
    return 'dixon' in d or 'wfi' in d or 'flex' in d
