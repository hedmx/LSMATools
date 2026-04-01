"""
序列类型工具函数
从 series_description 解析序列类型、前缀和序号
"""
import re


def _get_series_type(series_desc: str):
    """
    从 series_description 最后一个'_'之后的字符判断序列类型。
    返回 'W'（压脂）/ 'F'（压水）/ None
    """
    if not series_desc or '_' not in series_desc:
        return None
    suffix = series_desc.rsplit('_', 1)[-1].upper()
    if suffix in ('W', 'WATER'):
        return 'W'
    if suffix in ('F', 'FAT'):
        return 'F'
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
