"""
元数据解析工具函数
从 metadata.json 中提取常用字段
"""
import json
from pathlib import Path


def load_metadata(meta_path: str) -> dict:
    """加载 metadata.json，返回 dict；文件不存在则返回空 dict"""
    p = Path(meta_path)
    if not p.exists():
        return {}
    with open(p, 'r') as f:
        return json.load(f)


def parse_pixel_spacing(meta: dict, default: float = 0.9375) -> float:
    """
    从 metadata 中解析像素间距（mm）。
    优先读取 acquisition_params.pixel_spacing_mm，fallback 到 default。
    """
    acq = meta.get('acquisition_params', {})
    sp  = acq.get('pixel_spacing_mm', default)
    if isinstance(sp, list):
        return float(sp[0])
    return float(sp)


def parse_patient_id(meta: dict) -> str:
    """从 metadata 中解析患者 ID"""
    return meta.get('patient_info', {}).get('patient_id', '')


def parse_series_desc(meta: dict) -> str:
    """从 metadata 中解析序列描述字符串"""
    return meta.get('series_info', {}).get('series_description', '')


def parse_image_origin(meta: dict) -> list:
    """从 metadata 中解析图像坐标原点 [x, y, z]"""
    acq = meta.get('acquisition_params', {})
    pos = acq.get('imagepositionpatient', [0.0, 0.0, 0.0])
    return [float(v) for v in pos]
