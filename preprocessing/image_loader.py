"""
图像加载模块
提供 NIfTI 文件加载和压水/压脂图像配对功能
"""
import json
import numpy as np
import nibabel as nib
from pathlib import Path

from .series_utils import _get_series_type, _get_series_prefix, _get_series_number


def load_nifti(nifti_path: str, slice_idx: int = None):
    """
    加载 NIfTI 文件，返回 (data_3d, img_2d, pixel_spacing)。

    参数：
        nifti_path  - .nii.gz 文件路径
        slice_idx   - 若指定，则直接取该切片；否则取中间切片

    返回：
        data        - 3D numpy 数组 shape=(H,W,N)
        img_2d      - 选定切片的 2D float32 数组
        pixel_spacing - 像素间距（mm），从同目录 metadata.json 读取，默认 0.9375
    """
    data  = nib.load(nifti_path).get_fdata()
    n_sl  = data.shape[2]
    if slice_idx is None:
        slice_idx = n_sl // 2
    slice_idx = max(0, min(slice_idx, n_sl - 1))
    img_2d = data[:, :, slice_idx].astype(np.float32)

    # 读取像素间距
    meta_path = Path(nifti_path).parent / 'metadata.json'
    pixel_spacing = 0.9375
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        acq = meta.get('acquisition_params', {})
        sp  = acq.get('pixel_spacing_mm', pixel_spacing)
        pixel_spacing = float(sp[0]) if isinstance(sp, list) else float(sp)

    return data, img_2d, pixel_spacing


def find_fat_water_image(w_nii_path: str, slice_idx: int = None):
    """
    根据压脂图路径，在同一患者目录下找到最佳配对的压水图（F 序列）。

    配对规则：
      1. 从 metadata series_description 最后一个'_'之后字符判断类型(W/F)
      2. 必须是 T2 Dixon 序列（series_description 同时含 't2' 和 'dixon'，不区分大小写）
      3. 患者 ID 交叉校验（metadata patient_id 一致）
      4. 同前缀优先（series_description 去掉末尾 _W/_F 后的前缀相同）
      5. 多对同前缀时，按 series number 差值最小的 F 与当前 W 配对

    返回：
        (f_img_2d, f_meta, f_nii_path) 或 (None, None, None)
    """
    w_path      = Path(w_nii_path)
    seq_dir     = w_path.parent
    patient_dir = w_path.parent.parent

    # 读取压脂图自身 metadata
    w_meta_path = seq_dir / 'metadata.json'
    w_meta = {}
    if w_meta_path.exists():
        with open(w_meta_path, 'r') as fp:
            w_meta = json.load(fp)
    w_pid     = w_meta.get('patient_info', {}).get('patient_id', '')
    w_desc    = w_meta.get('series_info', {}).get('series_description', '')
    w_prefix  = _get_series_prefix(w_desc)
    w_ser_num = _get_series_number(seq_dir.name)

    print(f"\n搜索压水图(F序列): {patient_dir}")
    print(f"   压脂图: folder={seq_dir.name}, series='{w_desc}', "
          f"prefix='{w_prefix}', ser_num={w_ser_num}, pid='{w_pid}'")

    candidates = []
    for d in sorted(patient_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / 'metadata.json'
        nii_path  = d / 'scan.nii.gz'
        if not meta_path.exists() or not nii_path.exists():
            continue
        with open(meta_path, 'r') as fp:
            f_meta = json.load(fp)

        f_desc    = f_meta.get('series_info', {}).get('series_description', '')
        f_pid     = f_meta.get('patient_info', {}).get('patient_id', '')
        f_type    = _get_series_type(f_desc)
        f_prefix  = _get_series_prefix(f_desc)
        f_ser_num = _get_series_number(d.name)

        if f_type != 'F':
            continue
        if 't2' not in f_desc.lower() or 'dixon' not in f_desc.lower():
            print(f"   跳过 {d.name}: series='{f_desc}' 不是T2 Dixon序列")
            continue
        if w_pid and f_pid and w_pid != f_pid:
            print(f"   跳过 {d.name}: 患者ID不匹配 ({f_pid} != {w_pid})")
            continue

        prefix_match = 0 if f_prefix == w_prefix else 1
        ser_diff     = abs(f_ser_num - w_ser_num)
        candidates.append((prefix_match, ser_diff, d, f_meta, str(nii_path)))
        print(f"   候选 {d.name}: series='{f_desc}', "
              f"prefix_match={prefix_match==0}, ser_diff={ser_diff}")

    if not candidates:
        print("   [WARN] 未找到压水图F序列")
        return None, None, None

    candidates.sort(key=lambda x: (x[0], x[1]))
    best = candidates[0]
    d_best, f_meta_best, nii_best = best[2], best[3], best[4]
    print(f"   [OK] 最优配对压水图: {d_best.name} "
          f"(prefix_match={best[0]==0}, ser_diff={best[1]})")

    f_img  = nib.load(nii_best).get_fdata()
    n_f    = f_img.shape[2]
    if slice_idx is not None and 0 <= slice_idx < n_f:
        use_idx = slice_idx
        print(f"   压水图使用传入切片索引={use_idx}")
    else:
        use_idx = n_f // 2
        print(f"   压水图退回默认中间切片索引={use_idx}")
    f_img_2d = f_img[:, :, use_idx].astype(np.float32)
    return f_img_2d, f_meta_best, nii_best
