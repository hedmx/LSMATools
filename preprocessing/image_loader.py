"""
图像加载模块
提供 NIfTI 文件加载和 W/IN 序列查找功能
"""
import json
import numpy as np
import nibabel as nib
from pathlib import Path

from .series_utils import (
    _get_series_type, _get_series_prefix, _get_series_number, _is_dixon_sequence
)


def load_nifti(nifti_path: str, slice_idx: int = None):
    """
    加载 NIfTI 文件，返回 (data_3d, affine, header, img_2d, pixel_spacing)。

    参数：
        nifti_path  - .nii.gz 文件路径
        slice_idx   - 若指定，则直接取该切片；否则取中间切片

    返回：
        data        - 3D numpy 数组 shape=(H,W,N)
        affine      - NIfTI affine 矩阵
        header      - NIfTI header 对象
        img_2d      - 选定切片的 2D float32 数组
        pixel_spacing - 像素间距（mm），从同目录 metadata.json 读取，默认 0.9375
    """
    img_obj = nib.load(nifti_path)
    data    = img_obj.get_fdata()
    affine  = img_obj.affine
    header  = img_obj.header
    n_sl    = data.shape[2]
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

    return data, affine, header, img_2d, pixel_spacing


def find_in_image(w_nii_path: str, slice_idx: int = None):
    """
    根据压脂图（W序列）路径，在同一患者目录下找到最佳配对的 IN（in-phase）序列。

    配对规则（三级排序）：
      1. imagepositionpatient 精确匹配优先（三分量差值均 < 1mm → 0，否则 1）
         ── 同一次采集的 W/IN/OPP/F 四个序列 imagepositionpatient 完全相同，
            可精确区分同目录下两组 Dixon 序列
      2. 同前缀次之（series_description 去掉末尾 _W/_IN 后的前缀相同 → 0）
      3. series number 差值最小兜底

    其他条件：
      - 必须是 T2 Dixon/WFI 序列
      - 患者 ID 交叉校验

    返回：
        (in_img_2d, in_meta, in_nii_path) 或 (None, None, None)
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
    w_pos     = w_meta.get('sampling_parameters', {}).get('imagepositionpatient', None)

    print(f"\n搜索IN序列: {patient_dir}")
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
            in_meta = json.load(fp)

        in_desc    = in_meta.get('series_info', {}).get('series_description', '')
        in_pid     = in_meta.get('patient_info', {}).get('patient_id', '')
        in_type    = _get_series_type(in_desc)
        in_prefix  = _get_series_prefix(in_desc)
        in_ser_num = _get_series_number(d.name)

        # 接受 IN 类型或后缀含 'in'/'ip' 的序列（联影设备用 IP 表示 in-phase）
        is_in_type = (in_type == 'IN') or ('_in' in in_desc.lower().split('_')[-1:])
        # 更宽松的 IN/IP 匹配：序列名末尾含 'in' 或 'ip'
        if in_desc and in_desc.rsplit('_', 1)[-1].lower() in ('in', 'inphase', 'in_phase', 'ip'):
            is_in_type = True

        if not is_in_type:
            continue
        if 't2' not in in_desc.lower() or not _is_dixon_sequence(in_desc):
            print(f"   跳过 {d.name}: series='{in_desc}' 不是T2 Dixon/WFI序列")
            continue
        if w_pid and in_pid and w_pid != in_pid:
            print(f"   跳过 {d.name}: 患者ID不匹配 ({in_pid} != {w_pid})")
            continue

        # 计算 imagepositionpatient 精确匹配
        #   0=同位置, 1=无位置数据(允许), 2=有数据但位置不同(拒绝)
        in_pos = in_meta.get('sampling_parameters', {}).get('imagepositionpatient', None)
        if w_pos and in_pos and len(w_pos) == 3 and len(in_pos) == 3:
            pos_exact = 0 if all(abs(w_pos[i] - in_pos[i]) < 1.0 for i in range(3)) else 2
        else:
            pos_exact = 1

        prefix_match = 0 if in_prefix == w_prefix else 1
        ser_diff     = abs(in_ser_num - w_ser_num)
        candidates.append((pos_exact, prefix_match, ser_diff, d, in_meta, str(nii_path)))
        print(f"   候选 {d.name}: series='{in_desc}', "
              f"pos_exact={pos_exact}, prefix_match={prefix_match == 0}, ser_diff={ser_diff}")

    if not candidates:
        # GE Flex fallback: 无标准IN时，查找同位置的非Dixon T2矢状位作为IN
        ge_candidates = []
        for d in sorted(patient_dir.iterdir()):
            if not d.is_dir():
                continue
            meta_path = d / 'metadata.json'
            nii_path = d / 'scan.nii.gz'
            if not meta_path.exists() or not nii_path.exists():
                continue
            with open(meta_path, 'r') as fp:
                ge_meta = json.load(fp)
            ge_desc = ge_meta.get('series_info', {}).get('series_description', '').lower()
            ge_pid = ge_meta.get('patient_info', {}).get('patient_id', '')
            
            # 必须: T2 + sagittal，排除自身(W序列)
            if 't2' not in ge_desc or 'sag' not in ge_desc:
                continue
            if d.name == seq_dir.name:
                continue
            if w_pid and ge_pid and w_pid != ge_pid:
                continue
            
            # 同位置检查
            ge_pos = ge_meta.get('sampling_parameters', {}).get('imagepositionpatient', None)
            if w_pos and ge_pos and len(w_pos) == 3 and len(ge_pos) == 3:
                if not all(abs(w_pos[i] - ge_pos[i]) < 1.0 for i in range(3)):
                    continue
            else:
                continue
            
            ge_ser = _get_series_number(d.name)
            ge_candidates.append((abs(ge_ser - w_ser_num), d, ge_meta, str(nii_path)))
            print(f"   [GE fallback] {d.name}: series='{ge_desc}' 同位置T2，候选IN替代")
        
        if ge_candidates:
            ge_candidates.sort(key=lambda x: x[0])
            _, d_ge, ge_meta_best, nii_ge = ge_candidates[0]
            print(f"   [OK] GE fallback 最优替代IN: {d_ge.name} (ser_diff={ge_candidates[0][0]})")
            in_img = nib.load(nii_ge).get_fdata()
            n_in = in_img.shape[2]
            use_idx = slice_idx if (slice_idx is not None and 0 <= slice_idx < n_in) else n_in // 2
            return in_img[:, :, use_idx].astype(np.float32), ge_meta_best, nii_ge
        
        print("   [WARN] 未找到IN序列")
        return None, None, None

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    best = candidates[0]

    if best[0] == 2:
        print(f"   [WARN] 最佳候选 {best[3].name} 与压脂图位置不同 "
              f"(pos_exact=2)，拒绝跨采集配对")
        return None, None, None

    d_best, in_meta_best, nii_best = best[3], best[4], best[5]
    print(f"   [OK] 最优配对IN序列: {d_best.name} "
          f"(pos_exact={best[0]}, prefix_match={best[1] == 0}, ser_diff={best[2]})")

    in_img  = nib.load(nii_best).get_fdata()
    n_in    = in_img.shape[2]
    if slice_idx is not None and 0 <= slice_idx < n_in:
        use_idx = slice_idx
    else:
        use_idx = n_in // 2
    in_img_2d = in_img[:, :, use_idx].astype(np.float32)
    return in_img_2d, in_meta_best, nii_best
