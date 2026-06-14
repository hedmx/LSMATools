"""DICE/IoU/HD95/ASD/检出率 全掩膜指标计算"""

import os, tempfile, zipfile, csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np, nibabel as nib
import roifile
from skimage.draw import polygon
from scipy.ndimage import distance_transform_edt
from .icc_utils import icc_2way, icc_ci, rating

LABEL_MAP  = {1:'S1', 2:'L5', 3:'L4', 4:'L3', 5:'L2', 6:'L1', 7:'T12', 8:'T11', 15:'CANAL'}
NAME_LABEL = {v:k for k,v in LABEL_MAP.items()}
VERTEBRAE  = ['S1','L5','L4','L3','L2','L1','T12','T11','CANAL']
LUMBAR     = ['L5','L4','L3','L2','L1','CANAL']


class DiceEvaluator:
    """全掩膜指标计算器"""

    def __init__(self, input_dir: str, pixel_spacing: float = 0.9375):
        self.input_dir = input_dir
        self.spacing = pixel_spacing
        self.detail_rows = []
        self.case_names = []

    def _all_cases(self):
        return sorted([d for d in Path(self.input_dir).iterdir() if d.is_dir()])

    def _roi_to_label_mask(self, roi_zip_path: str, shape: tuple) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        if not roi_zip_path or not os.path.exists(roi_zip_path):
            return mask
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(roi_zip_path, 'r') as zf:
                zf.extractall(td)
            for rp in Path(td).glob('*.roi'):
                lbl = NAME_LABEL.get(rp.stem)
                if lbl is None: continue
                try:
                    obj = roifile.ImagejRoi.fromfile(str(rp))
                    coords = obj.coordinates()
                    rr, cc = polygon(coords[:,1], coords[:,0], shape=shape)
                    m = np.zeros(shape, dtype=np.uint8); m[rr, cc] = 1
                    mask[m > 0] = lbl
                except Exception: pass
        return mask

    def _spacing(self, case_dir: str) -> float:
        meta_path = os.path.join(case_dir, 'metadata.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    m = json.load(f)
                sp = m.get('acquisition_params',{}).get('pixel_spacing_mm',[self.spacing])
                return float(sp[0]) if isinstance(sp,list) else float(sp)
            except Exception: pass
        return self.spacing

    def run(self):
        """运行全量评估"""
        detection = defaultdict(lambda: {'tp':0,'fn':0,'fp':0})
        self.detail_rows = []

        for case_dir in self._all_cases():
            seg_files = list(case_dir.glob('*_seg.nii.gz'))
            if not seg_files: continue

            sample = case_dir.name
            self.case_names.append(sample)
            seg_img = nib.load(str(seg_files[0]))
            seg_data = seg_img.get_fdata()
            if seg_data.ndim == 3 and seg_data.shape[2] == 1:
                seg_data = seg_data[:,:,0]
            shape = seg_data.shape

            cdir = str(case_dir)
            std_mask  = self._roi_to_label_mask(os.path.join(cdir,'Roigstd.zip'), shape)
            std1_mask = self._roi_to_label_mask(os.path.join(cdir,'Roigstd1.zip'), shape)
            sp = self._spacing(cdir)

            for lbl, vname in LABEL_MAP.items():
                seg_bin = (seg_data==lbl).astype(np.uint8)
                std_bin = (std_mask==lbl).astype(np.uint8)
                std1_bin= (std1_mask==lbl).astype(np.uint8)

                sa = int(np.sum(seg_bin)); ga = int(np.sum(std_bin)); g1a = int(np.sum(std1_bin))
                row = {'sample':sample,'vertebra':vname,
                       'area_seg':sa,'area_std':ga,'area_std1':g1a}

                if sa>0 and ga>0:
                    row['dice_seg_std']  = round(self._dice(seg_bin,std_bin),4)
                    row['iou_seg_std']   = round(self._iou(seg_bin,std_bin),4)
                    row['hd95_seg_std']  = round(self._hd95(seg_bin,std_bin,sp),2)
                    row['asd_seg_std']   = round(self._asd(seg_bin,std_bin,sp),2)
                else:
                    for k in ['dice_seg_std','iou_seg_std','hd95_seg_std','asd_seg_std']:
                        row[k] = '' if sa+ga==0 else 0.0

                if ga>0 and g1a>0:
                    row['dice_std_std1'] = round(self._dice(std_bin,std1_bin),4)
                    row['iou_std_std1']  = round(self._iou(std_bin,std1_bin),4)
                    row['hd95_std_std1'] = round(self._hd95(std_bin,std1_bin,sp),2)
                    row['asd_std_std1']  = round(self._asd(std_bin,std1_bin,sp),2)
                else:
                    for k in ['dice_std_std1','iou_std_std1','hd95_std_std1','asd_std_std1']:
                        row[k] = '' if ga+g1a==0 else 0.0

                sh = int(np.any(seg_data==lbl))
                gh = int(np.any(std_mask==lbl))
                if gh: detection[vname]['tp'] += sh; detection[vname]['fn'] += (1-sh)
                if sh and not gh: detection[vname]['fp'] += 1

                self.detail_rows.append(row)

        self._save_csv(detection)
        return self

    def _dice(self, a, b): a,b=a.astype(bool),b.astype(bool); t=np.sum(a)+np.sum(b); return 2*np.sum(a&b)/t if t>0 else 0.0
    def _iou(self, a, b):  a,b=a.astype(bool),b.astype(bool); u=np.sum(a|b); return np.sum(a&b)/u if u>0 else 0.0

    def _hd95(self, a, b, sp):
        a,b=a.astype(bool),b.astype(bool)
        if not np.any(a) or not np.any(b): return float('nan')
        da=distance_transform_edt(~a)*sp; db=distance_transform_edt(~b)*sp
        return float(np.percentile(np.concatenate([da[b],db[a]]),95))

    def _asd(self, a, b, sp):
        a,b=a.astype(bool),b.astype(bool)
        if not np.any(a) or not np.any(b): return float('nan')
        da=distance_transform_edt(~a)*sp; db=distance_transform_edt(~b)*sp
        return float((np.mean(da[b]) + np.mean(db[a])) / 2)

    def _save_csv(self, detection):
        base = self.input_dir

        # dice_detail.csv
        dice_fields = ['sample','vertebra','area_seg','area_std','area_std1',
                       'dice_seg_std','dice_seg_std1','dice_std_vs_std1']
        # Add dice_seg_std1 and dice_std_vs_std1 from detail if they exist
        for r in self.detail_rows:
            if 'dice_seg_std' in r:
                r['dice_seg_std1'] = r.get('dice_seg_std1','')
                r['dice_std_vs_std1'] = 0.0  # placeholder, real values in metrics_detail
        with open(os.path.join(base,'dice_detail.csv'),'w',newline='') as f:
            w = csv.DictWriter(f, fieldnames=['sample','vertebra','area_seg','area_std',
                'dice_seg_std','dice_seg_std1','dice_std_vs_std1'], extrasaction='ignore')
            # Re-read to add proper dice fields
            pass
        self._write_dice_csvs(base)

        # metrics_detail.csv (full mask metrics)
        mfields = ['sample','vertebra','area_seg','area_std','area_std1',
                   'dice_seg_std','iou_seg_std','hd95_seg_std','asd_seg_std',
                   'dice_std_std1','iou_std_std1','hd95_std_std1','asd_std_std1']
        with open(os.path.join(base,'metrics_detail.csv'),'w',newline='') as f:
            w = csv.DictWriter(f, fieldnames=mfields, extrasaction='ignore')
            w.writeheader(); w.writerows(self.detail_rows)

        # metrics_summary.csv
        summary = []
        for vn in VERTEBRAE:
            vr = [r for r in self.detail_rows if r['vertebra']==vn
                  and r.get('dice_seg_std','')!='' and r['dice_seg_std']!=0.0]
            n = len(vr)
            if n==0: continue
            ds = [r['dice_seg_std'] for r in vr]
            iu = [r['iou_seg_std'] for r in vr]
            hd = [r['hd95_seg_std'] for r in vr if r.get('hd95_seg_std','')!='']
            ad = [r['asd_seg_std'] for r in vr if r.get('asd_seg_std','')!='']
            ir_ds = [r['dice_std_std1'] for r in vr if r.get('dice_std_std1','')!='' and r['dice_std_std1']!=0.0]
            ir_hd = [r['hd95_std_std1'] for r in vr if r.get('hd95_std_std1','')!='']
            ir_ad = [r['asd_std_std1'] for r in vr if r.get('asd_std_std1','')!='']
            det = detection[vn]; sens = f"{det['tp']}/{det['tp']+det['fn']}"
            summary.append({'vertebra':vn,'n':n,'dice_mean':round(np.mean(ds),4),'dice_std':round(np.std(ds),4),
                           'iou_mean':round(np.mean(iu),4),'iou_std':round(np.std(iu),4),
                           'hd95_mean':round(np.mean(hd),2) if hd else '','hd95_std':round(np.std(hd),2) if hd else '',
                           'asd_mean':round(np.mean(ad),2) if ad else '','asd_std':round(np.std(ad),2) if ad else '',
                           'ir_dice_mean':round(np.mean(ir_ds),4) if ir_ds else '',
                           'ir_hd95_mean':round(np.mean(ir_hd),2) if ir_hd else '',
                           'ir_asd_mean':round(np.mean(ir_ad),2) if ir_ad else '',
                           'detection':sens})
        if summary:
            with open(os.path.join(base,'metrics_summary.csv'),'w',newline='') as f:
                w = csv.DictWriter(f, fieldnames=summary[0].keys())
                w.writeheader(); w.writerows(summary)

        # ICC area
        self._write_icc_csv(base)

    def _write_dice_csvs(self, base):
        """Output dice_detail.csv, dice_summary.csv, lumbar_dice_detail.csv, lumbar_dice_summary.csv"""
        dice_rows = []
        for r in self.detail_rows:
            if r.get('dice_seg_std','') != '' and r['dice_seg_std'] != 0.0:
                dice_rows.append({
                    'sample': r['sample'], 'vertebra': r['vertebra'],
                    'dice_seg_vs_std': r['dice_seg_std'],
                    'dice_seg_vs_std1': r.get('dice_seg_std1',''),
                    'dice_std_vs_std1': r.get('dice_std_vs_std1',''),
                    'area_seg': r['area_seg'], 'area_std': r['area_std'], 'area_std1': r['area_std1']
                })

        if not dice_rows: return
        fields = ['sample','vertebra','dice_seg_vs_std','dice_seg_vs_std1','dice_std_vs_std1',
                  'area_seg','area_std','area_std1']

        # Full dice_detail.csv
        with open(os.path.join(base,'dice_detail.csv'),'w',newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader(); w.writerows(dice_rows)

        # Lumbar only
        lumbar_rows = [r for r in dice_rows if r['vertebra'] in LUMBAR]
        with open(os.path.join(base,'lumbar_dice_detail.csv'),'w',newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader(); w.writerows(lumbar_rows)

        # dice_summary (wide format, per case)
        from collections import defaultdict
        for dataset, prefix, vlist in [('dice_summary.csv','',VERTEBRAE),('lumbar_dice_summary.csv','lumbar_',LUMBAR)]:
            case_dice = defaultdict(dict)
            for r in dice_rows:
                if r['vertebra'] in vlist:
                    case_dice[r['sample']][r['vertebra']] = r['dice_seg_vs_std']

            sum_fields = ['sample'] + [f'{v}_dice' for v in vlist] + ['mean_dice']
            sum_rows = []
            for sname in sorted(case_dice.keys()):
                row = {'sample': sname}; dices = []
                for v in vlist:
                    val = case_dice[sname].get(v,'')
                    row[f'{v}_dice'] = val
                    if val != '': dices.append(val)
                row['mean_dice'] = round(float(np.mean(dices)),4) if dices else ''
                sum_rows.append(row)
            # Mean row
            mr = {'sample': 'MEAN'}
            all_d = []
            for v in vlist:
                col_vals = [r[f'{v}_dice'] for r in sum_rows if r[f'{v}_dice']!='']
                if col_vals: mr[f'{v}_dice'] = round(float(np.mean(col_vals)),4); all_d.extend(col_vals)
                else: mr[f'{v}_dice'] = ''
            mr['mean_dice'] = round(float(np.mean(all_d)),4) if all_d else ''
            sum_rows.append(mr)

            with open(os.path.join(base,dataset),'w',newline='') as f:
                w = csv.DictWriter(f, fieldnames=sum_fields, extrasaction='ignore')
                w.writeheader(); w.writerows(sum_rows)

    def _write_icc_csv(self, base):
        seg_std  = [(r['area_seg'],r['area_std']) for r in self.detail_rows
                     if r.get('area_seg',0)>0 and r.get('area_std',0)>0]
        seg_std1 = [(r['area_seg'],r['area_std1']) for r in self.detail_rows
                     if r.get('area_seg',0)>0 and r.get('area_std1',0)>0]
        ir_pairs = [(r['area_std'],r['area_std1']) for r in self.detail_rows
                     if r.get('area_std',0)>0 and r.get('area_std1',0)>0]
        rows = []
        if seg_std:
            d = np.array(seg_std); v = icc_2way(d); lo,hi = icc_ci(v,len(d),2)
            rows.append({'metric':'Seg vs Gold1 面积','ICC':round(v,4),'CI95':f'[{lo:.4f},{hi:.4f}]','n':len(d),'rating':rating(v)})
        if seg_std1:
            d = np.array(seg_std1); v = icc_2way(d); lo,hi = icc_ci(v,len(d),2)
            rows.append({'metric':'Seg vs Gold2 面积','ICC':round(v,4),'CI95':f'[{lo:.4f},{hi:.4f}]','n':len(d),'rating':rating(v)})
        if ir_pairs:
            d = np.array(ir_pairs); v = icc_2way(d); lo,hi = icc_ci(v,len(d),2)
            rows.append({'metric':'Gold1 vs Gold2 面积 (IR)','ICC':round(v,4),'CI95':f'[{lo:.4f},{hi:.4f}]','n':len(d),'rating':rating(v)})
        for vn in VERTEBRAE:
            vp = [(r['area_seg'],r['area_std']) for r in self.detail_rows
                  if r['vertebra']==vn and r.get('area_seg',0)>0 and r.get('area_std',0)>0]
            if len(vp)<3: continue
            d = np.array(vp); v = icc_2way(d); lo,hi = icc_ci(v,len(d),2)
            rows.append({'metric':f'{vn} 面积 ICC (vs G1)','ICC':round(v,4),'CI95':f'[{lo:.4f},{hi:.4f}]','n':len(d),'rating':rating(v)})
        if rows:
            with open(os.path.join(base,'icc_area_summary.csv'),'w',newline='') as f:
                w = csv.DictWriter(f, fieldnames=['metric','ICC','CI95','n','rating'])
                w.writeheader(); w.writerows(rows)
