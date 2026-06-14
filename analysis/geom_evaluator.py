"""形态学对比: 椎体垂线高度 ICC (算法 vs Gold1, 算法 vs Gold2)"""

import os, csv, json, tempfile, zipfile
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
import roifile as rf
from skimage.draw import polygon as skpoly
from .icc_utils import icc_2way, icc_ci, rating

LABEL_MAP = {'S1':1,'L5':2,'L4':3,'L3':4,'L2':5,'L1':6,'T12':7,'T11':8,'CANAL':15}
VERTEBRAE = ['S1','L5','L4','L3','L2','L1','T12','T11','CANAL']


class GeomEvaluator:

    def __init__(self, input_dir: str):
        self.input_dir = input_dir

    def _spacing(self, case_dir):
        meta_path = os.path.join(case_dir, 'metadata.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f: m = json.load(f)
                sp = m.get('acquisition_params',{}).get('pixel_spacing_mm',[0.9375])
                return float(sp[0]) if isinstance(sp, list) else float(sp)
            except: pass
        return 0.9375

    def _load_roi_zip(self, zip_path, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        if not os.path.exists(zip_path): return mask
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(zip_path, 'r') as zf: zf.extractall(td)
            for rp in Path(td).glob('*.roi'):
                lbl = LABEL_MAP.get(rp.stem)
                if lbl is None: continue
                try:
                    obj = rf.ImagejRoi.fromfile(str(rp))
                    coords = obj.coordinates()
                    rr, cc = skpoly(coords[:,1], coords[:,0], shape=shape)
                    m = np.zeros(shape, dtype=np.uint8); m[rr, cc] = 1
                    mask[m > 0] = lbl
                except: pass
        return mask

    def _vertical_height(self, mask, col_center, spacing):
        if col_center is None: return None
        c = int(round(col_center))
        if c < 0 or c >= mask.shape[1]: return None
        rows_with_mask = np.where(mask[:, c])[0]
        if len(rows_with_mask) < 2: return None
        return (rows_with_mask[-1] - rows_with_mask[0] + 1) * spacing

    def _multi_col_height(self, mask, spacing):
        """多列垂线均值高度: 取5列(10%/30%/50%/70%/90%宽度)分别做垂线, 取均值"""
        c_idx = np.where(mask.any(axis=0))[0]
        if len(c_idx) < 3: return None
        vals = []
        for pct in [30, 50, 70]:
            c = int(round(np.percentile(c_idx, pct)))
            h = self._vertical_height(mask, c, spacing)
            if h is not None: vals.append(h)
        if len(vals) < 3: return None
        return np.mean(vals)

    def run(self):
        rows = []
        for case_dir in sorted(Path(self.input_dir).iterdir()):
            if not case_dir.is_dir(): continue
            sample = case_dir.name
            geom_files = list(case_dir.glob('*_geom.csv'))
            seg_files  = list(case_dir.glob('*_seg.nii.gz'))
            if not geom_files or not seg_files: continue

            cdir = str(case_dir)
            seg_img = nib.load(str(seg_files[0]))
            seg_data = seg_img.get_fdata()
            if seg_data.ndim == 3 and seg_data.shape[2] == 1:
                seg_data = seg_data[:,:,0]
            shape = seg_data.shape
            sp = self._spacing(cdir)

            gold1 = self._load_roi_zip(os.path.join(cdir,'Roigstd.zip'), shape)
            gold2 = self._load_roi_zip(os.path.join(cdir,'Roigstd1.zip'), shape)

            try: gdf = pd.read_csv(str(geom_files[0]))
            except: continue

            for _, g in gdf.iterrows():
                level = str(g.get('level','')).strip()
                if level not in VERTEBRAE or level == 'CANAL': continue
                lbl = LABEL_MAP.get(level)

                if lbl and np.any(gold1 == lbl):
                    gc_idx = np.where(gold1 == lbl)[1]
                    cc = gc_idx.mean()
                    row = {'sample': sample, 'vertebra': level,
                           'height_seg': round(self._vertical_height(seg_data==lbl, cc, sp), 2),
                           'height_gold1': round(self._vertical_height(gold1==lbl, cc, sp), 2),
                           'height_gold2': round(self._vertical_height(gold2==lbl, cc, sp), 2) if np.any(gold2==lbl) else None,
                           'multi_h_seg': round(self._multi_col_height(seg_data==lbl, sp), 2) if self._multi_col_height(seg_data==lbl, sp) else None,
                           'multi_h_gold1': round(self._multi_col_height(gold1==lbl, sp), 2) if self._multi_col_height(gold1==lbl, sp) else None,
                           'multi_h_gold2': round(self._multi_col_height(gold2==lbl, sp), 2) if np.any(gold2==lbl) and self._multi_col_height(gold2==lbl, sp) else None,
                    }
                    rows.append(row)

        self._save(rows)
        return self

    def _save(self, rows):
        base = self.input_dir
        fields = ['sample','vertebra','height_seg','height_gold1','height_gold2',
                  'multi_h_seg','multi_h_gold1','multi_h_gold2']
        with open(os.path.join(base,'geom_comparison.csv'),'w',newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader(); w.writerows(rows)

        icc_rows = []
        # 单列垂线高度 ICC
        for pair_label, gold_col in [('vs Gold1','height_gold1'),('vs Gold2','height_gold2')]:
            vals, wts = [], []
            for vn in VERTEBRAE:
                if vn == 'CANAL': continue
                vp = [(r['height_seg'], r[gold_col]) for r in rows
                      if r['vertebra']==vn and r.get('height_seg') and r.get(gold_col)
                      and not (isinstance(r['height_seg'],float) and np.isnan(r['height_seg']))]
                if len(vp)<3: continue
                d=np.array(vp,dtype=float); v=icc_2way(d); lo,hi=icc_ci(v,len(d),2)
                icc_rows.append({'metric':f'{vn} 单列垂线 ICC ({pair_label})',
                                 'ICC':round(v,4),'CI95':f'[{lo:.4f},{hi:.4f}]',
                                 'n':len(d),'rating':rating(v)})
                vals.append(v); wts.append(len(vp))
            if vals:
                wavg = sum(v*w for v,w in zip(vals,wts))/sum(wts)
                icc_rows.append({'metric':f'单列垂线 ICC 加权均值 ({pair_label})',
                                 'ICC':round(wavg,4),'CI95':'—','n':sum(wts),
                                 'rating':rating(wavg)})
        # 多列垂线均值高度 ICC
        for pair_label, gold_col in [('vs Gold1','multi_h_gold1'),('vs Gold2','multi_h_gold2')]:
            vals, wts = [], []
            for vn in VERTEBRAE:
                if vn == 'CANAL': continue
                vp = [(r['multi_h_seg'], r[gold_col]) for r in rows
                      if r.get('multi_h_seg') and r.get(gold_col)
                      and r['vertebra']==vn
                      and not (isinstance(r['multi_h_seg'],float) and np.isnan(r['multi_h_seg']))]
                if len(vp)<3: continue
                d=np.array(vp,dtype=float); v=icc_2way(d); lo,hi=icc_ci(v,len(d),2)
                icc_rows.append({'metric':f'{vn} 多列垂线 ICC ({pair_label})',
                                 'ICC':round(v,4),'CI95':f'[{lo:.4f},{hi:.4f}]',
                                 'n':len(d),'rating':rating(v)})
                vals.append(v); wts.append(len(vp))
            if vals:
                wavg = sum(v*w for v,w in zip(vals,wts))/sum(wts)
                icc_rows.append({'metric':f'多列垂线 ICC 加权均值 ({pair_label})',
                                 'ICC':round(wavg,4),'CI95':'—','n':sum(wts),
                                 'rating':rating(wavg)})
        if icc_rows:
            with open(os.path.join(base,'icc_height_summary.csv'),'w',newline='') as f:
                w = csv.DictWriter(f, fieldnames=['metric','ICC','CI95','n','rating'])
                w.writeheader(); w.writerows(icc_rows)
