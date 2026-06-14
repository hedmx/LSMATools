"""报告生成: CSV → Markdown + HTML"""

import os, numpy as np, pandas as pd

VERTEBRAE = ['S1','L5','L4','L3','L2','L1','T12','T11','CANAL']
LUMBAR = ['L5','L4','L3','L2','L1','CANAL']


def generate_report(input_dir: str, output_dir: str = None):
    """从 CSV 生成 .md + .html 报告"""
    if output_dir is None:
        output_dir = input_dir

    metrics = _read_csv(os.path.join(input_dir, 'metrics_summary.csv'))
    dice = _read_csv(os.path.join(input_dir, 'dice_summary.csv'))
    detail = _read_csv(os.path.join(input_dir, 'dice_detail.csv'))
    icc_area = _read_csv(os.path.join(input_dir, 'icc_area_summary.csv'))
    icc_height = _read_csv(os.path.join(input_dir, 'icc_height_summary.csv'))
    geom = _read_csv(os.path.join(input_dir, 'geom_comparison.csv'))

    n_cases = _count_cases(detail)
    n_verts = len(detail) if detail is not None else 0

    # Build Markdown
    md = _build_md(n_cases, n_verts, metrics, dice, detail, icc_area, icc_height, geom)
    md_path = os.path.join(output_dir, '分析报告.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)

    # Build HTML
    html = _build_html(n_cases, n_verts, metrics, dice, detail, icc_area, icc_height, geom)
    html_path = os.path.join(output_dir, '分析报告.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"报告: {md_path}\n     {html_path}")


def _read_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def _count_cases(detail):
    if detail is None: return 0
    return detail['sample'].nunique()


def _fmt(v, d=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v:.{d}f}'


def _build_md(n_cases, n_verts, metrics, dice, detail, icc_a, icc_h, geom):
    lines = [
        f'# LSMATools_WIFS 分割性能分析报告',
        f'',
        f'**样本量**: {n_cases} 例 | **椎体数**: {n_verts} | **双金标准** (Roigstd + Roigstd1)',
        f'',
        '## 一、DICE 重叠度',
    ]

    # DICE table from dice_summary
    if dice is not None and 'MEAN' in dice['sample'].values:
        mean_row = dice[dice['sample'] == 'MEAN']
        lines.append('')
        lines.append('| 椎体 | DICE (Seg vs Gold) |')
        lines.append('|------|:---:|')
        dice_cols = [c for c in dice.columns if c.endswith('_dice') and c != 'mean_dice']
        for col in dice_cols:
            vname = col.replace('_dice', '')
            val = mean_row[col].values[0] if col in mean_row.columns else '—'
            lines.append(f'| {vname} | {val} |')
        # Overall mean
        if 'mean_dice' in mean_row.columns:
            lines.append(f'| **总体** | **{mean_row["mean_dice"].values[0]}** |')

    # Metrics table
    if metrics is not None:
        lines.append('')
        lines.append('## 二、全掩膜指标')
        lines.append('')
        lines.append('| 椎体 | n | DICE | IoU | HD95(mm) | ASD(mm) | IR_DICE | 检出率 |')
        lines.append('|------|:--:|:---:|:---:|:---:|:---:|:---:|:---:|')
        for _, r in metrics.iterrows():
            lines.append(f'| {r["vertebra"]} | {int(r["n"])} | {_fmt(r.get("dice_mean"))} | {_fmt(r.get("iou_mean"))} | {_fmt(r.get("hd95_mean"),2)} | {_fmt(r.get("asd_mean"),2)} | {_fmt(r.get("ir_dice_mean"))} | {r.get("detection","—")} |')

    # ICC
    if icc_a is not None:
        lines.append('')
        lines.append('## 三、ICC (面积)')
        lines.append('')
        lines.append('| 指标 | ICC | 95% CI | n | 评级 |')
        lines.append('|------|:---:|------|:--:|:--:|')
        for _, r in icc_a.iterrows():
            lines.append(f'| {r["metric"]} | {r["ICC"]} | {r["CI95"]} | {int(r["n"])} | {r["rating"]} |')

    if icc_h is not None:
        lines.append('')
        lines.append('## 四、ICC (高度)')
        lines.append('')
        lines.append('| 指标 | ICC | 95% CI | n | 评级 |')
        lines.append('|------|:---:|------|:--:|:--:|')
        for _, r in icc_h.iterrows():
            lines.append(f'| {r["metric"]} | {r["ICC"]} | {r["CI95"]} | {int(r["n"])} | {r["rating"]} |')

    lines.append('')
    lines.append('---')
    lines.append('*报告由 analysis 模块自动生成*')

    return '\n'.join(lines)


def _build_html(n_cases, n_verts, metrics, dice, detail, icc_a, icc_h, geom):
    body = f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>LSMATools_WIFS 分析报告</title>
<style>
body{{font-family:-apple-system,'Microsoft YaHei',sans-serif;background:#f5f5f5;color:#333;max-width:960px;margin:0 auto;padding:20px}}
h1{{background:linear-gradient(135deg,#1a365d,#2c5282);color:#fff;padding:30px;border-radius:10px}}
h2{{color:#1a365d;border-bottom:2px solid #e2e8f0;padding-bottom:8px;margin-top:30px}}
table{{width:100%;border-collapse:collapse;margin:15px 0;background:#fff;border-radius:8px;overflow:hidden}}
th{{background:#edf2f7;padding:10px 8px;border:1px solid #e2e8f0;text-align:center}}
td{{padding:9px 8px;border:1px solid #e2e8f0;text-align:center}}
tr:nth-child(even){{background:#f7fafc}}
.meta{{font-size:14px;opacity:.85;margin-top:-15px;margin-bottom:20px}}
</style></head><body>
<h1>LSMATools_WIFS 分割性能分析报告</h1>
<p class="meta">样本量: {n_cases} 例 | 椎体数: {n_verts} | 双金标准 (Roigstd + Roigstd1) | 自动生成</p>
'''

    # DICE
    if dice is not None:
        body += '<h2>一、DICE 重叠度</h2><table><tr><th>椎体</th><th>DICE (Seg vs Gold)</th></tr>'
        if 'MEAN' in dice['sample'].values:
            mr = dice[dice['sample']=='MEAN']
            for c in [c for c in dice.columns if c.endswith('_dice') and c!='mean_dice']:
                v = c.replace('_dice','')
                val = mr[c].values[0] if c in mr.columns else '—'
                body += f'<tr><td>{v}</td><td>{val}</td></tr>'
            if 'mean_dice' in mr.columns:
                body += f'<tr style="font-weight:700;background:#ebf8ff"><td>总体</td><td>{mr["mean_dice"].values[0]}</td></tr>'
        body += '</table>'

    # Metrics
    if metrics is not None:
        body += '<h2>二、全掩膜指标</h2><table><tr><th>椎体</th><th>n</th><th>DICE</th><th>IoU</th><th>HD95</th><th>ASD</th><th>IR_DICE</th><th>检出</th></tr>'
        for _, r in metrics.iterrows():
            body += f'<tr><td>{r["vertebra"]}</td><td>{int(r["n"])}</td><td>{_fmt(r.get("dice_mean"))}</td><td>{_fmt(r.get("iou_mean"))}</td><td>{_fmt(r.get("hd95_mean"),2)}</td><td>{_fmt(r.get("asd_mean"),2)}</td><td>{_fmt(r.get("ir_dice_mean"))}</td><td>{r.get("detection","—")}</td></tr>'
        body += '</table>'

    # ICC
    if icc_a is not None:
        body += '<h2>三、ICC (面积)</h2><table><tr><th>指标</th><th>ICC</th><th>95% CI</th><th>n</th><th>评级</th></tr>'
        for _, r in icc_a.iterrows():
            body += f'<tr><td>{r["metric"]}</td><td>{r["ICC"]}</td><td>{r["CI95"]}</td><td>{int(r["n"])}</td><td>{r["rating"]}</td></tr>'
        body += '</table>'

    if icc_h is not None:
        body += '<h2>四、ICC (高度)</h2><table><tr><th>指标</th><th>ICC</th><th>95% CI</th><th>n</th><th>评级</th></tr>'
        for _, r in icc_h.iterrows():
            body += f'<tr><td>{r["metric"]}</td><td>{r["ICC"]}</td><td>{r["CI95"]}</td><td>{int(r["n"])}</td><td>{r["rating"]}</td></tr>'
        body += '</table>'

    body += '</body></html>'
    return body
