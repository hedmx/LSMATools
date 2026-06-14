"""CLI 入口: python -m analysis.run_analysis --input-dir <path>"""

import os, sys, argparse

# 兼容直接运行和模块运行两种方式
if __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis.dice_evaluator import DiceEvaluator
    from analysis.geom_evaluator import GeomEvaluator
    from analysis.report_generator import generate_report
else:
    from .dice_evaluator import DiceEvaluator
    from .geom_evaluator import GeomEvaluator
    from .report_generator import generate_report

def main():
    parser = argparse.ArgumentParser(description='LSMATools_WIFS 分割性能评估')
    parser.add_argument('--input-dir', required=True, help='病例输出目录（含 *_seg.nii.gz, Roigstd.zip, *_geom.csv）')
    parser.add_argument('--spacing', type=float, default=0.9375, help='像素间距 mm')
    parser.add_argument('--skip-geom', action='store_true', help='跳过形态学分析')
    parser.add_argument('--skip-report', action='store_true', help='跳过报告生成')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: {args.input_dir} 不存在")
        sys.exit(1)

    print("=" * 60)
    print(f"LSMATools_WIFS 分割性能评估")
    print(f"输入: {args.input_dir}")
    print("=" * 60)

    # Step 1: DICE + mask metrics
    print("\n[1/3] 计算掩膜指标 (DICE/IoU/HD95/ASD)...")
    de = DiceEvaluator(args.input_dir, args.spacing)
    de.run()
    print(f"      完成: {len(de.detail_rows)} 条记录, {len(de.case_names)} 例")

    # Step 2: Geom analysis (optional)
    if not args.skip_geom:
        print("\n[2/3] 计算形态学指标 (高度 ICC)...")
        ge = GeomEvaluator(args.input_dir)
        ge.run()
        print("      完成")

    # Step 3: Report
    if not args.skip_report:
        print("\n[3/3] 生成报告...")
        generate_report(args.input_dir)
        print("      完成")

    print(f"\n{'='*60}")
    print(f"全部完成。输出在: {args.input_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
