"""
LSMATOOLS - Basic Usage Example
LSMATOOLS 基础使用示例
"""

import os
from LSMATOOLS_CP.main import test_single_image, batch_process


def example_single_case():
    """
    单病例处理示例 - Single Case Processing Example
    
    演示如何处理单个 NIfTI 文件
    """
    # 方法 1: 使用交互式菜单
    print("=" * 60)
    print("LSMATOOLS V15 - 单病例处理")
    print("=" * 60)
    print("\n提示：运行 python main.py 选择模式 1 进入交互式处理\n")
    
    # 方法 2: 直接调用函数
    test_single_image(
        nifti_path='input/T2_Dixon_W.nii.gz',  # NIfTI 文件路径
        metadata_path='input/metadata.json',    # Metadata JSON（可选）
        output_dir='output/single_case'         # 输出目录
    )
    
    print("✓ 单病例处理完成")
    print(f"输出目录：output/single_case")


def example_batch_processing():
    """
    批量处理示例 - Batch Processing Example
    
    演示如何批量处理多个病例
    """
    input_directory = 'patients/'      # 输入目录（包含多个子目录）
    output_directory = 'batch_results/' # 输出目录
    
    print("=" * 60)
    print("LSMATOOLS V15 - 批量处理")
    print("=" * 60)
    print(f"\n输入目录：{input_directory}")
    print(f"输出目录：{output_directory}\n")
    
    # 批量处理
    batch_process(
        input_dir=input_directory,
        output_dir=output_directory
    )
    
    print("\n✓ 批量处理完成")
    print(f"结果保存在：{output_directory}")


def example_custom_parameters():
    """
    自定义参数示例 - Custom Parameters Example
    
    演示如何调整关键算法参数
    """
    from LSMATOOLS_CP.config.parameters import DEFAULT_PARAMS
    
    # 复制默认参数
    params = DEFAULT_PARAMS.copy()
    
    # 调整前缘检测参数
    params['rise_ratio'] = 0.55       # 上升沿触发比例（默认 0.50）
    params['scan_mm'] = 45.0          # 扫描距离（默认 40.0mm）
    params['window_mm'] = 7.0         # 密集窗口宽度（默认 6.0mm）
    
    # 注意：当前版本通过修改配置文件或源代码使用自定义参数
    # 完整 API 封装将在后续版本中提供
    
    print("自定义参数配置:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    return params


def example_output_files():
    """
    输出文件说明 - Output Files Description
    
    展示 LSMATOOLS 生成的所有输出文件
    """
    output_dir = 'output_example'
    
    print("=" * 60)
    print("LSMATOOLS 输出文件说明")
    print("=" * 60)
    
    files_description = {
        'TRACED.png': '可视化结果图（双图对比：左压脂/右压水）',
        'vertebra_masks.nii.gz': '多标签椎体掩模（NIfTI 格式）',
        'geometric_data.csv': '椎体几何数据 CSV（面积、角度、高度）',
        'analysis_log.json': '分析日志（包含所有中间结果和参数）'
    }
    
    for filename, description in files_description.items():
        print(f"\n{filename}:")
        print(f"  {description}")
    
    print("\n提示：所有输出文件均保存在指定的 output_dir 目录下")


if __name__ == '__main__':
    print("=" * 60)
    print("LSMATOOLS V15 - 使用示例")
    print("=" * 60)
    print()
    
    # 示例 1: 单病例处理
    print("【示例 1】单病例处理")
    print("-" * 60)
    # 取消注释以运行（需准备实际数据）
    # example_single_case()
    print("提示：取消注释 example_single_case() 运行单病例处理\n")
    
    # 示例 2: 批量处理
    print("【示例 2】批量处理")
    print("-" * 60)
    # example_batch_processing()
    print("提示：取消注释 example_batch_processing() 运行批量处理\n")
    
    # 示例 3: 自定义参数
    print("【示例 3】自定义参数")
    print("-" * 60)
    # custom_params = example_custom_parameters()
    print("提示：取消注释 example_custom_parameters() 查看参数配置\n")
    
    # 示例 4: 输出文件说明
    print("【示例 4】输出文件说明")
    print("-" * 60)
    example_output_files()
    
    print("\n" + "=" * 60)
    print("更多文档请查看 docs/ 目录:")
    print("  - docs/algorithm_design.md  (算法设计文档)")
    print("  - docs/api_reference.md     (API 参考文档)")
    print("  - README.md                 (项目说明)")
    print("=" * 60)
