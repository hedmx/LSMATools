"""
日志导出模块

export_log: 将处理日志输出为纯文本 .txt 文件。
"""

import os
from datetime import datetime


def export_log(log_content: str, output_dir: str, stem: str) -> str:
    """
    将日志内容保存为纯文本文件。

    参数：
        log_content - 日志字符串内容
        output_dir  - 输出目录
        stem        - 文件名前缀

    返回：
        保存的文件路径

    输出：{stem}_log.txt
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{stem}_log.txt")
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"[处理时间] {ts}\n{'─' * 50}\n")
        f.write(log_content)
    print(f"   [log] 已保存: {os.path.basename(log_path)}")
    return log_path
