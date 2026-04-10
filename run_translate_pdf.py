#!/usr/bin/env python3
"""向后兼容薄壳:核心逻辑已搬到 gjr/translate_pipeline.py。

用法:
    python3 run_translate_pdf.py input.pdf [-o output_dir]
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gjr.translate_pipeline import process_translate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="工程图纸翻译")
    parser.add_argument("input", help="输入 PDF 路径")
    parser.add_argument("-o", "--output", help="输出目录", default=None)
    parser.add_argument("-p", "--page", type=int, default=1, help="要翻译的页码(1-based,默认 1)")
    parser.add_argument("-d", "--debug-log", action="store_true", help="写 API 调用日志到 api_log/")
    args = parser.parse_args()

    pdf_path = Path(args.input).resolve()
    if not pdf_path.exists():
        print(f"文件不存在: {pdf_path}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else pdf_path.parent / "output"
    num_blocks, num_trans, output_pdf = process_translate(
        pdf_path, output_dir, page_num=args.page - 1, debug_log=args.debug_log,
    )

    print(f"\n{'='*50}")
    print(f"完成: {pdf_path.name}")
    print(f"Block: {num_blocks} 个, 翻译: {num_trans} 个")
    print(f"输出: {output_pdf}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
