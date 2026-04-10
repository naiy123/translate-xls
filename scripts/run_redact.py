#!/usr/bin/env python3
"""工程图纸脱敏 CLI 入口。

用法:
    python3 scripts/run_redact.py input.pdf [-o output_dir] [--debug-preview]

或(从项目根):
    python3 -m scripts.run_redact input.pdf
"""
import argparse
import shutil
import sys
from pathlib import Path

# 允许直接 `python3 scripts/run_redact.py ...`:把项目根加到 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import fitz  # noqa: E402

from gjr.pipeline import process_page  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="工程图纸脱敏")
    parser.add_argument("input", help="输入 PDF 路径")
    parser.add_argument("-o", "--output", help="输出目录", default=None)
    parser.add_argument("-d", "--debug-preview", action="store_true",
                        help="输出 bbox / cluster / class 三张 debug PNG + API 日志到输出目录")
    args = parser.parse_args()

    pdf_path = Path(args.input).resolve()
    if not pdf_path.exists():
        print(f"文件不存在: {pdf_path}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else pdf_path.parent / "output"
    output_dir.mkdir(exist_ok=True)
    stem = pdf_path.stem

    doc = fitz.open(str(pdf_path))
    num_pages = len(doc)
    doc.close()

    output_pdf = output_dir / f"{stem}_redacted.pdf"
    shutil.copy2(str(pdf_path), str(output_pdf))

    print(f"{'='*50}")
    print(f"输入: {pdf_path.name} ({num_pages} 页)")
    print(f"{'='*50}")

    all_products = []
    total_keep, total_delete = 0, 0

    for page_num in range(num_pages):
        product_name, decisions = process_page(
            pdf_path, output_pdf, page_num, output_dir, stem,
            debug_preview=args.debug_preview,
        )
        if product_name:
            all_products.append(product_name)
        total_keep += sum(1 for d in decisions.values() if d == "KEEP")
        total_delete += sum(1 for d in decisions.values() if d == "DELETE")

    # 压缩输出(去除 incremental save 冗余)
    doc = fitz.open(str(output_pdf))
    doc.save(str(output_pdf) + ".tmp", garbage=4, deflate=True)
    doc.close()
    shutil.move(str(output_pdf) + ".tmp", str(output_pdf))

    product = all_products[0] if all_products else "未识别"
    print(f"\n{'='*50}")
    print(f"完成: {pdf_path.name}")
    print(f"页数: {num_pages}")
    print(f"产品: {product}")
    print(f"Block: {total_keep} 保留, {total_delete} 删除")
    print(f"输出: {output_pdf}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
