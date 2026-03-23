#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 转 Markdown 工具 —— 基于 PaddleOCR-VL + 跨页表格合并

整体流程：
  1. 读取 config15.ini 配置（GPU设备、vLLM服务地址等）
  2. 调用 PaddleOCR-VL 逐页识别 PDF（版面检测 + VLM 内容识别）
  3. 将各页 markdown 拼接为完整文档
  4. 后处理：检测并合并因分页断裂的 HTML 表格
  5. 输出最终 .md 文件

关于跨页表格合并：
  PDF 分页时，一个表格可能被切成两半，VLM 会分别识别为两个独立的 <table>。
  第二个 <table> 的第一行通常是残余行（上一页最后一行的后半部分），特征是第一列为空。
  本脚本通过比较相邻表格的表头来判断是否属于同一个表，并将残余行追加到上一个表格的末尾行。

依赖：
  - paddleocr（需要安装对应 conda 环境）
  - 远程 vLLM 推理服务（提供 PaddleOCR-VL 模型）
  - config15.ini 配置文件

使用：
  python paddle_ocr.py --input /path/to/file.pdf --output ./output
"""

import os
import re
import sys
import argparse
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from paddleocr import PaddleOCRVL
import configparser


# ══════════════════════════════════════════════════════════════════════════════
# 预编译正则表达式
# 所有 HTML 解析用的正则统一在此定义，避免运行时重复编译。
# 处理几百页 PDF 时，这些正则会被调用上千次，预编译能显著提升性能。
# ══════════════════════════════════════════════════════════════════════════════

RE_TABLE = re.compile(r'<table\b[^>]*>.*?</table>', re.DOTALL)        # 匹配完整的 <table>...</table>
RE_ROW = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL)                # 匹配 <tr> 行，捕获内部内容
RE_CELL = re.compile(r'<td[^>]*>.*?</td>', re.DOTALL)                # 匹配完整的 <td>（含标签）
RE_CELL_GROUPED = re.compile(r'(<td[^>]*>)(.*?)(</td>)', re.DOTALL)  # 分组捕获：(开标签)(内容)(闭标签)
RE_HTML_TAG = re.compile(r'<[^>]+>')                                  # 匹配任意 HTML 标签（用于去标签）
RE_TABLE_OPEN = re.compile(r'<table\b[^>]*>')                        # 匹配 <table> 开标签（保留属性用）

# 页间噪音模式：页码、报告编号、版本号、页眉页脚等固定文本
# 这些内容出现在两个表格之间时，不影响"同一个表"的判断
RE_PAGE_NOISE = re.compile(
    r'^(第\s*\d+\s*页|共\s*\d+\s*页|【.*版】|正文|附录)$|报告编号',
)


# ══════════════════════════════════════════════════════════════════════════════
# 配置加载 & 日志初始化
# ══════════════════════════════════════════════════════════════════════════════

def load_config(path='config15.ini'):
    """
    加载 INI 配置文件。

    配置文件示例（config15.ini）：
      [paddleocr_vl]
      device = 2
      vl_rec_server_url = http://127.0.0.1:8118/v1

    其中：
      - device: 指定版面检测模型使用的 GPU 编号
      - vl_rec_server_url: vLLM 推理服务的地址（VLM 内容识别通过 HTTP 调用此服务）
    """
    config = configparser.ConfigParser()
    if not config.read(path, encoding='utf-8'):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    return config


def setup_logger():
    """
    初始化日志系统：文件日志（按天轮转）+ 控制台输出。
    日志文件保存在 ./logs/task.log，每6天轮转一次，保留7份。
    """
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    os.makedirs('./logs', exist_ok=True)

    fh = TimedRotatingFileHandler(
        './logs/task.log', when='D', interval=6, backupCount=7, encoding='utf-8'
    )
    fh.setFormatter(fmt)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())

    # pdfminer 的日志太多了，只保留错误级别
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    return logger


# 模块加载时执行：读配置、设 GPU、初始化日志
config = load_config()
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"  # 跳过模型源连通性检查，内网环境必须
os.environ['CUDA_VISIBLE_DEVICES'] = config['paddleocr_vl']['device']
logger = setup_logger()


# ══════════════════════════════════════════════════════════════════════════════
# HTML 表格辅助函数
#
# PaddleOCR-VL 输出的 markdown 中，表格是 HTML 格式（<table><tr><td>...），
# 以下函数用于解析和操作这些 HTML 表格。
# ══════════════════════════════════════════════════════════════════════════════

def cell_text(html):
    """
    去掉 HTML 标签，返回纯文本。
    例如：'<td style="...">终端安全</td>' → '终端安全'
    """
    return RE_HTML_TAG.sub('', html).strip()


def row_cells_text(row_html):
    """
    从一个 <tr> 行中提取所有单元格的纯文本列表。
    例如：'<td>序号</td><td>设备名称</td>' → ['序号', '设备名称']
    用于表头比较。
    """
    return [cell_text(c) for c in RE_CELL.findall(row_html)]


def is_page_noise_only(text):
    """
    判断两个表格之间的文本是否只包含页间噪音。

    页间噪音包括：页码（"第21页 共1292页"）、报告编号、版本号（"【2025版】"）、
    页眉页脚文字（"正文"、"附录"）等。

    如果两个表格之间只有这些噪音，说明它们原本是同一个表格被分页切断了。
    如果中间有标题（如"2.3.2.3 安全设备"）或正文内容，说明是不同的表格。
    """
    clean = RE_HTML_TAG.sub('', text).strip()
    return all(
        not line or RE_PAGE_NOISE.search(line)
        for line in (l.strip() for l in clean.split('\n'))
    )


def merge_residual_row(last_row, residual_row):
    """
    将残余行的内容逐列追加到上一行。

    跨页断裂时，上一页最后一行可能不完整（如"终端安全"、"病毒防"），
    下一页第一行是残余内容（如"管理"、"aServer-R-2305"）。
    本函数将残余内容按列追加到对应的单元格中。

    注意：严格逐列操作，绝不跨列。保留原始 <td> 的 style 属性。

    示例：
      上一行: | 4 | 终端安全 | 否 | 3.7.12R3 | 深信服         | 病毒防 | 重要 |
      残余行: |   | 管理     |   |          | aServer-R-2305 |       |      |
      结果:   | 4 | 终端安全管理 | 否 | 3.7.12R3 | 深信服aServer-R-2305 | 病毒防 | 重要 |
    """
    last_cells = RE_CELL_GROUPED.findall(last_row)
    res_cells = RE_CELL_GROUPED.findall(residual_row)

    merged = []
    for j in range(max(len(last_cells), len(res_cells))):
        # 保留上一行的 <td> 开标签（含 style 等属性）
        tag_open = last_cells[j][0] if j < len(last_cells) else '<td>'
        c1 = cell_text(last_cells[j][1]) if j < len(last_cells) else ''
        c2 = cell_text(res_cells[j][1]) if j < len(res_cells) else ''
        merged.append(f'{tag_open}{c1}{c2}</td>')

    return ''.join(merged)


# ══════════════════════════════════════════════════════════════════════════════
# 跨页表格合并（核心后处理逻辑）
#
# 这是整个脚本最关键的函数。PaddleOCR-VL 逐页识别 PDF，当一个表格跨越两页时，
# 会被识别为两个独立的 <table>。本函数检测并合并这种情况。
# ══════════════════════════════════════════════════════════════════════════════

def merge_cross_page_tables(md, header_threshold=0.9):
    """
    合并跨页断裂的 HTML 表格。

    必须同时满足以下三个条件才会合并（AND 关系，任一不满足则跳过）：

      条件1 - 表头匹配：两个表格的表头列数相同，且 ≥90% 的列名一致。
              90% 是为了兜底 OCR 偶尔识别错表头的情况（如"序号"→"席号"）。
              7列表头允许1列不匹配。

      条件2 - 残余行（必要条件）：第二个表格去掉表头后，第一行数据的第一列（序号列）为空。
              这是跨页断裂最明确的信号 —— 正常表格的序号列不会为空。
              如果第一列有内容，说明这是一个正常的新表格，绝不合并。

      条件3 - 页间噪音：两个表格之间只有页眉页脚等噪音，没有标题或正文内容。
              如果中间有"2.3.2.3 安全设备"这样的标题，说明是不同的表。

    合并过程：
      1. 去掉第二个表格的重复表头行
      2. 将残余行逐列追加到第一个表格的最后一行
      3. 将第二个表格的剩余数据行拼接到第一个表格末尾
      4. 删除两表之间的页间噪音内容

    参数：
      md: 完整的 markdown 文本
      header_threshold: 表头匹配阈值，默认 0.9（90%）

    返回：
      处理后的 markdown 文本
    """
    tables = list(RE_TABLE.finditer(md))
    if len(tables) < 2:
        return md

    # 从后往前遍历，因为合并会改变字符串长度，从后往前可以避免前面的偏移被影响
    i = len(tables) - 2
    while i >= 0:
        t1, t2 = tables[i], tables[i + 1]
        rows1 = RE_ROW.findall(t1.group())
        rows2 = RE_ROW.findall(t2.group())

        # 基本校验：两个表都要有行，第二个表至少要有表头+1行数据
        if not rows1 or len(rows2) < 2:
            i -= 1
            continue

        # ── 条件1：表头匹配 ≥ threshold ──
        h1, h2 = row_cells_text(rows1[0]), row_cells_text(rows2[0])
        if len(h1) != len(h2) or not h1:
            i -= 1
            continue
        match_ratio = sum(a == b for a, b in zip(h1, h2)) / len(h1)
        if match_ratio < header_threshold:
            i -= 1
            continue

        # ── 条件3：两表之间只有页间噪音 ──
        if not is_page_noise_only(md[t1.end():t2.start()]):
            i -= 1
            continue

        # ── 条件2（必要条件）：第二个表格第一行数据的第一列为空 ──
        body2 = rows2[1:]  # 去掉表头
        if not body2:
            i -= 1
            continue
        first_data_cells = RE_CELL_GROUPED.findall(body2[0])
        if not first_data_cells or cell_text(first_data_cells[0][1]):
            # 第一列有内容 → 不是残余行 → 是正常的新表格，不合并
            i -= 1
            continue

        # ── 三个条件全部满足，执行合并 ──
        logger.info(f"合并跨页表格: 表头匹配率={match_ratio:.0%}, 表1共{len(rows1)}行, 表2共{len(rows2)}行")

        # 步骤1：残余行追加到上一个表格的末尾行
        rows1[-1] = merge_residual_row(rows1[-1], body2[0])
        body2 = body2[1:]  # 残余行已处理，从 body 中移除

        # 步骤2：拼接 —— 第一个表格的所有行 + 第二个表格的剩余数据行
        all_rows = ''.join(f'<tr>{r}</tr>' for r in rows1 + body2)

        # 步骤3：保留第一个表格的 <table> 开标签（含 border、style 等属性）
        table_tag = RE_TABLE_OPEN.match(t1.group()).group()
        merged_table = f'{table_tag}{all_rows}</table>'

        # 步骤4：替换原文 —— 从第一个表格开始到第二个表格结束（含中间噪音）全部替换
        md = md[:t1.start()] + merged_table + md[t2.end():]

        # 字符串变了，重新扫描所有表格位置
        tables = list(RE_TABLE.finditer(md))
        i = min(i, len(tables) - 2)
        continue

    return md


# ══════════════════════════════════════════════════════════════════════════════
# OCR Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def create_pipeline():
    """
    创建 PaddleOCR-VL pipeline。

    连接远程 vLLM 推理服务。版面检测模型（PP-DocLayoutV3）会自动加载到本地 GPU。
    VLM 内容识别通过 HTTP 调用 vLLM 服务完成。
    """
    return PaddleOCRVL(
        vl_rec_backend="vllm-server",
        vl_rec_server_url=config['paddleocr_vl']['vl_rec_server_url'],
    )


def parse_pdf_to_markdown(file_path, output_dir='./output'):
    """
    PDF → Markdown 主流程。

    步骤：
      1. 创建 pipeline（版面检测 + VLM）
      2. 逐页识别 PDF，每页返回一个 markdown 结果
      3. 拼接所有页的 markdown
      4. 跨页表格合并（后处理）
      5. 写入 .md 文件

    参数：
      file_path: 输入 PDF 文件路径
      output_dir: 输出目录，默认 ./output

    返回：
      输出的 .md 文件路径
    """
    logger.info(f"开始处理: {file_path}")
    os.makedirs(output_dir, exist_ok=True)

    pipeline = create_pipeline()

    # 逐页识别
    results = pipeline.predict(
        input=file_path,
        use_queues=False,                     # 必须 False，否则容易内存异常
        max_pixels=2048 * 2048,               # 增大图像分辨率，提升小字识别率
        markdown_ignore_labels=[              # 忽略这些版面元素（不输出到 markdown）
            'footnote', 'header_image', 'footer', 'footer_image', 'aside_text',
        ],
        layout_unclip_ratio=1.5,              # 扩大检测框，减少边缘内容被裁掉
    )

    # 收集各页 markdown 并拼接
    markdown_list = [res.markdown for res in results]
    md = pipeline.concatenate_markdown_pages(markdown_list)

    # 跨页表格合并（核心后处理）
    md = merge_cross_page_tables(md)

    # 写入文件
    stem = Path(file_path).stem
    out_path = Path(output_dir) / f"{stem}.md"
    out_path.write_text(md, encoding='utf-8')
    logger.info(f"Markdown 已保存: {out_path}")

    return str(out_path)


# ══════════════════════════════════════════════════════════════════════════════
# 命令行入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='PDF to Markdown (PaddleOCR-VL)')
    parser.add_argument("--input", required=True, help="输入 PDF 文件路径")
    parser.add_argument("--output", default="./output", help="输出目录（默认 ./output）")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"错误: 文件不存在 {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        parse_pdf_to_markdown(args.input, args.output)
    except Exception as e:
        logger.exception(f"处理失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
