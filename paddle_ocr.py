#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 转 Markdown 工具 —— 基于 PaddleOCR-VL + 跨页表格合并

整体流程：
  1. 读取 config15.ini 配置（GPU设备、vLLM服务地址等）
  2. 调用 PaddleOCR-VL 逐页识别 PDF（版面检测 + VLM 内容识别）
  3. 将各页 markdown 拼接为完整文档
  4. 后处理：基于 block_label 检测并合并因分页断裂的 HTML 表格
  5. 输出最终 .md 文件

关于跨页表格合并：
  PDF 分页时，一个表格可能被切成两半，VLM 会分别识别为两个独立的 <table>。
  第二个 <table> 的第一行通常是残余行（上一页最后一行的后半部分），特征是第一列为空。
  本脚本通过 VLM 返回的 block_label（而非正则猜测）判断两表之间是否只有页间噪音，
  再结合表头匹配和残余行检测来决定是否合并。

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
# ══════════════════════════════════════════════════════════════════════════════

RE_TABLE = re.compile(r'<table\b[^>]*>.*?</table>', re.DOTALL)
RE_ROW = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL)
RE_CELL = re.compile(r'<td[^>]*>.*?</td>', re.DOTALL)
RE_CELL_GROUPED = re.compile(r'(<td[^>]*>)(.*?)(</td>)', re.DOTALL)
RE_HTML_TAG = re.compile(r'<[^>]+>')
RE_TABLE_OPEN = re.compile(r'<table\b[^>]*>')

# 噪音标签：这些 block_label 出现在两个表格之间时，不影响"同一个表"的判断
# 来自 PP-DocLayoutV3 的 23 种检测类别，以下是页间噪音类
NOISE_LABELS = frozenset({
    'header', 'header_image',
    'footer', 'footer_image',
    'number',       # 页码（PP-DocLayoutV3 实际输出的标签名）
    'page_number',  # 兼容可能的其他标签名
    'footnote',
    'aside_text',
})


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
    """
    config = configparser.ConfigParser()
    if not config.read(path, encoding='utf-8'):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    return config


def setup_logger():
    """初始化日志：文件（按天轮转）+ 控制台。"""
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
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    return logger


config = load_config()
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ['CUDA_VISIBLE_DEVICES'] = config['paddleocr_vl']['device']
logger = setup_logger()


# ══════════════════════════════════════════════════════════════════════════════
# block_label 噪音判断
#
# 每个 block 都有 VLM 分配的 block_label（如 header、table、paragraph_title 等），
# 这是最可靠的判断依据——不需要正则猜测内容类型。
# ══════════════════════════════════════════════════════════════════════════════

def build_table_adjacency(results):
    """
    从 VLM 结果的 block_label 判断哪些相邻表格之间只有噪音。

    遍历所有页面的所有 block，找到所有 table block 的位置，
    检查每对相邻 table 之间的 block 是否全部是噪音标签。

    参数：
      results: list of predict() 返回的 res 对象

    返回：
      set of int，包含可以与下一个 table 合并的 table 序号。
      如 {0, 2} 表示第0个 table 可以和第1个合并，第2个可以和第3个合并。
    """
    # 收集所有 block 的 label
    all_labels = []
    for res in results:
        data = res.json
        res_data = data.get('res', data)
        for block in res_data.get('parsing_res_list', []):
            all_labels.append(block.get('block_label', ''))

    # 找到所有 table block 的位置
    table_positions = [i for i, label in enumerate(all_labels) if label == 'table']

    # 检查每对相邻 table 之间是否只有噪音
    mergeable = set()
    for k in range(len(table_positions) - 1):
        idx1, idx2 = table_positions[k], table_positions[k + 1]
        between = all_labels[idx1 + 1 : idx2]
        if all(label in NOISE_LABELS for label in between):
            mergeable.add(k)  # 第 k 个 table 可以和第 k+1 个 table 合并

    return mergeable


# ══════════════════════════════════════════════════════════════════════════════
# HTML 表格辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def cell_text(html):
    """去掉 HTML 标签，返回纯文本。"""
    return RE_HTML_TAG.sub('', html).strip()


def row_cells_text(row_html):
    """从 <tr> 行中提取所有单元格的纯文本列表，用于表头比较。"""
    return [cell_text(c) for c in RE_CELL.findall(row_html)]


def merge_residual_row(last_row, residual_row):
    """
    将残余行的内容逐列追加到上一行。严格逐列，保留 <td> 样式。

    示例：
      上一行: | 4 | 终端安全 | 否 | 深信服         | 病毒防 | 重要 |
      残余行: |   | 管理     |   | aServer-R-2305 |       |      |
      结果:   | 4 | 终端安全管理 | 否 | 深信服aServer-R-2305 | 病毒防 | 重要 |
    """
    last_cells = RE_CELL_GROUPED.findall(last_row)
    res_cells = RE_CELL_GROUPED.findall(residual_row)

    merged = []
    for j in range(max(len(last_cells), len(res_cells))):
        tag_open = last_cells[j][0] if j < len(last_cells) else '<td>'
        c1 = cell_text(last_cells[j][1]) if j < len(last_cells) else ''
        c2 = cell_text(res_cells[j][1]) if j < len(res_cells) else ''
        merged.append(f'{tag_open}{c1}{c2}</td>')

    return ''.join(merged)


def is_residual_row(row_html):
    """
    判断一行是否是跨页残余行。

    残余行特征（区别于 rowspan 造成的正常空首列）：
      - 第一列为空
      - 列数 ≥ 5（3-4列表格的 rowspan 行无法区分，不做合并）
      - 非空单元格占比 ≤ 50%
    """
    cells = RE_CELL_GROUPED.findall(row_html)
    cell_texts = [cell_text(c[1]) for c in cells]
    total = len(cell_texts)
    if total < 5:
        return False
    non_empty = sum(1 for t in cell_texts if t)
    return not cell_texts[0] and non_empty <= total * 0.5


# ══════════════════════════════════════════════════════════════════════════════
# 跨页表格合并
#
# 合并条件（AND，全部满足才合并）：
#   1. 表头 ≥90% 匹配
#   2. 第二个表格第一行数据是残余行（第一列空 + 大部分空 + 列数≥5）
#   3. 两表之间只有噪音标签的 block（由 build_table_adjacency 预计算）
# ══════════════════════════════════════════════════════════════════════════════

def merge_cross_page_tables(md, mergeable_set, header_threshold=0.9):
    """
    合并跨页断裂的 HTML 表格。

    参数：
      md: 完整的 markdown 文本
      mergeable_set: build_table_adjacency 返回的可合并 table 序号集合
      header_threshold: 表头匹配阈值，默认 0.9
    """
    tables = list(RE_TABLE.finditer(md))
    if len(tables) < 2:
        return md

    i = len(tables) - 2
    while i >= 0:
        # 条件3：block_label 级别的噪音判断（最可靠）
        if i not in mergeable_set:
            i -= 1
            continue

        t1, t2 = tables[i], tables[i + 1]
        rows1 = RE_ROW.findall(t1.group())
        rows2 = RE_ROW.findall(t2.group())

        if not rows1 or len(rows2) < 2:
            i -= 1
            continue

        # 条件1：表头匹配
        h1, h2 = row_cells_text(rows1[0]), row_cells_text(rows2[0])
        if len(h1) != len(h2) or not h1:
            i -= 1
            continue
        match_ratio = sum(a == b for a, b in zip(h1, h2)) / len(h1)
        if match_ratio < header_threshold:
            i -= 1
            continue

        # 条件2：第二个表格第一行数据是残余行
        body2 = rows2[1:]
        if not body2 or not is_residual_row(body2[0]):
            i -= 1
            continue

        # ── 三个条件全部满足，执行合并 ──
        logger.info(f"合并跨页表格: 表头匹配率={match_ratio:.0%}, "
                     f"表1共{len(rows1)}行, 表2共{len(rows2)}行")

        rows1[-1] = merge_residual_row(rows1[-1], body2[0])
        body2 = body2[1:]

        all_rows = ''.join(f'<tr>{r}</tr>' for r in rows1 + body2)
        table_tag = RE_TABLE_OPEN.match(t1.group()).group()
        md = md[:t1.start()] + f'{table_tag}{all_rows}</table>' + md[t2.end():]

        # 重新扫描 + 更新 mergeable_set（合并后序号偏移）
        tables = list(RE_TABLE.finditer(md))
        # 合并了第 i 和 i+1 个表格，后续序号全部 -1
        mergeable_set = {(k - 1 if k > i else k) for k in mergeable_set if k != i}
        i = min(i, len(tables) - 2)
        continue

    return md


def fix_internal_residual_rows(md):
    """
    修复单个表格内部的残余行。

    当 restructure_pages 或 merge_cross_page_tables 已经合并了跨页表格，
    但残余行仍在表格内部时，本函数扫描每个表格并将残余行追加到上一行。
    """
    def fix_table(match):
        table_html = match.group()
        table_tag = RE_TABLE_OPEN.match(table_html).group()
        rows = RE_ROW.findall(table_html)
        if len(rows) < 3:
            return table_html

        fixed_rows = [rows[0]]
        for idx in range(1, len(rows)):
            if is_residual_row(rows[idx]) and len(fixed_rows) > 1:
                fixed_rows[-1] = merge_residual_row(fixed_rows[-1], rows[idx])
            else:
                fixed_rows.append(rows[idx])

        all_rows = ''.join(f'<tr>{r}</tr>' for r in fixed_rows)
        return f'{table_tag}{all_rows}</table>'

    return RE_TABLE.sub(fix_table, md)


# ══════════════════════════════════════════════════════════════════════════════
# OCR Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def create_pipeline():
    """创建 PaddleOCR-VL pipeline，连接远程 vLLM 推理服务。"""
    return PaddleOCRVL(
        vl_rec_backend="vllm-server",
        vl_rec_server_url=config['paddleocr_vl']['vl_rec_server_url'],
    )


def parse_pdf_to_markdown(file_path, output_dir='./output'):
    """
    PDF → Markdown 主流程。

    步骤：
      1. VLM 逐页识别
      2. 从 block_label 预计算可合并的表格对
      3. 拼接 markdown
      4. 跨页表格合并（基于 block_label + 表头匹配 + 残余行检测）
      5. 表内残余行修复
      6. 写入 .md 文件
    """
    logger.info(f"开始处理: {file_path}")
    os.makedirs(output_dir, exist_ok=True)

    pipeline = create_pipeline()

    results = list(pipeline.predict(
        input=file_path,
        use_queues=False,
        temperature=0.0,
        top_p=0.1,
        # 注意：不传 markdown_ignore_labels，让页眉页脚等正常输出到 md
        # 合并逻辑基于 block_label 判断噪音，不依赖 md 中是否有这些内容
    ))

    # 从 block_label 判断哪些相邻表格之间只有噪音（最可靠，不依赖正则）
    mergeable = build_table_adjacency(results)
    logger.info(f"共 {len(mergeable)} 对相邻表格可能需要合并")

    # 拼接 markdown
    markdown_list = [res.markdown for res in results]
    md = pipeline.concatenate_markdown_pages(markdown_list)

    # 后处理
    md = merge_cross_page_tables(md, mergeable)
    md = fix_internal_residual_rows(md)

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
