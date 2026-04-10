"""独立的图纸翻译 pipeline:不做脱敏,只在原图上叠加中文 FreeText 注释。

和 gjr.pipeline(脱敏)的关键差异:
  - 聚类算法是 **Y-band + X-split** (不是 DBSCAN),因此分组粒度不同
  - **一次性批量**把所有 block 发给 GPT 翻译,不按 KEEP/DELETE 分类
  - 输出是 annotations over 原图,原内容保留

入口函数:
    process_translate(pdf_path, output_dir, page_num=0, debug_log=False)
      → 返回 (num_blocks, num_translated, output_path)
"""
import json
import re
import time
from pathlib import Path

import fitz

from gjr.api_log import log_api_call, set_log_context
from gjr.clients import get_client
from gjr.config import PRICING, load_prompt
from gjr.ocr import ocr_pdf, parse_items

# 翻译专用聚类参数(和 gjr.config.DBSCAN_EPS 不共享,算法不同)
Y_GAP = 30
X_GAP = 80
ROW_OVERLAP = 0.3


# ── 噪音过滤 + 聚类 ────────────────────────────────

def _filter_noise(items):
    pat = re.compile(r'^[A-Za-z0-9↑↓←→]+$')
    return [it for it in items
            if not (len(it["text"].strip()) <= 3 and pat.match(it["text"].strip()))]


def _cluster_blocks(items):
    """Y-band → X-split 聚类(和 gjr.cluster.cluster_blocks 的 DBSCAN 不同)。"""
    if not items:
        return []
    sorted_y = sorted(items, key=lambda it: it["top"])
    bands = []
    band = [sorted_y[0]]
    for it in sorted_y[1:]:
        prev_bottom = max(x["bottom"] for x in band)
        if it["top"] - prev_bottom > Y_GAP:
            bands.append(band)
            band = [it]
        else:
            band.append(it)
    bands.append(band)

    blocks = []
    for band in bands:
        sx = sorted(band, key=lambda it: it["left"])
        blk = [sx[0]]
        for it in sx[1:]:
            prev_right = max(x["right"] for x in blk)
            if it["left"] - prev_right > X_GAP:
                blocks.append(blk)
                blk = [it]
            else:
                blk.append(it)
        blocks.append(blk)
    return blocks


def _items_overlap_y(a, b):
    overlap = min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])
    if overlap <= 0:
        return 0
    shorter = min(a["h"], b["h"])
    return overlap / shorter if shorter > 0 else 0


def _group_into_rows(block_items):
    si = sorted(block_items, key=lambda it: it["top"])
    rows = []
    row = [si[0]]
    for it in si[1:]:
        if any(_items_overlap_y(it, r) >= ROW_OVERLAP for r in row):
            row.append(it)
        else:
            rows.append(row)
            row = [it]
    rows.append(row)
    for r in rows:
        r.sort(key=lambda it: it["left"])
    return rows


def _block_to_spatial(block):
    rows = _group_into_rows(block)
    lines = []
    for row in rows:
        parts = []
        for it in row:
            conf_mark = "" if it["conf"] >= 0.9 else f"(?{it['conf']:.0%})"
            parts.append(f"{it['text']}{conf_mark}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _block_bbox(block):
    return (
        min(it["left"] for it in block),
        min(it["top"] for it in block),
        max(it["right"] for it in block),
        max(it["bottom"] for it in block),
    )


# ── GPT 批量翻译 ───────────────────────────────────

def _translate_blocks(blocks):
    cfg = load_prompt("translate_layout")
    block_texts = []
    for i, block in enumerate(blocks):
        spatial = _block_to_spatial(block)
        block_texts.append(f"[Block {i+1}]\n{spatial}")
    all_text = "\n\n".join(block_texts)

    print(f"发送 {len(blocks)} 个 block 给 GPT 翻译...")
    _t0 = time.monotonic()
    response = get_client().chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": cfg["system"]},
            {"role": "user", "content": all_text},
        ],
        temperature=cfg["temperature"],
        max_completion_tokens=cfg["max_completion_tokens"],
    )
    _dur = int((time.monotonic() - _t0) * 1000)
    usage = response.usage
    cost = None
    if usage:
        cost = usage.prompt_tokens * PRICING["input"] / 1e6 + usage.completion_tokens * PRICING["output"] / 1e6
        print(f"Token: input={usage.prompt_tokens}, output={usage.completion_tokens}, 费用=${cost:.4f}")

    result = response.choices[0].message.content or ""

    log_api_call(
        step="gpt_translate_layout",
        api="openai.chat.completions",
        request={
            "model": cfg["model"],
            "temperature": cfg["temperature"],
            "max_completion_tokens": cfg["max_completion_tokens"],
            "system": cfg["system"],
            "user_text": all_text,
            "num_blocks": len(blocks),
        },
        response={
            "text": result,
            "finish_reason": response.choices[0].finish_reason,
        },
        usage=({
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "cost_usd": round(cost, 6) if cost else None,
        } if usage else None),
        duration_ms=_dur,
    )
    return result


def _parse_translated(text, num_blocks):
    result = {}
    current_block = None
    current_lines = []
    for line in text.strip().split("\n"):
        m = re.match(r'\[Block\s*(\d+)\]', line)
        if m:
            if current_block is not None:
                result[current_block] = "\n".join(current_lines)
            current_block = int(m.group(1)) - 1
            current_lines = []
        else:
            if current_block is not None and line.strip():
                current_lines.append(line.strip())
    if current_block is not None:
        result[current_block] = "\n".join(current_lines)
    return result


# ── FreeText 注释写入 ─────────────────────────────

_BG_COLORS = [
    (0.9, 0.95, 1.0),   # 浅蓝
    (1.0, 0.95, 0.9),   # 浅橙
    (0.9, 1.0, 0.9),    # 浅绿
    (1.0, 1.0, 0.85),   # 浅黄
]


def _add_translation_annots(pdf_path, output_path, blocks, translations, page_w, page_h):
    doc = fitz.open(pdf_path)
    page = doc[0]
    font_name = "china-s"  # PyMuPDF 内建简体中文(不嵌入字体文件)
    print(f"使用 CJK 内建字体: {font_name}")

    for bi, block in enumerate(blocks):
        if bi not in translations:
            continue
        trans_text = translations[bi]
        bx1, by1, bx2, by2 = _block_bbox(block)
        block_w = bx2 - bx1
        block_h = by2 - by1

        trans_lines = trans_text.split("\n")
        num_lines = len(trans_lines)
        fontsize = min(10, max(6, int(block_h / max(num_lines, 1) * 0.8)))
        line_height = fontsize * 1.5
        trans_h = num_lines * line_height + 10
        trans_w = block_w * 1.2  # 中文通常比英文宽一点

        # 位置:优先正下方
        tx1 = bx1
        ty1 = by2 + 5
        if ty1 + trans_h > page_h:
            ty1 = by1 - trans_h - 5
        if ty1 < 0:
            ty1 = 5
        if tx1 + trans_w > page_w:
            trans_w = page_w - tx1 - 5
        if tx1 < 0:
            tx1 = 5

        rect = fitz.Rect(tx1, ty1, tx1 + trans_w, ty1 + trans_h)
        bg_color = _BG_COLORS[bi % len(_BG_COLORS)]

        annot = page.add_freetext_annot(
            rect, trans_text,
            fontsize=fontsize, fontname=font_name,
            text_color=(0, 0, 0), fill_color=bg_color,
            align=fitz.TEXT_ALIGN_LEFT,
        )
        annot.set_info(title=f"Block {bi+1} 翻译", subject="OCR Translation")
        annot.update()
        print(f"  Block {bi+1}: 注释位置 ({tx1:.0f},{ty1:.0f})-({tx1+trans_w:.0f},{ty1+trans_h:.0f}), "
              f"{num_lines} 行, 字号 {fontsize}")

    doc.save(output_path)
    doc.close()
    print(f"\n翻译 PDF 已保存: {output_path}")


# ── 入口 ──────────────────────────────────────────

def process_translate(pdf_path, output_dir, page_num=0, debug_log=False):
    """
    翻译单页 PDF:OCR → 聚类 → GPT 批量翻译 → 叠加 FreeText 注释。

    pdf_path     — 输入 PDF 路径
    output_dir   — 输出目录(自动创建)
    page_num     — 要翻译的页码(0-based,默认首页)
    debug_log    — True 时把 API 调用写到 output_dir/api_log/

    Returns: (num_blocks, num_translated, output_pdf_path)
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    stem = pdf_path.stem

    set_log_context(debug_log, page_num + 1, output_dir)

    # 1. OCR(吃缓存)
    json_path = output_dir / f"{stem}_p{page_num+1}_ocr.json"
    if json_path.exists():
        print(f"OCR 缓存: {json_path.name}")
        with open(json_path) as f:
            ocr_result = json.load(f)
    else:
        print(f"调用百度 OCR...")
        _t0 = time.monotonic()
        ocr_result = ocr_pdf(str(pdf_path), page_num=page_num + 1)
        json_path.write_text(json.dumps(ocr_result, ensure_ascii=False, indent=2), encoding="utf-8")
        log_api_call(
            step="baidu_ocr",
            api="baidu.ocr.accurate",
            request={
                "endpoint": "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
                "page_num": page_num + 1,
                "pdf": pdf_path.name,
                "note": "完整响应见同目录 OCR 缓存文件,此处不重复存储",
            },
            response={
                "words_result_num": ocr_result.get("words_result_num", 0),
                "cache_file": json_path.name,
            },
            duration_ms=int((time.monotonic() - _t0) * 1000),
        )

    # 2. 解析 + 过滤 + 聚类
    items = parse_items(ocr_result)
    items = _filter_noise(items)
    blocks = _cluster_blocks(items)
    print(f"{len(items)} 行 → {len(blocks)} 个 block")

    # 3. GPT 批量翻译
    raw_translation = _translate_blocks(blocks)

    # 落盘原始翻译文本(便于 debug)
    trans_txt = output_dir / f"{stem}_translated.txt"
    trans_txt.write_text(raw_translation, encoding="utf-8")

    translations = _parse_translated(raw_translation, len(blocks))
    print(f"解析到 {len(translations)} 个 block 的翻译")

    # 4. 获取页面尺寸
    doc = fitz.open(str(pdf_path))
    page = doc[page_num]
    page_w, page_h = page.rect.width, page.rect.height
    doc.close()

    # 5. 写 FreeText 注释
    output_pdf = output_dir / f"{stem}_translated.pdf"
    _add_translation_annots(str(pdf_path), str(output_pdf),
                            blocks, translations, page_w, page_h)

    return len(blocks), len(translations), output_pdf
