#!/usr/bin/env python3
"""
工程图纸脱敏 v2 —— 完整 pipeline

流程：
  1. 百度 OCR（accurate 接口）→ 文字 + bbox
  2. 预处理：噪音过滤 → 空间聚类 → 二次 Y 切分 → 矢量线 snap
  3. GPT 判断：每个 block 裁切图片 + OCR 文本 → KEEP/DELETE + 产品名
  4. 脱敏：DELETE block 用 snapped bbox 整块涂白
  5. 输出：脱敏 PDF + 产品名
"""
import os
import re
import sys
import json
import base64
import requests
import fitz
from pathlib import Path
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════
# API 客户端
# ══════════════════════════════════════════════════════════════

def _get_openai_key():
    key = os.getenv("OPENAI_API") or os.getenv("OPENAI_API_KEY")
    if key:
        return key
    config_path = Path.home() / ".translate_xls_key"
    if config_path.exists():
        key = config_path.read_text().strip()
        if key:
            return key
    print("错误：未找到 OPENAI_API_KEY")
    sys.exit(1)


# 延迟初始化，支持外部传入 client
_client = None
MODEL = "gpt-5.4-mini"
PRICING = {"input": 0.75, "output": 4.50}


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=_get_openai_key())
    return _client


def set_client(c):
    global _client
    _client = c

BAIDU_API_KEY = os.getenv("BAIDU_API_KEY")
BAIDU_SECRET_KEY = os.getenv("BAIDU_SECRET_KEY")


# ══════════════════════════════════════════════════════════════
# Step 1: 百度 OCR
# ══════════════════════════════════════════════════════════════

def baidu_access_token():
    resp = requests.post("https://aip.baidubce.com/oauth/2.0/token", params={
        "grant_type": "client_credentials",
        "client_id": BAIDU_API_KEY,
        "client_secret": BAIDU_SECRET_KEY,
    })
    data = resp.json()
    if "access_token" in data:
        return data["access_token"]
    print(f"百度 token 获取失败: {data}")
    sys.exit(1)


def ocr_pdf(pdf_path, page_num=1):
    """百度高精度 OCR，先渲染为图片再识别（兼容纯矢量轮廓化 PDF）"""
    import fitz as _fitz
    token = baidu_access_token()

    # 渲染指定页为 PNG（控制最大边 4000px，百度 API 限制 4096）
    doc = _fitz.open(pdf_path)
    page = doc[page_num - 1]
    max_dim = max(page.rect.width, page.rect.height)
    scale = min(4000 / max_dim, 300 / 72)  # 最大 4000px 或 300DPI
    pix = page.get_pixmap(matrix=_fitz.Matrix(scale, scale))
    img_data = base64.b64encode(pix.tobytes("png")).decode()
    doc.close()

    resp = requests.post(
        f"https://aip.baidubce.com/rest/2.0/ocr/v1/accurate?access_token={token}",
        data={
            "image": img_data,
            "recognize_granularity": "small",
            "detect_direction": "false",
            "probability": "true",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=120,
    )
    result = resp.json()

    # 坐标从图片像素空间转回 PDF 点空间
    if "words_result" in result:
        for w in result["words_result"]:
            loc = w.get("location", {})
            loc["left"] = loc.get("left", 0) / scale
            loc["top"] = loc.get("top", 0) / scale
            loc["width"] = loc.get("width", 0) / scale
            loc["height"] = loc.get("height", 0) / scale
            if "chars" in w:
                for ch in w["chars"]:
                    cl = ch.get("location", {})
                    cl["left"] = cl.get("left", 0) / scale
                    cl["top"] = cl.get("top", 0) / scale
                    cl["width"] = cl.get("width", 0) / scale
                    cl["height"] = cl.get("height", 0) / scale

    return result


# ══════════════════════════════════════════════════════════════
# Step 2: 预处理 — 聚类 + snap
# ══════════════════════════════════════════════════════════════

Y_GAP = 30
X_GAP = 80
INTERNAL_Y_GAP = 50
SNAP_MARGIN = 50
ROW_OVERLAP = 0.3


def parse_items(ocr_result):
    items = []
    for w in ocr_result["words_result"]:
        loc = w["location"]
        items.append({
            "text": w["words"],
            "conf": w.get("probability", {}).get("average", 0),
            "left": loc["left"], "top": loc["top"],
            "w": loc["width"], "h": loc["height"],
            "right": loc["left"] + loc["width"],
            "bottom": loc["top"] + loc["height"],
        })
    return items


BRAND_PATTERNS = [
    re.compile(r'^GE\s+APPLIANCES$', re.IGNORECASE),
    re.compile(r'^GEA$', re.IGNORECASE),
    re.compile(r'^GENERAL\s+ELECTRIC$', re.IGNORECASE),
    re.compile(r'^ROPER\s+CORP\.?$', re.IGNORECASE),
    re.compile(r'^HAIER$', re.IGNORECASE),
]


def filter_noise(items):
    pat = re.compile(r'^[A-Za-z0-9↑↓←→]+$')
    result = []
    for it in items:
        text = it["text"].strip()
        # 过滤短符号
        if len(text) <= 3 and pat.match(text):
            continue
        # 过滤纯品牌名 bbox（聚类前剔除，避免撑大 block）
        if any(p.match(text) for p in BRAND_PATTERNS):
            continue
        result.append(it)
    return result


def cluster_blocks(items):
    if not items:
        return []
    sorted_y = sorted(items, key=lambda it: it["top"])
    bands, band = [], [sorted_y[0]]
    for it in sorted_y[1:]:
        if it["top"] - max(x["bottom"] for x in band) > Y_GAP:
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
            if it["left"] - max(x["right"] for x in blk) > X_GAP:
                blocks.append(blk)
                blk = [it]
            else:
                blk.append(it)
        blocks.append(blk)

    # 二次 Y 切分
    split = []
    for blk in blocks:
        sy = sorted(blk, key=lambda it: it["top"])
        sub = [sy[0]]
        for it in sy[1:]:
            if it["top"] - max(x["bottom"] for x in sub) > INTERNAL_Y_GAP:
                split.append(sub)
                sub = [it]
            else:
                sub.append(it)
        split.append(sub)

    if len(split) > len(blocks):
        print(f"  二次切分: {len(blocks)} → {len(split)} 个 block")
    return split


def extract_lines(pdf_path):
    """从 PDF 提取水平/垂直矢量线段"""
    doc = fitz.open(pdf_path)
    page = doc[0]
    h_lines, v_lines = [], []
    for d in page.get_drawings():
        for item in d["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                x1, y1, x2, y2 = p1.x, p1.y, p2.x, p2.y
                if abs(y2 - y1) < 1 and abs(x2 - x1) > 10:
                    h_lines.append((round((y1+y2)/2, 1), min(x1,x2), max(x1,x2)))
                elif abs(x2 - x1) < 1 and abs(y2 - y1) > 10:
                    v_lines.append((round((x1+x2)/2, 1), min(y1,y2), max(y1,y2)))
    doc.close()
    return h_lines, v_lines


def snap_bbox(bx1, by1, bx2, by2, h_lines, v_lines, margin=SNAP_MARGIN):
    """将 bbox 扩展到最近的表格边框线"""
    c = [h for h in h_lines if h[0] < by1 and h[0] > by1-margin and h[1] < bx1+50 and h[2] > bx2-50]
    if c: by1 = min(x[0] for x in c)
    c = [h for h in h_lines if h[0] > by2 and h[0] < by2+margin and h[1] < bx1+50 and h[2] > bx2-50]
    if c: by2 = max(x[0] for x in c)
    c = [v for v in v_lines if v[0] < bx1 and v[0] > bx1-margin and v[1] < by1+50 and v[2] > by2-50]
    if c: bx1 = min(x[0] for x in c)
    c = [v for v in v_lines if v[0] > bx2 and v[0] < bx2+margin and v[1] < by1+50 and v[2] > by2-50]
    if c: bx2 = max(x[0] for x in c)
    return bx1, by1, bx2, by2


def block_bbox_raw(block):
    return (
        min(it["left"] for it in block),
        min(it["top"] for it in block),
        max(it["right"] for it in block),
        max(it["bottom"] for it in block),
    )


def block_position(block, page_w=2448, page_h=1584):
    cx = sum(it["left"] + it["w"] / 2 for it in block) / len(block)
    cy = sum(it["top"] + it["h"] / 2 for it in block) / len(block)
    v = "上部" if cy < page_h * 0.35 else ("中部" if cy < page_h * 0.65 else "下部")
    h = "左侧" if cx < page_w * 0.35 else ("中间" if cx < page_w * 0.65 else "右侧")
    return f"{v}{h}"


def items_overlap_y(a, b):
    overlap = min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])
    if overlap <= 0:
        return 0
    shorter = min(a["h"], b["h"])
    return overlap / shorter if shorter > 0 else 0


def group_into_rows(block_items):
    si = sorted(block_items, key=lambda it: it["top"])
    rows, row = [], [si[0]]
    for it in si[1:]:
        if any(items_overlap_y(it, r) >= ROW_OVERLAP for r in row):
            row.append(it)
        else:
            rows.append(row)
            row = [it]
    rows.append(row)
    for r in rows:
        r.sort(key=lambda it: it["left"])
    return rows


WORD_PAT = re.compile(r'[A-Za-z]{4,}')
DWG_PAT = re.compile(r'\d{3}[A-Z]\d{4}')


# 已知要强制删除的关键词（不需要 GPT 判断）
# 支持拆分识别：即使 OCR 把文字拆成多个 item，拼接后仍能匹配
AUTO_DELETE_PATTERNS = [
    re.compile(r'UNCONTROLLED', re.IGNORECASE),  # 不要求完整短语，单词足够判断
    re.compile(r'DO\s*NOT\s*SCALE', re.IGNORECASE),
]


def classify_block(block):
    """
    小 block 分类：
      "GPT"         → 大块(>4 items)，发 GPT 判断 KEEP/DELETE
      "AUTO_KEEP"   → 纯尺寸，自动保留，不翻译
      "TRANSLATE"   → 有意义英文文字，自动保留，纯文本翻译
      "AUTO_DELETE" → 已知要删除的内容（UNCONTROLLED IF PRINTED 等）
    """
    all_text = " ".join(it["text"] for it in block)
    # 强制删除的已知内容
    if any(p.search(all_text) for p in AUTO_DELETE_PATTERNS):
        return "AUTO_DELETE"
    if len(block) > 4:
        return "GPT"
    # 含图号 → 可能是标题栏混合内容，发 GPT 判断
    if DWG_PAT.search(all_text):
        return "GPT"
    # 有意义的英文单词 → 保留 + 翻译
    if WORD_PAT.search(all_text):
        # 平均置信度太低（OCR 乱码）→ 不翻译
        avg_conf = sum(it["conf"] for it in block) / len(block)
        if avg_conf < 0.5:
            return "AUTO_KEEP"
        return "TRANSLATE"
    # 纯尺寸/符号
    if len(all_text) < 30:
        return "AUTO_KEEP"
    return "GPT"


def block_to_spatial(block):
    rows = group_into_rows(block)
    lines = []
    for row in rows:
        parts = []
        for it in row:
            conf_mark = "" if it["conf"] >= 0.9 else f" (?{it['conf']:.0%})"
            parts.append(f"{it['text']}{conf_mark}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# Step 3: GPT 判断 + 产品名提取
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是工程图纸脱敏专家。

## 背景
用户需要将上游供应商的工程图纸发给下游施工方。
目标：隐藏上游身份信息，保留施工所需的技术内容。

## 任务 1：对每个 block 判断 KEEP 或 DELETE

DELETE（整块删除）的区域：
- 签名栏 / 签核栏（SIGNOFF BLOCK）—— 包含审批人名、职能角色
- 标题栏（Title Block）—— 包含绘图人、日期、版本、公司名、发布状态等

KEEP（整块保留）的区域：
- NOTES / 技术注释 —— 材料规格、加工要求、公差、表面处理等施工必须信息
- 零件表 / Parts List —— SEE NOTES、CAD FILE NAME、PART、REV 等列的表格
- 图号如果在边框区域已有保留，图纸内部的图号区域可以删除
- 图形区域、尺寸标注

## 任务 2：提取产品名称

在标题栏中通常有一个产品名称（如 MOUNT EVAP FAN、MULLION BOTTOM QC 等），是图纸中最大最显眼的文字。请从所有 block 中找到这个产品名称并返回。

注意：OCR 可能漏字或误读，请结合图片判断完整的产品名。如果不确定，返回你从图片中看到的内容。

## few-shot 示例

Block: 上部中间，SIGNOFF BLOCK / FUNCTION / NAME / PROJECT ENGINEER Zell,Yuriy / DESIGN MANAGER 等
→ DELETE（签名栏）

Block: 上部右侧，DWG NO 224D4488
→ DELETE（图号区域，边框已有保留）

Block: 上部右侧，SEE NOTES / CAD FILE NAME 224D4488P001-P011 / PART / REV
→ KEEP（零件表）

Block: 中部右侧，NOTES / 1. MATERIAL: GALVANIZED STEEL... / 公差、折弯半径等
→ KEEP（技术注释）

Block: 下部右侧，MOUNT EVAP FAN / 224D4488 / DRAWN C.VELMURUGAM / 06/06/16 / Released 等
→ DELETE（标题栏）
→ 产品名: MOUNT EVAP FAN

## 输出格式
先逐行输出 block 判断：
[Block N] KEEP 或 DELETE — 理由

最后一行输出产品名：
[Product] 产品名称"""


def crop_block_images(pdf_path, blocks, snapped_bboxes, dpi=200, pad=5):
    """用 snapped bbox 裁切 block 图片"""
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72
    images = []
    for bi, block in enumerate(blocks):
        bx1, by1, bx2, by2 = snapped_bboxes[bi]
        clip = fitz.Rect(
            max(0, bx1 - pad), max(0, by1 - pad),
            min(page.rect.width, bx2 + pad), min(page.rect.height, by2 + pad),
        )
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
        images.append(base64.b64encode(pix.tobytes("png")).decode())
    doc.close()
    return images


def ask_gpt(blocks, block_images, block_texts, snapped_bboxes):
    content = []
    content.append({"type": "text", "text": "以下是一张工程图纸中识别出的各个区域（block），请判断每个 block 应该 KEEP 还是 DELETE，并提取产品名称。\n"})

    for bi in range(len(blocks)):
        bbox = snapped_bboxes[bi]
        pos = block_position(blocks[bi])
        content.append({"type": "text", "text": f"\n--- [Block {bi+1}] 位置: {pos}, 坐标: ({bbox[0]:.0f},{bbox[1]:.0f})-({bbox[2]:.0f},{bbox[3]:.0f}), {len(blocks[bi])} 项 ---\nOCR 文本:\n{block_texts[bi]}\n\n裁切图片:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{block_images[bi]}", "detail": "high"},
        })

    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=0,
        max_completion_tokens=1000,
    )
    usage = response.usage
    if usage:
        cost = usage.prompt_tokens * PRICING["input"] / 1e6 + usage.completion_tokens * PRICING["output"] / 1e6
        print(f"Token: input={usage.prompt_tokens}, output={usage.completion_tokens}, 费用=${cost:.4f}")
    return response.choices[0].message.content


def parse_gpt_result(text, num_blocks):
    decisions = {}
    product_name = None
    for line in text.strip().split("\n"):
        m = re.match(r'\[Block\s*(\d+)\]\s*(KEEP|DELETE)', line.strip())
        if m:
            decisions[int(m.group(1)) - 1] = m.group(2)
        m2 = re.match(r'\[Product\]\s*(.+)', line.strip())
        if m2:
            product_name = m2.group(1).strip()
    for i in range(num_blocks):
        if i not in decisions:
            decisions[i] = "DELETE"
            print(f"  警告: Block {i+1} 未提及，默认 DELETE")
    return decisions, product_name


# ══════════════════════════════════════════════════════════════
# Step 4: 脱敏
# ══════════════════════════════════════════════════════════════

def detect_frame(pdf_path):
    """
    检测图纸的外框线和内框边界。
    返回:
      frame_lines: 所有长框线（用于 redact 后重画）
      inner_rect: 内框矩形 (x1,y1,x2,y2)，redact 不能超出此范围
    """
    doc = fitz.open(pdf_path)
    page = doc[0]
    pw, ph = page.rect.width, page.rect.height

    h_lines, v_lines = [], []
    for d in page.get_drawings():
        for item in d["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                x1, y1, x2, y2 = p1.x, p1.y, p2.x, p2.y
                if abs(y2-y1) < 1 and abs(x2-x1) > pw * 0.5:
                    h_lines.append((round((y1+y2)/2, 1), min(x1,x2), max(x1,x2)))
                elif abs(x2-x1) < 1 and abs(y2-y1) > ph * 0.5:
                    v_lines.append((round((x1+x2)/2, 1), min(y1,y2), max(y1,y2)))
    doc.close()

    # 收集所有框线（用于 redact 后重画）
    frame_lines = []
    for y, x1, x2 in h_lines:
        frame_lines.append(("h", y, x1, x2))
    for x, y1, y2 in v_lines:
        frame_lines.append(("v", x, y1, y2))

    # 内框：非页面边缘的最长线段组成的矩形
    inner_h = sorted([y for y, x1, x2 in h_lines if y > 10 and y < ph - 10])
    inner_v = sorted([x for x, y1, y2 in v_lines if x > 10 and x < pw - 10])

    if len(inner_h) >= 2 and len(inner_v) >= 2:
        inner_rect = (inner_v[0], inner_h[0], inner_v[-1], inner_h[-1])
    else:
        # fallback
        inner_rect = (72, 36, pw - 72, ph - 36)

    return frame_lines, inner_rect


def apply_redactions(pdf_path, output_path, blocks, snapped_bboxes, decisions):
    # 检测外框线和内框边界
    frame_lines, inner_rect = detect_frame(pdf_path)
    ix1, iy1, ix2, iy2 = inner_rect
    print(f"  内框边界: ({ix1:.0f},{iy1:.0f})-({ix2:.0f},{iy2:.0f})")

    doc = fitz.open(pdf_path)
    page = doc[0]

    for bi, block in enumerate(blocks):
        action = decisions.get(bi, "DELETE")
        if action == "DELETE":
            bx1, by1, bx2, by2 = snapped_bboxes[bi]
            # 限制 redact 在内框以内，保护边框区域的数字/字母
            rx1 = max(bx1, ix1)
            ry1 = max(by1, iy1)
            rx2 = min(bx2, ix2)
            ry2 = min(by2, iy2)
            if rx1 < rx2 and ry1 < ry2:
                rect = fitz.Rect(rx1, ry1, rx2, ry2)
                page.add_redact_annot(rect, fill=(1, 1, 1))
                print(f"  Block {bi+1}: DELETE → 涂白 ({rx1:.0f},{ry1:.0f})-({rx2:.0f},{ry2:.0f})")
            else:
                print(f"  Block {bi+1}: DELETE → 跳过（完全在内框外）")
        else:
            print(f"  Block {bi+1}: KEEP")

    page.apply_redactions()

    # 重画可能被擦掉的框线
    shape = page.new_shape()
    for line in frame_lines:
        if line[0] == "h":
            _, y, x1, x2 = line
            shape.draw_line(fitz.Point(x1, y), fitz.Point(x2, y))
        else:
            _, x, y1, y2 = line
            shape.draw_line(fitz.Point(x, y1), fitz.Point(x, y2))
    shape.finish(color=(0, 0, 0), width=0.5)
    shape.commit()

    doc.save(output_path)
    doc.close()


def find_product_bbox(items, product_name):
    """
    在 OCR items 中匹配产品名，返回合并 bbox。

    策略：
      1. 产品名拆成单词集合
      2. 每个 item 文本匹配产品名中的单词 → 候选
      3. 候选按空间距离聚类（Y 重叠 + X 间距 < 阈值）
      4. 取最大聚类的合并 bbox
    """
    if not product_name:
        return None

    product_words = set(product_name.upper().split())
    candidates = []

    for it in items:
        text = it["text"].strip().upper()
        # 完全匹配某个产品名单词，或产品名单词包含在 item 中
        if text in product_words or any(pw in text for pw in product_words):
            candidates.append(it)

    if not candidates:
        return None

    # 按空间聚类：Y 重叠且 X 间距 < 200 → 同一组
    candidates.sort(key=lambda it: (it["top"], it["left"]))
    groups = [[candidates[0]]]
    for it in candidates[1:]:
        last_group = groups[-1]
        # 检查和当前组中任意 item 是否空间接近
        close = False
        for g in last_group:
            y_overlap = min(it["bottom"], g["bottom"]) - max(it["top"], g["top"])
            x_dist = abs(it["left"] - g["right"])
            if y_overlap > 0 and x_dist < 200:
                close = True
                break
        if close:
            last_group.append(it)
        else:
            groups.append([it])

    # 取匹配单词数最多的组
    best = max(groups, key=lambda g: len(g))
    matched_words = [it["text"] for it in best]
    bbox = (
        min(it["left"] for it in best),
        min(it["top"] for it in best),
        max(it["right"] for it in best),
        max(it["bottom"] for it in best),
    )
    print(f"  产品名匹配: {matched_words} → bbox ({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})")
    return bbox


def render_product_name(pdf_path, product_name, product_bbox):
    """在脱敏后的 PDF 上原位渲染产品名"""
    if not product_name or not product_bbox:
        return

    doc = fitz.open(pdf_path)
    page = doc[0]

    bx1, by1, bx2, by2 = product_bbox
    box_h = by2 - by1
    box_w = bx2 - bx1

    # 字号：根据原始 bbox 高度
    fontsize = max(10, box_h * 0.8)

    # 居中渲染
    text_w = len(product_name) * fontsize * 0.5  # 粗估
    x = bx1 + (box_w - text_w) / 2 if text_w < box_w else bx1
    y = by2 - box_h * 0.15  # 基线

    page.insert_text(
        fitz.Point(x, y),
        product_name,
        fontsize=fontsize,
        fontname="helv",
        color=(0, 0, 0),
    )

    doc.saveIncr()
    doc.close()
    print(f"  产品名已渲染: \"{product_name}\" 字号={fontsize:.0f}")


# ══════════════════════════════════════════════════════════════
# Step 6: 翻译 + 放置
# ══════════════════════════════════════════════════════════════

TRANSLATE_PROMPT = """你是工程图纸翻译专家。用户给你一个工程图纸区域的 OCR 文本和对应图片。

请将内容翻译为简洁整齐的中文。

规则：
1. 保持原文结构（表格保持行列对齐，编号列表保持编号）
2. 专有名词保留英文（零件号如 224D4488P001、规格代码如 B8A26、标准号如 ETP 0900D00I）
3. OCR 有误读的地方，先根据图片纠正再翻译
4. 表格用简洁的格式：每行用 " | " 分隔列
5. 输出纯翻译文本，不要加解释、标题、分隔线
6. 保持紧凑，不要多余空行
7. 跳过任何包含上游公司/制造商信息的行（如公司名、专属程序编号等），这些已被标记为 [已过滤]"""


# 敏感词模式：公司名 / 毒性程序 / 制造商相关
SENSITIVE_REPLACEMENTS = [
    (re.compile(r'\bGE\s+APPLIANCES\b', re.IGNORECASE), '[品牌]'),
    (re.compile(r'\bGENERAL\s+ELECTRIC\b', re.IGNORECASE), '[品牌]'),
    (re.compile(r'\bGEA\b', re.IGNORECASE), '[品牌]'),
    (re.compile(r'(?<!\w)GE(?!\w)', re.IGNORECASE), '[品牌]'),
    (re.compile(r'\bROPER\s+CORP\.?\b', re.IGNORECASE), '[品牌]'),
    (re.compile(r'\bHAIER\b', re.IGNORECASE), '[品牌]'),
]


def filter_sensitive_lines(text):
    """
    替换文本中的敏感关键词为占位符，保留句子其余内容。
    返回 (替换后文本, 被替换的关键词列表)
    """
    replaced = []
    result = text
    for pat, placeholder in SENSITIVE_REPLACEMENTS:
        matches = pat.findall(result)
        if matches:
            replaced.extend(matches)
            result = pat.sub(placeholder, result)
    return result, replaced


def translate_block(block_text, block_image):
    """翻译单个 block，自动过滤敏感内容"""
    # 翻译前过滤敏感行
    clean_text, filtered = filter_sensitive_lines(block_text)
    if filtered:
        print(f"      过滤 {len(filtered)} 行敏感内容: {filtered[:3]}")
    if not clean_text.strip():
        return None

    content = [
        {"type": "text", "text": f"请翻译以下工程图纸区域的内容：\n\n{clean_text}\n\n对应图片："},
        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{block_image}", "detail": "high",
        }},
    ]
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": TRANSLATE_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=0.1,
        max_completion_tokens=2000,
    )
    usage = response.usage
    if usage:
        cost = usage.prompt_tokens * PRICING["input"] / 1e6 + usage.completion_tokens * PRICING["output"] / 1e6
        print(f"    Token: in={usage.prompt_tokens}, out={usage.completion_tokens}, ${cost:.4f}")
    result = response.choices[0].message.content or ""
    # 翻译后再替换一次（GPT 可能从图片里看到品牌名）
    result, post_filtered = filter_sensitive_lines(result)
    if post_filtered:
        print(f"      翻译后替换 {len(post_filtered)} 处品牌名")
    # 清理占位符
    result = result.replace("[品牌]", "").replace("[品牌] ", "").replace(" [品牌]", "")
    # 清理多余空格
    result = re.sub(r'  +', ' ', result).strip()
    return result


def find_white_space(pdf_path, block_bbox, all_bboxes, page_w, page_h, trans_w, trans_h):
    """
    在 block 附近找到白色空白区域放置翻译。

    优先级：左侧 → 下方 → 上方 → 右侧
    """
    bx1, by1, bx2, by2 = block_bbox
    gap = 10  # 和原 block 的间距

    candidates = [
        # 左侧
        (bx1 - trans_w - gap, by1, bx1 - gap, by1 + trans_h),
        # 下方
        (bx1, by2 + gap, bx1 + trans_w, by2 + gap + trans_h),
        # 上方
        (bx1, by1 - trans_h - gap, bx1 + trans_w, by1 - gap),
    ]

    for cx1, cy1, cx2, cy2 in candidates:
        # 边界检查
        if cx1 < 30 or cy1 < 30 or cx2 > page_w - 30 or cy2 > page_h - 30:
            continue
        # 检查是否和其他 block 重叠
        overlap = False
        for obx1, oby1, obx2, oby2 in all_bboxes:
            if not (cx2 < obx1 or cx1 > obx2 or cy2 < oby1 or cy1 > oby2):
                overlap = True
                break
        if not overlap:
            return (cx1, cy1, cx2, cy2)

    # fallback: 强制放在左侧，即使超出边界也 clamp
    return (max(30, bx1 - trans_w - gap), by1,
            max(30 + trans_w, bx1 - gap), by1 + trans_h)


def batch_translate_labels(blocks, indices, block_texts):
    """
    批量翻译小 block 标注文字（纯文本，不发图片，一次调用）。
    返回 {block_index: translated_text}
    """
    if not indices:
        return {}

    # 构建输入（替换品牌名后翻译）
    lines = []
    valid_indices = []
    for i, bi in enumerate(indices):
        text = " ".join(it["text"] for it in blocks[bi])
        clean_text, replaced = filter_sensitive_lines(text)
        if replaced:
            print(f"      替换品牌名 [{bi+1}]: {replaced}")
        lines.append(f"[{len(lines)+1}] {clean_text}")
        valid_indices.append(bi)

    if not lines:
        return {}
    input_text = "\n".join(lines)

    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是工程图纸翻译专家。将以下工程图纸标注文字翻译为中文。\n"
             "规则：\n"
             "- 每行格式 [N] 英文，输出对应 [N] 中文\n"
             "- 专有名词保留英文（零件号、规格代码、字母编号如 A/B/C）\n"
             "- 简洁准确，不加解释"},
            {"role": "user", "content": input_text},
        ],
        temperature=0.1,
        max_completion_tokens=1000,
    )
    usage = response.usage
    if usage:
        cost = usage.prompt_tokens * PRICING["input"] / 1e6 + usage.completion_tokens * PRICING["output"] / 1e6
        print(f"  Token: in={usage.prompt_tokens}, out={usage.completion_tokens}, ${cost:.4f}")

    # 解析结果
    result_text = response.choices[0].message.content or ""
    translations = {}
    for line in result_text.strip().split("\n"):
        m = re.match(r'\[(\d+)\]\s*(.+)', line.strip())
        if m:
            idx = int(m.group(1)) - 1
            if idx < len(valid_indices):
                translated = m.group(2).strip()
                # 清理占位符
                translated = translated.replace("[品牌]", "").replace("[品牌] ", "").replace(" [品牌]", "")
                translated = re.sub(r'  +', ' ', translated).strip()
                translations[valid_indices[idx]] = translated

    return translations


def add_translation_annotations(pdf_path, blocks, snapped_bboxes, decisions,
                                block_texts, block_images, page_w, page_h):
    """翻译 KEEP block 并放置到空白区域"""
    doc = fitz.open(pdf_path)
    page = doc[0]

    # 注册中文字体
    font_name = "china-s"

    keep_indices = [i for i, d in decisions.items() if d == "KEEP"]
    all_bboxes = list(snapped_bboxes)  # 用于碰撞检测

    for bi in keep_indices:
        block = blocks[bi]
        bbox = snapped_bboxes[bi]
        print(f"\n  翻译 Block {bi+1}...")

        # 翻译
        trans_text = translate_block(block_texts[bi], block_images[bi])
        print(f"    翻译结果 ({len(trans_text)} 字):")
        for line in trans_text.split("\n")[:3]:
            print(f"      {line}")
        if len(trans_text.split("\n")) > 3:
            print(f"      ...")

        # 估算翻译区域大小
        lines = trans_text.split("\n")
        fontsize = 9
        line_height = fontsize * 1.6
        trans_h = len(lines) * line_height + 10
        max_line_len = max(len(l) for l in lines) if lines else 10
        trans_w = max(200, min(max_line_len * fontsize * 0.55, bbox[2] - bbox[0]))

        # 找空白位置
        pos = find_white_space(pdf_path, bbox, all_bboxes, page_w, page_h, trans_w, trans_h)
        rect = fitz.Rect(pos[0], pos[1], pos[2], pos[3])

        # 添加到已占用区域
        all_bboxes.append(pos)

        # FreeText 注释
        annot = page.add_freetext_annot(
            rect,
            trans_text,
            fontsize=fontsize,
            fontname=font_name,
            text_color=(0, 0, 0),
            fill_color=(1, 1, 1),  # 白底
            align=fitz.TEXT_ALIGN_LEFT,
        )
        annot.set_opacity(0.92)
        annot.update()

        print(f"    放置位置: ({pos[0]:.0f},{pos[1]:.0f})-({pos[2]:.0f},{pos[3]:.0f})")

    doc.saveIncr()
    doc.close()


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def process_page(pdf_path, output_path, page_num, output_dir, stem):
    """处理单页 PDF 的完整 pipeline"""
    page_label = f"P{page_num+1}"

    # ── Step 1: OCR ──
    print(f"\n{'─'*50}")
    print(f"  [{page_label}] Step 1: 百度 OCR")
    json_path = output_dir / f"{stem}_p{page_num+1}_ocr.json"
    if json_path.exists():
        print(f"    使用缓存: {json_path}")
        with open(json_path) as f:
            ocr_result = json.load(f)
    else:
        print(f"    识别第 {page_num+1} 页...")
        ocr_result = ocr_pdf(str(pdf_path), page_num=page_num + 1)
        json_path.write_text(json.dumps(ocr_result, ensure_ascii=False, indent=2), encoding="utf-8")
    num_words = ocr_result.get("words_result_num", 0)
    print(f"    OCR: {num_words} 行")

    if num_words == 0:
        print(f"    跳过（无文字）")
        return None, {}

    # ── Step 2: 预处理 ──
    print(f"  [{page_label}] Step 2: 预处理")

    # 提取矢量线 + 检测内框
    doc = fitz.open(str(pdf_path))
    page = doc[page_num]
    pw, ph = page.rect.width, page.rect.height
    h_lines, v_lines = [], []
    for d in page.get_drawings():
        for item in d["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                x1, y1, x2, y2 = p1.x, p1.y, p2.x, p2.y
                if abs(y2-y1) < 1 and abs(x2-x1) > 10:
                    h_lines.append((round((y1+y2)/2, 1), min(x1,x2), max(x1,x2)))
                elif abs(x2-x1) < 1 and abs(y2-y1) > 10:
                    v_lines.append((round((x1+x2)/2, 1), min(y1,y2), max(y1,y2)))
    doc.close()

    # 检测内框边界
    frame_h = [h for h in h_lines if h[2] - h[1] > pw * 0.5]
    frame_v = [v for v in v_lines if v[2] - v[1] > ph * 0.5]
    inner_h = sorted([y for y, x1, x2 in frame_h if y > 10 and y < ph - 10])
    inner_v = sorted([x for x, y1, y2 in frame_v if x > 10 and x < pw - 10])
    if len(inner_h) >= 2 and len(inner_v) >= 2:
        ix1, iy1, ix2, iy2 = inner_v[0], inner_h[0], inner_v[-1], inner_h[-1]
    else:
        ix1, iy1, ix2, iy2 = 72, 36, pw - 72, ph - 36
    print(f"    内框: ({ix1:.0f},{iy1:.0f})-({ix2:.0f},{iy2:.0f})")

    # 解析 OCR items，只保留内框内的
    all_items = parse_items(ocr_result)
    items = [it for it in all_items
             if it["left"] >= ix1 - 5 and it["right"] <= ix2 + 5
             and it["top"] >= iy1 - 5 and it["bottom"] <= iy2 + 5]
    print(f"    OCR: {len(all_items)} 行, 内框内: {len(items)} 行 (过滤 {len(all_items)-len(items)} 行框外)")

    items = filter_noise(items)

    if not items:
        print(f"    跳过（过滤后无内容）")
        return None, {}

    blocks = cluster_blocks(items)

    snapped_bboxes = []
    for bi, block in enumerate(blocks):
        raw = block_bbox_raw(block)
        snapped = snap_bbox(*raw, h_lines, v_lines)
        snapped_bboxes.append(snapped)

    print(f"    {len(items)} 行 → {len(blocks)} 个 block")

    # ── 预分类 ──
    gpt_indices, auto_keep_indices, translate_indices, auto_delete_indices = [], [], [], []
    for bi, block in enumerate(blocks):
        cat = classify_block(block)
        if cat == "GPT":
            gpt_indices.append(bi)
        elif cat == "TRANSLATE":
            translate_indices.append(bi)
        elif cat == "AUTO_DELETE":
            auto_delete_indices.append(bi)
        else:
            auto_keep_indices.append(bi)
    print(f"    分类: {len(auto_keep_indices)} 尺寸, {len(translate_indices)} 标注, {len(gpt_indices)} 大块, {len(auto_delete_indices)} 强制删除")

    # ── block 文本 ──
    block_texts = [block_to_spatial(block) for block in blocks]

    # ── Step 3: GPT 判断 ──
    product_name = None
    if gpt_indices:
        print(f"  [{page_label}] Step 3: GPT 判断 ({len(gpt_indices)} 个 block)")
        gpt_blocks = [blocks[i] for i in gpt_indices]
        gpt_bboxes = [snapped_bboxes[i] for i in gpt_indices]
        gpt_texts = [block_texts[i] for i in gpt_indices]
        # 裁切图片（指定页码）
        doc = fitz.open(str(pdf_path))
        page_obj = doc[page_num]
        zoom = 200 / 72
        gpt_images = []
        for bbox in gpt_bboxes:
            clip = fitz.Rect(max(0, bbox[0]-5), max(0, bbox[1]-5),
                             min(pw, bbox[2]+5), min(ph, bbox[3]+5))
            pix = page_obj.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
            gpt_images.append(base64.b64encode(pix.tobytes("png")).decode())
        doc.close()

        gpt_result = ask_gpt(gpt_blocks, gpt_images, gpt_texts, gpt_bboxes)
        print(f"    {gpt_result}")
        gpt_decisions, product_name = parse_gpt_result(gpt_result, len(gpt_blocks))
    else:
        print(f"  [{page_label}] Step 3: 跳过（无大块）")
        gpt_decisions = {}

    # 合并决策
    decisions = {}
    for bi in auto_keep_indices:
        decisions[bi] = "KEEP"
    for bi in translate_indices:
        decisions[bi] = "KEEP"
    for bi in auto_delete_indices:
        decisions[bi] = "DELETE"
    for gi, bi in enumerate(gpt_indices):
        decisions[bi] = gpt_decisions.get(gi, "DELETE")

    print(f"    产品名: {product_name or '未识别'}")
    delete_count = sum(1 for d in decisions.values() if d == "DELETE")
    keep_count = sum(1 for d in decisions.values() if d == "KEEP")
    print(f"    决策: {keep_count} 保留, {delete_count} 删除")

    # ── Step 4/5/6: 脱敏 + 产品名 + 翻译（一次打开、一次保存）──
    print(f"  [{page_label}] Step 4: 脱敏 + 产品名 + 翻译")

    # 先收集翻译结果（需要在打开 output_path 之前完成 GPT 调用）
    big_translate = [bi for bi in range(len(blocks))
                     if decisions.get(bi) == "KEEP" and len(blocks[bi]) > 4]

    # 6a: 小 block 批量翻译
    label_translations = {}
    if translate_indices:
        label_translations = batch_translate_labels(blocks, translate_indices, block_texts)
        for bi, trans in label_translations.items():
            print(f"    标注 [{bi+1}] → {trans}")

    # 6b: 大 block 翻译（先裁切图片，调 GPT，收集结果）
    big_translations = {}  # bi -> trans_text
    if big_translate:
        # 从原始 PDF 裁切图片
        src_doc = fitz.open(str(pdf_path))
        src_page = src_doc[page_num]
        zoom = 200 / 72
        for bi in big_translate:
            bbox = snapped_bboxes[bi]
            clip = fitz.Rect(max(0, bbox[0]-5), max(0, bbox[1]-5),
                             min(pw, bbox[2]+5), min(ph, bbox[3]+5))
            pix = src_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
            img_b64 = base64.b64encode(pix.tobytes("png")).decode()
            trans_text = translate_block(block_texts[bi], img_b64)
            if trans_text:
                big_translations[bi] = trans_text
                print(f"    大块 Block {bi+1} 翻译完成")
        src_doc.close()

    # 产品名匹配
    product_bbox = None
    if product_name:
        all_items = parse_items(ocr_result)
        product_bbox = find_product_bbox(all_items, product_name)

    # ── 一次性打开 output PDF，执行所有修改 ──
    doc = fitz.open(str(output_path))
    page_obj = doc[page_num]

    # (1) 脱敏涂白
    for bi in range(len(blocks)):
        if decisions.get(bi) == "DELETE":
            bx1, by1, bx2, by2 = snapped_bboxes[bi]
            rx1 = max(bx1, ix1)
            ry1 = max(by1, iy1)
            rx2 = min(bx2, ix2)
            ry2 = min(by2, iy2)
            if rx1 < rx2 and ry1 < ry2:
                page_obj.add_redact_annot(fitz.Rect(rx1, ry1, rx2, ry2), fill=(1, 1, 1))

    # (1.5) KEEP block 内品牌关键词精确遮盖
    BRAND_INLINE_PATS = [
        re.compile(r'\bGE\s+APPLIANCES\b', re.IGNORECASE),
        re.compile(r'\bGENERAL\s+ELECTRIC\b', re.IGNORECASE),
        re.compile(r'\bGEA\b', re.IGNORECASE),
        re.compile(r'(?<!\w)GE(?!\w)'),  # 独立 GE（不匹配 EDGE 等）
        re.compile(r'\bROPER\s+CORP\.?\b', re.IGNORECASE),
        re.compile(r'\bHAIER\b', re.IGNORECASE),
    ]
    brand_mask_count = 0
    for w in ocr_result.get("words_result", []):
        text = w["words"]
        chars = w.get("chars", [])
        if not chars:
            continue
        # 建立 text index → chars index 的映射（chars 不含空格）
        text_to_char = {}
        ci = 0
        for ti, ch in enumerate(text):
            if ch == ' ':
                continue
            if ci < len(chars):
                text_to_char[ti] = ci
                ci += 1
        for pat in BRAND_INLINE_PATS:
            for m in pat.finditer(text):
                # 找匹配范围内第一个和最后一个非空格字符对应的 chars 索引
                first_ci, last_ci = None, None
                for ti in range(m.start(), m.end()):
                    if ti in text_to_char:
                        if first_ci is None:
                            first_ci = text_to_char[ti]
                        last_ci = text_to_char[ti]
                if first_ci is not None and last_ci is not None:
                    c_start = chars[first_ci]["location"]
                    c_end = chars[last_ci]["location"]
                    # 用行级 bbox 高度的 75% 居中，匹配矢量文字实际高度
                    line_loc = w["location"]
                    line_h = line_loc["height"]
                    shrink = line_h * 0.125  # 上下各缩 12.5%
                    rx1 = c_start["left"] - 1
                    ry1 = line_loc["top"] + shrink
                    rx2 = c_end["left"] + c_end["width"] + 1
                    ry2 = line_loc["top"] + line_h - shrink
                    page_obj.add_redact_annot(fitz.Rect(rx1, ry1, rx2, ry2), fill=(1, 1, 1))
                    brand_mask_count += 1
    if brand_mask_count:
        print(f"    品牌关键词遮盖: {brand_mask_count} 处")

    page_obj.apply_redactions()

    # (2) 重画框线
    shape = page_obj.new_shape()
    for y, x1, x2 in frame_h:
        shape.draw_line(fitz.Point(x1, y), fitz.Point(x2, y))
    for x, y1, y2 in frame_v:
        shape.draw_line(fitz.Point(x, y1), fitz.Point(x, y2))
    shape.finish(color=(0, 0, 0), width=0.5)
    shape.commit()

    # (3) 渲染产品名
    if product_name and product_bbox:
        bx1, by1, bx2, by2 = product_bbox
        fontsize = max(10, (by2 - by1) * 0.8)
        text_w = len(product_name) * fontsize * 0.5
        x = bx1 + ((bx2-bx1) - text_w) / 2 if text_w < (bx2-bx1) else bx1
        page_obj.insert_text(fitz.Point(x, by2 - (by2-by1)*0.15), product_name,
                             fontsize=fontsize, fontname="helv", color=(0, 0, 0))
        print(f"    产品名: \"{product_name}\" 字号={fontsize:.0f}")

    # (4) 大块翻译注释
    all_bboxes = list(snapped_bboxes)
    for bi, trans_text in big_translations.items():
        lines = trans_text.split("\n")
        fontsize = 9
        line_height = fontsize * 1.6
        trans_h = len(lines) * line_height + 10
        max_line_len = max(len(l) for l in lines) if lines else 10
        trans_w = max(200, min(max_line_len * fontsize * 0.55, snapped_bboxes[bi][2] - snapped_bboxes[bi][0]))

        pos = find_white_space(str(pdf_path), snapped_bboxes[bi], all_bboxes, pw, ph, trans_w, trans_h)
        rect = fitz.Rect(pos[0], pos[1], pos[2], pos[3])
        all_bboxes.append(pos)

        annot = page_obj.add_freetext_annot(rect, trans_text, fontsize=fontsize,
            fontname="china-s", text_color=(0,0,0), fill_color=(1,1,1), align=fitz.TEXT_ALIGN_LEFT)
        annot.set_opacity(0.92)
        annot.update()
        print(f"    翻译 Block {bi+1} → ({pos[0]:.0f},{pos[1]:.0f})")

    # (5) 小块翻译注释
    for bi, trans in label_translations.items():
        bx1, by1, bx2, by2 = snapped_bboxes[bi]
        fontsize = 8
        tw = len(trans) * fontsize * 0.7
        th = fontsize * 1.5
        ty1 = by2 + 2
        if ty1 + th > ph - 30:
            ty1 = by1 - th - 2
        rect = fitz.Rect(bx1, ty1, bx1 + max(tw, 50), ty1 + th)
        annot = page_obj.add_freetext_annot(rect, trans, fontsize=fontsize,
            fontname="china-s", text_color=(0,0,0), fill_color=(1,1,1), align=fitz.TEXT_ALIGN_LEFT)
        annot.set_opacity(0.9)
        annot.update()

    # ── 一次保存 ──
    doc.save(output_path, incremental=True, encryption=0)
    doc.close()

    print(f"    翻译: {len(label_translations)} 标注 + {len(big_translations)} 大块")

    return product_name, decisions


def main():
    import argparse
    parser = argparse.ArgumentParser(description="工程图纸脱敏")
    parser.add_argument("input", help="输入 PDF 路径")
    parser.add_argument("-o", "--output", help="输出目录", default=None)
    args = parser.parse_args()

    pdf_path = Path(args.input).resolve()
    if not pdf_path.exists():
        print(f"文件不存在: {pdf_path}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else pdf_path.parent / "output"
    output_dir.mkdir(exist_ok=True)
    stem = pdf_path.stem

    # 获取页数
    doc = fitz.open(str(pdf_path))
    num_pages = len(doc)
    doc.close()

    # 复制原 PDF 作为输出基础（多页时在原件上逐页修改）
    output_pdf = output_dir / f"{stem}_redacted.pdf"
    import shutil
    shutil.copy2(str(pdf_path), str(output_pdf))

    print(f"{'='*50}")
    print(f"输入: {pdf_path.name} ({num_pages} 页)")
    print(f"{'='*50}")

    all_products = []
    total_keep, total_delete = 0, 0

    for page_num in range(num_pages):
        product_name, decisions = process_page(
            pdf_path, output_pdf, page_num, output_dir, stem
        )
        if product_name:
            all_products.append(product_name)
        total_keep += sum(1 for d in decisions.values() if d == "KEEP")
        total_delete += sum(1 for d in decisions.values() if d == "DELETE")

    # 压缩输出（去除 incremental save 冗余）
    doc = fitz.open(str(output_pdf))
    doc.save(str(output_pdf) + ".tmp", garbage=4, deflate=True)
    doc.close()
    import shutil as _shutil
    _shutil.move(str(output_pdf) + ".tmp", str(output_pdf))

    # 汇总
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
