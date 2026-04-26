"""百度高精度 OCR 调用 + 结果解析 + 噪音过滤。

核心注意点:百度 OCR 返回的 location 是"渲染后图片像素"空间,
这里统一除回 200 DPI 的 PDF 点空间,下游所有模块(cluster/classify/render/preview)
都在 PDF 点空间工作,和 fitz 的 page.rect 对齐。
"""
import base64
import re

import fitz
import requests

from gjr.clients import baidu_access_token
from gjr.config import load_sensitive


def ocr_pdf(pdf_path, page_num=1):
    """百度高精度 OCR，先渲染为图片再识别（兼容纯矢量轮廓化 PDF）"""
    token = baidu_access_token()

    # 渲染指定页为 PNG（控制最大边 4000px，百度 API 限制 4096）
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    scale = 200 / 72  # 固定 200 DPI（图片约 3MB，在百度 10MB 限制内）
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img_data = base64.b64encode(pix.tobytes("png")).decode()
    doc.close()

    resp = requests.post(
        f"https://aip.baidubce.com/rest/2.0/ocr/v1/accurate?access_token={token}",
        data={
            "image": img_data,
            "recognize_granularity": "small",
            "detect_direction": "false",
            "probability": "true",
            "language_type": "ENG",
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


_SHORT_SYM_PAT = re.compile(r'^[A-Za-z0-9↑↓←→]+$')


def _in_rect(it, rect):
    if rect is None:
        return False
    rx1, ry1, rx2, ry2 = rect
    return it["left"] >= rx1 and it["top"] >= ry1 and it["right"] <= rx2 and it["bottom"] <= ry2


def filter_noise(items, title_rect=None):
    """剔除短符号和纯品牌名 bbox。

    short 符号过滤是为了清掉图纸绘图区的区位标识(↑/↓/A/D/1/2..),
    但标题栏里 FAN/CAD 列 P001/P002 等 3 字短词也会误伤。
    传入 title_rect 后,落在该矩形内的短 item 不再剔除,
    避免后续 DBSCAN 因缺失"垫脚石 item"把标题栏切碎。
    品牌名过滤对全页生效(品牌哪儿冒头都得删)。
    """
    brand_patterns = load_sensitive()["brand_patterns"]
    result = []
    for it in items:
        text = it["text"].strip()
        if not _in_rect(it, title_rect):
            if len(text) <= 3 and _SHORT_SYM_PAT.match(text):
                continue
        if any(p.match(text) for p in brand_patterns):
            continue
        result.append(it)
    return result
