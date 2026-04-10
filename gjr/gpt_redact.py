"""GPT 脱敏判断:裁切 block 图 + 拼 user 消息 + 调用 + 解析。

提示词(system / user_intro)从 gjr/prompts/classify.yaml 加载。
"""
import base64
import re
import time

import fitz

from gjr.api_log import log_api_call
from gjr.classify import block_to_spatial  # noqa: F401  (历史依赖,避免循环)
from gjr.clients import get_client
from gjr.cluster import block_position
from gjr.config import PRICING, load_prompt


def crop_block_images(pdf_path, blocks, snapped_bboxes, dpi=200, pad=5):
    """用 snapped bbox 裁切 block 图片(第 0 页),返回 base64 列表。"""
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
    cfg = load_prompt("classify")
    system_prompt = cfg["system"]
    user_intro = cfg["user_intro"]

    content = [{"type": "text", "text": user_intro}]
    for bi in range(len(blocks)):
        bbox = snapped_bboxes[bi]
        pos = block_position(blocks[bi])
        content.append({
            "type": "text",
            "text": (f"\n--- [Block {bi+1}] 位置: {pos}, 坐标: "
                     f"({bbox[0]:.0f},{bbox[1]:.0f})-({bbox[2]:.0f},{bbox[3]:.0f}), "
                     f"{len(blocks[bi])} 项 ---\nOCR 文本:\n{block_texts[bi]}\n\n裁切图片:"),
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{block_images[bi]}", "detail": "high"},
        })

    _t0 = time.monotonic()
    response = get_client().chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
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
    answer = response.choices[0].message.content

    req_text_parts = [c["text"] for c in content if c.get("type") == "text"]
    log_api_call(
        step="gpt_classify",
        api="openai.chat.completions",
        request={
            "model": cfg["model"],
            "temperature": cfg["temperature"],
            "max_completion_tokens": cfg["max_completion_tokens"],
            "system": system_prompt,
            "user_text": "".join(req_text_parts),
            "num_blocks": len(blocks),
        },
        response={
            "text": answer,
            "finish_reason": response.choices[0].finish_reason,
        },
        usage=({
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "cost_usd": round(cost, 6) if cost else None,
        } if usage else None),
        images_b64=block_images,
        duration_ms=_dur,
    )
    return answer


def parse_gpt_result(text, num_blocks):
    """解析 GPT 返回的 [Block N] KEEP/DELETE 和 [Product] 行。"""
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
