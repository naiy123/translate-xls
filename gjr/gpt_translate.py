"""GPT 翻译:大块(含图片) + 批量小标注(纯文本) + 敏感词前后过滤。

两个入口函数:
  translate_block(text, image_b64)  — 单个大块(含图片上下文)
  batch_translate_labels(blocks, indices, block_texts) — 一次翻译多个小 block
"""
import re
import time

from gjr.api_log import log_api_call
from gjr.clients import get_client
from gjr.config import PRICING, load_prompt, load_sensitive


def filter_sensitive_lines(text):
    """
    替换文本中的敏感关键词为占位符,保留句子其余内容。
    返回 (替换后文本, 被替换的关键词列表)
    """
    replaced = []
    result = text
    for pat, placeholder in load_sensitive()["sensitive_replacements"]:
        matches = pat.findall(result)
        if matches:
            replaced.extend(matches)
            result = pat.sub(placeholder, result)
    return result, replaced


def _clean_translation(text):
    """清理 [品牌] 占位符和多余空格。"""
    text = text.replace("[品牌]", "").replace("[品牌] ", "").replace(" [品牌]", "")
    text = re.sub(r'  +', ' ', text).strip()
    return text


def translate_block(block_text, block_image):
    """翻译单个 block,自动过滤敏感内容。返回翻译文本或 None。"""
    clean_text, filtered = filter_sensitive_lines(block_text)
    if filtered:
        print(f"      过滤 {len(filtered)} 行敏感内容: {filtered[:3]}")
    if not clean_text.strip():
        return None

    cfg = load_prompt("translate_block")
    user_text = cfg["user_template"].format(text=clean_text)
    content = [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{block_image}", "detail": "high",
        }},
    ]

    _t0 = time.monotonic()
    response = get_client().chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": cfg["system"]},
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
        print(f"    Token: in={usage.prompt_tokens}, out={usage.completion_tokens}, ${cost:.4f}")
    result = response.choices[0].message.content or ""

    log_api_call(
        step="gpt_translate_block",
        api="openai.chat.completions",
        request={
            "model": cfg["model"],
            "temperature": cfg["temperature"],
            "max_completion_tokens": cfg["max_completion_tokens"],
            "system": cfg["system"],
            "user_text": user_text,
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
        images_b64=[block_image],
        duration_ms=_dur,
    )

    # 翻译后再替换一次(GPT 可能从图片里看到品牌名)
    result, post_filtered = filter_sensitive_lines(result)
    if post_filtered:
        print(f"      翻译后替换 {len(post_filtered)} 处品牌名")
    return _clean_translation(result)


def batch_translate_labels(blocks, indices, block_texts):
    """
    批量翻译小 block 标注文字(纯文本,不发图片,一次调用)。
    返回 {block_index: translated_text}
    """
    if not indices:
        return {}

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

    cfg = load_prompt("translate_labels")

    _t0 = time.monotonic()
    response = get_client().chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": cfg["system"]},
            {"role": "user", "content": input_text},
        ],
        temperature=cfg["temperature"],
        max_completion_tokens=cfg["max_completion_tokens"],
    )
    _dur = int((time.monotonic() - _t0) * 1000)
    usage = response.usage
    cost = None
    if usage:
        cost = usage.prompt_tokens * PRICING["input"] / 1e6 + usage.completion_tokens * PRICING["output"] / 1e6
        print(f"  Token: in={usage.prompt_tokens}, out={usage.completion_tokens}, ${cost:.4f}")

    log_api_call(
        step="gpt_translate_labels",
        api="openai.chat.completions",
        request={
            "model": cfg["model"],
            "temperature": cfg["temperature"],
            "max_completion_tokens": cfg["max_completion_tokens"],
            "system": cfg["system"],
            "user_text": input_text,
            "num_labels": len(lines),
        },
        response={
            "text": response.choices[0].message.content or "",
            "finish_reason": response.choices[0].finish_reason,
        },
        usage=({
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "cost_usd": round(cost, 6) if cost else None,
        } if usage else None),
        duration_ms=_dur,
    )

    result_text = response.choices[0].message.content or ""
    translations = {}
    for line in result_text.strip().split("\n"):
        m = re.match(r'\[(\d+)\]\s*(.+)', line.strip())
        if m:
            idx = int(m.group(1)) - 1
            if idx < len(valid_indices):
                translations[valid_indices[idx]] = _clean_translation(m.group(2).strip())

    return translations
