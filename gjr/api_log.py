"""API 调用日志:每次调用一个 JSON 文件 + base64 图片外置为 PNG。

使用方式:
    set_log_context(enabled=True, page=1, output_dir=Path("output/zip2"))
    log_api_call(step="gpt_classify", api="openai.chat.completions",
                 request={...}, response={...}, usage={...}, images_b64=[...])

每次调用会在 <output_dir>/api_log/ 下生成:
    p1_01_gpt_classify.json              元数据 + request/response 文本
    p1_01_gpt_classify_crop_b1.png       第 1 张裁切图(base64 外置)
    p1_01_gpt_classify_crop_b2.png       ...
"""
import base64
import hashlib
import json
from datetime import datetime
from pathlib import Path

_LOG_CTX = {"enabled": False, "page": 0, "seq": 0, "dir": None}


def set_log_context(enabled, page, output_dir):
    """每页开始时调用,重置 seq 和目标目录。"""
    _LOG_CTX["enabled"] = bool(enabled)
    _LOG_CTX["page"] = page
    _LOG_CTX["seq"] = 0
    if enabled:
        log_dir = Path(output_dir) / "api_log"
        log_dir.mkdir(exist_ok=True)
        _LOG_CTX["dir"] = log_dir
    else:
        _LOG_CTX["dir"] = None


def log_api_call(step, api, request, response, usage=None, images_b64=None, duration_ms=None):
    """
    记录一次 API 调用。

    step        — 步骤名(baidu_ocr / gpt_classify / gpt_translate_labels / gpt_translate_block)
    api         — API 标识(如 openai.chat.completions)
    request     — 请求结构(任意可 JSON 序列化的字典)
    response    — 响应结构
    usage       — {"input_tokens", "output_tokens", "cost_usd"} 或 None
    images_b64  — [base64_str, ...] 会被外置成 PNG,JSON 里只留引用
    duration_ms — 调用耗时(毫秒)
    """
    if not _LOG_CTX["enabled"] or _LOG_CTX["dir"] is None:
        return
    _LOG_CTX["seq"] += 1
    seq = _LOG_CTX["seq"]
    page = _LOG_CTX["page"]
    base = f"p{page}_{seq:02d}_{step}"
    log_dir = _LOG_CTX["dir"]

    img_refs = []
    for i, b64 in enumerate(images_b64 or [], 1):
        try:
            data = base64.b64decode(b64)
        except Exception:
            continue
        fname = f"{base}_crop_b{i}.png"
        (log_dir / fname).write_bytes(data)
        img_refs.append({
            "index": i,
            "file": fname,
            "sha256": hashlib.sha256(data).hexdigest()[:12],
            "bytes": len(data),
        })

    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "page": page,
        "seq": seq,
        "step": step,
        "api": api,
        "duration_ms": duration_ms,
        "usage": usage or {},
        "request": request,
        "response": response,
    }
    if img_refs:
        record["request_images"] = img_refs

    (log_dir / f"{base}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
