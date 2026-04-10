"""Excel 翻译核心:表格上下文构建 + 术语表 + 批量翻译 + 结果解析。

两阶段流程:
  1. extract_glossary —— 先从整表抽 20-40 个术语,供后续翻译保持一致性
  2. translate_batch  —— 按行分批翻译,system prompt 里带术语表

Prompt 从 gjr/prompts/xlsx_*.yaml 加载,用 string.Template 的 $var 语法替换。
"""
import re
import time
from collections import defaultdict
from string import Template

from gjr.api_log import log_api_call
from gjr.clients import get_client
from gjr.config import (
    MODEL_PRICING,
    XLSX_BATCH_SIZE,
    XLSX_LANGUAGES,
    XLSX_MODEL_DEFAULT,
    XLSX_MODEL_PER_LANG,
    load_prompt,
)


def build_table_context(all_cells, nrows, ncols):
    """把整个表格渲染成 `| R1 | A | B | ...` 的 markdown-ish 文本,给 GPT 看上下文。"""
    used_cols = sorted(set(col for _, col in all_cells.keys()))
    if not used_cols:
        return ""
    lines = []
    for row in range(nrows):
        row_cells = []
        has_content = False
        for col in used_cols:
            val = all_cells.get((row, col), "")
            val = str(val).replace('\n', ' ').strip()[:30]
            if val:
                has_content = True
            row_cells.append(val)
        if has_content:
            lines.append(f"| R{row+1} | " + " | ".join(row_cells) + " |")
    return '\n'.join(lines)


def format_cells_for_prompt(cells, keys=None):
    """把 cells 渲染成 `R1C1: "值"` 行,给 GPT 逐格翻译。"""
    if keys is None:
        keys = sorted(cells.keys())
    lines = []
    for row, col in keys:
        val = cells[(row, col)]
        val_escaped = val.replace('\n', '\\n')
        lines.append(f'R{row+1}C{col+1}: "{val_escaped}"')
    return '\n'.join(lines)


def group_by_rows(keys, max_cells=XLSX_BATCH_SIZE):
    """按行分批,保证一行的单元格尽量在同一批里(翻译上下文连贯)。"""
    row_groups = defaultdict(list)
    for row, col in keys:
        row_groups[row].append((row, col))
    batches = []
    current_batch = []
    for row in sorted(row_groups.keys()):
        row_keys = row_groups[row]
        if current_batch and len(current_batch) + len(row_keys) > max_cells:
            batches.append(current_batch)
            current_batch = []
        current_batch.extend(row_keys)
    if current_batch:
        batches.append(current_batch)
    return batches


def _call_llm(system_prompt, user_prompt, temperature, lang, step):
    """单次 OpenAI 调用封装,自动记日志 + 返回 tokens。"""
    model = XLSX_MODEL_PER_LANG.get(lang, XLSX_MODEL_DEFAULT)
    _t0 = time.monotonic()
    response = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    _dur = int((time.monotonic() - _t0) * 1000)
    usage = response.usage
    tokens = {"input": 0, "output": 0, "model": model}
    cost = None
    if usage:
        tokens["input"] = usage.prompt_tokens
        tokens["output"] = usage.completion_tokens
        pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
        cost = usage.prompt_tokens * pricing["input"] / 1e6 + usage.completion_tokens * pricing["output"] / 1e6

    text = response.choices[0].message.content or ""

    log_api_call(
        step=step,
        api="openai.chat.completions",
        request={
            "model": model,
            "temperature": temperature,
            "lang": lang,
            "system": system_prompt,
            "user": user_prompt,
        },
        response={
            "text": text,
            "finish_reason": response.choices[0].finish_reason,
        },
        usage=({
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "cost_usd": round(cost, 6) if cost else None,
        } if usage else None),
        duration_ms=_dur,
    )
    return text, tokens


def _resolve_fallback(prompt_cfg, key, lang_name, target_lang):
    """如果目标语言是 th/my,返回 fallback 规则(已把 $lang_name 展开);否则返回空串。"""
    if target_lang not in ("th", "my"):
        return ""
    raw = prompt_cfg.get(key, "")
    return Template(raw).safe_substitute(lang_name=lang_name)


def extract_glossary(cells, target_lang):
    """第一阶段:从整表抽术语表。返回 (glossary_text, tokens_dict)。"""
    cfg = load_prompt("xlsx_glossary")
    lang_name = XLSX_LANGUAGES[target_lang]

    all_text = format_cells_for_prompt(cells)
    fallback_rule = _resolve_fallback(cfg, "fallback_glossary_rule_th_my", lang_name, target_lang)

    system_prompt = Template(cfg["system"]).safe_substitute(lang_name=lang_name)
    user_prompt = Template(cfg["user"]).safe_substitute(
        lang_name=lang_name,
        fallback_glossary_rule=fallback_rule,
        all_text=all_text,
    )

    return _call_llm(
        system_prompt, user_prompt,
        temperature=cfg.get("temperature", 0.1),
        lang=target_lang,
        step=f"xlsx_glossary_{target_lang}",
    )


def translate_batch(batch_keys, all_cells, table_context, glossary, target_lang):
    """第二阶段:按批翻译单元格。返回 (raw_text, tokens_dict)。"""
    cfg = load_prompt("xlsx_translate_batch")
    lang_name = XLSX_LANGUAGES[target_lang]
    batch_text = format_cells_for_prompt(all_cells, keys=batch_keys)
    fallback_rule = _resolve_fallback(cfg, "fallback_rule_th_my", lang_name, target_lang)

    system_prompt = Template(cfg["system"]).safe_substitute(
        lang_name=lang_name,
        glossary=glossary,
        fallback_rule=fallback_rule,
    )
    user_prompt = Template(cfg["user"]).safe_substitute(
        lang_name=lang_name,
        table_context=table_context,
        batch_text=batch_text,
    )

    return _call_llm(
        system_prompt, user_prompt,
        temperature=cfg.get("temperature", 0.2),
        lang=target_lang,
        step=f"xlsx_translate_batch_{target_lang}",
    )


_RESULT_LINE_QUOTED = re.compile(r'R(\d+)C(\d+):\s*"(.+)"')
_RESULT_LINE_UNQUOTED = re.compile(r'R(\d+)C(\d+):\s*(.+)')


def parse_translation_result(result_text):
    """把 `R1C2: "译文"` 解析成 {(row, col): translated}。"""
    translations = {}
    for line in result_text.strip().splitlines():
        line = line.strip()
        match = _RESULT_LINE_QUOTED.match(line)
        if not match:
            match = _RESULT_LINE_UNQUOTED.match(line)
        if match:
            row = int(match.group(1)) - 1
            col = int(match.group(2)) - 1
            val = match.group(3).strip().strip('"')
            val = val.replace('\\n', '\n')
            translations[(row, col)] = val
    return translations
