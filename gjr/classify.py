"""Block 分类规则 + 空间描述生成。

classify_block 输出四种标签:
  GPT         — 大块(>4 items),交给 GPT 判断 KEEP/DELETE
  AUTO_KEEP   — 纯尺寸/符号,自动保留且不翻译
  TRANSLATE   — 有意义英文文字,保留 + 送翻译
  AUTO_DELETE — 已知要删的内容(UNCONTROLLED / DO NOT SCALE 等)
"""
import re

from gjr.cluster import group_into_rows
from gjr.config import load_sensitive

WORD_PAT = re.compile(r'[A-Za-z]{4,}')
DWG_PAT = re.compile(r'\d{3}[A-Z]\d{4}')


def classify_block(block):
    all_text = " ".join(it["text"] for it in block)
    # 大块始终发 GPT(即使含 UNCONTROLLED 等,GPT 还需要提取产品名)
    if len(block) > 4:
        return "GPT"
    auto_delete_patterns = load_sensitive()["auto_delete_patterns"]
    if any(p.search(all_text) for p in auto_delete_patterns):
        return "AUTO_DELETE"
    if DWG_PAT.search(all_text):
        return "GPT"
    if WORD_PAT.search(all_text):
        avg_conf = sum(it["conf"] for it in block) / len(block)
        if avg_conf < 0.5:
            return "AUTO_KEEP"
        return "TRANSLATE"
    if len(all_text) < 30:
        return "AUTO_KEEP"
    return "GPT"


def block_to_spatial(block):
    """把 block 里的 items 按行分组,输出人类可读的多行字符串(供 GPT prompt 使用)。"""
    rows = group_into_rows(block)
    lines = []
    for row in rows:
        parts = []
        for it in row:
            conf_mark = "" if it["conf"] >= 0.9 else f" (?{it['conf']:.0%})"
            parts.append(f"{it['text']}{conf_mark}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)
