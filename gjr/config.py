"""集中管理调参常量和提示词加载。

任何涉及"这个数要调一下"的常量都放这里,不要散落在业务代码里。
提示词以 YAML 存在 gjr/prompts/ 下,运行时 lazy 加载并缓存。
"""
import re
from pathlib import Path

import yaml

# ── 模型与计费 ────────────────────────────────────────
# 图纸脱敏/翻译流水线默认用 mini(要看图,精度高)
MODEL = "gpt-5.4-mini"
PRICING = {"input": 0.75, "output": 4.50}

# 每个 model 的价格表(per-million tokens, USD);Excel 流水线会按实际 model 查表
MODEL_PRICING = {
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4":      {"input": 2.50, "output": 15.00},
    "gpt-4o":       {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":  {"input": 0.15, "output": 0.60},
}

# ── Excel 翻译专用配置 ────────────────────────────
# Excel 翻译成本敏感(大表可能 500+ 单元格),用更便宜的 nano
XLSX_MODEL_DEFAULT = "gpt-5.4-nano"
# 按语言可以指定不同 model(保留扩展空间,默认都用 nano)
XLSX_MODEL_PER_LANG = {
    "en": "gpt-5.4-nano",
    "th": "gpt-5.4-nano",
    "my": "gpt-5.4-nano",
}
XLSX_BATCH_SIZE = 25  # 每批翻译的最大单元格数
XLSX_LANGUAGES = {
    "en": "English",
    "th": "Thai (ภาษาไทย)",
    "my": "Burmese (မြန်မာဘာသာ)",
}

# ── 聚类 / snap 参数 ─────────────────────────────────
DBSCAN_EPS = 30      # DBSCAN 邻域半径(bbox 边缘间距,单位 pt)
SNAP_MARGIN = 50     # snap 到表格线时的最大搜索距离
ROW_OVERLAP = 0.3    # 判断两个 item 同一行的垂直重叠阈值

# ── Debug 预览调色板 ─────────────────────────────────
PREVIEW_PALETTE = [
    (0.90, 0.20, 0.20), (0.20, 0.55, 0.90), (0.20, 0.75, 0.30),
    (0.90, 0.60, 0.10), (0.65, 0.30, 0.85), (0.10, 0.70, 0.70),
    (0.90, 0.40, 0.60), (0.50, 0.55, 0.25),
]

CLASS_COLORS = {
    "AUTO_KEEP":   (0.15, 0.70, 0.20),
    "TRANSLATE":   (0.15, 0.45, 0.90),
    "GPT":         (0.90, 0.65, 0.10),
    "AUTO_DELETE": (0.90, 0.15, 0.15),
}


# ── 提示词与敏感词 YAML 加载 ─────────────────────────
_PROMPTS_DIR = Path(__file__).parent / "prompts"
_prompt_cache: dict = {}


def load_prompt(name: str) -> dict:
    """加载 gjr/prompts/{name}.yaml,返回 dict(含 system / user_template / model 等)。"""
    if name not in _prompt_cache:
        path = _PROMPTS_DIR / f"{name}.yaml"
        with open(path, encoding="utf-8") as f:
            _prompt_cache[name] = yaml.safe_load(f)
    return _prompt_cache[name]


def _compile_patterns(patterns):
    """把字符串列表编译成 re.Pattern;支持 (?i) 等内联标志。"""
    return [re.compile(p) for p in patterns]


_sensitive_cache = None


def load_sensitive() -> dict:
    """
    加载 sensitive.yaml 并编译正则。返回:
      {
        "brand_patterns":          [re.Pattern, ...],
        "auto_delete_patterns":    [re.Pattern, ...],
        "sensitive_replacements":  [(re.Pattern, replacement_str), ...],
        "brand_inline_patterns":   [re.Pattern, ...],
      }
    """
    global _sensitive_cache
    if _sensitive_cache is not None:
        return _sensitive_cache
    raw = yaml.safe_load((_PROMPTS_DIR / "sensitive.yaml").read_text(encoding="utf-8"))
    _sensitive_cache = {
        "brand_patterns": _compile_patterns(raw.get("brand_patterns", [])),
        "auto_delete_patterns": _compile_patterns(raw.get("auto_delete_patterns", [])),
        "sensitive_replacements": [
            (re.compile(r["pattern"]), r["replacement"])
            for r in raw.get("sensitive_replacements", [])
        ],
        "brand_inline_patterns": _compile_patterns(raw.get("brand_inline_patterns", [])),
    }
    return _sensitive_cache
