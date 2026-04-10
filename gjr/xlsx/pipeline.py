"""Excel 翻译 pipeline 的编排者。

对外入口:
    translate_xlsx_file(
        input_path,          # 上传的 .xls 或 .xlsx
        output_dir,          # 输出目录
        target_langs,        # ["en", "th", "my"] 等,来自 XLSX_LANGUAGES
        progress_callback=None,  # 可选 Streamlit 进度回调
        debug_log=False,     # True 时 API 调用写 api_log/
    ) -> {
        "outputs":  {lang: Path},   # 每个语言一个 xlsx
        "costs":    {"input_tokens", "output_tokens", "cost_usd"},
        "glossaries": {lang: str},   # 术语表(调试用)
        "meta":     {"num_cells", "num_translatable"},
    }
"""
from pathlib import Path

from gjr.api_log import set_log_context
from gjr.config import MODEL_PRICING, XLSX_BATCH_SIZE, XLSX_LANGUAGES, XLSX_MODEL_DEFAULT
from gjr.xlsx.reader import ensure_xlsx, is_translatable, read_file
from gjr.xlsx.translate import (
    build_table_context,
    extract_glossary,
    group_by_rows,
    parse_translation_result,
    translate_batch,
)
from gjr.xlsx.writer import save_translated_xlsx


class XlsxTranslateProgress:
    """进度回调协议(duck typed):
      on_phase(lang, phase_name)
      on_batch(lang, batch_idx, total_batches)
      on_warning(msg)
    """

    def on_phase(self, lang, phase_name): ...
    def on_batch(self, lang, batch_idx, total_batches): ...
    def on_warning(self, msg): ...


def translate_xlsx_file(input_path, output_dir, target_langs,
                        progress=None, debug_log=False):
    """
    把一份 Excel 翻译成多种目标语言。

    返回一个 dict(结构见模块 docstring)。
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    stem = input_path.stem

    set_log_context(debug_log, 1, output_dir)

    # 1. .xls 尝试转 .xlsx 保留图片;失败(libreoffice 缺失)则原样送到 save 层走 fallback
    effective_path = str(input_path)
    if input_path.suffix.lower() == '.xls':
        try:
            effective_path = ensure_xlsx(str(input_path))
            print(f".xls → .xlsx 转换成功(保留图片): {effective_path}")
        except Exception as e:
            print(f"LibreOffice 不可用,降级为无图片路径: {e}")
            # effective_path 保持原 .xls,下游 save_translated_xlsx 走 xls_to_xlsx fallback

    # 2. 读文件
    cells, nrows, ncols, sheet_name = read_file(effective_path)
    translatable = {k: v for k, v in cells.items() if is_translatable(v)}
    print(f"共 {len(cells)} 个单元格, {len(translatable)} 个需要翻译")

    table_context = build_table_context(cells, nrows, ncols)

    total_tokens = {"input": 0, "output": 0}
    outputs = {}
    glossaries = {}

    for lang in target_langs:
        if lang not in XLSX_LANGUAGES:
            if progress:
                progress.on_warning(f"未知语言 {lang},跳过")
            continue
        lang_name = XLSX_LANGUAGES[lang]

        # 阶段 1:术语表
        if progress:
            progress.on_phase(lang, f"提取 {lang_name} 术语表")
        try:
            glossary, g_tokens = extract_glossary(translatable, lang)
        except Exception as e:
            if progress:
                progress.on_warning(f"{lang_name} 术语提取失败: {e}")
            continue
        glossaries[lang] = glossary
        total_tokens["input"] += g_tokens["input"]
        total_tokens["output"] += g_tokens["output"]

        # 阶段 2:分批翻译
        keys = sorted(translatable.keys())
        batches = group_by_rows(keys, XLSX_BATCH_SIZE)
        all_translations = {}

        for batch_idx, batch_keys in enumerate(batches):
            if progress:
                progress.on_batch(lang, batch_idx, len(batches))
            try:
                result, b_tokens = translate_batch(
                    batch_keys, translatable, table_context, glossary, lang,
                )
                total_tokens["input"] += b_tokens["input"]
                total_tokens["output"] += b_tokens["output"]
                parsed = parse_translation_result(result)
                all_translations.update(parsed)
            except Exception as e:
                # 单次重试
                if progress:
                    progress.on_warning(f"批次 {batch_idx+1} 失败: {e},重试")
                try:
                    result, b_tokens = translate_batch(
                        batch_keys, translatable, table_context, glossary, lang,
                    )
                    total_tokens["input"] += b_tokens["input"]
                    total_tokens["output"] += b_tokens["output"]
                    parsed = parse_translation_result(result)
                    all_translations.update(parsed)
                except Exception as e2:
                    if progress:
                        progress.on_warning(f"批次 {batch_idx+1} 再次失败: {e2},跳过")

        # 补上不需要翻译的单元格(纯数字/标点等)保持原值
        for k, v in cells.items():
            if k not in all_translations:
                all_translations[k] = v

        # 写结果
        output_path = output_dir / f"{stem}_{lang}.xlsx"
        try:
            save_translated_xlsx(effective_path, all_translations, str(output_path))
            outputs[lang] = output_path
            print(f"{lang_name}: {len(all_translations)} 个单元格写入 {output_path.name}")
        except Exception as e:
            if progress:
                progress.on_warning(f"保存 {lang_name} 结果失败: {e}")

    # 成本
    pricing = MODEL_PRICING.get(XLSX_MODEL_DEFAULT, {"input": 0, "output": 0})
    cost_usd = (
        total_tokens["input"] / 1_000_000 * pricing["input"]
        + total_tokens["output"] / 1_000_000 * pricing["output"]
    )

    return {
        "outputs": outputs,
        "costs": {
            "input_tokens": total_tokens["input"],
            "output_tokens": total_tokens["output"],
            "cost_usd": round(cost_usd, 6),
        },
        "glossaries": glossaries,
        "meta": {
            "num_cells": len(cells),
            "num_translatable": len(translatable),
        },
    }
