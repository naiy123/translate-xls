"""Streamlit UI for gjr:工程图纸脱敏 + 翻译。

布局参考上一版 "图纸 & 表格翻译工具":
  - 侧边栏:API keys (OpenAI + 百度)
  - 主区域:标题 + 两个 tab(脱敏 / 翻译)

本地运行:
    streamlit run streamlit_app.py
"""
import io
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import fitz
import streamlit as st

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gjr import clients  # noqa: E402  (用于重置 _client 缓存)
from gjr.config import XLSX_LANGUAGES  # noqa: E402
from gjr.pipeline import process_page  # noqa: E402  (脱敏+翻译合一)
from gjr.xlsx import translate_xlsx_file  # noqa: E402


st.set_page_config(page_title="图纸 & 翻译工具", page_icon="📐", layout="wide")


# ══════════════════════════════════════════════════════════════
# 侧边栏:API Keys
# ══════════════════════════════════════════════════════════════

def _apply_key(env_name, value):
    """把 UI 上填写的 key 同步到环境变量,并使 OpenAI client 缓存失效。"""
    if value:
        os.environ[env_name] = value
        if env_name in ("OPENAI_API", "OPENAI_API_KEY"):
            clients._client = None  # 下次 get_client() 会用新 key 重建


with st.sidebar:
    st.header("设置")

    openai_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API", ""),
        type="password",
        help="用于 GPT 判断和翻译。优先级:UI > .env",
    )
    _apply_key("OPENAI_API", openai_key)

    st.divider()
    st.subheader("百度 OCR(图纸用)")
    baidu_key = st.text_input(
        "BAIDU_API_KEY",
        value=os.getenv("BAIDU_API_KEY", ""),
        type="password",
    )
    _apply_key("BAIDU_API_KEY", baidu_key)
    # 百度凭据在 clients 模块里是模块级常量,需要同步刷新
    clients.BAIDU_API_KEY = os.getenv("BAIDU_API_KEY")

    baidu_secret = st.text_input(
        "BAIDU_SECRET_KEY",
        value=os.getenv("BAIDU_SECRET_KEY", ""),
        type="password",
    )
    _apply_key("BAIDU_SECRET_KEY", baidu_secret)
    clients.BAIDU_SECRET_KEY = os.getenv("BAIDU_SECRET_KEY")

    st.divider()
    debug_preview = st.checkbox(
        "输出 debug 预览",
        value=True,
        help="生成 bbox / cluster / class 三张 PNG 和 API 调用日志",
    )


# ══════════════════════════════════════════════════════════════
# 主区域:标题 + Tabs
# ══════════════════════════════════════════════════════════════

st.title("📊 图纸 & 表格翻译工具")
st.caption("上传 Excel 表格或工程图纸 PDF,自动脱敏翻译")

tab_excel, tab_pdf = st.tabs([
    "📊 Excel 表格翻译",
    "📐 工程图纸脱敏翻译",
])


def _require_keys():
    """运行前校验必要 key 已填。"""
    missing = []
    if not (os.getenv("OPENAI_API") or os.getenv("OPENAI_API_KEY")):
        missing.append("OpenAI API Key")
    if not os.getenv("BAIDU_API_KEY"):
        missing.append("BAIDU_API_KEY")
    if not os.getenv("BAIDU_SECRET_KEY"):
        missing.append("BAIDU_SECRET_KEY")
    if missing:
        st.error(f"缺少配置: {', '.join(missing)} — 请在左侧 '设置' 中填写")
        return False
    return True


# ── Tab 0: Excel 翻译 ───────────────────────────
with tab_excel:
    st.markdown("**流程**:提取术语表 → 分批翻译(保持一致性) → 写回 xlsx(保留图片/格式)")
    st.caption("⚠️ `.xls` 老格式需要系统装 LibreOffice 才能保留图片,否则走降级路径")

    uploaded_x = st.file_uploader(
        "上传 Excel 文件",
        type=["xlsx", "xls"],
        key="excel_upload",
    )

    lang_options = {
        "en": "English (英语)",
        "th": "Thai (泰语)",
        "my": "Burmese (缅甸语)",
    }
    selected_langs = st.multiselect(
        "目标语言",
        options=list(lang_options.keys()),
        default=["en"],
        format_func=lambda x: lang_options[x],
    )

    if uploaded_x is not None:
        st.info(f"已选择: **{uploaded_x.name}** ({uploaded_x.size / 1024:.0f} KB)")
        run_x = st.button(
            "开始翻译",
            type="primary",
            key="run_excel",
            disabled=not selected_langs,
        )

        if run_x and _require_keys():
            with tempfile.TemporaryDirectory() as tmp_str:
                tmp_dir = Path(tmp_str)
                input_path = tmp_dir / uploaded_x.name
                input_path.write_bytes(uploaded_x.getbuffer())
                output_dir = tmp_dir / "output"
                output_dir.mkdir()

                # Streamlit 进度回调
                progress_bar = st.progress(0.0, text="初始化")
                status_box = st.empty()
                warnings = []

                class UIProgress:
                    def __init__(self):
                        self.lang_order = list(selected_langs)
                        self.cur_lang_idx = 0
                        self.cur_batch_total = 1

                    def on_phase(self, lang, msg):
                        status_box.info(f"[{lang}] {msg}")
                        if lang in self.lang_order:
                            self.cur_lang_idx = self.lang_order.index(lang)

                    def on_batch(self, lang, i, total):
                        self.cur_batch_total = total
                        pct = (self.cur_lang_idx + (i + 1) / total) / len(self.lang_order)
                        progress_bar.progress(
                            min(pct, 1.0),
                            text=f"[{lang}] 批次 {i+1}/{total}",
                        )

                    def on_warning(self, msg):
                        warnings.append(msg)
                        st.warning(msg)

                log_buf = io.StringIO()
                with redirect_stdout(log_buf):
                    result = translate_xlsx_file(
                        input_path, output_dir,
                        target_langs=selected_langs,
                        progress=UIProgress(),
                        debug_log=debug_preview,
                    )
                progress_bar.progress(1.0, text="完成")
                status_box.success("翻译完成")

                # 指标
                costs = result["costs"]
                meta = result["meta"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("总单元格", meta["num_cells"])
                c2.metric("已翻译", meta["num_translatable"])
                c3.metric("本次费用", f"${costs['cost_usd']:.4f}")
                c4.metric("生成语言", len(result["outputs"]))

                # 下载:多个文件打 zip
                import zipfile as _zf
                if len(result["outputs"]) == 1:
                    lang, out_path = next(iter(result["outputs"].items()))
                    st.download_button(
                        f"⬇️ 下载 {lang_options[lang]} xlsx",
                        data=out_path.read_bytes(),
                        file_name=out_path.name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="dl_excel_single",
                    )
                elif len(result["outputs"]) > 1:
                    zip_buf = io.BytesIO()
                    with _zf.ZipFile(zip_buf, "w", _zf.ZIP_DEFLATED) as zf:
                        for lang, out_path in result["outputs"].items():
                            zf.writestr(out_path.name, out_path.read_bytes())
                    zip_buf.seek(0)
                    st.download_button(
                        f"⬇️ 下载全部 ({len(result['outputs'])} 语言) zip",
                        data=zip_buf.getvalue(),
                        file_name=f"{input_path.stem}_translated.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key="dl_excel_zip",
                    )

                # 术语表展示
                for lang, glossary in result["glossaries"].items():
                    with st.expander(f"{lang_options[lang]} 术语表", expanded=False):
                        st.text(glossary)

                with st.expander("详细日志", expanded=False):
                    st.text(log_buf.getvalue())


# ── Tab 1: 图纸脱敏翻译(合并流程 + zip in-place 替换)──
with tab_pdf:
    st.markdown("**流程**:百度 OCR → 聚类 → GPT 判断 KEEP/DELETE → 涂白敏感 + 翻译标注 (两步合一)")
    st.caption(
        "⚠️ **zip 保持原结构**:只替换里面的 `lay`/`drw` PDF,其他文件(docx/igs/stp 等)原样保留,"
        "zip 名称、顺序、内部文件名全部不变"
    )

    uploaded_files = st.file_uploader(
        "上传 PDF 文件或 ZIP 压缩包(可多选)",
        type=["pdf", "zip"],
        accept_multiple_files=True,
        key="pdf_upload",
    )

    def _should_process(filename):
        """只处理文件名含 lay 或 drw 的 PDF。"""
        name_lower = filename.lower()
        return "lay" in name_lower or "drw" in name_lower

    if uploaded_files:
        total_size = sum(f.size for f in uploaded_files)
        st.info(f"已选择 **{len(uploaded_files)}** 个文件,总计 {total_size / 1024:.0f} KB")
        run_p = st.button("开始处理", type="primary", key="run_pdf", use_container_width=True)

        if run_p and _require_keys():
            import zipfile as _zf
            with tempfile.TemporaryDirectory() as tmp_str:
                tmp_dir = Path(tmp_str)
                work_dir = tmp_dir / "work"
                work_dir.mkdir()

                progress = st.progress(0.0, text="初始化")
                log_buf = io.StringIO()
                # outputs: list of {type, name, bytes, file_metas: [{name, pages, product, keep, delete}...]}
                outputs = []
                # 全局跳过的文件(不含 lay/drw 的独立上传 PDF)
                skipped_loose = []

                def _process_pdf_bytes(pdf_bytes, display_name):
                    """
                    把一段 PDF bytes 写到磁盘 → 跑 process_page 全页 → 压缩 → 返回 (new_bytes, meta)。
                    display_name 只用于日志/进度显示,不影响文件路径。
                    """
                    # 每个 PDF 单独子目录,避免 debug PNG / api_log 互相覆盖
                    safe_stem = Path(display_name).stem
                    work_sub = work_dir / safe_stem
                    work_sub.mkdir(parents=True, exist_ok=True)
                    in_path = work_sub / "input.pdf"
                    in_path.write_bytes(pdf_bytes)
                    out_path = work_sub / "output.pdf"
                    shutil.copy2(str(in_path), str(out_path))

                    doc = fitz.open(str(in_path))
                    num_pages = len(doc)
                    doc.close()

                    products = []
                    keep_count, delete_count = 0, 0
                    for pn in range(num_pages):
                        try:
                            with redirect_stdout(log_buf):
                                product_name, decisions = process_page(
                                    in_path, out_path, pn, work_sub, safe_stem,
                                    debug_preview=debug_preview,
                                )
                            if product_name:
                                products.append(product_name)
                            keep_count += sum(1 for d in decisions.values() if d == "KEEP")
                            delete_count += sum(1 for d in decisions.values() if d == "DELETE")
                        except Exception as e:
                            st.warning(f"{display_name} 第 {pn+1} 页出错: {e}")

                    # 压缩(去掉 incremental save 冗余)
                    _doc = fitz.open(str(out_path))
                    _doc.save(str(out_path) + ".tmp", garbage=4, deflate=True)
                    _doc.close()
                    shutil.move(str(out_path) + ".tmp", str(out_path))

                    return out_path.read_bytes(), {
                        "name": display_name,
                        "pages": num_pages,
                        "product": products[0] if products else "未识别",
                        "keep": keep_count,
                        "delete": delete_count,
                        "work_sub": work_sub,
                        "safe_stem": safe_stem,
                    }

                # ── 遍历每个上传 ──
                for ui, uf in enumerate(uploaded_files):
                    ufname_lower = uf.name.lower()
                    progress.progress(
                        ui / max(len(uploaded_files), 1),
                        text=f"[{ui+1}/{len(uploaded_files)}] {uf.name}",
                    )

                    if ufname_lower.endswith(".zip"):
                        # ── ZIP 路径:in-place 替换 ──
                        orig_bytes = uf.getbuffer().tobytes()
                        with _zf.ZipFile(io.BytesIO(orig_bytes), "r") as zf_in:
                            all_entries = zf_in.infolist()  # 保留原顺序
                            # 找 lay/drw PDFs
                            pdfs_to_process = [
                                zi for zi in all_entries
                                if not zi.is_dir()
                                and zi.filename.lower().endswith(".pdf")
                                and not zi.filename.startswith("__MACOSX")
                                and _should_process(Path(zi.filename).name)
                            ]
                            skipped_in_zip = [
                                Path(zi.filename).name for zi in all_entries
                                if not zi.is_dir() and not zi.filename.startswith("__MACOSX")
                                and Path(zi.filename).name not in [Path(p.filename).name for p in pdfs_to_process]
                            ]

                            if not pdfs_to_process:
                                st.warning(f"`{uf.name}` 里没有 lay/drw PDF,跳过")
                                continue

                            # 处理每个 lay/drw PDF
                            processed_bytes = {}  # entry.filename → new bytes
                            file_metas = []
                            for fi, zi in enumerate(pdfs_to_process):
                                frac = (ui + (fi + 1) / max(len(pdfs_to_process), 1)) / max(len(uploaded_files), 1)
                                progress.progress(
                                    min(frac, 1.0),
                                    text=f"[{uf.name}] {Path(zi.filename).name} ({fi+1}/{len(pdfs_to_process)})",
                                )
                                pdf_bytes = zf_in.read(zi)
                                new_bytes, meta = _process_pdf_bytes(pdf_bytes, Path(zi.filename).name)
                                processed_bytes[zi.filename] = new_bytes
                                file_metas.append(meta)

                            # 重建新 zip:按原顺序,替换处理过的
                            new_zip_buf = io.BytesIO()
                            with _zf.ZipFile(new_zip_buf, "w", _zf.ZIP_DEFLATED) as zf_out:
                                for zi in all_entries:
                                    if zi.filename in processed_bytes:
                                        zf_out.writestr(zi, processed_bytes[zi.filename])
                                    else:
                                        zf_out.writestr(zi, zf_in.read(zi))
                            new_zip_buf.seek(0)

                        outputs.append({
                            "type": "zip",
                            "name": uf.name,   # 完全同名
                            "bytes": new_zip_buf.getvalue(),
                            "file_metas": file_metas,
                            "skipped_inside": skipped_in_zip,
                        })

                    elif ufname_lower.endswith(".pdf"):
                        # ── 独立 PDF 路径 ──
                        if not _should_process(uf.name):
                            skipped_loose.append(uf.name)
                            continue
                        progress.progress(
                            (ui + 0.5) / max(len(uploaded_files), 1),
                            text=f"[{ui+1}/{len(uploaded_files)}] {uf.name}",
                        )
                        new_bytes, meta = _process_pdf_bytes(uf.getbuffer().tobytes(), uf.name)
                        outputs.append({
                            "type": "pdf",
                            "name": uf.name,   # 完全同名,零后缀
                            "bytes": new_bytes,
                            "file_metas": [meta],
                            "skipped_inside": [],
                        })

                progress.progress(1.0, text="完成")

                # ── 汇总 ──
                if skipped_loose:
                    st.info(
                        f"跳过 {len(skipped_loose)} 个不含 lay/drw 的独立文件: "
                        f"{', '.join(skipped_loose[:5])}" + (" 等" if len(skipped_loose) > 5 else "")
                    )
                if not outputs:
                    st.warning("没有任何文件被处理")
                    st.stop()

                total_pages = sum(m["pages"] for out in outputs for m in out["file_metas"])
                total_keep = sum(m["keep"] for out in outputs for m in out["file_metas"])
                total_delete = sum(m["delete"] for out in outputs for m in out["file_metas"])
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("输出产物", len(outputs))
                c2.metric("总页数", total_pages)
                c3.metric("保留", total_keep)
                c4.metric("删除", total_delete)

                # ── 下载:每个输出一个按钮,文件名完全保留原名 ──
                st.subheader("下载")
                for oi, out in enumerate(outputs):
                    icon = "📦" if out["type"] == "zip" else "📄"
                    mime = "application/zip" if out["type"] == "zip" else "application/pdf"
                    size_kb = len(out["bytes"]) / 1024
                    n_inner = len(out["file_metas"])
                    label = (
                        f"⬇️ {icon} {out['name']}  "
                        f"({size_kb:.0f} KB" + (f", {n_inner} 文件处理" if out["type"] == "zip" else "") + ")"
                    )
                    st.download_button(
                        label,
                        data=out["bytes"],
                        file_name=out["name"],
                        mime=mime,
                        use_container_width=True,
                        key=f"dl_{oi}",
                    )

                # ── 各输出明细 ──
                with st.expander(f"处理明细 (共 {sum(len(o['file_metas']) for o in outputs)} 个 PDF)", expanded=False):
                    for out in outputs:
                        st.markdown(f"**{out['type'].upper()}: `{out['name']}`**")
                        for m in out["file_metas"]:
                            st.markdown(
                                f"- `{m['name']}` — {m['pages']} 页 | "
                                f"保留 {m['keep']} / 删除 {m['delete']} | "
                                f"产品:{m['product']}"
                            )
                        if out["skipped_inside"]:
                            st.caption(
                                f"  (zip 内 {len(out['skipped_inside'])} 个其他文件原样保留: "
                                f"{', '.join(out['skipped_inside'][:3])}"
                                + (" 等" if len(out['skipped_inside']) > 3 else "") + ")"
                            )

                with st.expander("详细日志", expanded=False):
                    st.text(log_buf.getvalue())

                # ── Debug 预览(单 PDF / 单 zip 含单 PDF 时展示)──
                if debug_preview and len(outputs) == 1 and len(outputs[0]["file_metas"]) == 1:
                    st.subheader("Debug 预览")
                    meta = outputs[0]["file_metas"][0]
                    work_sub = meta["work_sub"]
                    safe_stem = meta["safe_stem"]
                    for pn in range(meta["pages"]):
                        page_label = f"p{pn+1}"
                        st.markdown(f"**第 {pn+1} 页**")
                        preview_tabs = st.tabs(["原始 bbox", "聚类", "分类"])
                        for tab, suffix in zip(preview_tabs, ["bbox", "cluster", "class"]):
                            png_path = work_sub / f"{safe_stem}_{page_label}_{suffix}.png"
                            if png_path.exists():
                                tab.image(str(png_path), use_container_width=True)
