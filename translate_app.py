#!/usr/bin/env python3
"""Streamlit Cloud 入口薄壳 — 真正的 UI 代码在 streamlit_app.py。

Streamlit Cloud 的 Main file 路径是 translate_app.py(历史原因),
这里通过 exec 把 streamlit_app.py 的内容注入当前命名空间,
避免用户手动改 Cloud 设置。

等价于 "Main file path = streamlit_app.py",零侧效应。
"""
from pathlib import Path

_app_file = Path(__file__).resolve().parent / "streamlit_app.py"
exec(compile(_app_file.read_text(encoding="utf-8"), str(_app_file), "exec"))
