#!/usr/bin/env python3
"""向后兼容薄壳:真正的实现已搬到 gjr/ 包和 scripts/run_redact.py。

保留此文件只是为了让习惯 `python3 run_redact_v2.py foo.pdf` 的调用方继续可用。
新代码请直接用 `python3 scripts/run_redact.py` 或 `python3 -m scripts.run_redact`。
"""
from scripts.run_redact import main

if __name__ == "__main__":
    main()
