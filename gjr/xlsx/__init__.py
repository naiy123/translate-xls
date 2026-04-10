"""Excel 表格翻译子包。

对外暴露:
    from gjr.xlsx import translate_xlsx_file
"""
from gjr.xlsx.pipeline import translate_xlsx_file

__all__ = ["translate_xlsx_file"]
