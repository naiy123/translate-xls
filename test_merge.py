#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨页表格合并效果验证工具

用法：
  python test_merge.py <原始md> <合并后md>

示例：
  python test_merge.py output/xxx_baidu.md output/xxx_merged.md

验证项：
  1. 差异检测：找出合并后新产生的行（原始中不存在的），即被合并修改的行
  2. 误拼接检测：数字直接跟中文、中文直接跟英文等异常模式
  3. 残余行检测：合并后是否还有残留的空首列行
  4. 表格数量统计：合并前后表格数变化
"""

import re
import sys

RE_TABLE = re.compile(r'<table\b[^>]*>.*?</table>', re.DOTALL)
RE_ROW = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL)
RE_CELL = re.compile(r'<td[^>]*>.*?</td>', re.DOTALL)
RE_HTML_TAG = re.compile(r'<[^>]+>')


def cell_text(html):
    return RE_HTML_TAG.sub('', html).strip()


def extract_all_rows(md):
    """提取 md 中所有表格的所有行，返回 set of tuples"""
    rows = set()
    for t in RE_TABLE.findall(md):
        for r in RE_ROW.findall(t):
            cells = tuple(cell_text(c) for c in RE_CELL.findall(r))
            if cells:
                rows.add(cells)
    return rows


def check_new_rows(raw_md, merged_md):
    """检查合并后新产生的行（被合并修改的行）"""
    raw_rows = extract_all_rows(raw_md)
    issues = []

    for t in RE_TABLE.findall(merged_md):
        rows = RE_ROW.findall(t)
        for j, r in enumerate(rows):
            cells = tuple(cell_text(c) for c in RE_CELL.findall(r))
            if cells and cells not in raw_rows:
                issues.append(('NEW_ROW', j, list(cells)))

    return issues


def check_bad_concat(merged_md):
    """检查疑似误拼接的单元格内容"""
    issues = []

    for i, t in enumerate(RE_TABLE.findall(merged_md)):
        rows = RE_ROW.findall(t)
        for j, r in enumerate(rows):
            cells = [cell_text(c) for c in RE_CELL.findall(r)]
            for k, c in enumerate(cells):
                if not c:
                    continue
                # 数字后直接跟中文（如"4入侵防范"）
                if re.search(r'\d[\u4e00-\u9fff]', c) and len(c) > 3:
                    issues.append(('NUM_CN', f'表{i+1}行{j}列{k}', c[:50]))
                # 英文/数字后无空格直接跟中文（如"aServer-R-2305深信服"，但排除正常品牌名）
                if re.search(r'[a-zA-Z0-9][\u4e00-\u9fff]{2,}', c) and len(c) > 15:
                    issues.append(('EN_CN', f'表{i+1}行{j}列{k}', c[:50]))

    return issues


def check_remaining_residuals(merged_md):
    """检查合并后是否还有残留的空首列行"""
    issues = []

    for i, t in enumerate(RE_TABLE.findall(merged_md)):
        rows = RE_ROW.findall(t)
        for j in range(1, len(rows)):  # 跳过表头
            cells = [cell_text(c) for c in RE_CELL.findall(rows[j])]
            if cells and not cells[0]:
                non_empty = [c for c in cells if c]
                ne_pct = len(non_empty) / len(cells) * 100
                issues.append(('RESIDUAL', f'表{i+1}行{j}', f'非空{len(non_empty)}/{len(cells)}={ne_pct:.0f}%', non_empty[:3]))

    return issues


def main():
    if len(sys.argv) < 3:
        print("用法: python test_merge.py <原始md> <合并后md>")
        sys.exit(1)

    raw_path, merged_path = sys.argv[1], sys.argv[2]

    with open(raw_path, encoding='utf-8') as f:
        raw_md = f.read()
    with open(merged_path, encoding='utf-8') as f:
        merged_md = f.read()

    raw_tables = RE_TABLE.findall(raw_md)
    merged_tables = RE_TABLE.findall(merged_md)
    print(f"表格数量: {len(raw_tables)} → {len(merged_tables)} (合并了 {len(raw_tables) - len(merged_tables)} 个)")
    print()

    # 1. 新产生的行
    new_rows = check_new_rows(raw_md, merged_md)
    print(f"=== 合并产生的新行: {len(new_rows)} 个 ===")
    for typ, row_idx, cells in new_rows:
        summary = [c[:15] for c in cells if c]
        print(f"  行{row_idx}: {summary}")
    print()

    # 2. 疑似误拼接
    bad_concat = check_bad_concat(merged_md)
    print(f"=== 疑似误拼接: {len(bad_concat)} 个 ===")
    for typ, loc, content in bad_concat:
        tag = "数字+中文" if typ == 'NUM_CN' else "英文+中文"
        print(f"  {loc} [{tag}]: {content}")
    print()

    # 3. 残留空首列行
    residuals = check_remaining_residuals(merged_md)
    print(f"=== 残留空首列行: {len(residuals)} 个 ===")
    for typ, loc, pct, content in residuals:
        print(f"  {loc} {pct}: {content}")
    print()

    # 总结
    total_issues = len(bad_concat)  # 只有误拼接算真正的问题
    if total_issues == 0:
        print("✅ 合并结果无异常")
    else:
        print(f"⚠️ 发现 {total_issues} 个疑似问题，需人工核对")


if __name__ == "__main__":
    main()
