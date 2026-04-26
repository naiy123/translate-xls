"""单页脱敏流水线的编排者。

和被拆分后的各模块关系:
  ocr.py        → OCR + 解析
  cluster.py    → 聚类/几何/框线
  classify.py   → 小 block 分类
  gpt_redact    → GPT 判断 KEEP/DELETE + 产品名
  gpt_translate → 标注/大块翻译
  render.py     → 产品名 fallback / bbox 查找 / 白底空间
  preview.py    → debug PNG
  api_log.py    → API 日志开关

这里只负责串流程 + 一次打开/一次保存 PDF。
"""
import base64
import json
import re
import time
from pathlib import Path

import fitz

from gjr.api_log import log_api_call, set_log_context
from gjr.classify import DWG_PAT, block_to_spatial, classify_block
from gjr.cluster import (
    absorb_nested_blocks,
    block_bbox_raw,
    cluster_blocks,
    merge_by_shared_edges,
    snap_bbox,
)
from gjr.geometry import PageGeometry
from gjr.config import load_sensitive
from gjr.gpt_redact import ask_gpt, parse_gpt_result
from gjr.gpt_translate import batch_translate_labels, translate_block
from gjr.ocr import filter_noise, ocr_pdf, parse_items
from gjr.preview import render_debug_previews
from gjr.render import fallback_product_name, find_product_bbox, find_white_space


def process_page(pdf_path, output_path, page_num, output_dir, stem, debug_preview=False):
    """处理单页 PDF 的完整 pipeline"""
    page_label = f"P{page_num+1}"
    set_log_context(debug_preview, page_num + 1, output_dir)

    # ── Step 1: OCR ──
    print(f"\n{'─'*50}")
    print(f"  [{page_label}] Step 1: 百度 OCR")
    json_path = output_dir / f"{stem}_p{page_num+1}_ocr.json"
    if json_path.exists():
        print(f"    使用缓存: {json_path}")
        with open(json_path) as f:
            ocr_result = json.load(f)
    else:
        print(f"    识别第 {page_num+1} 页...")
        _t0 = time.monotonic()
        ocr_result = ocr_pdf(str(pdf_path), page_num=page_num + 1)
        json_path.write_text(json.dumps(ocr_result, ensure_ascii=False, indent=2), encoding="utf-8")
        log_api_call(
            step="baidu_ocr",
            api="baidu.ocr.accurate",
            request={
                "endpoint": "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
                "page_num": page_num + 1,
                "pdf": Path(pdf_path).name,
                "note": "完整响应见同目录 OCR 缓存文件,此处不重复存储",
            },
            response={
                "words_result_num": ocr_result.get("words_result_num", 0),
                "cache_file": json_path.name,
            },
            duration_ms=int((time.monotonic() - _t0) * 1000),
        )
    num_words = ocr_result.get("words_result_num", 0)
    print(f"    OCR: {num_words} 行")

    if num_words == 0:
        print(f"    跳过（无文字）")
        return None, {}

    # ── Step 2: 预处理 ──
    print(f"  [{page_label}] Step 2: 预处理")

    geom = PageGeometry.from_pdf_page(pdf_path, page_num)
    pw, ph = geom.pw, geom.ph
    ix1, iy1, ix2, iy2 = geom.inner
    h_lines, v_lines = geom.h_lines, geom.v_lines
    frame_h, frame_v = geom.frame_h, geom.frame_v
    title_rect = geom.title_rect
    print(f"    内框: ({ix1:.0f},{iy1:.0f})-({ix2:.0f},{iy2:.0f})")

    all_items = parse_items(ocr_result)
    items = [it for it in all_items if geom.is_in_inner(it)]
    print(f"    OCR: {len(all_items)} 行, 内框内: {len(items)} 行 (过滤 {len(all_items)-len(items)} 行框外)")

    items = filter_noise(items, title_rect=title_rect)
    if not items:
        print(f"    跳过（过滤后无内容）")
        return None, {}

    # 聚类时把矢量框线喂进去:被框线隔开的 items 走原始距离,
    # 未被分隔的(同表格单元/同连通区)距离打 0.3 折,更容易归一块。
    blocks = cluster_blocks(items, h_lines=h_lines, v_lines=v_lines)

    snapped_bboxes = [snap_bbox(*block_bbox_raw(b), h_lines, v_lines) for b in blocks]

    # snap 后两步收敛:
    #  ① absorb_nested_blocks - 消除"snap 互嵌":若 A 完全包含 B,把 B 吞入 A
    #  ② merge_by_shared_edges  - 标题栏内共享 snap 边的 block 合并
    n_raw = len(blocks)
    blocks, snapped_bboxes = absorb_nested_blocks(blocks, snapped_bboxes)
    n_after_absorb = len(blocks)
    blocks, snapped_bboxes = merge_by_shared_edges(blocks, snapped_bboxes, pw, ph)
    n_after_merge = len(blocks)

    parts = [f"{len(items)} 行 → {n_raw} 块"]
    if n_after_absorb < n_raw:
        parts.append(f"嵌套吞并 {n_raw - n_after_absorb}")
    if n_after_merge < n_after_absorb:
        parts.append(f"边贴合并 {n_after_absorb - n_after_merge}")
    parts.append(f"最终 {n_after_merge} 块")
    print("    " + " → ".join(parts))

    # ── 预分类 ──
    gpt_indices, auto_keep_indices, translate_indices, auto_delete_indices = [], [], [], []
    for bi, block in enumerate(blocks):
        cat = classify_block(block, in_title_rect=geom.is_in_title(snapped_bboxes[bi]))
        if cat == "GPT":
            gpt_indices.append(bi)
        elif cat == "TRANSLATE":
            translate_indices.append(bi)
        elif cat == "AUTO_DELETE":
            auto_delete_indices.append(bi)
        else:
            auto_keep_indices.append(bi)
    print(f"    分类: {len(auto_keep_indices)} 尺寸, {len(translate_indices)} 标注, {len(gpt_indices)} 大块, {len(auto_delete_indices)} 强制删除")

    if debug_preview:
        render_debug_previews(pdf_path, page_num, output_dir, stem,
                              items, blocks, snapped_bboxes,
                              raw_items=all_items, title_rect=title_rect)

    # ── block 文本 ──
    block_texts = [block_to_spatial(block) for block in blocks]

    # ── Step 3: GPT 判断 ──
    product_name = None
    if gpt_indices:
        print(f"  [{page_label}] Step 3: GPT 判断 ({len(gpt_indices)} 个 block)")
        gpt_blocks = [blocks[i] for i in gpt_indices]
        gpt_bboxes = [snapped_bboxes[i] for i in gpt_indices]
        gpt_texts = [block_texts[i] for i in gpt_indices]
        doc = fitz.open(str(pdf_path))
        page_obj = doc[page_num]
        zoom = 200 / 72
        gpt_images = []
        for bbox in gpt_bboxes:
            clip = fitz.Rect(max(0, bbox[0]-5), max(0, bbox[1]-5),
                             min(pw, bbox[2]+5), min(ph, bbox[3]+5))
            pix = page_obj.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
            gpt_images.append(base64.b64encode(pix.tobytes("png")).decode())
        doc.close()

        gpt_result = ask_gpt(gpt_blocks, gpt_images, gpt_texts, gpt_bboxes)
        print(f"    {gpt_result}")
        gpt_decisions, product_name = parse_gpt_result(gpt_result, len(gpt_blocks))
    else:
        print(f"  [{page_label}] Step 3: 跳过（无大块）")
        gpt_decisions = {}

    # 合并决策
    decisions = {}
    for bi in auto_keep_indices:
        decisions[bi] = "KEEP"
    for bi in translate_indices:
        decisions[bi] = "KEEP"
    for bi in auto_delete_indices:
        decisions[bi] = "DELETE"
    for gi, bi in enumerate(gpt_indices):
        decisions[bi] = gpt_decisions.get(gi, "DELETE")

    # GPT 未返回有效产品名时,用 OCR fallback
    def _is_valid_product(name):
        if not name:
            return False
        if re.search(r'[\u4e00-\u9fff]', name):
            return False
        if DWG_PAT.search(name):
            return False
        return True

    if not _is_valid_product(product_name):
        fb = fallback_product_name(ocr_result, ph)
        if fb:
            product_name = fb

    print(f"    产品名: {product_name or '未识别'}")
    delete_count = sum(1 for d in decisions.values() if d == "DELETE")
    keep_count = sum(1 for d in decisions.values() if d == "KEEP")
    print(f"    决策: {keep_count} 保留, {delete_count} 删除")

    # ── Step 4/5/6: 脱敏 + 产品名 + 翻译(一次打开、一次保存)──
    print(f"  [{page_label}] Step 4: 脱敏 + 产品名 + 翻译")

    big_translate = [bi for bi in range(len(blocks))
                     if decisions.get(bi) == "KEEP" and len(blocks[bi]) > 4]

    # 6a: 小 block 批量翻译
    label_translations = {}
    if translate_indices:
        label_translations = batch_translate_labels(blocks, translate_indices, block_texts)
        for bi, trans in label_translations.items():
            print(f"    标注 [{bi+1}] → {trans}")

    # 6b: 大 block 翻译
    big_translations = {}
    if big_translate:
        src_doc = fitz.open(str(pdf_path))
        src_page = src_doc[page_num]
        zoom = 200 / 72
        for bi in big_translate:
            bbox = snapped_bboxes[bi]
            clip = fitz.Rect(max(0, bbox[0]-5), max(0, bbox[1]-5),
                             min(pw, bbox[2]+5), min(ph, bbox[3]+5))
            pix = src_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
            img_b64 = base64.b64encode(pix.tobytes("png")).decode()
            trans_text = translate_block(block_texts[bi], img_b64)
            if trans_text:
                big_translations[bi] = trans_text
                print(f"    大块 Block {bi+1} 翻译完成")
        src_doc.close()

    # 产品名匹配
    product_bbox = None
    if product_name:
        all_items_for_match = parse_items(ocr_result)
        product_bbox = find_product_bbox(all_items_for_match, product_name)

    # ── 一次性打开 output PDF,执行所有修改 ──
    doc = fitz.open(str(output_path))
    page_obj = doc[page_num]

    # (1) 脱敏涂白
    for bi in range(len(blocks)):
        if decisions.get(bi) == "DELETE":
            bx1, by1, bx2, by2 = snapped_bboxes[bi]
            rx1 = max(bx1, ix1)
            ry1 = max(by1, iy1)
            rx2 = min(bx2, ix2)
            ry2 = min(by2, iy2)
            if rx1 < rx2 and ry1 < ry2:
                page_obj.add_redact_annot(fitz.Rect(rx1, ry1, rx2, ry2), fill=(1, 1, 1))

    # (1.5) KEEP block 内品牌关键词精确遮盖
    brand_inline_pats = load_sensitive()["brand_inline_patterns"]
    brand_mask_count = 0
    for w in ocr_result.get("words_result", []):
        text = w["words"]
        chars = w.get("chars", [])
        if not chars:
            continue
        # 建立 text index → chars index 的映射(chars 不含空格)
        text_to_char = {}
        ci = 0
        for ti, ch in enumerate(text):
            if ch == ' ':
                continue
            if ci < len(chars):
                text_to_char[ti] = ci
                ci += 1
        for pat in brand_inline_pats:
            for m in pat.finditer(text):
                first_ci, last_ci = None, None
                for ti in range(m.start(), m.end()):
                    if ti in text_to_char:
                        if first_ci is None:
                            first_ci = text_to_char[ti]
                        last_ci = text_to_char[ti]
                if first_ci is not None and last_ci is not None:
                    c_start = chars[first_ci]["location"]
                    c_end = chars[last_ci]["location"]
                    line_loc = w["location"]
                    line_h = line_loc["height"]
                    rx1 = c_start["left"] - 1
                    ry1 = line_loc["top"] + line_h * 0.2
                    rx2 = c_end["left"] + c_end["width"] + 1
                    ry2 = line_loc["top"] + line_h
                    page_obj.add_redact_annot(fitz.Rect(rx1, ry1, rx2, ry2), fill=(1, 1, 1))
                    brand_mask_count += 1
    if brand_mask_count:
        print(f"    品牌关键词遮盖: {brand_mask_count} 处")

    page_obj.apply_redactions()

    # (2) 重画框线
    shape = page_obj.new_shape()
    for y, x1, x2 in frame_h:
        shape.draw_line(fitz.Point(x1, y), fitz.Point(x2, y))
    for x, y1, y2 in frame_v:
        shape.draw_line(fitz.Point(x, y1), fitz.Point(x, y2))
    shape.finish(color=(0, 0, 0), width=0.5)
    shape.commit()

    # (3) 渲染产品名
    #   优先级 1:find_product_bbox 找到的位置在某个 DELETE block 内 → 原位回写
    #   优先级 2(fallback):找不到或不在 DELETE 内 → 推定右下角最大的 DELETE block 是标题栏,写到它里面
    #   优先级 3:完全没 DELETE block → 只打印产品名,不写
    def _write_product_name_in_block(bi):
        dbx1, dby1, dbx2, dby2 = snapped_bboxes[bi]
        block_w = dbx2 - dbx1
        block_h = dby2 - dby1
        # 字号按 block 高度的一小部分
        fontsize = max(10, min(block_h * 0.25, 22))
        text_w = len(product_name) * fontsize * 0.5
        x = dbx1 + (block_w - text_w) / 2 if text_w < block_w else dbx1 + 2
        # 垂直居中偏下
        y = dby2 - block_h * 0.15
        page_obj.insert_text(
            fitz.Point(x, y), product_name,
            fontsize=fontsize, fontname="helv", color=(0, 0, 0),
        )
        return fontsize

    if product_name:
        in_deleted_bi = None
        if product_bbox:
            bx1, by1, bx2, by2 = product_bbox
            for bi2 in range(len(blocks)):
                if decisions.get(bi2) == "DELETE":
                    dbx1, dby1, dbx2, dby2 = snapped_bboxes[bi2]
                    if bx1 >= dbx1 - 10 and by1 >= dby1 - 10 and bx2 <= dbx2 + 10 and by2 <= dby2 + 10:
                        in_deleted_bi = bi2
                        break

        if in_deleted_bi is not None:
            # 优先级 1:原位回写
            bx1, by1, bx2, by2 = product_bbox
            fontsize = max(10, (by2 - by1) * 0.8)
            text_w = len(product_name) * fontsize * 0.5
            x = bx1 + ((bx2-bx1) - text_w) / 2 if text_w < (bx2-bx1) else bx1
            page_obj.insert_text(
                fitz.Point(x, by2 - (by2-by1)*0.15), product_name,
                fontsize=fontsize, fontname="helv", color=(0, 0, 0),
            )
            print(f"    产品名: \"{product_name}\" 字号={fontsize:.0f} (原位回写)")
        else:
            # 优先级 2:fallback 到右下角最大的 DELETE block
            delete_indices = [bi for bi, d in decisions.items() if d == "DELETE"]
            # 只考虑右下角区域的 block(x 中心 > 页宽 50%,y 中心 > 页高 50%)
            bottom_right = [
                bi for bi in delete_indices
                if (snapped_bboxes[bi][0] + snapped_bboxes[bi][2]) / 2 > pw * 0.5
                and (snapped_bboxes[bi][1] + snapped_bboxes[bi][3]) / 2 > ph * 0.5
            ]
            candidates = bottom_right or delete_indices  # 没有右下角就退一步用全部
            if candidates:
                # 取面积最大的(最像标题栏)
                def _area(bi):
                    b = snapped_bboxes[bi]
                    return (b[2]-b[0]) * (b[3]-b[1])
                title_bi = max(candidates, key=_area)
                fontsize = _write_product_name_in_block(title_bi)
                print(f"    产品名: \"{product_name}\" 字号={fontsize:.0f} (fallback → Block {title_bi+1})")
            else:
                print(f"    产品名: \"{product_name}\" (无 DELETE block,不写回)")

    # 渲染原始页面图片用于白色检测
    from PIL import Image as _Image
    _src_doc = fitz.open(str(pdf_path))
    _src_pix = _src_doc[page_num].get_pixmap(matrix=fitz.Matrix(1, 1))
    page_img = _Image.frombytes("RGB", (_src_pix.width, _src_pix.height), _src_pix.samples)
    _src_doc.close()
    _inner_rect = (ix1, iy1, ix2, iy2)

    # (4) 大块翻译注释
    all_bboxes = list(snapped_bboxes)
    for bi, trans_text in big_translations.items():
        lines = trans_text.split("\n")
        fontsize = 11
        line_height = fontsize * 1.5
        trans_h = len(lines) * line_height + 10
        max_line_len = max(len(l) for l in lines) if lines else 10
        trans_w = max(250, min(max_line_len * fontsize * 0.6, snapped_bboxes[bi][2] - snapped_bboxes[bi][0]))

        pos = find_white_space(snapped_bboxes[bi], all_bboxes, pw, ph, trans_w, trans_h,
                               page_img=page_img, inner_rect=_inner_rect)
        rect = fitz.Rect(pos[0], pos[1], pos[2], pos[3])
        all_bboxes.append(pos)

        annot = page_obj.add_freetext_annot(rect, trans_text, fontsize=fontsize,
            fontname="china-s", text_color=(0, 0, 0), fill_color=(1, 1, 1), align=fitz.TEXT_ALIGN_LEFT)
        annot.set_opacity(0.92)
        annot.update()
        print(f"    翻译 Block {bi+1} → ({pos[0]:.0f},{pos[1]:.0f})")

    # (5) 小块翻译注释
    for bi, trans in label_translations.items():
        bx1, by1, bx2, by2 = snapped_bboxes[bi]
        fontsize = 8
        tw = len(trans) * fontsize * 0.7
        th = fontsize * 1.5
        ty1 = by2 + 2
        if ty1 + th > ph - 30:
            ty1 = by1 - th - 2
        rect = fitz.Rect(bx1, ty1, bx1 + max(tw, 50), ty1 + th)
        annot = page_obj.add_freetext_annot(rect, trans, fontsize=fontsize,
            fontname="china-s", text_color=(0, 0, 0), fill_color=(1, 1, 1), align=fitz.TEXT_ALIGN_LEFT)
        annot.set_opacity(0.9)
        annot.update()

    # ── 一次保存 ──
    doc.save(output_path, incremental=True, encryption=0)
    doc.close()

    print(f"    翻译: {len(label_translations)} 标注 + {len(big_translations)} 大块")

    return product_name, decisions
