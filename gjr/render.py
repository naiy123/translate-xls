"""所有与"往 PDF 上画东西"相关的辅助函数。

这里只放**纯辅助函数**(计算 + 单一职责),真正"打开 PDF → 修改 → 保存"的编排
发生在 gjr/pipeline.py::process_page 里,以保证一次打开/一次保存的语义。
"""
import numpy as np

from gjr.classify import DWG_PAT, WORD_PAT


def fallback_product_name(ocr_result, page_h):
    """
    当 GPT 未返回产品名时,从 OCR 结果中推断。

    策略:在标题栏区域(页面下部 20%)找 TITLE 标签附近、
    字号最大的、非编码非公司名的英文词组。
    """
    title_y_start = page_h * 0.8
    candidates = []

    for w in ocr_result.get("words_result", []):
        loc = w["location"]
        text = w["words"].strip()
        if loc["top"] < title_y_start:
            continue
        if DWG_PAT.search(text):
            continue
        skip_words = {"TITLE", "UNCONTROLLED", "PRINTED", "RELEASED", "STATUS",
                      "SHEET", "DRAWN", "CHECKED", "ENGRG", "APPROVED", "DATE",
                      "CLASS", "DO NOT SCALE", "THIS DRAWING", "ROPER", "CORP",
                      "GE", "GEA", "GENERAL", "ELECTRIC", "APPLIANCES", "HAIER"}
        if text.upper() in skip_words or any(s in text.upper() for s in [
            "TOLERANCE", "DECIMAL", "ANGLE", "FRACTION", "PROJECTION",
            "DIMENSION", "CONFORM", "PROCEDURE", "DISCLOSURE", "AGREEMENT", "INTERNAL",
        ]):
            continue
        if not WORD_PAT.search(text):
            continue
        h = loc["height"]
        candidates.append((h, text))

    if not candidates:
        return None

    candidates.sort(key=lambda x: -x[0])
    best_h = candidates[0][0]
    best_text = candidates[0][1]

    # 合并同一 Y 行、相近字号的相邻 item(如 MOUNT + EVAP + FAN)
    best_y = None
    for w in ocr_result.get("words_result", []):
        if w["words"].strip() == best_text:
            best_y = w["location"]["top"]
            break

    if best_y is not None:
        row_items = []
        for w in ocr_result.get("words_result", []):
            loc = w["location"]
            if abs(loc["top"] - best_y) < best_h and abs(loc["height"] - best_h) < best_h * 0.3:
                if loc["top"] > title_y_start:
                    text = w["words"].strip()
                    if DWG_PAT.search(text):
                        continue
                    row_items.append((loc["left"], text))
        if row_items:
            row_items.sort()
            merged = " ".join(t for _, t in row_items)
            if len(merged) > len(best_text):
                best_text = merged

    print(f"    OCR fallback 产品名: \"{best_text}\" (字高={best_h})")
    return best_text


def find_product_bbox(items, product_name):
    """
    在 OCR items 中匹配产品名,返回合并 bbox。

    策略:
      1. 产品名拆成单词集合
      2. 每个 item 文本匹配产品名中的单词 → 候选
      3. 候选按空间距离聚类(Y 重叠 + X 间距 < 阈值)
      4. 取最大聚类的合并 bbox
    """
    if not product_name:
        return None

    product_words = set(product_name.upper().split())
    # Bug #2 fix: 短 token(如 '24' / 'RH' / 'A')只允许**完整匹配**,不允许 substring,
    # 否则会误中 '224D4774P012' 这种带 '24' 的图号
    long_words = {w for w in product_words if len(w) >= 3}
    short_words = product_words - long_words

    def _item_matches(text):
        """返回该 item text 命中的 product_words 集合(可能多个)。"""
        hit = set()
        # 短词要求完整等于
        if text in short_words:
            hit.add(text)
        # 长词允许 substring
        for lw in long_words:
            if lw in text:
                hit.add(lw)
        return hit

    candidates = []  # [(item, matched_set)]
    for it in items:
        text = it["text"].strip().upper()
        hit = _item_matches(text)
        if hit:
            candidates.append((it, hit))

    if not candidates:
        return None

    # 空间聚类:Y 重叠 + X 间距 < 200 → 同一 group
    candidates.sort(key=lambda c: (c[0]["top"], c[0]["left"]))
    groups = [[candidates[0]]]
    for pair in candidates[1:]:
        it = pair[0]
        last_group = groups[-1]
        close = False
        for g_pair in last_group:
            g = g_pair[0]
            y_overlap = min(it["bottom"], g["bottom"]) - max(it["top"], g["top"])
            x_dist = abs(it["left"] - g["right"])
            if y_overlap > 0 and x_dist < 200:
                close = True
                break
        if close:
            last_group.append(pair)
        else:
            groups.append([pair])

    # Bug #3 fix: 按"去重后的 matched_words 数量"评分,而不是 group size
    def _group_score(g):
        all_hits = set()
        for pair in g:
            all_hits |= pair[1]
        # 先看去重匹配词数,再看 group 大小(命中相同时更大的 group 更可信)
        return (len(all_hits), len(g))

    best = max(groups, key=_group_score)
    matched_words = [pair[0]["text"] for pair in best]
    all_hits = set()
    for pair in best:
        all_hits |= pair[1]
    bbox = (
        min(pair[0]["left"] for pair in best),
        min(pair[0]["top"] for pair in best),
        max(pair[0]["right"] for pair in best),
        max(pair[0]["bottom"] for pair in best),
    )
    print(f"  产品名匹配: {matched_words} 命中{sorted(all_hits)} → bbox ({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})")
    return bbox


def find_white_space(block_bbox, all_bboxes, page_w, page_h, trans_w, trans_h,
                     page_img=None, inner_rect=None):
    """
    在 block 附近找到白色空白区域放置翻译。
    使用像素级白色检测避免覆盖图纸内容。

    优先级:左侧 → 下方 → 上方
    """
    bx1, by1, bx2, by2 = block_bbox
    gap = 15

    ix1 = inner_rect[0] if inner_rect else 72
    iy1 = inner_rect[1] if inner_rect else 36
    ix2 = inner_rect[2] if inner_rect else page_w - 72
    iy2 = inner_rect[3] if inner_rect else page_h - 36

    candidates = [
        (bx1 - trans_w - gap, by1, bx1 - gap, by1 + trans_h),        # 左侧
        (bx1, by2 + gap, bx1 + trans_w, by2 + gap + trans_h),        # 下方
        (bx1, by1 - trans_h - gap, bx1 + trans_w, by1 - gap),        # 上方
    ]

    def is_area_clear(cx1, cy1, cx2, cy2):
        if cx1 < ix1 + 5 or cy1 < iy1 + 5 or cx2 > ix2 - 5 or cy2 > iy2 - 5:
            return False
        for obx1, oby1, obx2, oby2 in all_bboxes:
            if not (cx2 < obx1 or cx1 > obx2 or cy2 < oby1 or cy1 > oby2):
                return False
        if page_img is not None:
            img_w, img_h = page_img.size
            sx, sy = img_w / page_w, img_h / page_h
            px1 = max(0, int(cx1 * sx))
            py1 = max(0, int(cy1 * sy))
            px2 = min(img_w, int(cx2 * sx))
            py2 = min(img_h, int(cy2 * sy))
            if px2 > px1 and py2 > py1:
                region = np.array(page_img.crop((px1, py1, px2, py2)))
                white_ratio = np.mean(region > 230)
                if white_ratio < 0.9:
                    return False
        return True

    for cx1, cy1, cx2, cy2 in candidates:
        if is_area_clear(cx1, cy1, cx2, cy2):
            return (cx1, cy1, cx2, cy2)

    return (max(ix1 + 5, bx1 - trans_w - gap), by1,
            max(ix1 + 5 + trans_w, bx1 - gap), by1 + trans_h)
