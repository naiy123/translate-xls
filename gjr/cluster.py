"""空间聚类 + 几何工具 + 框线/内框检测。

- cluster_blocks: 手写 DBSCAN,距离度量是 bbox 切比雪夫边缘间距
- snap_bbox: 把聚类后的 bbox 吸附到最近的表格线,得到更干净的外框
- extract_lines / detect_frame: PDF 矢量线分析,找内外框
- 行内工具:block_bbox_raw / block_position / group_into_rows 等
"""
import fitz

from gjr.config import DBSCAN_EPS, SNAP_MARGIN, ROW_OVERLAP


def _bbox_gap(a, b):
    """两个 bbox 边缘间的切比雪夫距离(X/Y 间距取较大者,重叠返回 0)"""
    dx = max(0, max(a["left"] - b["right"], b["left"] - a["right"]))
    dy = max(0, max(a["top"] - b["bottom"], b["top"] - a["bottom"]))
    return max(dx, dy)


def cluster_blocks(items, eps=DBSCAN_EPS):
    """
    DBSCAN 二维空间聚类(手写实现,零依赖)。

    距离度量:两个 bbox 边缘间的切比雪夫距离。
    解决旧算法的桥接问题——不同区域即使 Y 重叠,X 间距大也不会连通。
    """
    if not items:
        return []
    if len(items) == 1:
        return [items]

    n = len(items)
    visited = [False] * n
    labels = [-1] * n
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        labels[i] = cluster_id
        queue = [i]
        while queue:
            p = queue.pop(0)
            for j in range(n):
                if visited[j]:
                    continue
                if _bbox_gap(items[p], items[j]) <= eps:
                    visited[j] = True
                    labels[j] = cluster_id
                    queue.append(j)
        cluster_id += 1

    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(items[i])

    blocks = sorted(clusters.values(), key=lambda b: min(it["top"] for it in b))
    return blocks


def extract_lines(pdf_path):
    """从 PDF 第 0 页提取水平/垂直矢量线段。"""
    doc = fitz.open(pdf_path)
    page = doc[0]
    h_lines, v_lines = [], []
    for d in page.get_drawings():
        for item in d["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                x1, y1, x2, y2 = p1.x, p1.y, p2.x, p2.y
                if abs(y2 - y1) < 1 and abs(x2 - x1) > 10:
                    h_lines.append((round((y1+y2)/2, 1), min(x1, x2), max(x1, x2)))
                elif abs(x2 - x1) < 1 and abs(y2 - y1) > 10:
                    v_lines.append((round((x1+x2)/2, 1), min(y1, y2), max(y1, y2)))
    doc.close()
    return h_lines, v_lines


def snap_bbox(bx1, by1, bx2, by2, h_lines, v_lines, margin=SNAP_MARGIN):
    """将 bbox 扩展到最近的表格边框线"""
    c = [h for h in h_lines if h[0] < by1 and h[0] > by1-margin and h[1] < bx1+50 and h[2] > bx2-50]
    if c: by1 = min(x[0] for x in c)
    c = [h for h in h_lines if h[0] > by2 and h[0] < by2+margin and h[1] < bx1+50 and h[2] > bx2-50]
    if c: by2 = max(x[0] for x in c)
    c = [v for v in v_lines if v[0] < bx1 and v[0] > bx1-margin and v[1] < by1+50 and v[2] > by2-50]
    if c: bx1 = min(x[0] for x in c)
    c = [v for v in v_lines if v[0] > bx2 and v[0] < bx2+margin and v[1] < by1+50 and v[2] > by2-50]
    if c: bx2 = max(x[0] for x in c)
    return bx1, by1, bx2, by2


def block_bbox_raw(block):
    return (
        min(it["left"] for it in block),
        min(it["top"] for it in block),
        max(it["right"] for it in block),
        max(it["bottom"] for it in block),
    )


def block_position(block, page_w=2448, page_h=1584):
    cx = sum(it["left"] + it["w"] / 2 for it in block) / len(block)
    cy = sum(it["top"] + it["h"] / 2 for it in block) / len(block)
    v = "上部" if cy < page_h * 0.35 else ("中部" if cy < page_h * 0.65 else "下部")
    h = "左侧" if cx < page_w * 0.35 else ("中间" if cx < page_w * 0.65 else "右侧")
    return f"{v}{h}"


def items_overlap_y(a, b):
    overlap = min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])
    if overlap <= 0:
        return 0
    shorter = min(a["h"], b["h"])
    return overlap / shorter if shorter > 0 else 0


def group_into_rows(block_items):
    si = sorted(block_items, key=lambda it: it["top"])
    rows, row = [], [si[0]]
    for it in si[1:]:
        if any(items_overlap_y(it, r) >= ROW_OVERLAP for r in row):
            row.append(it)
        else:
            rows.append(row)
            row = [it]
    rows.append(row)
    for r in rows:
        r.sort(key=lambda it: it["left"])
    return rows


def detect_frame(pdf_path):
    """
    检测图纸的外框线和内框边界(第 0 页)。

    返回:
      frame_lines: [("h", y, x1, x2), ("v", x, y1, y2), ...] 用于 redact 后重画
      inner_rect:  (x1, y1, x2, y2) 内框矩形,redact 不能超出此范围
    """
    doc = fitz.open(pdf_path)
    page = doc[0]
    pw, ph = page.rect.width, page.rect.height

    h_lines, v_lines = [], []
    for d in page.get_drawings():
        for item in d["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                x1, y1, x2, y2 = p1.x, p1.y, p2.x, p2.y
                if abs(y2-y1) < 1 and abs(x2-x1) > pw * 0.5:
                    h_lines.append((round((y1+y2)/2, 1), min(x1, x2), max(x1, x2)))
                elif abs(x2-x1) < 1 and abs(y2-y1) > ph * 0.5:
                    v_lines.append((round((x1+x2)/2, 1), min(y1, y2), max(y1, y2)))
    doc.close()

    frame_lines = []
    for y, x1, x2 in h_lines:
        frame_lines.append(("h", y, x1, x2))
    for x, y1, y2 in v_lines:
        frame_lines.append(("v", x, y1, y2))

    inner_h = sorted([y for y, x1, x2 in h_lines if y > 10 and y < ph - 10])
    inner_v = sorted([x for x, y1, y2 in v_lines if x > 10 and x < pw - 10])

    if len(inner_h) >= 2 and len(inner_v) >= 2:
        inner_rect = (inner_v[0], inner_h[0], inner_v[-1], inner_h[-1])
    else:
        inner_rect = (72, 36, pw - 72, ph - 36)

    return frame_lines, inner_rect
