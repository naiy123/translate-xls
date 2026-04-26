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


def _line_separates(a, b, h_lines, v_lines):
    """判定两个 bbox 的中心连线是否被矢量框线穿过。

    用线段 vs 框线的交点测试:对每条垂直线看连线是否跨越其 x,
    并检查交点的 y 是否落在线段范围内;水平线同理。
    命中任一条即认为"中间有框线"→ 属于不同表格单元/不同区域。
    """
    if not h_lines and not v_lines:
        return False
    ax = (a["left"] + a["right"]) / 2
    ay = (a["top"] + a["bottom"]) / 2
    bx = (b["left"] + b["right"]) / 2
    by = (b["top"] + b["bottom"]) / 2
    if v_lines and ax != bx:
        dx = bx - ax
        lo_x, hi_x = (ax, bx) if ax < bx else (bx, ax)
        for vx, y1, y2 in v_lines:
            if vx < lo_x or vx > hi_x:
                continue
            t = (vx - ax) / dx
            iy = ay + t * (by - ay)
            if y1 <= iy <= y2:
                return True
    if h_lines and ay != by:
        dy = by - ay
        lo_y, hi_y = (ay, by) if ay < by else (by, ay)
        for hy, x1, x2 in h_lines:
            if hy < lo_y or hy > hi_y:
                continue
            t = (hy - ay) / dy
            ix = ax + t * (bx - ax)
            if x1 <= ix <= x2:
                return True
    return False


def cluster_blocks(items, eps=DBSCAN_EPS, h_lines=None, v_lines=None,
                   same_cell_factor=0.3):
    """
    DBSCAN 二维空间聚类(手写实现,零依赖)。

    距离度量:两个 bbox 边缘间的切比雪夫距离。
    解决旧算法的桥接问题——不同区域即使 Y 重叠,X 间距大也不会连通。

    若传入 h_lines/v_lines,对"中心连线未被任何框线穿过"的 item 对
    把距离打一个折扣(same_cell_factor,默认 0.3),让同一连通区域内
    偏远的 items 更容易进同一块;被框线隔开的 items 仍按原距离判断。
    """
    if not items:
        return []
    if len(items) == 1:
        return [items]

    n = len(items)
    visited = [False] * n
    labels = [-1] * n
    cluster_id = 0

    def _gap(a, b):
        base = _bbox_gap(a, b)
        if base == 0:
            return 0
        if h_lines is None and v_lines is None:
            return base
        if _line_separates(a, b, h_lines or [], v_lines or []):
            return base
        return base * same_cell_factor

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
                if _gap(items[p], items[j]) <= eps:
                    visited[j] = True
                    labels[j] = cluster_id
                    queue.append(j)
        cluster_id += 1

    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(items[i])

    blocks = sorted(clusters.values(), key=lambda b: min(it["top"] for it in b))
    return blocks


def absorb_nested_blocks(blocks, snapped_bboxes, slack=2.0):
    """若 cluster A 的 snap bbox 完全包含 cluster B(slack 容差),
    把 B 的 items 并入 A、移除 B,保证聚类输出"互不相交分区"语义。

    病灶:snap_bbox 是逐块独立调用的,每块各自往最近矢量线扩张,
    互不感知。结果出现 #6 完全嵌套在 #8 内、redact 双涂、annot 双写。
    这一步在 snap 后扫一遍嵌套关系,把内套块吞入外包块。
    """
    n = len(blocks)
    if n <= 1:
        return blocks, snapped_bboxes

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def contains(big, small):
        return (big[0] - slack <= small[0] and small[2] <= big[2] + slack
                and big[1] - slack <= small[1] and small[3] <= big[3] + slack)

    # 按面积降序遍历:先处理大的,小的去找包含它的最小父亲
    def _area(bi):
        b = snapped_bboxes[bi]
        return (b[2] - b[0]) * (b[3] - b[1])

    order = sorted(range(n), key=lambda i: -_area(i))
    for i_pos, i in enumerate(order):
        # 在比 i 大的 block 里找包含 i 的(取最小的那个 = 直接父亲)
        best_parent = None
        best_area = float("inf")
        for j in order[:i_pos]:
            if contains(snapped_bboxes[j], snapped_bboxes[i]):
                a = _area(j)
                if a < best_area:
                    best_area = a
                    best_parent = j
        if best_parent is not None:
            parent[find(i)] = find(best_parent)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    new_blocks, new_bboxes = [], []
    for root, members in groups.items():
        merged_items = []
        for m in members:
            merged_items.extend(blocks[m])
        # snap bbox 取根(外包块)的,保留扩张结果
        new_blocks.append(merged_items)
        new_bboxes.append(snapped_bboxes[root])

    order2 = sorted(range(len(new_blocks)),
                    key=lambda k: (new_bboxes[k][1], new_bboxes[k][0]))
    new_blocks = [new_blocks[k] for k in order2]
    new_bboxes = [new_bboxes[k] for k in order2]
    return new_blocks, new_bboxes


def merge_by_shared_edges(blocks, snapped_bboxes, pw, ph,
                          region=None, tol=1.5, min_overlap=4.0):
    """Snap 后共享边合并:两个 snapped_bbox 若在同一条吸附线上贴边,
    且相邻边投影有足够重叠,视为同一表格连通区域,合并为一个 block。

    只在 region 矩形内触发(默认右下 45% 宽 × 30% 高,覆盖标题栏),
    避免绘图区边缘两个 bbox 偶然 snap 到同一条外框而误粘。
    """
    n = len(blocks)
    if n <= 1:
        return blocks, snapped_bboxes

    if region is None:
        rx1 = pw * 0.55
        ry1 = ph * 0.70
        rx2 = pw
        ry2 = ph
    else:
        rx1, ry1, rx2, ry2 = region

    def _center_in_region(bbox):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    eligible = [i for i in range(n) if _center_in_region(snapped_bboxes[i])]
    for idx_a, i in enumerate(eligible):
        ax1, ay1, ax2, ay2 = snapped_bboxes[i]
        for j in eligible[idx_a + 1:]:
            bx1, by1, bx2, by2 = snapped_bboxes[j]
            shared_v = abs(ax2 - bx1) < tol or abs(bx2 - ax1) < tol
            y_over = min(ay2, by2) - max(ay1, by1)
            if shared_v and y_over > min_overlap:
                union(i, j)
                continue
            shared_h = abs(ay2 - by1) < tol or abs(by2 - ay1) < tol
            x_over = min(ax2, bx2) - max(ax1, bx1)
            if shared_h and x_over > min_overlap:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    new_blocks, new_bboxes = [], []
    for members in groups.values():
        merged_items = []
        for m in members:
            merged_items.extend(blocks[m])
        x1 = min(snapped_bboxes[m][0] for m in members)
        y1 = min(snapped_bboxes[m][1] for m in members)
        x2 = max(snapped_bboxes[m][2] for m in members)
        y2 = max(snapped_bboxes[m][3] for m in members)
        new_blocks.append(merged_items)
        new_bboxes.append((x1, y1, x2, y2))

    order = sorted(range(len(new_blocks)), key=lambda k: (new_bboxes[k][1], new_bboxes[k][0]))
    new_blocks = [new_blocks[k] for k in order]
    new_bboxes = [new_bboxes[k] for k in order]
    return new_blocks, new_bboxes


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
