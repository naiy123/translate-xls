"""页面几何抽象。

把页面尺寸 / 内框 / 矢量框线 / 标题栏区聚合成一个 `PageGeometry` 对象,
避免 pipeline / cluster / preview / ocr 各自重复实现一份提取逻辑。

历史病灶:
  - pipeline.py 内联抄了一份 cluster.extract_lines + detect_frame
    (因为后者写死 page_num=0)
  - title_rect 比例 (0.55/0.70) 在 pipeline 用 inner_box 做基准、
    merge_by_shared_edges 默认参数用 page_size 做基准, 数值相近但不一致
  - "中心点是否在标题栏内" 的判定在三处复刻 (filter_noise / pipeline / preview)

统一从这里读, 上述三个问题一次解决。
"""
from dataclasses import dataclass, field
from typing import List, Tuple

import fitz


HLine = Tuple[float, float, float]   # (y, x1, x2)
VLine = Tuple[float, float, float]   # (x, y1, y2)
Rect = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


# 标题栏区域占内框的比例(右下 45% 宽 × 30% 高):
# - filter_noise 在此区内不剔除短符号
# - classify_block 在此区内的小块统一升 GPT
# - merge_by_shared_edges 在此区内才触发共享边合并
TITLE_RECT_RATIO = (0.55, 0.70, 1.0, 1.0)


@dataclass
class PageGeometry:
    pw: float
    ph: float
    inner: Rect
    h_lines: List[HLine] = field(default_factory=list)
    v_lines: List[VLine] = field(default_factory=list)
    # frame_h/frame_v: 跨过页宽/高一半的"贯通线"。redact 后用它们重画外框,
    # 因为 apply_redactions 会把矩形内的矢量也擦掉。
    frame_h: List[HLine] = field(default_factory=list)
    frame_v: List[VLine] = field(default_factory=list)
    title_rect: Rect = (0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_pdf_page(cls, pdf_path, page_num: int) -> "PageGeometry":
        """从 PDF 指定页提取所有几何信息。"""
        doc = fitz.open(str(pdf_path))
        page = doc[page_num]
        pw, ph = page.rect.width, page.rect.height

        h_lines: List[HLine] = []
        v_lines: List[VLine] = []
        for d in page.get_drawings():
            for it in d["items"]:
                if it[0] != "l":
                    continue
                p1, p2 = it[1], it[2]
                x1, y1, x2, y2 = p1.x, p1.y, p2.x, p2.y
                if abs(y2 - y1) < 1 and abs(x2 - x1) > 10:
                    h_lines.append(
                        (round((y1 + y2) / 2, 1), min(x1, x2), max(x1, x2))
                    )
                elif abs(x2 - x1) < 1 and abs(y2 - y1) > 10:
                    v_lines.append(
                        (round((x1 + x2) / 2, 1), min(y1, y2), max(y1, y2))
                    )
        doc.close()

        # 贯通线 = 跨过页面一半以上的水平/竖直线;内框由其最外两条围成。
        frame_h = [h for h in h_lines if h[2] - h[1] > pw * 0.5]
        frame_v = [v for v in v_lines if v[2] - v[1] > ph * 0.5]
        inner_h = sorted([y for y, _, _ in frame_h if 10 < y < ph - 10])
        inner_v = sorted([x for x, _, _ in frame_v if 10 < x < pw - 10])
        if len(inner_h) >= 2 and len(inner_v) >= 2:
            inner: Rect = (inner_v[0], inner_h[0], inner_v[-1], inner_h[-1])
        else:
            inner = (72, 36, pw - 72, ph - 36)

        ix1, iy1, ix2, iy2 = inner
        rx1, ry1, rx2, ry2 = TITLE_RECT_RATIO
        title_rect: Rect = (
            ix1 + (ix2 - ix1) * rx1,
            iy1 + (iy2 - iy1) * ry1,
            ix1 + (ix2 - ix1) * rx2,
            iy1 + (iy2 - iy1) * ry2,
        )
        return cls(
            pw=pw, ph=ph, inner=inner,
            h_lines=h_lines, v_lines=v_lines,
            frame_h=frame_h, frame_v=frame_v,
            title_rect=title_rect,
        )

    def is_in_inner(self, item: dict, slack: float = 5) -> bool:
        """item bbox 是否落在内框内(slack 容差)。"""
        ix1, iy1, ix2, iy2 = self.inner
        return (item["left"] >= ix1 - slack
                and item["right"] <= ix2 + slack
                and item["top"] >= iy1 - slack
                and item["bottom"] <= iy2 + slack)

    def is_in_title(self, bbox: Rect) -> bool:
        """bbox 中心是否落在标题栏区。"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        tx1, ty1, tx2, ty2 = self.title_rect
        return tx1 <= cx <= tx2 and ty1 <= cy <= ty2
