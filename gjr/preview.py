"""Debug 预览 PNG 渲染:bbox / cluster / class 三张图。

所有标注都画在 fitz.Shape 上然后 pixmap 导出,不保存原 PDF。
颜色表从 gjr.config 读取。
"""
import fitz

from gjr.classify import classify_block
from gjr.config import CLASS_COLORS, PREVIEW_PALETTE


def render_debug_previews(pdf_path, page_num, output_dir, stem,
                          items, blocks, snapped_bboxes, dpi=200,
                          raw_items=None, title_rect=None):
    """
    输出三张 debug PNG:原始 bbox / 聚类 / 分类。不修改原 PDF。

    参数:
      items       — 内框过滤 + 噪音过滤后的 items(用于 cluster/class 两张图)
      blocks      — 聚类后的 block 列表
      snapped_bboxes — snap 到表格线后的大框
      raw_items   — 百度 OCR 原始 items(过滤前)。传入后 bbox.png 显示所有原始行,
                    否则 fallback 到 items(过滤后)。
                    推荐始终传 raw_items,便于观察内框过滤是否误伤
      title_rect  — 右下标题栏区。传入后 class 图按位置感知分类(中心在区内的
                    小块升级为 GPT),与主流水线决策一致
    """
    page_label = f"p{page_num+1}"
    mat = fitz.Matrix(dpi / 72, dpi / 72)

    def _in_title(bbox):
        if title_rect is None:
            return False
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return (title_rect[0] <= cx <= title_rect[2]
                and title_rect[1] <= cy <= title_rect[3])

    block_cats = [classify_block(b, in_title_rect=_in_title(snapped_bboxes[bi]))
                  for bi, b in enumerate(blocks)]
    # bbox 图显示原始 OCR(过滤前),能看到哪些被内框/噪音过滤器剔除了
    bbox_items = raw_items if raw_items is not None else items

    def _render(draw_fn, suffix):
        doc = fitz.open(str(pdf_path))
        page = doc[page_num]
        shape = page.new_shape()
        draw_fn(shape)
        shape.commit()
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = output_dir / f"{stem}_{page_label}_{suffix}.png"
        pix.save(str(out_path))
        doc.close()
        return out_path.name

    def draw_bbox(shape):
        # 全部显示为浅红,被过滤掉的行同样会出现在图里
        for it in bbox_items:
            shape.draw_rect(fitz.Rect(it["left"], it["top"], it["right"], it["bottom"]))
        shape.finish(color=(0.85, 0.15, 0.15), width=0.5, fill=None)

    def _draw_blocks_colored(shape, color_fn, label_fn):
        for bi, block in enumerate(blocks):
            color = color_fn(bi)
            for it in block:
                shape.draw_rect(fitz.Rect(it["left"], it["top"], it["right"], it["bottom"]))
            shape.finish(color=color, width=0.4, fill=None)
            bx1, by1, bx2, by2 = snapped_bboxes[bi]
            shape.draw_rect(fitz.Rect(bx1, by1, bx2, by2))
            shape.finish(color=color, width=1.2, fill=None)
            shape.insert_text(fitz.Point(bx1 + 2, by1 + 8),
                              label_fn(bi), fontsize=7, color=color)

    def draw_cluster(shape):
        _draw_blocks_colored(
            shape,
            color_fn=lambda bi: PREVIEW_PALETTE[bi % len(PREVIEW_PALETTE)],
            label_fn=lambda bi: f"#{bi}",
        )

    def draw_class(shape):
        _draw_blocks_colored(
            shape,
            color_fn=lambda bi: CLASS_COLORS.get(block_cats[bi], (0.5, 0.5, 0.5)),
            label_fn=lambda bi: block_cats[bi],
        )

    names = [
        _render(draw_bbox, "bbox"),
        _render(draw_cluster, "cluster"),
        _render(draw_class, "class"),
    ]
    print(f"    [debug] 预览: {', '.join(names)}")
