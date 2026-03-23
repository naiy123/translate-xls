192.168.13.170
root/Keyten@2025
conda 虚拟环境：
0.9B是paddlevl
1.5B是paddleocrvl31212
/test_ocr


 1. 在 VSCode 中打开生成的 .md 文件
 2. 按 Cmd + Shift + V 直接预览（表格会渲染成可读格式）
 3. 或按 Cmd + K V 左右分屏对照（左边源码，右边预览）

入 PDF
    │
    ▼
  ① PDF 转图片（每页一张）
    │
    ▼
  ② 文档预处理（use_doc_preprocessor=false，你没开）
    ├─ 方向分类：检测图片是否旋转
    └─ 扭曲校正：拉直拍照的弯曲文档
    │
    ▼
  ③ 版面检测（PP-DocLayoutV3，在你的 GPU 2 上跑）
    │  输入：整页图片
    │  输出：每个区域的 bbox + label + 置信度
    │
    │  例如本页检测到：
    │  ├─ [175,87,513,110]   label=header   "报告编号..."
    │  ├─ [168,757,1023,1528] label=table    ← 表格区域
    │  ├─ [174,679,383,712]  label=paragraph_title  "2.3.2.3 安全设备"
    │  └─ [517,1560,672,1583] label=number   "第21页"
    │
    ▼
  ④ 按区域裁剪图片 → 分别送给 VLM
    │
    │  table 区域 → 裁剪出表格图片 → VLM（prompt="table"）
    │  text 区域  → 裁剪出文字图片 → VLM（prompt="ocr"）
    │  formula    → 裁剪出公式图片 → VLM（prompt="formula"）
    │
    │  VLM 就是你的 vLLM 服务（GPU 0+1，http://127.0.0.1:8118）
    │  VLM 看图 → 自回归生成文本
    │
    ▼
  ⑤ VLM 返回每个区域的文本内容
    │
    │  table → "<table><tr><td>序号</td><td>设备名称</td>...</table>"
    │  text  → "终端计算机的管理和控制..."
    │  title → "2.3.2.3 安全设备"
    │
    ▼
  ⑥ 组装成 parsing_res_list（就是你在 debug json 里看到的）
    │
    │  [
    │    {block_label: "header", block_content: "报告编号...", block_bbox: [...]},
    │    {block_label: "table",  block_content: "<table>...</table>", block_bbox: [...]},
    │    ...
    │  ]
    │
    ▼
  ⑦ 后处理
    ├─ restructure_pages() → 跨页表格合并（按 global_group_id）
    ├─ concatenate_markdown_pages() → 多页拼接
    ├─ markdown_ignore_labels 过滤 → 去掉不需要的标签
    └─ 生成最终 markdown / html / xlsx

  关键点

  - 版面检测在本地 GPU 跑（PP-DocLayoutV3，轻量模型）
  - VLM 识别走远程 vLLM 服务（通过 HTTP 调用）
  - VLM 对每个区域分别识别，不是看整页，所以表格和文字是独立的
  - 表格内容是 VLM 一次性生成整个 HTML，不是逐单元格检测

  这也解释了为什么表格内部会有残余行 — VLM 看到的是裁剪后的表格图片，如果这张图片底部刚好有跨页断裂的半行，VLM 就会把它当成表格的一行输出。