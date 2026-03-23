# PaddleOCR-VL 参数指导文档

> 适用版本：PaddleOCR 3.x / PaddleOCR-VL 0.9B & 1.5B
> 官方文档：https://paddlepaddle.github.io/PaddleX/3.3/en/pipeline_usage/tutorials/ocr_pipelines/PaddleOCR-VL.html

---

## 一、初始化参数（PaddleOCRVL / create_pipeline）

创建 pipeline 时传入，作用于整个生命周期。

### 1.1 模型配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vl_rec_model_name` | str\|None | None | VL 识别模型名称，如 `PaddleOCR-VL-1.5-0.9B` |
| `vl_rec_model_dir` | str\|None | None | VL 识别模型本地目录路径 |
| `vl_rec_backend` | str\|None | None | 推理后端，可选：`paddle`、`vllm`、`sglang`、`vllm-server` 等 |
| `vl_rec_server_url` | str\|None | None | 远程推理服务地址（配合 `vllm-server` 使用） |
| `vl_rec_max_concurrency` | str\|None | None | 最大并发请求数 |
| `layout_detection_model_name` | str\|None | None | 版面检测模型名称 |
| `layout_detection_model_dir` | str\|None | None | 版面检测模型本地目录路径 |
| `doc_orientation_classify_model_name` | str\|None | None | 文档方向分类模型名称 |
| `doc_orientation_classify_model_dir` | str\|None | None | 文档方向分类模型本地目录路径 |
| `doc_unwarping_model_name` | str\|None | None | 文档扭曲校正模型名称 |
| `doc_unwarping_model_dir` | str\|None | None | 文档扭曲校正模型本地目录路径 |
| `paddlex_config` | str\|None | None | PaddleX 配置文件路径 |

### 1.2 模块开关

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_doc_orientation_classify` | bool | False | 是否启用文档方向分类（处理旋转文档） |
| `use_doc_unwarping` | bool | False | 是否启用文档扭曲校正（处理弯曲/拍照文档） |
| `use_layout_detection` | bool | True | 是否启用版面检测（识别标题、表格、公式等区域） |
| `use_chart_recognition` | bool | False | 是否启用图表识别（解析柱状图、饼图等） |

### 1.3 版面检测参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `layout_threshold` | float\|dict\|None | None | 版面检测置信度阈值（0~1），低于此值的区域会被丢弃。可传 dict 为不同类别设置不同阈值 |
| `layout_nms` | bool\|None | None | 是否对版面检测结果做 NMS（非极大值抑制）去重 |
| `layout_unclip_ratio` | float\|Tuple\|dict\|None | None | 检测框扩展系数，值越大框越大 |
| `layout_merge_bboxes_mode` | str\|dict\|None | None | 检测框合并模式：`large`（取大框）/ `small`（取小框）/ `union`（取并集） |
| `merge_layout_blocks` | bool | True | 是否合并相邻的版面块 |

### 1.4 输出控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `format_block_content` | bool | False | 是否将识别结果格式化为 Markdown |
| `markdown_ignore_labels` | list\|None | None | 在 Markdown 输出中忽略的标签列表 |

可忽略的标签包括：
- `footnote` — 脚注
- `header` / `header_image` — 页眉
- `footer` / `footer_image` — 页脚
- `aside_text` — 旁注文字

### 1.5 性能与硬件

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `device` | str | None | 推理设备：`cpu`、`gpu:0`、`gpu:1`、`npu:0` 等 |
| `enable_hpi` | bool | False | 是否启用高性能推理（High Performance Inference） |
| `use_tensorrt` | bool | False | 是否启用 TensorRT 加速（需 NVIDIA GPU） |
| `precision` | str | "fp32" | 计算精度：`fp32`（全精度）/ `fp16`（半精度，更快但精度略降） |
| `enable_mkldnn` | bool | True | 是否启用 MKL-DNN 加速（仅 CPU 生效） |
| `mkldnn_cache_capacity` | int | 10 | MKL-DNN 缓存容量 |
| `cpu_threads` | int | 8 | CPU 推理线程数 |

---

## 二、predict() 方法参数

调用 `pipeline.predict()` 时传入，可逐次调整。

### 2.1 输入

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input` | str\|list\|numpy | **必填** | 输入文件，支持：PDF 路径、图片路径、图片 URL、目录、numpy 数组、列表 |

### 2.2 模块开关（覆盖初始化设置）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_doc_orientation_classify` | bool\|None | None | 同初始化，传入时覆盖初始化设置 |
| `use_doc_unwarping` | bool\|None | None | 同上 |
| `use_layout_detection` | bool\|None | None | 同上 |
| `use_chart_recognition` | bool\|None | None | 同上 |

### 2.3 版面检测参数（覆盖初始化设置）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `layout_threshold` | float\|dict\|None | None | 同初始化 |
| `layout_nms` | bool\|None | None | 同初始化 |
| `layout_unclip_ratio` | float\|Tuple\|dict\|None | None | 同初始化 |
| `layout_merge_bboxes_mode` | str\|dict\|None | None | 同初始化 |
| `merge_layout_blocks` | bool\|None | None | 同初始化 |

### 2.4 VLM 生成参数（关键！影响识别质量）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | float\|None | None | 采样温度。`0.0` = 最确定性输出（减少幻觉/脑补）；`1.0` = 更随机。**建议设为 0.0** |
| `top_p` | float\|None | None | 核采样。`0.1` = 只从最高概率的 10% token 中选择；`1.0` = 不限制。**建议 0.1~0.3** |
| `repetition_penalty` | float\|None | None | 重复惩罚。`1.0` = 无惩罚；`>1.0` = 抑制重复输出。**建议 1.0** |
| `max_new_tokens` | int\|None | None | 最大生成 token 数。限制单次识别输出长度，防止无限生成 |

### 2.5 图像处理参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `min_pixels` | int\|None | None | 图像最小像素数，低于此值会被放大 |
| `max_pixels` | int\|None | None | 图像最大像素数，超过此值会被缩小。影响显存占用和识别精度 |

### 2.6 输出与提示控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_queues` | bool\|None | None | 是否启用队列异步处理。**建议设为 False**，否则容易出内存异常 |
| `format_block_content` | bool\|None | None | 是否格式化为 Markdown |
| `markdown_ignore_labels` | list\|None | None | 忽略的标签列表 |
| `prompt_label` | str\|None | None | VL 模型提示类型（仅在 `use_layout_detection=False` 时生效）。可选值：`ocr`、`formula`、`table`、`chart` |

---

## 三、推荐配置方案

### 3.1 高精度文档解析（减少幻觉）

```python
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server",
    vl_rec_server_url="http://your-server:8000/v1"
)

output = pipeline.predict(
    input="document.pdf",
    use_queues=False,
    temperature=0.0,
    top_p=0.1,
    repetition_penalty=1.0,
    markdown_ignore_labels=['footnote', 'header_image', 'footer', 'footer_image']
)
```

### 3.2 处理拍照/扭曲文档

```python
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server",
    vl_rec_server_url="http://your-server:8000/v1",
    use_doc_orientation_classify=True,
    use_doc_unwarping=True
)
```

### 3.3 只识别表格（跳过版面检测）

```python
output = pipeline.predict(
    input="table_image.png",
    use_layout_detection=False,
    prompt_label="table",
    temperature=0.0
)
```

### 3.4 只识别公式

```python
output = pipeline.predict(
    input="formula_image.png",
    use_layout_detection=False,
    prompt_label="formula",
    temperature=0.0
)
```

---

## 四、常见问题

### Q: 识别结果出现原文没有的字（如"备"变成"备案"）？
**A:** 这是 VLM 的幻觉现象。设置 `temperature=0.0` + `top_p=0.1` 可减少但无法完全消除。

### Q: 多页 PDF 处理时内存报错？
**A:** 确保 `use_queues=False`，并关注 `max_pixels` 参数控制图像大小。

### Q: 如何加速推理？
**A:** 使用 vLLM/SGLang 后端 + `precision="fp16"` + 提高 `vl_rec_max_concurrency`。

### Q: prompt_label 不生效？
**A:** `prompt_label` 仅在 `use_layout_detection=False` 时有效，因为开启版面检测时模型会自动判断区域类型。


