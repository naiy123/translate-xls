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
| `layout_unclip_ratio` | float\|Tuple\|dict\|None | 1.0 | 检测框扩展系数。详见下方说明 |
| `layout_merge_bboxes_mode` | str\|dict\|None | None | 检测框合并模式：`large`（取大框）/ `small`（取小框）/ `union`（取并集） |
| `merge_layout_blocks` | bool | True | 是否合并相邻的版面块 |

**`layout_unclip_ratio` 详解：**

默认值 `1.0` 表示不扩展。支持三种传参方式：
- `float`：所有类别统一扩展，如 `1.5`
- `Tuple[float, float]`：(水平倍数, 垂直倍数)，如 `(1.0, 2.0)` 表示水平不变、垂直扩2倍
- `dict`：按 cls_id 为不同类别设置，如 `{21: (1.0, 2.0)}` 只对 table 垂直扩展

**建议值：**
- 跨页表格场景：`(1.0, 2.0)` — 垂直扩展抓住页面底部边缘字
- 太大（>2.0）风险：可能把表格外的内容框进来，导致 VLM 识别混乱
- 太小（1.0）风险：页面边缘的文字被裁掉

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
| `min_pixels` | int\|None | 112,896 (28\*28\*144) | 图像最小像素数，低于此值会被放大 |
| `max_pixels` | int\|None | 1,003,520 (28\*28\*1280) | 图像最大像素数，超过此值会被缩小。**注意：超出训练范围可能降低精度** |

**默认值说明（源码确认）：**
- 官方默认 `max_pixels = 1,003,520`（约 1M），对应 `1280 * 28 * 28`
- `spotting` 任务特殊处理：`max_pixels = 1,605,632`（约 1.6M），对应 `2048 * 28 * 28`
- `max_pixels` 作用于**裁剪后的区域图片**（不是整页），如果裁剪图已经小于此值则不缩放
- 超过 ~2M 后效果可能退化（超出模型训练时见过的分辨率范围）

**按区域类型独立设置（通过 vlm_extra_args）：**

```python
output = pipeline.predict(
    input="document.pdf",
    vlm_extra_args={
        "table_max_pixels": 2560000,   # 单独给表格设更大的像素上限
        "ocr_max_pixels": 1003520,     # 文本区域用默认值
    }
)
```

支持的 key：`ocr_min/max_pixels`、`table_min/max_pixels`、`formula_min/max_pixels`、`chart_min/max_pixels`、`seal_min/max_pixels`

### 2.6 输出与提示控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_queues` | bool\|None | None | 是否启用队列异步处理。**建议设为 False**，否则容易出内存异常 |
| `format_block_content` | bool\|None | None | 是否格式化为 Markdown |
| `markdown_ignore_labels` | list\|None | None | 忽略的标签列表 |
| `prompt_label` | str\|None | None | VL 模型提示类型（仅在 `use_layout_detection=False` 时生效）。可选值：`ocr`、`formula`、`table`、`chart` |

---

## 三、推荐配置方案

### 3.1 生产环境推荐（跨页表格场景）

```python
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server",
    vl_rec_server_url="http://your-server:8000/v1"
)

output = pipeline.predict(
    input="document.pdf",
    use_queues=False,
    max_pixels=1003520,                    # 官方默认值，不要超出训练范围
    layout_unclip_ratio=(1.0, 2.0),        # 垂直扩展，抓住跨页底部边缘字
    markdown_ignore_labels=['footnote', 'header_image', 'footer', 'footer_image', 'aside_text']
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

### Q: 跨页表格底部单字丢失（如"病毒防护"只识别出"病毒防"）？
**A:** 根本原因是版面检测框没有完全覆盖页面底部边缘的文字。设置 `layout_unclip_ratio=(1.0, 2.0)` 垂直扩展检测框可缓解。这是 0.9B 小模型的已知局限，无法完全消除。

### Q: 识别结果出现原文没有的字（如"备"变成"备案"）？
**A:** 这是 VLM 的幻觉现象。设置 `temperature=0.0` + `top_p=0.1` 可减少但无法完全消除。

### Q: max_pixels 设多大合适？
**A:** 官方默认 1,003,520（约1M像素）。超过 ~2M 可能超出模型训练范围导致效果退化。不要盲目调大。如果只想提升表格精度，用 `vlm_extra_args={"table_max_pixels": 2560000}` 单独调表格。

### Q: 多页 PDF 处理时内存报错？
**A:** 确保 `use_queues=False`，并关注 `max_pixels` 参数控制图像大小。

### Q: 如何加速推理？
**A:** 使用 vLLM/SGLang 后端 + `precision="fp16"` + 提高 `vl_rec_max_concurrency`。

### Q: prompt_label 不生效？
**A:** `prompt_label` 仅在 `use_layout_detection=False` 时有效。开启版面检测时模型自动按区域类型选择 prompt（table → "Table Recognition:"，ocr → "OCR:" 等）。

### Q: 百度在线 API 和本地部署有什么区别？
**A:** 百度 API 不暴露 `temperature`、`top_p`、`max_pixels`、`layout_unclip_ratio` 等参数，精度由服务端固定。本地部署可以完全调参。API 支持 `merge_tables=true` 做跨页表格合并，但不做残余行追加。


