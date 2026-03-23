import os
import json
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddlex import create_pipeline

pipeline = create_pipeline("table_recognition_v2", device="gpu:3")

input_file = "/test_ocr/page_1_only.pdf"
output_dir = "/test_ocr/output/table_test"
os.makedirs(output_dir, exist_ok=True)

count = 0
for res in pipeline.predict(input=input_file):
    count += 1
    print(f"--- Result {count} ---")
    print(f"Type: {type(res)}")

    # 保存所有格式
    res.save_to_json(output_dir)
    res.save_to_html(output_dir)
    res.save_to_xlsx(output_dir)
    res.save_to_img(output_dir)

    # 打印结果概要
    res.print()
    print(f"--- Result {count} Done ---")

if count == 0:
    print("No results returned! Check if the PDF contains tables.")
else:
    print(f"Total {count} results saved to {output_dir}")
