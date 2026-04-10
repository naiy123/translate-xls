"""OpenAI 和百度 OCR 的客户端/凭据。

- OpenAI client 延迟初始化,允许测试时注入 mock(`set_client`)
- 百度 token 每次调用 OCR 时现取(不缓存,十分钟过期无大碍)
"""
import os
import sys
from pathlib import Path

import requests
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


def _get_openai_key():
    key = os.getenv("OPENAI_API") or os.getenv("OPENAI_API_KEY")
    if key:
        return key
    config_path = Path.home() / ".translate_xls_key"
    if config_path.exists():
        key = config_path.read_text().strip()
        if key:
            return key
    print("错误：未找到 OPENAI_API_KEY")
    sys.exit(1)


_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=_get_openai_key())
    return _client


def set_client(c):
    """测试注入点:外部可替换为 mock client。"""
    global _client
    _client = c


BAIDU_API_KEY = os.getenv("BAIDU_API_KEY")
BAIDU_SECRET_KEY = os.getenv("BAIDU_SECRET_KEY")


def baidu_access_token():
    resp = requests.post("https://aip.baidubce.com/oauth/2.0/token", params={
        "grant_type": "client_credentials",
        "client_id": BAIDU_API_KEY,
        "client_secret": BAIDU_SECRET_KEY,
    })
    data = resp.json()
    if "access_token" in data:
        return data["access_token"]
    print(f"百度 token 获取失败: {data}")
    sys.exit(1)
