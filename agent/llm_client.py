# -*- coding: utf-8 -*-
"""工具层 LLM 客户端(酒店评价生成等内部调用)。

复用 TRAVEL_LLM_* 环境变量:云端 qwen-plus / 本地 Ollama 一键切换。
"""
import os

from openai import OpenAI

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def call_llm(model_name: str = "qwen-plus", prompt: str | None = None,
             system_prompt: str | None = None, messages=None,
             stream: bool = False, show_thinking: bool = False) -> str:
    base_url = os.environ.get("TRAVEL_LLM_BASE_URL", DEFAULT_BASE_URL)
    api_key = os.environ.get("TRAVEL_LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    model = os.environ.get("TRAVEL_LLM_MODEL", model_name)

    if messages is None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if prompt:
            messages.append({"role": "user", "content": prompt})

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""
