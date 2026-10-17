import os

from langchain_openai import ChatOpenAI

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def build_llm() -> ChatOpenAI:
    """复用现有 TRAVEL_LLM_* 环境变量:云端 qwen-plus / 本地 Ollama 一键切换。"""
    return ChatOpenAI(
        model=os.environ.get("TRAVEL_LLM_MODEL", "qwen-plus"),
        api_key=os.environ.get("TRAVEL_LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", ""),
        base_url=os.environ.get("TRAVEL_LLM_BASE_URL", DEFAULT_BASE_URL),
    )
