# -*- coding: utf-8 -*-
"""MCP 天气执行层的纯函数与开关(不触网、不起子进程)。"""
from agent_graph import mcp_weather


def test_content_to_text_text_blocks():
    raw = [
        {"type": "text", "text": "天气信息(北京)"},
        {"type": "text", "text": "温度: 7℃ ~ 13℃"},
    ]
    assert mcp_weather.content_to_text(raw) == "天气信息(北京)\n温度: 7℃ ~ 13℃"


def test_content_to_text_plain_and_empty():
    assert mcp_weather.content_to_text("纯文本") == "纯文本"
    assert mcp_weather.content_to_text(None) == ""
    assert mcp_weather.content_to_text([]) == ""


def test_content_to_text_sdk_objects():
    class Item:
        def __init__(self, type_, text_):
            self.type = type_
            self.text = text_

    assert mcp_weather.content_to_text([Item("text", "a"), Item("image", None)]) == "a"


def test_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(mcp_weather, "ENABLED", False)
    assert mcp_weather.get_weather_via_mcp("北京", "2026-10-20", 1) is None


def test_child_env_excludes_secrets(monkeypatch):
    monkeypatch.setenv("AMAP_KEY", "secret-amap")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-dashscope")
    env = mcp_weather._child_env()
    assert "AMAP_KEY" not in env
    assert "DASHSCOPE_API_KEY" not in env
    assert env.get("PYTHONIOENCODING") == "utf-8"
