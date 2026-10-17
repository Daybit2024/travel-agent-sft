# -*- coding: utf-8 -*-
"""评测脚本的工具调用解析逻辑（纯函数）"""
import json

import pytest

pytest.importorskip("torch")  # 评测模块依赖 torch / transformers

import eval_toolcall_accuracy as ev


def test_parse_valid_toolcall():
    text = '好的<tool_call>{"name": "query_route", "arguments": {"end_location": "天安门"}}</tool_call>'
    got = ev.parse_toolcall(text)
    assert got == {"name": "query_route", "arguments": {"end_location": "天安门"}}


def test_parse_invalid_json_returns_none():
    assert ev.parse_toolcall("<tool_call>{broken</tool_call>") is None


def test_parse_no_toolcall_returns_none():
    assert ev.parse_toolcall("请问您要去哪里？") is None


def test_parse_arguments_string_is_normalized():
    inner = json.dumps({"a": 1})
    text = "<tool_call>" + json.dumps({"name": "x", "arguments": inner}) + "</tool_call>"
    got = ev.parse_toolcall(text)
    assert got["arguments"] == {"a": 1}


def test_reference_call():
    msg = {"tool_calls": [{"function": {"name": "get_weather_info",
                                        "arguments": json.dumps({"location": "北京"})}}]}
    got = ev.reference_call(msg)
    assert got == {"name": "get_weather_info", "arguments": {"location": "北京"}}
