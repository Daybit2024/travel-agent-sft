# -*- coding: utf-8 -*-
"""Agent 工具链"是否继续"判断策略（纯逻辑，不实例化、不触网）"""
from agent_graph.toolkit import should_continue_tool_chain as _should


def test_hotel_chain_continues_on_valid_result():
    result = "根据您的需求，推荐以下酒店（数据来源：高德）:\n\n酒店1: 示例酒店\n位置: 某路1号\n评分: 4.5/5"
    assert _should("recommend_hotels", result) is True


def test_hotel_chain_stops_on_empty_result():
    assert _should("recommend_hotels", "暂时没有找到符合您需求的酒店推荐") is False


def test_guide_chain_continues_on_valid_result():
    result = "婺源旅行攻略：篁岭晒秋、石城红枫。" + "更多攻略内容。" * 10
    assert _should("search_travel_guide", result) is True


def test_guide_chain_stops_on_missing_result():
    assert _should("search_travel_guide", "未找到相关旅行攻略信息") is False


def test_other_tools_do_not_continue_chain():
    assert _should("query_route", "从 A 到 B 的路线: ...") is False
    assert _should("get_weather_info", "天气信息(北京 - 2026-09-17起1天): ...") is False
