# -*- coding: utf-8 -*-
"""护栏策略:只有结构化事实工具(路线/酒店)才做终答保真核对。"""
from agent_graph.nodes import finalize_node

DEGRADE_PREFIX = "（以下为查询到的原始结果"


def _state(tool_names, results, answer):
    return {
        "final_answer": answer,
        "last_tool_results": results,
        "last_tool_names": tool_names,
        "messages": [],
    }


def test_route_scene_degrades_on_number_mismatch():
    out = finalize_node(
        _state(["query_route"], ["方案1 全程62分钟 | 票价4元"], "全程61分钟，票价4元")
    )
    assert out["final_answer"].startswith(DEGRADE_PREFIX)


def test_route_scene_keeps_faithful_answer():
    out = finalize_node(
        _state(["query_route"], ["方案1 全程62分钟 | 票价4元"], "全程62分钟，票价4元")
    )
    assert out["final_answer"].startswith("全程62分钟")


def test_hotel_scene_still_checked():
    out = finalize_node(
        _state(["recommend_hotels"], ["价格: 300-500元"], "价格 999 元")
    )
    assert out["final_answer"].startswith(DEGRADE_PREFIX)


def test_planning_scene_skips_fidelity_check():
    out = finalize_node(
        _state(
            ["search_travel_guide", "get_weather_info"],
            ["天气: 晴，21℃ ~ 31℃"],
            "第一天去西湖，可以坐2号线，步行632米，票价20元。",
        )
    )
    assert out["final_answer"].startswith("第一天去西湖")
