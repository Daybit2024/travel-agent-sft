# -*- coding: utf-8 -*-
from langchain_core.messages import HumanMessage

from agent_graph.prompts import WORKFLOW_PROMPT, build_system_prompt, build_user_info_text
from agent_graph.skills import (
    build_skill_system_prompt,
    route_skill,
    tool_schemas_for,
)


def _msgs(text):
    return [HumanMessage(content=text)]


def test_review_intent_routes_to_itinerary_review():
    assert route_skill(_msgs("帮我看看这个行程合理吗")) == "itinerary-review"
    assert route_skill(_msgs("杭州三日游，第一天西湖，第二天灵隐寺，这么安排合理吗")) == "itinerary-review"


def test_regular_requests_stay_default():
    assert route_skill(_msgs("你好")) == "default"
    assert route_skill(_msgs("上海三日游有什么推荐")) == "default"
    assert route_skill(_msgs("推荐北京的商务酒店")) == "default"


def test_local_backend_forces_default(monkeypatch):
    monkeypatch.setenv("TRAVEL_LLM_BASE_URL", "http://localhost:11434/v1")
    assert route_skill(_msgs("帮我看看这个行程合理吗")) == "default"


def test_tool_subset_by_skill():
    schemas = [
        {"type": "function", "function": {"name": name}}
        for name in [
            "search_travel_guide",
            "get_weather_info",
            "query_route",
            "recommend_hotels",
            "get_hotel_reviews",
        ]
    ]
    review = {
        s["function"]["name"]
        for s in tool_schemas_for(schemas, "itinerary-review")
    }
    assert review == {"search_travel_guide", "get_weather_info", "query_route"}
    assert len(tool_schemas_for(schemas, "default")) == 5


def test_default_prompt_stays_byte_identical():
    assert build_system_prompt({}) == build_user_info_text({}) + WORKFLOW_PROMPT


def test_cloud_default_prompt_has_cross_city_rule(monkeypatch):
    monkeypatch.delenv("TRAVEL_LLM_BASE_URL", raising=False)
    prompt = build_skill_system_prompt("default", {})
    assert "跨城出行补充规则" in prompt


def test_local_default_prompt_is_byte_identical(monkeypatch):
    monkeypatch.setenv("TRAVEL_LLM_BASE_URL", "http://localhost:11434/v1")
    assert build_skill_system_prompt("default", {}) == build_system_prompt({})
