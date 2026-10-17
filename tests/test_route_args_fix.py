# -*- coding: utf-8 -*-
"""图版参数纠偏:模型把地名传成坐标时,用用户原话里的「从X到Y」覆盖。"""
from langchain_core.messages import HumanMessage

from agent_graph.nodes import _fix_route_args, extract_from_to


def _msgs(text):
    return [HumanMessage(content=text)]


def test_extract_from_to_variants():
    assert extract_from_to("从郑州大学主校区到北京理工大学怎么走") == (
        "郑州大学主校区",
        "北京理工大学",
    )
    assert extract_from_to("郑州大学到北京理工怎么走") == ("郑州大学", "北京理工")
    assert extract_from_to("从北京南站到天安门") == ("北京南站", "天安门")
    assert extract_from_to("你好") is None


def test_fix_route_args_replaces_hallucinated_coords():
    args = {
        "start_location": "113.6793,36.13011",
        "end_location": "116.28614,39.94258",
        "city_code": "110000",
    }
    fixed = _fix_route_args(args, _msgs("从郑州大学主校区到北京理工大学怎么走"))
    assert fixed["start_location"] == "郑州大学主校区"
    assert fixed["end_location"] == "北京理工大学"


def test_fix_route_args_keeps_place_names():
    args = {"start_location": "郑州大学", "end_location": "北京理工大学"}
    assert _fix_route_args(args, _msgs("从郑州大学到北京理工怎么走")) == args
