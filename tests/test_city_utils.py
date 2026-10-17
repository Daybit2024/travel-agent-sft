# -*- coding: utf-8 -*-
"""城市映射加载（依赖 rag-system/city_code_mapping.json）"""
from tools.city_utils import load_city_map, resolve_city


def test_city_map_loads_and_contains_major_cities():
    cmap = load_city_map()
    assert len(cmap) > 300
    assert cmap.get("北京市") == "110000"
    assert cmap.get("上海市") == "310000"


def test_resolve_city_known_city():
    assert resolve_city("上海") == ("上海市", "310000")
    assert resolve_city("北京") == ("北京市", "110000")


def test_resolve_city_empty_returns_empty():
    """空输入 → 空结果（不是默认城市）"""
    assert resolve_city("") == ("", "")
    assert resolve_city("   ") == ("", "")


def test_resolve_city_unresolvable_returns_empty_not_default():
    """解析失败必须返回空、交给"反问所在城市"层——不能静默退默认城市

    静默退默认值等于伪造一条"事实"喂给模型：prompt 里写着"以用户信息为准"，
    模型会当真并按错城市查询，且没有任何报错。
    """
    assert resolve_city("这不是一个城市名ZZZ") == ("", "")
