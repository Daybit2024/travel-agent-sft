# -*- coding: utf-8 -*-
"""终答保真护栏与地名补全（纯函数，不触网）"""
from agent_graph.toolkit import (_answer_mismatches, _fallback_text,
                                 _qualify_place)

_FACTS = [
    "【公交】(策略: 不乘地铁)\n共2条方案，列出2条:\n"
    "- 方案1【时间最短|少换乘|步行最少】全程241分钟 | 票价313元 | 换乘1次 | 步行1151米\n"
    "    1) 地铁8号线(朱辛庄--瀛海): 前门 上车 → 永定门外 下车（共3站，途经珠市口、天桥）\n"
    "    2) 步行 986 米\n"
]


# ── 地名补全（修跨城误配）────────────────────────────────
def test_qualify_adds_city_prefix():
    assert _qualify_place("颐和园", "北京市") == "北京市颐和园"
    assert _qualify_place("天坛", "北京市") == "北京市天坛"


def test_qualify_keeps_region_qualified():
    assert _qualify_place("北京市海淀区颐和园", "北京市") == "北京市海淀区颐和园"
    assert _qualify_place("江苏省南京市玄武湖", "北京市") == "江苏省南京市玄武湖"


def test_qualify_skips_coords_and_own_city():
    assert _qualify_place("116.48,39.99", "北京市") == "116.48,39.99"
    assert _qualify_place("北京南站", "北京市") == "北京南站"      # 已是本市 → 不重复加
    assert _qualify_place("颐和园", "") == "颐和园"                # 无城市信息 → 不补


def test_qualify_keeps_cross_city_name():
    """裸解析所在城市的名字就出现在地名里 → 跨城，不补本市

    否则 "上海外滩" 会变成 "北京市上海外滩"（实测落到北京，白白错城）
    """
    fake = lambda s: {"city": "上海市", "province": "上海市"}
    assert _qualify_place("上海外滩", "北京市", lookup=fake) == "上海外滩"
    assert _qualify_place("上海路", "北京市", lookup=fake) == "上海路"      # 同理


def test_qualify_prefixes_when_bare_resolves_elsewhere():
    """"北海公园"含"北海"但在北京：裸解析跑到新疆、名字里没有"乌鲁木齐" → 必须补本市"""
    fake = lambda s: {"city": "乌鲁木齐市", "province": "新疆维吾尔自治区"}
    assert _qualify_place("北海公园", "北京市", lookup=fake) == "北京市北海公园"


def test_qualify_prefixes_when_province_matches_result():
    """跨城判据同时看 province（如"云南大理古城"裸解析落在云南省 → 跨城）"""
    fake = lambda s: {"city": "大理白族自治州", "province": "云南省"}
    assert _qualify_place("云南大理古城", "北京市", lookup=fake) == "云南大理古城"


def test_qualify_bare_lookup_failure_prefixes_local():
    def boom(s):
        raise ValueError("无法找到地址")

    assert _qualify_place("北海公园", "北京市", lookup=boom) == "北京市北海公园"


# ── 终答一致性护栏 ──────────────────────────────────────
def test_guard_passes_faithful_answer():
    ans = "推荐方案1：全程241分钟，票价313元，步行1151米，乘地铁8号线从永定门外上车，途经珠市口。"
    assert _answer_mismatches(ans, _FACTS) == []


def test_guard_catches_wrong_number():
    bad = _answer_mismatches("方案一：票价314元，步行760米。", _FACTS)
    assert "314元" in bad
    assert "760米" in bad


def test_guard_catches_fabricated_line():
    bad = _answer_mismatches("乘坐地铁1号线八通线到北京南站。", _FACTS)
    assert any("地铁1号线" in b for b in bad)


def test_guard_prefix_not_swallowed():
    """中文无词边界：'乘坐332路' 不得把'乘坐'吞进线路名（否则忠实回答被误判）"""
    facts = ["- 方案1: 全程100分钟 | 票价5元\n    1) 332路(前门--地铁北宫门站): 前门 上车 → 北宫门 下车"]
    assert _answer_mismatches("可以乘坐332路直达，票价5元。", facts) == []


def test_guard_noop_without_tool_results():
    assert _answer_mismatches("票价999元，乘地铁99号线", []) == []


# ── 站名校验（补最大的漏放面）────────────────────────────
_FACTS_WITH_STATIONS = [
    "【公交】(策略: 不乘地铁)\n共2条方案，列出2条:\n"
    "- 方案1【时间最短】全程241分钟 | 票价313元 | 步行1151米\n"
    "    1) 地铁8号线: 前门 上车 → 永定门外 下车（共3站，途经珠市口、天桥）"
]


def test_guard_catches_fabricated_station():
    """只编站名、数字全对——数字/线路名判据下完全隐形"""
    bad = _answer_mismatches("乘地铁8号线，从望京SOHO步行到天安门东站，241分钟。", _FACTS_WITH_STATIONS)
    assert "天安门东" in bad


def test_guard_station_faithful_answer_passes():
    ans = "乘地铁8号线，前门上车、永定门外下车，全程241分钟，票价313元。"
    assert _answer_mismatches(ans, _FACTS_WITH_STATIONS) == []


def test_guard_station_verb_prefix_not_flagged():
    """站名前带动词/状语（"建议您在永定门外下车"）不得误判——误杀比漏放更糟"""
    assert _answer_mismatches("建议您在永定门外下车，票价313元。", _FACTS_WITH_STATIONS) == []


def test_guard_generic_station_words_not_flagged():
    """'到达终点站''换乘站'是通用说法，不是站名"""
    assert _answer_mismatches("到达终点站后步行1151米。", _FACTS_WITH_STATIONS) == []


def test_guard_markdown_bold_station_not_flagged():
    """模型用 markdown 加粗站名（**天桥站**）不得误判为编造"""
    facts = ["方案1: 全程62分钟 | 票价4元 | 106路: 北京南站 上车 → 天桥 下车 → 天安门东 下车"]
    ans = "乘坐106路，在**天桥站**下车，最终到达**天安门东站**，全程62分钟。"
    assert _answer_mismatches(ans, facts) == []


def test_guard_markdown_numbers_still_checked():
    """剥离强调符后，真实数字改写仍必须被抓到"""
    facts = ["方案1: 全程62分钟 | 票价4元"]
    bad = _answer_mismatches("全程**61分钟**，票价**4元**。", facts)
    assert "61分钟" in bad
    assert "4元" not in bad


# ── 跨工具类型的误报（只验证过路线场景，这层容易误杀）────
def test_guard_hotel_faithful_answer_passes():
    facts = ["根据您的需求，推荐以下酒店（数据来源：高德）:\n\n酒店1: 示例酒店\n"
             "位置: 某路1号\n价格: 300-500元\n评分: 4.5/5"]
    assert _answer_mismatches("为您推荐示例酒店（数据来源：高德），价格300-500元，评分4.5/5。", facts) == []


def test_guard_weather_faithful_answer_passes():
    facts = ["天气信息(北京 - 2026-09-24起1天):\n第1天 (2026-09-24):\n  天气: 晴\n  温度: 18 ~ 28"]
    assert _answer_mismatches("北京9月24日晴，温度18到28度。", facts) == []


# ── 降级渲染 ────────────────────────────────────────────
def test_fallback_strips_model_facing_notes():
    text = _fallback_text(_FACTS_WITH_STATIONS)
    assert "未经汇总整理" in text                  # 告诉用户这不是整理过的答复
    assert "策略:" not in text                     # 去掉面向模型的注记
    assert "共2条方案" not in text
    assert "永定门外" in text                      # 真实数据保留


def test_fallback_empty_results():
    assert "未经汇总整理" in _fallback_text([])
