# -*- coding: utf-8 -*-
"""公交多方案的解析与展示格式化（纯函数，不触网）"""
from tools.get_route import _annotate_plans, _is_coord, _parse_transit_plan, query_routes
from agent_graph.toolkit import format_routes

_fmt = format_routes


def _bus(name, dep, arr, via=None):
    b = {"name": name, "departure_stop": {"name": dep}, "arrival_stop": {"name": arr}}
    if via is not None:
        b["via_stops"] = [{"name": n} for n in via]
        b["via_num"] = str(len(via))
    return {"bus": {"buslines": [b]}}


def _walk(distance):
    return {"walking": {"distance": str(distance)}}


def _raw(duration, cost, segs, walking_distance=None):
    t = {"duration": str(duration), "cost": cost, "segments": segs}
    if walking_distance is not None:
        t["walking_distance"] = str(walking_distance)
    return t


def _mk_plan(t, tr, w, tags=None):
    """直接构造 plan（测展示层时不必经过解析）"""
    return {"总时间(分钟)": t, "票价(元)": "5", "步行距离(米)": w, "换乘次数": tr,
            "标签": tags or [], "详细路线": [f"L{t}路: A 上车 → B 下车（共3站）"]}


def _routes(plans, walking=None, driving=None):
    return {"步行": walking, "公交": plans, "驾车/打车": driving}


# ── 解析 ────────────────────────────────────────────────
def test_parse_basic_fields():
    p = _parse_transit_plan(_raw(5457, "7.0",
                                 [_bus("专212路", "A", "B", ["x", "y", "z"]), _walk(400)],
                                 walking_distance=1489))
    assert p["总时间(分钟)"] == 90
    assert p["票价(元)"] == "7"
    assert p["步行距离(米)"] == 1489
    assert p["换乘次数"] == 0
    assert p["详细路线"][0].startswith("专212路: A 上车 → B 下车")
    assert "共4站" in p["详细路线"][0]        # via_num 3 + 1（via_stops 不含首末站）
    assert "步行 400 米" in p["详细路线"][1]


def test_transfer_count():
    two = _parse_transit_plan(_raw(100, "5", [_bus("L1", "A", "B"), _bus("L2", "B", "C")]))
    assert two["换乘次数"] == 1
    three = _parse_transit_plan(_raw(100, "5", [_bus("L1", "A", "B"), _bus("L2", "B", "C"),
                                                _bus("L3", "C", "D")]))
    assert three["换乘次数"] == 2
    assert _parse_transit_plan(_raw(100, "5", [_walk(300)]))["换乘次数"] == 0


def _railway(name, dep, arr, seconds):
    return {"railway": {
        "name": name,
        "trip": name.split("(")[0],
        "time": str(seconds),
        "departure_stop": {"name": dep},
        "arrival_stop": {"name": arr},
    }}


def test_parse_cross_city_railway_segment():
    """跨城方案:两端的市内接驳 + 中间的铁路段都要出现在详细路线里"""
    t = _raw(13620, "360", [
        _bus("地铁1号线", "黄河南路", "郑州东站"),
        _walk(1536),
        {"railway": {"via_stops": [], "alters": [], "spaces": []}},   # 高德的空壳铁路段
        _railway("G370(西安北-北京西)", "郑州东", "北京西", 8280),
        _walk(51),
        _bus("地铁7号线", "北京西站", "桥湾"),
    ], walking_distance=[])
    p = _parse_transit_plan(t)
    assert p["总时间(分钟)"] == 227
    assert p["票价(元)"] == "360"
    assert p["步行距离(米)"] == 1587          # 顶层 [] → 回退为各段步行累加
    assert p["换乘次数"] == 1                  # 两段公交/地铁,铁路不计换乘
    text = "\n".join(p["详细路线"])
    assert "G370(西安北-北京西): 郑州东 上车 → 北京西 下车（约138分钟）" in text
    assert "地铁1号线" in text and "地铁7号线" in text


def test_parse_railway_shell_skipped():
    p = _parse_transit_plan(_raw(1000, "5", [
        {"railway": {"via_stops": [], "alters": [], "spaces": []}},
        _walk(100),
    ]))
    assert p["详细路线"] == ["步行 100 米"]


def test_is_coord_detection():
    assert _is_coord("116.28,39.94")
    assert _is_coord(" 113.7, 34.7 ")
    assert not _is_coord("北京理工大学")
    assert not _is_coord("116.28")
    assert not _is_coord("")


def test_query_routes_accepts_coord_destination(monkeypatch):
    """终点是坐标时不得再当地址 geocode(4B 会把地名传成坐标)"""
    import tools.get_route as gr

    seen = {}

    def fake_walking(o, d):
        seen["walking"] = d
        return None

    def fake_transit(o, d, city_code, strategy=None):
        seen["transit"] = d
        return []

    def fake_driving(o, d):
        seen["driving"] = d
        return None

    def boom(address):
        raise AssertionError("坐标终点不应调用 geocode")

    monkeypatch.setattr(gr, "get_walking", fake_walking)
    monkeypatch.setattr(gr, "get_transit", fake_transit)
    monkeypatch.setattr(gr, "get_driving", fake_driving)
    monkeypatch.setattr(gr, "geocode", boom)

    gr.query_routes("113.7,34.7", "116.28,39.94", "110000")
    assert seen == {
        "walking": "116.28,39.94",
        "transit": "116.28,39.94",
        "driving": "116.28,39.94",
    }


def test_cost_normalized():
    assert _parse_transit_plan({"cost": "507.000000"})["票价(元)"] == "507"
    assert _parse_transit_plan({"cost": ""})["票价(元)"] == "未知"
    assert _parse_transit_plan({"cost": None})["票价(元)"] == "未知"
    assert _parse_transit_plan({"cost": "未知"})["票价(元)"] == "未知"


def test_walking_distance_fallback_forms():
    p = _parse_transit_plan(_raw(100, "5", [_walk(300), _walk(200)]))
    assert p["步行距离(米)"] == 500                      # 无 walking_distance → 累加
    p2 = _parse_transit_plan({"duration": "60", "cost": "5",
                              "segments": [{"walking": [{"distance": "100"}, {"distance": "50"}]}]})
    assert p2["步行距离(米)"] == 150                     # walking 为 list 形态


def test_via_stops_absent_no_fabrication():
    p = _parse_transit_plan(_raw(100, "5", [_bus("L1", "A", "B", via=[])]))
    assert "共" not in p["详细路线"][0]                  # 拿不到站数就不编


def test_via_stops_long_only_count():
    p = _parse_transit_plan(_raw(100, "5", [_bus("L1", "A", "B", via=[f"s{i}" for i in range(9)])]))
    assert "共10站" in p["详细路线"][0]
    assert "途经" not in p["详细路线"][0]                # 超过 6 站只给站数


# ── 标签 ────────────────────────────────────────────────
def test_annotate_three_dimensions():
    plans = [_mk_plan(85, 2, 1489), _mk_plan(102, 1, 2988), _mk_plan(104, 2, 1083)]
    _annotate_plans(plans)
    assert plans[0]["标签"] == ["时间最短"]
    assert plans[1]["标签"] == ["少换乘"]
    assert plans[2]["标签"] == ["步行最少"]


def test_annotate_single_plan_gets_all():
    plans = [_mk_plan(85, 1, 500)]
    _annotate_plans(plans)
    assert plans[0]["标签"] == ["时间最短", "少换乘", "步行最少"]


def test_annotate_tiebreak_keeps_labels_unique():
    """完全平局时标签不重复发给多条（每个维度唯一）"""
    plans = [_mk_plan(100, 1, 500), _mk_plan(100, 1, 500)]
    _annotate_plans(plans)
    assert plans[0]["标签"] == ["时间最短", "少换乘", "步行最少"]
    assert plans[1]["标签"] == []


# ── 展示 ────────────────────────────────────────────────
def test_format_keeps_all_segments():
    """一条方案 5 段 → 5 段必须全部出现（锁死 [:3] 截断回归）"""
    segs = [_bus("L1", "A", "B"), _walk(100), _bus("L2", "B", "C"), _walk(200), _bus("L3", "C", "D")]
    plan = _parse_transit_plan(_raw(6000, "7", segs, walking_distance=300))
    text = _fmt("A", "D", _routes([plan]))
    for i in (1, 2, 3, 4, 5):
        assert f"    {i}) " in text
    assert "L3" in text                                # 第 5 段（原 [:3] 会被砍掉）


def test_format_lists_at_most_three_plans():
    plans = [_mk_plan(85, 2, 1489), _mk_plan(102, 1, 2988), _mk_plan(104, 2, 1083),
             _mk_plan(125, 1, 2182), _mk_plan(130, 3, 2500)]
    _annotate_plans(plans)
    text = _fmt("A", "B", _routes(plans))
    assert "共5条方案，列出3条" in text
    assert text.count("- 方案") == 3


def test_format_tops_up_when_one_plan_dominates():
    """某条方案在三个维度都占优时，仍要补足到 3 条（而不是只列 1 条）"""
    plans = [_mk_plan(85, 1, 500), _mk_plan(200, 3, 3000), _mk_plan(300, 4, 4000)]
    _annotate_plans(plans)
    text = _fmt("A", "B", _routes(plans))
    assert "共3条方案，列出3条" in text
    assert text.count("- 方案") == 3
    assert "【】" not in text                              # 补足进来的方案不显示空标签


def test_format_strategy_suffix():
    text = _fmt("A", "B", _routes([_mk_plan(100, 1, 500, ["时间最短"])]))
    assert "【公交】\n" in text                          # 无策略 → 无后缀
    text2 = _fmt("A", "B", _routes([_mk_plan(100, 1, 500, ["时间最短"])]), strategy=2)
    assert "(策略: 最少换乘)" in text2


def test_format_empty_transit():
    assert "暂无路线信息" in _fmt("A", "B", _routes([]))
    assert "暂无路线信息" in _fmt("A", "B", _routes(None))


def test_format_walking_and_driving_unchanged():
    routes = {"步行": {"总时间(分钟)": 10, "总距离(米)": 800}, "公交": None,
              "驾车/打车": {"总时间(分钟)": 20, "总距离(米)": 9000, "过路费(元)": "15"}}
    text = _fmt("A", "B", routes)
    assert "【步行】\n总时间: 10分钟\n总距离: 800米" in text
    assert "过路费: 15元" in text
