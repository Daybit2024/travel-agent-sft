# -*- coding: utf-8 -*-
"""公交策略识别：规则层纯函数 + 多轮扫描 + 4B 兜底调用（不触网）"""
import json
from pathlib import Path

import pytest

from tools.transit_strategy import (detect_transit_strategy, match_transit_strategy,
                                    strategy_name)

ROOT = Path(__file__).resolve().parent.parent

# 测试集里允许命中的话术（新增命中必须显式登记 → 强制 review）
_WHITELIST = {
    "我要去KTV，最快的路线",     # 明确说了"最快" → 合法命中
    "医院有直达的公交吗？",       # 问"有直达吗"= 想要直达 → 合法命中（最少换乘）
}


@pytest.mark.parametrize("text,expected", [
    # 2 最少换乘
    ("少换乘", 2),
    ("尽量少换乘，谢谢", 2),
    ("不想换乘", 2),
    ("别老让我换乘", 2),
    ("能直达吗", 2),
    ("一趟车能到吗", 2),
    # 陷阱：疑问句 ≠ 偏好（"需要换乘吗"就是测试集里的原话）
    ("需要换乘吗？", None),
    ("换乘几次？", None),
    ("在哪换乘", None),
    # 3 最少步行
    ("少走路", 3),
    ("我不想走太多路", 3),
    ("腿脚不便，尽量少走", 3),
    ("走路要多久", None),
    ("从机场步行到地铁站要多久", None),      # 双重诱饵：不能命中 3，也不能命中 5
    # 5 不乘地铁
    ("不坐地铁", 5),
    ("尽量别坐地铁", 5),
    ("地铁太挤了，不想坐", 5),
    # 陷阱：宾语必须是地铁/轻轨 → "不坐公交" ≠ 不乘地铁
    ("不坐公交", None),
    ("不坐地铁，坐公交", 5),
    # 1 最经济
    ("最便宜", 1),
    ("省钱一点", 1),
    ("不便宜也行", None),
    ("经济型酒店离地铁近吗", None),
    ("广州有性价比高的酒店吗？", None),          # 酒店属性 ≠ 路线偏好
    ("医院有直达的公交吗？", 2),                # 问"有直达吗"= 想要直达
    # 0 最快捷
    ("最快", 0),
    ("赶时间，越快越好", 0),
    ("最佳路线是什么", None),                # 测试集里真实出现 2 次的诱饵
    ("不着急，慢慢来", None),
    ("别太快了", None),                      # 跨类别否定
    # 4 最舒适
    ("不要太累，轻松点", 4),
    # 优先级：5 > 3 > 2 > 1 > 4 > 0
    ("最快但别换乘", 2),
    ("少走路又不想换乘", 3),
    # 撤销
    ("算了随便吧", None),
    # 健壮性
    ("", None),
    ("   ", None),
])
def test_match_single(text, expected):
    hit = match_transit_strategy(text)
    assert (hit.code if hit else None) == expected


@pytest.mark.parametrize("texts,expected", [
    (["少换乘", "算了随便吧"], None),              # 撤销 → 不再翻出旧偏好
    (["少走路", "算了还是最快吧"], 0),              # 改口为新偏好
    (["少换乘", "换成少走路"], 3),                  # 新偏好覆盖旧的
    (["少走路", "酒店也要少走路的"], 3),            # 当前轮无偏好 → 回溯命中
    (["酒店要少走路的", "从A到B怎么走"], None),      # 跨话题污染：遇到酒店标记停止回溯
    (["少换乘"] * 4, 2),                            # 窗口内（新→旧）
    (["少换乘"] + ["随便聊点别的"] * 4, None),       # 窗口外不召回
    ([], None),
])
def test_multi_turn_scan(texts, expected):
    hit = detect_transit_strategy(texts)
    assert (hit.code if hit else None) == expected


def test_llm_fallback_only_on_rule_miss():
    calls = []

    def stub(q):
        calls.append(q)
        return 3

    assert detect_transit_strategy(["我不想折腾怎么去"], llm_fn=stub).code == 3
    assert len(calls) == 1
    detect_transit_strategy(["少走路"], llm_fn=stub)
    assert len(calls) == 1                    # 规则已命中 → 不再兜底
    detect_transit_strategy(["算了随便吧"], llm_fn=stub)
    assert len(calls) == 1                    # 撤销 → 跳过兜底


def test_llm_fallback_fail_open():
    assert detect_transit_strategy(["我不想折腾"], llm_fn=lambda q: None) is None
    assert detect_transit_strategy(["我不想折腾"]) is None      # 未注入 → 只走规则


def test_strategy_name():
    assert strategy_name(None) == "最快捷(默认)"
    assert strategy_name(2) == "最少换乘"


def _iter_user_texts(obj):
    if isinstance(obj, dict):
        if obj.get("role") == "user" and isinstance(obj.get("content"), str):
            yield obj["content"]
        for v in obj.values():
            yield from _iter_user_texts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_user_texts(v)


def test_dataset_utterances_no_unexpected_hit():
    """测试集全部用户话术：规则只允许命中白名单（误命中回归）"""
    path = ROOT / "datasets" / "merged_test_final.json"
    if not path.exists():
        pytest.skip("测试集不在本地")
    data = json.loads(path.read_text(encoding="utf-8"))
    hits = {t for t in set(_iter_user_texts(data)) if match_transit_strategy(t)}
    unexpected = hits - _WHITELIST
    assert not unexpected, f"出现未登记的命中：{sorted(unexpected)}"
