# -*- coding: utf-8 -*-
"""攻略文本的元数据清洗（纯函数，不触网）

背景：攻略文件头里的"生成时间"是**攻略生成那一刻**的时间戳，与用户要去的日期无关。
它随攻略进上下文等于给模型一个陈旧的时间线索；而"哪些行是元数据"完全可以被代码判断，
所以由代码剥掉，不靠模型自觉。
"""
from agent_graph.toolkit import _strip_guide_meta

_RAW = """# 北京市旅游攻略
城市编码: 110000
生成时间: 2025-09-11 21:33:50
生成用时: 26.44秒
生成模型: deepseek-v3
==================================================

# 北京市2日游攻略

## 🏙️ 城市简介
北京是中国的首都。"""


def test_strips_metadata_lines():
    out = _strip_guide_meta(_RAW)
    assert "生成时间" not in out
    assert "2025-09-11" not in out          # 陈旧时间戳必须消失
    assert "deepseek-v3" not in out
    assert "城市编码" not in out


def test_keeps_content_and_titles():
    out = _strip_guide_meta(_RAW)
    assert "# 北京市2日游攻略" in out        # 正文标题保留（天数信息属于攻略内容，靠 prompt 规则约束用法）
    assert "## 🏙️ 城市简介" in out
    assert "北京是中国的首都。" in out
    assert "=====" not in out               # 分隔线去掉


def test_handles_empty_and_plain_text():
    assert _strip_guide_meta("") == ""
    assert _strip_guide_meta(None) == ""
    plain = "## 🚗 交通指南\n地铁很方便。"
    assert _strip_guide_meta(plain) == plain   # 没有元数据时原样返回
