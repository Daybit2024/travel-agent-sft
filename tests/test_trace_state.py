# -*- coding: utf-8 -*-
"""trace_id 从图入口写入状态,供各节点在自身线程恢复。"""
from langchain_core.messages import HumanMessage

from agent_graph.graph import load_skill_node


def test_load_skill_sets_trace_id():
    out = load_skill_node({"messages": [HumanMessage(content="你好")]})
    assert out["active_skill"] == "default"
    assert out["trace_id"]
