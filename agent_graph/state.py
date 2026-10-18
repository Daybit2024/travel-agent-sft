from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """图共享状态。

    messages 用 add_messages reducer 累积(节点返回新消息列表即可);
    total=False 允许 invoke 时只给部分字段。
    其余字段为后续阶段预留:
      tool_rounds       工具轮次计数(阶段 2)
      route_strategy    公交偏好策略码(阶段 3)
      last_tool_results 最近一轮工具结果,供保真护栏核对(阶段 2)
      final_answer      出口节点产出的最终答复(阶段 2)
    """

    messages: Annotated[list, add_messages]
    trace_id: str
    user_info: dict
    active_skill: str
    tool_rounds: int
    route_strategy: int | None
    last_tool_results: list
    last_tool_names: list
    last_round_continue: bool
    final_answer: str
    guard_turns: int
    guard_hits: int
