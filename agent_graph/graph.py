from langgraph.graph import END, START, StateGraph

from .llm import build_llm
from .nodes import agent_node, finalize_node, tools_node, wrap_up_node
from .routing import after_agent, after_tools
from .skills import route_skill
from .state import AgentState
from .toolkit import load_tool_schemas, set_trace_id


def load_skill_node(state) -> dict:
    trace_id = set_trace_id()
    return {"active_skill": route_skill(state["messages"]), "trace_id": trace_id}


def build_graph(checkpointer=None, llm=None):
    llm = llm or build_llm()
    tool_schemas = load_tool_schemas()

    builder = StateGraph(AgentState)
    builder.add_node("load_skill", load_skill_node)
    builder.add_node("agent", lambda state: agent_node(state, llm, tool_schemas))
    builder.add_node("tools", tools_node)
    builder.add_node("wrap_up", lambda state: wrap_up_node(state, llm))
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "load_skill")
    builder.add_edge("load_skill", "agent")
    builder.add_conditional_edges("agent", after_agent, {"tools": "tools", "finalize": "finalize"})
    builder.add_conditional_edges("tools", after_tools, {"agent": "agent", "wrap_up": "wrap_up"})
    builder.add_edge("wrap_up", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)
