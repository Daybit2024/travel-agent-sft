"""阶段 2 冒烟:真实工具链(默认问路)+ 护栏。

用法:
    python -m agent_graph.smoke_agent
    python -m agent_graph.smoke_agent "从北京南站到天安门怎么走"
    python -m agent_graph.smoke_agent "上海三日游有什么推荐" my-thread
"""
import os
import sys
import time

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from langchain_core.messages import HumanMessage

from agent_graph.graph import build_graph
from agent_graph.mcp_weather import warm_up_mcp
from agent_graph.memory import make_checkpointer
from agent_graph.toolkit import set_trace_id

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "从北京南站到天安门怎么走"
    thread_id = sys.argv[2] if len(sys.argv) > 2 else f"smoke-tools-{int(time.time())}"

    set_trace_id()
    warm_up_mcp()
    graph = build_graph(checkpointer=make_checkpointer())
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke({"messages": [HumanMessage(content=question)]}, config)

    print("问题:", question)
    print("技能:", result.get("active_skill", "default"), "| 工具轮次:", result.get("tool_rounds", 0),
          "| 公交策略:", result.get("route_strategy"))
    print("最终答复:\n" + result["messages"][-1].content)


if __name__ == "__main__":
    main()
