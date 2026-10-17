"""阶段 1 冒烟:两轮对话,验证 SqliteSaver 跨轮记忆。

用法(项目根目录,激活 .venv 后):
    python -m agent_graph.smoke_chat
    python -m agent_graph.smoke_chat my-thread   # 指定 thread 续上次对话
"""
import os
import sys

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from langchain_core.messages import HumanMessage

from agent_graph.graph import build_graph
from agent_graph.memory import make_checkpointer
from agent_graph.toolkit import set_trace_id

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    thread_id = sys.argv[1] if len(sys.argv) > 1 else "smoke-1"
    set_trace_id()
    graph = build_graph(checkpointer=make_checkpointer())
    config = {"configurable": {"thread_id": thread_id}}

    graph.invoke({"messages": [HumanMessage(content="你好,用一句话介绍你自己")]}, config)
    result = graph.invoke({"messages": [HumanMessage(content="我刚才问了你什么?")]}, config)
    print("回答:", result["messages"][-1].content)


if __name__ == "__main__":
    main()
