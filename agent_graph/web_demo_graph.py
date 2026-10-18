# -*- coding: utf-8 -*-
"""旅行助手 · LangGraph 版网页 Demo(Gradio)。

用法:
    python -m agent_graph.web_demo_graph
浏览器打开 http://127.0.0.1:7860

会话状态落在 agent_memory.sqlite;启动时从库中列出历史会话供选择续接。
"""
import os
import sys
import uuid

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import gradio as gr
from langchain_core.messages import HumanMessage

from agent_graph.graph import build_graph
from agent_graph.mcp_weather import warm_up_mcp
from agent_graph.memory import make_checkpointer
from agent_graph.sessions import list_sessions
from agent_graph.toolkit import set_trace_id

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

checkpointer = make_checkpointer()
graph = build_graph(checkpointer=checkpointer)
warm_up_mcp()


def _chat_history(thread_id: str) -> list:
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", []) if state and state.values else []
    history = []
    for m in messages:
        if m.type == "human":
            history.append({"role": "user", "content": m.content})
        elif m.type == "ai" and not getattr(m, "tool_calls", None):
            history.append({"role": "assistant", "content": m.content})
    return history


def _session_choices() -> list:
    return [
        (f"{s['title']} · {s['thread_id'][:10]}", s["thread_id"])
        for s in list_sessions()
    ]


def refresh_sessions():
    return gr.Dropdown(choices=_session_choices())


def load_session(thread_id: str):
    if not thread_id:
        return [], thread_id
    return _chat_history(thread_id), thread_id


def respond(message: str, chat: list, thread_id: str):
    """流式输出;结束时以图状态里的 final_answer 为准(护栏降级可能改写文本)。"""
    message = (message or "").strip()
    if not message:
        yield chat, "", thread_id
        return
    if not thread_id:
        thread_id = f"web-{uuid.uuid4().hex[:8]}"
    set_trace_id()
    config = {"configurable": {"thread_id": thread_id}}

    chat = list(chat) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "…"},
    ]
    yield chat, "", thread_id

    partial = ""
    try:
        for chunk, metadata in graph.stream(
            {"messages": [HumanMessage(content=message)]},
            config,
            stream_mode="messages",
        ):
            if metadata.get("langgraph_node") not in ("agent", "wrap_up"):
                continue
            text = getattr(chunk, "content", "")
            if not isinstance(text, str) or not text:
                continue
            partial += text
            chat = chat[:-1] + [{"role": "assistant", "content": partial}]
            yield chat, "", thread_id
    except Exception as exc:
        chat = chat[:-1] + [{"role": "assistant", "content": f"处理请求时出错: {exc}"}]
        yield chat, "", thread_id
        return

    state = graph.get_state(config)
    final_answer = (state.values or {}).get("final_answer") or partial or "（无回复）"
    if final_answer != partial:
        chat = chat[:-1] + [{"role": "assistant", "content": final_answer}]
        yield chat, "", thread_id


def new_session():
    return [], f"web-{uuid.uuid4().hex[:8]}", gr.Dropdown(choices=_session_choices())


with gr.Blocks(title="旅行助手 · LangGraph") as demo:
    gr.Markdown(
        "# 🌍 旅行助手 · LangGraph 版\n"
        "选择历史会话继续,或直接开始新对话。会话状态持久化在 SQLite。"
    )
    with gr.Row():
        session_dd = gr.Dropdown(choices=_session_choices(), label="历史会话", scale=4)
        refresh_btn = gr.Button("刷新列表", scale=1)
        load_btn = gr.Button("加载所选会话", scale=1)

    thread_state = gr.State("")
    chatbot = gr.Chatbot(height=460, label="对话", type="messages", allow_tags=False)

    with gr.Row():
        msg = gr.Textbox(placeholder="输入你的旅行问题，回车发送…", scale=5)
        send = gr.Button("发送", variant="primary", scale=1)
        clear = gr.Button("新对话", scale=1)

    refresh_btn.click(refresh_sessions, None, session_dd)
    load_btn.click(load_session, session_dd, [chatbot, thread_state])
    send.click(respond, [msg, chatbot, thread_state], [chatbot, msg, thread_state])
    msg.submit(respond, [msg, chatbot, thread_state], [chatbot, msg, thread_state])
    clear.click(new_session, None, [chatbot, thread_state, session_dd])


if __name__ == "__main__":
    demo.launch()
