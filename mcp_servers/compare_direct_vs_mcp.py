# -*- coding: utf-8 -*-
"""对比:天气工具直连 vs MCP 裸接(同名、FastMCP 默认 schema、无包装)。"""
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent_graph.llm import build_llm
from agent_graph.prompts import build_system_prompt
from agent_graph.toolkit import call_function, load_tool_schemas

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUESTION = "帮我查一下北京 2026-10-20 的天气"
SERVER = PROJECT_ROOT / "mcp_servers" / "travel_mcp.py"


def _schema_of(tool):
    try:
        return tool.args_schema.model_json_schema()
    except Exception:
        return getattr(tool, "args", None)


def _print_call(tag, resp):
    print(f"[{tag}] content={resp.content!r}")
    print(f"[{tag}] tool_calls:")
    print(json.dumps(resp.tool_calls, ensure_ascii=False, indent=2, default=str))


def _direct():
    schemas = [s for s in load_tool_schemas() if s["function"]["name"] == "get_weather_info"]
    print("=== 直连 schema(all_tools.json) ===")
    print(json.dumps(schemas, ensure_ascii=False, indent=2))

    llm = build_llm()
    bound = llm.bind_tools(schemas)
    messages = [SystemMessage(content=build_system_prompt({})), HumanMessage(content=QUESTION)]
    resp = bound.invoke(messages)
    _print_call("直连", resp)

    tool_msgs = []
    for call in resp.tool_calls:
        result = call_function(call["name"], call["args"])
        print(f"[直连] 工具结果 type={type(result).__name__} repr={result[:300]!r}")
        tool_msgs.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    if tool_msgs:
        final = llm.invoke(messages + [resp] + tool_msgs)
        print(f"[直连] 最终答复: {final.content}")


async def _mcp():
    client = MultiServerMCPClient({
        "travel": {
            "command": sys.executable,
            "args": [str(SERVER)],
            "transport": "stdio",
            "cwd": str(PROJECT_ROOT),
            "env": os.environ.copy(),
        }
    })
    tools = await client.get_tools()
    weather_tools = [t for t in tools if t.name == "get_weather_info"]
    print("=== MCP 工具清单 ===")
    print([t.name for t in tools])
    for t in weather_tools:
        print(f"name={t.name!r}")
        print(f"description={t.description!r}")
        print("args_schema=")
        print(json.dumps(_schema_of(t), ensure_ascii=False, indent=2, default=str))

    llm = build_llm()
    bound = llm.bind_tools(weather_tools)
    messages = [SystemMessage(content=build_system_prompt({})), HumanMessage(content=QUESTION)]
    resp = await bound.ainvoke(messages)
    _print_call("MCP", resp)

    tool_msgs = []
    for call in resp.tool_calls:
        tool = next(t for t in weather_tools if t.name == call["name"])
        raw = await tool.ainvoke(call["args"])
        print(f"[MCP] 原始结果 type={type(raw).__name__} repr={str(raw)[:300]!r}")
        if isinstance(raw, list):
            text = "\n".join(
                item.get("text", "")
                for item in raw
                if isinstance(item, dict) and item.get("type") == "text"
            )
        else:
            text = str(raw)
        print(f"[MCP] 规范化 ToolMessage: {text[:300]!r}")
        tool_msgs.append(ToolMessage(content=text, tool_call_id=call["id"]))

    if tool_msgs:
        final = await llm.ainvoke(messages + [resp] + tool_msgs)
        print(f"[MCP] 最终答复: {final.content}")


if __name__ == "__main__":
    print("########## 直连 ##########")
    _direct()
    print("\n########## MCP 裸接 ##########")
    asyncio.run(_mcp())
