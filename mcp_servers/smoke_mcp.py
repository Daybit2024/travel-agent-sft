# -*- coding: utf-8 -*-
"""独立验证 MCP 天气工具:list_tools + call_tool。"""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "mcp_servers" / "travel_mcp.py"


async def main():
    params = StdioServerParameters(
        command=sys.executable, args=[str(SERVER)], cwd=str(ROOT)
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"{tool.name} | {tool.description}")
                print(json.dumps(tool.inputSchema, ensure_ascii=False, indent=2))
            result = await session.call_tool(
                "get_weather_info",
                {"location": "北京", "start_date": "2026-10-20", "num_days": 1},
            )
            if getattr(result, "isError", False):
                raise SystemExit(f"MCP tool error: {result}")
            print("\n".join(
                getattr(item, "text", "")
                for item in result.content
                if getattr(item, "type", None) == "text"
            ))


if __name__ == "__main__":
    asyncio.run(main())
