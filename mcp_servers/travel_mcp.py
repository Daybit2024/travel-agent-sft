# -*- coding: utf-8 -*-
"""MCP 天气工具(参数描述与 all_tools.json 对齐)。"""
import contextlib
import sys
from pathlib import Path
from typing import Annotated

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from agent_graph.toolkit import get_weather_info as _get_weather_info

mcp = FastMCP("travel-tools")


@mcp.tool()
def get_weather_info(
    location: Annotated[str, Field(description="城市或景区名称，如'北京'、'婺源篁岭'")],
    start_date: Annotated[
        str,
        Field(description="开始日期，格式 YYYY-MM-DD（用户说'今天/明天'时按用户信息中的当前日期换算）"),
    ],
    num_days: Annotated[
        int,
        Field(description="查询天数：用户说要几天就传几天（如“三日游”传3）；用户没明确说时按默认1天，不要用攻略文档里的天数（攻略天数是生成时随机选的）"),
    ] = 1,
) -> str:
    """查询指定地点和时间段的天气信息"""
    # stdio 模式下 stdout 专用于 JSON-RPC；工具内部的 print 必须改道 stderr
    with contextlib.redirect_stdout(sys.stderr):
        return _get_weather_info(location, start_date, num_days)


if __name__ == "__main__":
    mcp.run()
