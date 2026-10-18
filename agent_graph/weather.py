# -*- coding: utf-8 -*-
"""天气查询与格式化(直连与 MCP server 共用,避免 server 依赖整个工具层)。"""
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from tools.get_weather import get_weather_by_date_range  # noqa: E402


def fetch_weather(location: str, start_date: str, num_days: int = 1) -> str:
    try:
        weather_list = get_weather_by_date_range(location, start_date, num_days)
        if not weather_list:
            return f"无法获取{location}在{start_date}起{num_days}天的天气信息"
        text = f"天气信息({location} - {start_date}起{num_days}天):\n"
        for i, w in enumerate(weather_list, 1):
            text += f"\n第{i}天 ({w['日期']}):\n"
            text += f"  天气: {w.get('天气', w.get('白天天气', '未知'))}\n"
            text += f"  温度: {w['最低温']} ~ {w['最高温']}\n"
        return text
    except Exception as e:
        return f"天气查询出错: {e}"
