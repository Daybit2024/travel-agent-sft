# -*- coding: utf-8 -*-
"""天气工具 MCP 执行通道(只替换执行层,LLM 侧 schema 不变)。

- 常驻会话:后台事件循环 + 一次 stdio 初始化,后续调用复用;
- 失败/超时自动回退本地直连;
- 开关:TRAVEL_MCP_WEATHER=1(默认)/0,TRAVEL_MCP_TIMEOUT 秒。
"""
import asyncio
import atexit
import contextlib
import os
import sys
import threading
import time
from pathlib import Path

from .toolkit import record

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MCP_SERVER = PROJECT_ROOT / "mcp_servers" / "travel_mcp.py"
ENABLED = os.environ.get("TRAVEL_MCP_WEATHER", "1") != "0"
TIMEOUT = float(os.environ.get("TRAVEL_MCP_TIMEOUT", "30"))
INIT_TIMEOUT = 30.0

_lock = threading.Lock()
_client = None


def _item_text(item) -> str:
    if isinstance(item, dict):
        return item.get("text", "") if item.get("type") == "text" else ""
    if getattr(item, "type", None) == "text":
        return getattr(item, "text", "") or ""
    return ""


def content_to_text(raw) -> str:
    """MCP content blocks(dict 或 SDK 对象)→ 纯文本。"""
    if isinstance(raw, list):
        return "\n".join(text for text in (_item_text(i) for i in raw) if text)
    return str(raw or "")


class _WeatherMCPClient:
    def __init__(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._ClientSession = ClientSession
        self._stdio_client = stdio_client
        self._params = StdioServerParameters(
            command=sys.executable,
            args=[str(MCP_SERVER)],
            env=os.environ.copy(),
            cwd=str(PROJECT_ROOT),
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._stack = None
        self._session = None
        ready = asyncio.run_coroutine_threadsafe(self._open(), self._loop)
        ready.result(timeout=INIT_TIMEOUT)

    async def _open(self):
        self._stack = contextlib.AsyncExitStack()
        read, write = await self._stack.enter_async_context(
            self._stdio_client(self._params)
        )
        self._session = await self._stack.enter_async_context(
            self._ClientSession(read, write)
        )
        await self._session.initialize()

    def call(self, arguments: dict, timeout: float) -> str:
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool("get_weather_info", arguments), self._loop
        )
        result = future.result(timeout=timeout)
        if getattr(result, "isError", False):
            raise RuntimeError("MCP 工具返回 isError")
        return content_to_text(getattr(result, "content", result))

    def close(self):
        try:
            if self._stack is not None:
                asyncio.run_coroutine_threadsafe(
                    self._stack.aclose(), self._loop
                ).result(timeout=5)
        except Exception:
            pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2)
        except Exception:
            pass


def _reset_client():
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


def _ensure_client():
    global _client
    if _client is None:
        _client = _WeatherMCPClient()
    return _client


def warm_up_mcp():
    """后台预热常驻会话,避免首个请求承担初始化耗时。"""
    if not ENABLED:
        return

    def _warm():
        t0 = time.perf_counter()
        with _lock:
            try:
                _ensure_client()
                record("mcp", tool="get_weather_info", ok=True, warmup=True,
                       ms=round((time.perf_counter() - t0) * 1000))
            except Exception as exc:
                record("mcp", tool="get_weather_info", ok=False, warmup=True,
                       ms=round((time.perf_counter() - t0) * 1000),
                       error=f"{type(exc).__name__}: {exc}")
                print(f"[MCP] 预热失败，首次请求将回退直连: {exc}")

    threading.Thread(target=_warm, daemon=True).start()


def get_weather_via_mcp(location, start_date, num_days=1):
    """返回 MCP 结果文本;未启用/为空/失败返回 None,调用方回退直连。"""
    if not ENABLED:
        return None
    t0 = time.perf_counter()
    try:
        if not location or not start_date:
            raise ValueError("缺少 location/start_date")
        with _lock:
            try:
                result = _ensure_client().call(
                    {
                        "location": location,
                        "start_date": start_date,
                        "num_days": num_days,
                    },
                    timeout=TIMEOUT,
                )
            except Exception:
                _reset_client()
                raise
        ok = bool(result and result.strip())
        record("mcp", tool="get_weather_info", ok=ok,
               ms=round((time.perf_counter() - t0) * 1000))
        return result if ok else None
    except Exception as exc:
        record("mcp", tool="get_weather_info", ok=False,
               ms=round((time.perf_counter() - t0) * 1000),
               error=f"{type(exc).__name__}: {exc}")
        print(f"[MCP] 天气工具调用失败，回退本地直连: {exc}")
        return None


atexit.register(_reset_client)
