# -*- coding: utf-8 -*-
"""天气工具 MCP 执行通道(只替换执行层,LLM 侧 schema 不变)。

- 常驻会话 + 后台事件循环;初始化与重置加锁,工具调用并发执行;
- 调用超时保留会话(临时负载/网络问题不重建);其他错误重建;
- 子进程只继承运行必需的环境变量(不传密钥);
- 失败/超时自动回退直连并记录 mcp 事件;
- 开关:TRAVEL_MCP_WEATHER=1(默认)/0,TRAVEL_MCP_TIMEOUT 秒(默认 30)。
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

_init_lock = threading.Lock()
_client = None

# 只给 server 子进程传运行环境与代理,不传 AMAP_KEY / DASHSCOPE_API_KEY 等密钥
_ENV_KEYS = {
    "path", "systemroot", "systemdrive", "windir", "comspec", "pathext",
    "temp", "tmp", "userprofile", "homedrive", "homepath", "appdata",
    "localappdata", "programdata", "programfiles", "programfiles(x86)",
    "processor_architecture", "number_of_processors", "os",
    "http_proxy", "https_proxy", "no_proxy", "all_proxy",
    "requests_ca_bundle", "ssl_cert_file", "pythonioencoding",
}


def _child_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k.lower() in _ENV_KEYS}
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _item_text(item) -> str:
    if isinstance(item, dict):
        return item.get("text", "") if item.get("type") == "text" else ""
    if getattr(item, "type", None) == "text":
        return getattr(item, "text", "") or ""
    return ""


def content_to_text(raw) -> str:
    """MCP content blocks(dict 或 SDK 对象)→ 纯文本。"""
    if isinstance(raw, list):
        return "\n".join(t for t in (_item_text(i) for i in raw) if t)
    return str(raw or "")


class _Client:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.stack = None
        self.session = None
        asyncio.run_coroutine_threadsafe(self._open(), self.loop).result(timeout=TIMEOUT)

    async def _open(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self.stack = contextlib.AsyncExitStack()
        read, write = await self.stack.enter_async_context(
            stdio_client(
                StdioServerParameters(
                    command=sys.executable,
                    args=[str(MCP_SERVER)],
                    env=_child_env(),
                    cwd=str(PROJECT_ROOT),
                )
            )
        )
        self.session = await self.stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

    def call(self, arguments):
        result = asyncio.run_coroutine_threadsafe(
            self.session.call_tool("get_weather_info", arguments), self.loop
        ).result(timeout=TIMEOUT)
        if getattr(result, "isError", False):
            raise RuntimeError("MCP 工具返回 isError")
        return content_to_text(getattr(result, "content", result))

    def close(self):
        try:
            asyncio.run_coroutine_threadsafe(self.stack.aclose(), self.loop).result(timeout=5)
        except Exception:
            pass
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass


def _ensure_client():
    global _client
    with _init_lock:
        if _client is None:
            _client = _Client()
        return _client


def _drop_client():
    global _client
    with _init_lock:
        client, _client = _client, None
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


def warm_up_mcp():
    """后台预热常驻会话,避免首个请求承担初始化耗时。"""
    if not ENABLED:
        return

    def _warm():
        t0 = time.perf_counter()
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
    if not ENABLED or not location or not start_date:
        return None
    t0 = time.perf_counter()
    try:
        client = _ensure_client()
    except Exception as exc:
        record("mcp", tool="get_weather_info", ok=False,
               ms=round((time.perf_counter() - t0) * 1000),
               error=f"{type(exc).__name__}: {exc}")
        print(f"[MCP] 天气工具初始化失败，回退本地直连: {exc}")
        return None
    arguments = {
        "location": location,
        "start_date": start_date,
        "num_days": num_days,
    }
    try:
        result = client.call(arguments)
    except TimeoutError as exc:
        # 超时保留会话:多为临时负载/网络问题,下一次调用可直接复用
        record("mcp", tool="get_weather_info", ok=False, timeout=True,
               ms=round((time.perf_counter() - t0) * 1000),
               error=f"{type(exc).__name__}: {exc}")
        print(f"[MCP] 天气工具超时，回退本地直连: {exc}")
        return None
    except Exception as exc:
        _drop_client()
        record("mcp", tool="get_weather_info", ok=False,
               ms=round((time.perf_counter() - t0) * 1000),
               error=f"{type(exc).__name__}: {exc}")
        print(f"[MCP] 天气工具调用失败，回退本地直连: {exc}")
        return None
    ok = bool(result and result.strip())
    record("mcp", tool="get_weather_info", ok=ok,
           ms=round((time.perf_counter() - t0) * 1000))
    return result if ok else None


atexit.register(_drop_client)
