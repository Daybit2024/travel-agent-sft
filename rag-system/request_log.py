# -*- coding: utf-8 -*-
"""请求流水日志（RAG 侧）：一条事件一行 JSONL

与 agent 侧 `agent/tools/request_log.py` **接口一致、各自一份实现**——agent 与 RAG 是
两个进程，共享模块不现实；两边各写各自文件，靠 `trace_id` 事后关联。

trace_id 由 agent 通过请求头 `X-Trace-Id` 传进来（见 rag_api.search），
这样同一次用户请求在 agent 侧与 RAG 侧的事件能串起来。

设计取舍同 agent 侧：只记白名单字段（绝不落全文，块级只记 chunk_id/标题/名次）、
失败也落一行、写失败静默、`REQUEST_LOG=0` 可关。
落点：`logs/<name>/YYYY-MM-DD.jsonl`（name 默认 rag，可用 REQUEST_LOG_NAME 改）
"""
import contextvars
import io
import json
import os
import re
import time
import uuid

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_trace = contextvars.ContextVar("request_trace_id", default="")

_SECRET_RE = re.compile(r"(key|api_key|token|secret)=[^&\s\"']*", re.I)
_MAX_STR = 200


def _scrub(v):
    """字符串脱敏 + 截断；dict / list 递归处理"""
    if isinstance(v, str):
        v = _SECRET_RE.sub(r"\1=***", v)
        return v if len(v) <= _MAX_STR else v[:_MAX_STR] + f"…(+{len(v) - _MAX_STR})"
    if isinstance(v, dict):
        return {k: _scrub(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_scrub(x) for x in v]
    return v


def new_trace_id() -> str:
    return uuid.uuid4().hex[:8]


def set_trace_id(tid: str = "") -> str:
    """为当前上下文设置 trace id（默认自动生成）；返回设置后的值"""
    tid = tid or new_trace_id()
    _trace.set(tid)
    return tid


def get_trace_id() -> str:
    return _trace.get()


def elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def record(kind: str, **fields) -> None:
    """落一条事件。kind ∈ {rag, retr, llm, web}"""
    if os.environ.get("REQUEST_LOG", "1") == "0":
        return
    try:
        row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "trace": get_trace_id(), "kind": kind}
        row.update({k: _scrub(v) for k, v in fields.items() if v is not None})
        d = os.path.join(_ROOT, "logs", os.environ.get("REQUEST_LOG_NAME", "rag"))
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, time.strftime("%Y-%m-%d") + ".jsonl")
        with io.open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass          # 写日志绝不能影响检索
