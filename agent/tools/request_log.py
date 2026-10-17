# -*- coding: utf-8 -*-
"""请求流水日志：一条事件一行 JSONL

与评测集**互补**（不替代）：日志有真实分布但没有标准答案，评测集有标准答案但样本
与训练同源。日志的三个用途——监控（成功率/延迟/护栏触发率）、回归回放、评测集供给。

设计取舍：
- **只记白名单字段**，绝不整段落 request/response——高德 URL 里带 `key=`，
  本项目出过"真实密钥散落到意想不到的地方"的事故（问题记录第 1、2 条）。
  record() 内部还有一层兜底脱敏（`key=***`）+ 截断，防止调用方手滑。
- **失败也落一行**（带 error），否则"失败率"统计不到。
- **写失败静默**：磁盘/权限问题绝不能影响主流程（与 `_write_cache` 同一取向）。
- **trace_id 串联**：同一次用户请求的所有事件共用一个 id；agent 与 RAG 是
  两个进程，各写各自文件，靠 trace_id 事后关联（不写同一文件）。

落点：`logs/<name>/YYYY-MM-DD.jsonl`（name 默认 agent，可用 REQUEST_LOG_NAME 改）
关闭：`REQUEST_LOG=0`（测试里默认关，见 tests/conftest.py）
"""
import contextvars
import io
import json
import os
import re
import time
import uuid

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 每个线程/协程各自的 trace（转换脚本用线程池并发跑 agent，全局变量会串号）
_trace = contextvars.ContextVar("request_trace_id", default="")

# 兜底脱敏：即使调用方把整条 URL 传进来也不会漏 key
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
    """落一条事件。kind ∈ {turn, llm, amap, tool, rag, guard}

    只落调用方显式传入的字段；None 值略过；字符串统一脱敏 + 截断。
    """
    if os.environ.get("REQUEST_LOG", "1") == "0":
        return
    try:
        row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "trace": get_trace_id(), "kind": kind}
        row.update({k: _scrub(v) for k, v in fields.items() if v is not None})
        d = os.path.join(_ROOT, "logs", os.environ.get("REQUEST_LOG_NAME", "agent"))
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, time.strftime("%Y-%m-%d") + ".jsonl")
        with io.open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass          # 写日志绝不能影响主流程
