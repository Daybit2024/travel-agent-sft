"""LLM 请求埋点:LangChain 回调 → 现有 request_log(JSONL)。

补齐图版相对于旧实现缺失的 llm 事件(耗时 / tokens / 工具名),
保持 logs/ 分析与 scripts/request_log_report.py 的口径不变。
"""
import os
import time

from langchain_core.callbacks import BaseCallbackHandler

from .toolkit import record


def _backend() -> str:
    return "local" if "11434" in os.environ.get("TRAVEL_LLM_BASE_URL", "") else "cloud"


class RequestLogHandler(BaseCallbackHandler):
    def __init__(self):
        self._t0 = None
        self._msgs = 0

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self._t0 = time.perf_counter()
        self._msgs = sum(len(batch) for batch in messages)

    def on_llm_start(self, serialized, prompts, **kwargs):
        if self._t0 is None:
            self._t0 = time.perf_counter()
            self._msgs = len(prompts)

    @staticmethod
    def _usage(response):
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or {}
        tin = usage.get("prompt_tokens")
        tout = usage.get("completion_tokens")
        if tin is None:
            try:
                msg = response.generations[0][0].message
                meta = getattr(msg, "usage_metadata", None) or {}
                tin = meta.get("input_tokens")
                tout = meta.get("output_tokens")
            except Exception:
                pass
        return tin, tout

    @staticmethod
    def _call_names(response):
        names = []
        try:
            for gen_list in getattr(response, "generations", []):
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    for call in (getattr(msg, "tool_calls", None) or []):
                        names.append(call.get("name"))
        except Exception:
            pass
        return names

    def on_llm_end(self, response, **kwargs):
        ms = (time.perf_counter() - self._t0) * 1000 if self._t0 else None
        tin, tout = self._usage(response)
        names = self._call_names(response)
        record(
            "llm",
            backend=_backend(),
            model=os.environ.get("TRAVEL_LLM_MODEL", "qwen-plus"),
            ok=True,
            ms=round(ms) if ms is not None else None,
            msgs=self._msgs,
            tools=bool(names),
            call_names=names or None,
            tokens_in=tin,
            tokens_out=tout,
        )
        self._t0 = None

    def on_llm_error(self, error, **kwargs):
        ms = (time.perf_counter() - self._t0) * 1000 if self._t0 else None
        record(
            "llm",
            backend=_backend(),
            model=os.environ.get("TRAVEL_LLM_MODEL", "qwen-plus"),
            ok=False,
            ms=round(ms) if ms is not None else None,
            msgs=self._msgs,
            error=f"{type(error).__name__}: {error}",
        )
        self._t0 = None
