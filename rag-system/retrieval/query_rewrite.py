# -*- coding: utf-8 -*-
"""Query 改写（本地 4B）：
  - normalize_place：地名归一化兜底门（无地点输出"无"）
  - standardize：    语言标准化（地名替换后、检索前）

工程保障：
  - 磁盘缓存（.cache/rewrite/）：同 query 不重复调 4B，失败不落缓存
  - keep_alive：让 Ollama 常驻模型，避免被卸载后的冷加载（首查 10s+）
  - 任何失败都优雅降级（返回原输入/None），不阻塞检索
"""
import hashlib
import io
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (BASE_DIR, REWRITE_BASE_URL, REWRITE_ENABLED, REWRITE_MODEL,
                    REWRITE_TIMEOUT, STANDARDIZE_ENABLED)

_CACHE_DIR = os.path.join(BASE_DIR, ".cache", "rewrite")

_PROMPT = (
    "你是地名标准化助手。把用户说法中的地点转换为对应的地级市名称，只输出城市名；"
    "如果说法里没有任何地点，只输出：无\n用户：{q}\n回答："
)

_STD_PROMPT = (
    "把用户问题整理成简洁、规范的检索查询：保持原意与所有地点名称不变"
    "（地点已是标准城市名，不要改写），去掉口语化冗余。只输出整理后的查询。\n"
    "用户：{q}\n查询："
)


def _call_4b(prompt: str, max_tokens: int):
    """调用本地 4B；失败返回 None（不抛异常）。

    Ollama 端点走原生 /api/chat（keep_alive 才能生效，模型驻留 30 分钟）；
    其他 OpenAI 兼容端点走标准 /chat/completions。
    """
    base = REWRITE_BASE_URL.rstrip("/")
    try:
        if "11434" in base:  # Ollama
            url = (base[:-3] if base.endswith("/v1") else base) + "/api/chat"
            resp = requests.post(
                url,
                json={
                    "model": REWRITE_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": max_tokens},
                    "keep_alive": "30m",
                },
                timeout=REWRITE_TIMEOUT,
            )
            resp.raise_for_status()
            out = (resp.json().get("message", {}).get("content") or "")
        else:
            resp = requests.post(
                base + "/chat/completions",
                json={
                    "model": REWRITE_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                },
                timeout=REWRITE_TIMEOUT,
            )
            resp.raise_for_status()
            out = resp.json()["choices"][0]["message"]["content"] or ""
        return out.strip().strip("。.，, \n")
    except Exception:
        return None


def warmup():
    """服务启动预热：加载 4B 并置 keep_alive（不写缓存）"""
    return _call_4b("你好", 4) is not None


def _cache_path(ns: str, key_text: str) -> str:
    h = hashlib.md5((REWRITE_MODEL + "|" + key_text).encode("utf-8")).hexdigest()
    return os.path.join(_CACHE_DIR, ns, h + ".txt")


def _cached(ns: str, key_text: str):
    p = _cache_path(ns, key_text)
    if os.path.exists(p):
        try:
            return io.open(p, encoding="utf-8").read().strip()
        except Exception:
            return None
    return None


def _write_cache(ns: str, key_text: str, value: str):
    try:
        p = _cache_path(ns, key_text)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with io.open(p, "w", encoding="utf-8") as f:
            f.write(value)
    except Exception:
        pass


def normalize_place(query: str):
    """地名归一化：返回规范城市名；'无'/空/失败 一律返回 None（结果带缓存）"""
    q = (query or "").strip()
    if not REWRITE_ENABLED or not q:
        return None
    cached = _cached("norm", q)
    if cached is not None:
        return cached or None
    out = _call_4b(_PROMPT.format(q=q), 16)
    if not out or out == "无" or len(out) > 8:
        return None
    _write_cache("norm", q, out)
    return out


def standardize(query: str):
    """语言标准化：返回规范检索式；失败时原样返回输入（结果带缓存）"""
    q = (query or "").strip()
    if not STANDARDIZE_ENABLED or not q:
        return q
    cached = _cached("std", q)
    if cached:
        return cached
    out = _call_4b(_STD_PROMPT.format(q=q), 64)
    if not out or len(out) > 80:
        return q
    _write_cache("std", q, out)
    return out
