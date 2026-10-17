# -*- coding: utf-8 -*-
"""网络检索兜底：本地语料未覆盖的地点，由 RAG 内部转调搜索 API。

对 Agent 完全透明（工具契约不变）——何时走网络由 RAG 判断，
响应中 search_strategy="web_fallback" 全程可观测。

接入真实现（环境变量）：
    WEB_SEARCH_PROVIDER=bocha  + BOCHA_API_KEY     （推荐：国内、便宜）
    WEB_SEARCH_PROVIDER=tavily + TAVILY_API_KEY
未配置时返回 mock 占位结果（仅用于链路验证）。
"""
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (WEB_SEARCH_LIMIT, WEB_SEARCH_PROVIDER, WEB_SEARCH_TIMEOUT)


def _search_bocha(query, limit):
    key = os.environ.get("BOCHA_API_KEY", "")
    resp = requests.post(
        "https://api.bochaai.com/v1/web-search",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"query": query, "count": limit, "summary": True},
        timeout=WEB_SEARCH_TIMEOUT,
    )
    resp.raise_for_status()
    pages = (((resp.json().get("data") or {}).get("webPages") or {}).get("value")) or []
    return [
        {"title": p.get("name") or "", "url": p.get("url") or "",
         "snippet": p.get("summary") or p.get("snippet") or ""}
        for p in pages[:limit]
    ]


def _search_tavily(query, limit):
    key = os.environ.get("TAVILY_API_KEY", "")
    resp = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": key, "query": query, "max_results": limit},
        timeout=WEB_SEARCH_TIMEOUT,
    )
    resp.raise_for_status()
    return [
        {"title": r.get("title") or "", "url": r.get("url") or "",
         "snippet": r.get("content") or ""}
        for r in (resp.json().get("results") or [])[:limit]
    ]


def _search_mock(query, limit):
    return [{
        "title": f"[占位] {query}",
        "url": "",
        "snippet": "网络检索兜底已触发，但尚未配置搜索 API"
                   "（设置 WEB_SEARCH_PROVIDER=bocha 与 BOCHA_API_KEY）。此占位结果用于链路验证。",
    }][:limit]


def search_web(query: str, limit: int = 0):
    """返回 [{title, url, snippet}]；任何失败都返回空列表（不抛异常）"""
    limit = limit or WEB_SEARCH_LIMIT
    try:
        provider = (WEB_SEARCH_PROVIDER or "").lower()
        if provider == "bocha" and os.environ.get("BOCHA_API_KEY"):
            return _search_bocha(query, limit)
        if provider == "tavily" and os.environ.get("TAVILY_API_KEY"):
            return _search_tavily(query, limit)
        return _search_mock(query, limit)
    except Exception:
        return []


