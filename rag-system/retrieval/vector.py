# -*- coding: utf-8 -*-
"""向量检索：DashScope 嵌入 + Milvus（chunk 级，支持城市候选集过滤）。

查询嵌入带磁盘缓存（.cache/embed/）：评测重跑、演示复问不再重复消耗 API 额度。
"""
import hashlib
import io
import json
import os
import sys

from openai import OpenAI
from pymilvus import MilvusClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (BASE_DIR, COLLECTION_NAME, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
                    EMBEDDING_DIM, EMBEDDING_MODEL, METRIC_TYPE, MILVUS_URI)

_CACHE_DIR = os.path.join(BASE_DIR, ".cache", "embed")

OUTPUT_FIELDS = ["chunk_id", "city_code", "city_name", "province_name",
                 "chunk_index", "title", "content"]

_client = None
_openai = None


def _ensure():
    global _client, _openai
    if _client is None:
        _client = MilvusClient(uri=MILVUS_URI)
        _openai = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
    return _client


def get_client():
    return _ensure()


def ensure_loaded(client=None):
    """query/search 前需要先 load 集合（Milvus Lite 每次会话后集合会回到 released 状态）"""
    client = client or _ensure()
    try:
        client.load_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client


def embed_query(text: str):
    """查询嵌入（带磁盘缓存，命中则不消耗 API 额度）"""
    key = hashlib.md5((EMBEDDING_MODEL + "|" + text).encode("utf-8")).hexdigest()
    path = os.path.join(_CACHE_DIR, key + ".json")
    if os.path.exists(path):
        try:
            with io.open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    _ensure()
    resp = _openai.embeddings.create(
        model=EMBEDDING_MODEL, input=text, dimensions=EMBEDDING_DIM, encoding_format="float"
    )
    vec = resp.data[0].embedding
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with io.open(path, "w", encoding="utf-8") as f:
            json.dump(vec, f)
    except Exception:
        pass
    return vec


def search_vector(query: str, limit: int, city_codes=None):
    client = ensure_loaded()
    vec = embed_query(query)
    flt = ""
    if city_codes:
        codes = ", ".join(f'"{c}"' for c in city_codes)
        flt = f"city_code in [{codes}]"
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=[vec],
        limit=limit,
        filter=flt,
        output_fields=OUTPUT_FIELDS,
        search_params={"metric_type": METRIC_TYPE},
    )
    hits = []
    for h in (res[0] if res else []):
        ent = dict(h.get("entity") or {})
        ent["score"] = float(h.get("distance", 0.0))
        ent["source"] = "vector"
        hits.append(ent)
    return hits
