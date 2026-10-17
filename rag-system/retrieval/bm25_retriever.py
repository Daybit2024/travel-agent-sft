# -*- coding: utf-8 -*-
"""BM25 关键词检索：jieba 分词 + rank_bm25，内存索引（启动时从 Milvus 拉取全部块）。

相比原先"子串包含 + 词频计数"的朴素实现，BM25 提供：
  - IDF：稀有词（关键地名）权重高，通用词权重低
  - TF 饱和：词频增益递减，防止刷分
  - 文档长度归一化
"""
import os
import sys

import jieba
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLLECTION_NAME
from retrieval.vector import ensure_loaded

# 领域词典：避免"三日游/行程安排"等词被切碎（如 三日游行程 → 三日/游行/程）
_USERDICT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jieba_userdict.txt")
try:
    jieba.load_userdict(_USERDICT)
except Exception:
    pass

_FIELDS = ["chunk_id", "city_code", "city_name", "province_name",
           "chunk_index", "title", "content"]

_bm25 = None
_docs = []


def _tokenize(text: str):
    return [w for w in jieba.cut_for_search(text or "") if w.strip()]


def load_index():
    """启动时构建内存 BM25 索引（从 Milvus 拉取全部块）"""
    global _bm25, _docs
    client = ensure_loaded()
    rows = client.query(
        collection_name=COLLECTION_NAME,
        filter='chunk_id != ""',
        output_fields=_FIELDS,
        limit=16384,
    )
    _docs = sorted(rows, key=lambda r: r["chunk_id"])
    corpus = [_tokenize(r["content"]) for r in _docs]
    _bm25 = BM25Okapi(corpus)
    return len(_docs)


def search_bm25(query: str, limit: int, city_codes=None):
    if _bm25 is None:
        load_index()
    toks = _tokenize(query)
    scores = _bm25.get_scores(toks)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits = []
    for i in order:
        if scores[i] <= 0:
            break
        d = _docs[i]
        if city_codes and d["city_code"] not in city_codes:
            continue
        hit = dict(d)
        hit["score"] = float(scores[i])
        hit["source"] = "bm25"
        hits.append(hit)
        if len(hits) >= limit:
            break
    return hits
