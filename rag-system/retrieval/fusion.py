# -*- coding: utf-8 -*-
"""结果融合与组装：

  - RRF（倒数排名融合）：向量与 BM25 分数尺度不可比，按排名融合
  - 阈值过滤
  - small-to-big：命中块 → 返回其父文档（该城市整篇攻略，按块序拼回）
"""
from collections import defaultdict
from typing import Dict, List

from config import (BM25_SCORE_THRESHOLD, PARENT_MAX_CHARS, RRF_K,
                    VECTOR_SIM_THRESHOLD)


def rrf_fuse(vector_hits: List[Dict], bm25_hits: List[Dict],
             vector_weight: float, keyword_weight: float):
    """RRF 融合两路召回；返回按融合分排序的块命中列表（保留两路原始分数）"""
    scores = defaultdict(float)
    info = {}
    ranks = defaultdict(dict)
    raw = defaultdict(dict)

    for rank, h in enumerate(vector_hits):
        cid = h["chunk_id"]
        scores[cid] += vector_weight * (1.0 / (RRF_K + rank + 1))
        info.setdefault(cid, h)
        ranks[cid]["vector_rank"] = rank + 1
        raw[cid]["vector_score"] = h.get("score")
    for rank, h in enumerate(bm25_hits):
        cid = h["chunk_id"]
        scores[cid] += keyword_weight * (1.0 / (RRF_K + rank + 1))
        info.setdefault(cid, h)
        ranks[cid]["bm25_rank"] = rank + 1
        raw[cid]["bm25_score"] = h.get("score")

    fused = []
    for cid, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        h = dict(info[cid])
        h["rrf_score"] = round(s, 6)
        h["vector_rank"] = ranks[cid].get("vector_rank")
        h["bm25_rank"] = ranks[cid].get("bm25_rank")
        h["vector_score"] = raw[cid].get("vector_score")
        h["bm25_score"] = raw[cid].get("bm25_score")
        fused.append(h)
    return fused


def filter_hits(hits: List[Dict], mode: str):
    """阈值过滤：单路模式按本路阈值；融合模式要求"至少一路自己够格"

    注：RRF 分数只反映排名，直接设绝对阈值会在单通道命中时误杀——
    因此融合结果的过滤下探到原始分数。
    """
    if mode == "vector":
        return [h for h in hits if h.get("score", 0) >= VECTOR_SIM_THRESHOLD]
    if mode == "keyword":
        return [h for h in hits if h.get("score", 0) >= BM25_SCORE_THRESHOLD]
    kept = []
    for h in hits:
        vs = h.get("vector_score")
        ks = h.get("bm25_score")
        if (vs is not None and vs >= VECTOR_SIM_THRESHOLD) or \
           (ks is not None and ks >= BM25_SCORE_THRESHOLD) or \
           (vs is None and ks is None and h.get("score", 0) >= BM25_SCORE_THRESHOLD):
            kept.append(h)
    return kept


def group_top_cities(hits: List[Dict], limit: int):
    """按城市聚合：每城市取最佳块，按最佳分排序，取前 limit 个城市"""
    best = {}
    for h in hits:
        code = h["city_code"]
        key = h.get("rrf_score") or h.get("score", 0)
        if code not in best or key > best[code]["_best_score"]:
            best[code] = {**h, "_best_score": key}
    ordered = sorted(best.values(), key=lambda x: x["_best_score"], reverse=True)
    return ordered[:limit]


def fetch_parent(client, city_code: str, match_chunk_index=None) -> str:
    """small-to-big：把该城市所有块拼回整篇攻略

    命中块提到最前——否则"从文档头部截 PARENT_MAX_CHARS 字符"会把排在后面的
    小节（住宿/行程/预算等）整段截掉，检索挑中的那一块反而进不了上下文；
    其余块仍按文档顺序，行程规划需要的"整篇"不受影响。
    """
    rows = client.query(
        collection_name=__import__("config").COLLECTION_NAME,
        filter=f'city_code == "{city_code}"',
        output_fields=["chunk_index", "content"],
        limit=256,
    )
    rows = sorted(rows, key=lambda r: r["chunk_index"])
    if match_chunk_index is not None:
        head = [r for r in rows if r["chunk_index"] == match_chunk_index]
        rows = head + [r for r in rows if r["chunk_index"] != match_chunk_index]
    text = "\n\n".join(r["content"] for r in rows)
    if len(text) > PARENT_MAX_CHARS:
        text = text[:PARENT_MAX_CHARS] + "……"
    return text
