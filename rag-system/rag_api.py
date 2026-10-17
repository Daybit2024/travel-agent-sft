# -*- coding: utf-8 -*-
"""RAG 服务（重构版）：检索编排 + Flask HTTP 接口。

链路：地名识别 → 候选集过滤 → 向量 + BM25 召回 → RRF 融合 → 阈值过滤
      → 按城市聚合 → small-to-big 返回整篇攻略

接口契约（与旧版兼容，Agent 无需改动）：
  POST /search  {query, search_type: vector|keyword|hybrid, limit,
                 vector_weight?, keyword_weight?}
  GET  /health
  GET  /stats
  POST /search_by_location  {province? / city?}
"""
import logging
import os
import sys
import time

from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (COLLECTION_NAME, RECALL_LIMIT, RRF_KEYWORD_WEIGHT,
                    RRF_VECTOR_WEIGHT, WEB_QUERY_SUFFIX)
from request_log import elapsed_ms, record, set_trace_id
from retrieval import bm25_retriever, fusion, location, query_rewrite, vector, web_fallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_api")

app = Flask(__name__)


def _web_results(query, limit, search_type):
    """网络检索兜底结果：与本地结果同构，content 中标明来源（未核实）

    联网前：别名替换为标准城市名（鹰城→平顶山）+ 补旅游意图词（裸地名直搜会命中百科/新闻）。
    """
    web_query = location.canonicalize(query) + WEB_QUERY_SUFFIX
    hits = web_fallback.search_web(web_query, min(limit, 5))
    results = []
    for h in hits:
        src = f"\n来源: {h['url']}" if h.get("url") else ""
        results.append({
            "city_code": "", "city_name": "", "province_name": "",
            "location": "（网络检索）",
            "content": f"【网络检索 · 未核实】{h.get('title', '')}\n{h.get('snippet', '')}{src}",
            "source": "web",
            "url": h.get("url", ""),
            "search_type": search_type,
            "search_strategy": "web_fallback",
        })
    return results


def _search(query, limit, search_type, vector_weight, keyword_weight):
    """检索编排 + 请求流水埋点

    原实现有 4 个返回点（联网兜底 / 4B 改写后兜底 / 全库兜底 / 正常返回），
    分散埋点容易漏，所以在这里包一层：计时 → 调用 → 落一行事件。
    记录策略、命中城市、返回块的 chunk_id 与名次 —— **只记块级结构字段，不记正文**。
    """
    t0 = time.perf_counter()
    try:
        results, strategy, loc = _search_impl(query, limit, search_type, vector_weight, keyword_weight)
    except Exception as e:
        record("rag", query=query, ok=False, ms=elapsed_ms(t0),
               search_type=search_type, error=f"{type(e).__name__}: {e}")
        raise
    top = []
    for x in results[:limit]:
        mc = x.get("matched_chunk") or {}
        top.append({"chunk_id": mc.get("chunk_id"), "city": x.get("city_name"),
                    "title": mc.get("title"), "rrf": x.get("rrf_score"),
                    "vr": x.get("vector_rank"), "br": x.get("bm25_rank")})
    record("rag", query=query, ok=True, ms=elapsed_ms(t0), search_type=search_type,
           strategy=strategy, cities=(loc or {}).get("city_codes", [])[:5],
           n=len(results), top=top,
           content_chars=sum(len(str(x.get("content") or "")) for x in results),
           web=("web" in strategy))
    return results, strategy, loc


def _search_impl(query, limit, search_type, vector_weight, keyword_weight):
    """核心检索编排：

    ① 映射表：别名 → 标准城市名替换
    ② 表未识别 → 本地 4B 兜底门（判"无地点"→ 全库；识别出城市 → 校验路由；其余 → 网络兜底）
    ③ 本地 4B 语言标准化（地名替换后、检索前）
    ④ 候选集过滤 → 向量+BM25 → RRF → 阈值 → small-to-big
    """
    loc = location.extract_candidates(query)
    candidates = loc["city_codes"]
    q = location.canonicalize(query)              # ① 别名 → 标准城市名

    # 语料未覆盖的知名地点（广州/成都/西安等）→ 网络检索兜底（对 Agent 透明）
    if not candidates and loc.get("uncovered"):
        return _web_results(q, limit, search_type), "web_fallback", loc

    strategy = "location_priority" if candidates else f"{search_type}_fallback"

    # ② 表完全未识别 → 本地 4B 兜底门
    if not candidates:
        name = query_rewrite.normalize_place(q)
        if name:
            code = location.resolve_city_name(name)
            if code:
                candidates = [code]
                strategy = "llm_rewrite"
            else:
                # 识别出地点但不在语料覆盖内 → 网络兜底
                return _web_results(q, limit, search_type), "llm_rewrite_web", loc
        # name 为 None（无地点/无法利用）→ 保持全库检索

    # ③ 语言标准化（本地 4B；失败时保持替换后的 query）
    q = query_rewrite.standardize(q)

    vec_hits, kw_hits = [], []
    if search_type in ("vector", "hybrid"):
        vec_hits = vector.search_vector(q, RECALL_LIMIT, candidates or None)
    if search_type in ("keyword", "hybrid"):
        kw_hits = bm25_retriever.search_bm25(q, RECALL_LIMIT, candidates or None)

    if search_type == "vector":
        hits = fusion.filter_hits(vec_hits, "vector")
    elif search_type == "keyword":
        hits = fusion.filter_hits(kw_hits, "keyword")
    else:
        hits = fusion.filter_hits(
            fusion.rrf_fuse(vec_hits, kw_hits, vector_weight, keyword_weight), "hybrid"
        )

    top_cities = fusion.group_top_cities(hits, limit)
    client = vector.get_client()

    results = []
    for h in top_cities:
        results.append({
            "city_code": h["city_code"],
            "city_name": h["city_name"],
            "province_name": h.get("province_name", ""),
            "location": f'{h.get("province_name", "")}-{h.get("city_name", "")}',
            "content": fusion.fetch_parent(client, h["city_code"], h.get("chunk_index")),
            "matched_chunk": {"chunk_id": h["chunk_id"], "title": h.get("title", ""),
                              "chunk_index": h.get("chunk_index")},
            "score": h.get("score"),
            "rrf_score": h.get("rrf_score"),
            "vector_rank": h.get("vector_rank"),
            "bm25_rank": h.get("bm25_rank"),
            "search_type": search_type,
            "search_strategy": strategy,
        })

    # 边界兜底：本地有候选但检索为空，且 query 提到语料外地点 → 网络
    if not results and loc.get("uncovered"):
        return _web_results(query, limit, search_type), "web_fallback", loc
    return results, strategy, loc


@app.route("/health", methods=["GET"])
def health():
    try:
        client = vector.get_client()
        stats = client.get_collection_stats(COLLECTION_NAME)
        return jsonify({
            "status": "healthy",
            "service": "RAG Travel Guide API",
            "collection": COLLECTION_NAME,
            "collection_size": stats.get("row_count", 0),
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json() or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"error": "缺少query参数", "status": "error"}), 400
        limit = int(data.get("limit", 5))
        search_type = data.get("search_type", "hybrid")
        if search_type not in ("vector", "keyword", "hybrid"):
            return jsonify({"error": f"不支持的 search_type: {search_type}", "status": "error"}), 400
        vector_weight = float(data.get("vector_weight", RRF_VECTOR_WEIGHT))
        keyword_weight = float(data.get("keyword_weight", RRF_KEYWORD_WEIGHT))

        # 请求流水：沿用 agent 传来的 trace_id（便于跨进程把一次用户请求串起来）；
        # 没带就自己生成一个，RAG 独立被调用时也能单独成链。
        set_trace_id(request.headers.get("X-Trace-Id", ""))

        results, strategy, loc = _search(query, limit, search_type, vector_weight, keyword_weight)
        logger.info("query=%r type=%s strategy=%s 命中城市=%d", query, search_type, strategy, len(results))

        return jsonify({
            "status": "success",
            "query": query,
            "search_type": search_type,
            "search_strategy": strategy,
            "matched_locations": {"terms": loc["terms"], "prefixes": loc["prefixes"],
                                  "city_codes": loc["city_codes"][:20]},
            "weights": {"vector_weight": vector_weight, "keyword_weight": keyword_weight},
            "limit": limit,
            "results_count": len(results),
            "results": results,
        })
    except Exception as e:
        logger.exception("搜索接口错误")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/search_by_location", methods=["POST"])
def search_by_location():
    """按城市/省份查看攻略（调试与演示用）"""
    try:
        data = request.get_json() or {}
        province = (data.get("province") or "").strip()
        city = (data.get("city") or "").strip()
        client = vector.get_client()
        if city:
            loc = location.extract_candidates(city)
            codes = loc["city_codes"]
        elif province:
            loc = location.extract_candidates(province)
            codes = loc["city_codes"]
        else:
            return jsonify({"error": "请提供 province 或 city", "status": "error"}), 400
        results = []
        for code in codes[:20]:
            chunks = client.query(collection_name=COLLECTION_NAME,
                                  filter=f'city_code == "{code}"',
                                  output_fields=["city_code", "city_name", "province_name", "chunk_index", "content"],
                                  limit=256)
            if not chunks:
                continue
            chunks.sort(key=lambda r: r["chunk_index"])
            results.append({
                "city_code": code,
                "city_name": chunks[0]["city_name"],
                "province_name": chunks[0]["province_name"],
                "content": "\n\n".join(c["content"] for c in chunks)[:4000],
            })
        return jsonify({"status": "success", "results_count": len(results), "results": results})
    except Exception as e:
        logger.exception("按地点查询错误")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/stats", methods=["GET"])
def stats():
    try:
        client = vector.get_client()
        info = client.get_collection_stats(COLLECTION_NAME)
        return jsonify({
            "status": "success",
            "collection": COLLECTION_NAME,
            "chunks": info.get("row_count", 0),
            "bm25_docs": len(bm25_retriever._docs),
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


def _warmup():
    n = bm25_retriever.load_index()
    logger.info("BM25 索引构建完成：%d 个块", n)
    try:
        ok = query_rewrite.warmup()
        logger.info("4B 预热%s", "完成（模型驻留 30 分钟）" if ok else "跳过（Ollama 不可用，检索不受影响）")
    except Exception as e:
        logger.warning("4B 预热异常（不影响服务）：%s", e)


if __name__ == "__main__":
    _warmup()
    app.run(host="127.0.0.1", port=8010, debug=False)
