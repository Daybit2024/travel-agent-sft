# -*- coding: utf-8 -*-
"""RRF 权重网格搜索：离线复现检索链路，对比不同 keyword 权重下的城市/块级指标。

需要服务停止运行（milvus-lite 单进程独占）。用法（在 rag-system 目录）：
    python eval/grid_search_weights.py
"""
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval import bm25_retriever, fusion, location, vector  # noqa: E402
from retrieval.query_rewrite import standardize  # noqa: E402
from rag_eval import judge  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMBOS = [(1.0, w) for w in (0.3, 0.5, 0.7, 1.0, 1.5)]


def main():
    with io.open(os.path.join(HERE, "test_queries.json"), encoding="utf-8") as f:
        queries = json.load(f)["queries"]

    cases = []
    print("预处理：标准化 + 双路召回（约 1-2 分钟）...")
    t0 = time.time()
    for q in queries:
        loc = location.extract_candidates(q["query"])
        cands = loc["city_codes"]
        if not cands or q.get("expect_strategy"):
            continue  # 只评估本地检索路径（权重与网络/策略用例无关）
        qq = standardize(location.canonicalize(q["query"]))
        vh = vector.search_vector(qq, 20, cands)
        kh = bm25_retriever.search_bm25(qq, 20, cands)
        cases.append((q, vh, kh))
    print(f"有效本地用例 {len(cases)} 条，召回完成（{time.time() - t0:.0f}s）\n")

    print(f"{'vector_w':>8} {'keyword_w':>9} | {'Top1':>6} {'Top3':>6} | {'块级':>6}")
    for vw, kw in COMBOS:
        top1 = top3 = 0
        blk_total = blk_pass = 0
        for q, vh, kh in cases:
            fused = fusion.filter_hits(fusion.rrf_fuse(vh, kh, vw, kw), "hybrid")
            top = fusion.group_top_cities(fused, 5)
            ok, rank = judge(top, q)
            if ok is None:
                continue
            if ok:
                top3 += 1
                if rank == 1:
                    top1 += 1
            et = q.get("expected_chunk_titles")
            if et and top:
                blk_total += 1
                titles = [str(h.get("title", "")) for h in top]
                if any(any(k in t for t in titles) for k in et):
                    blk_pass += 1
        n = len(cases)
        print(f"{vw:>8} {kw:>9} | {top1:>3}/{n} {top3:>3}/{n} | {blk_pass:>3}/{blk_total}")


if __name__ == "__main__":
    main()
