# -*- coding: utf-8 -*-
"""RAG 检索质量评测。

对 test_queries.json 逐条调用 RAG 服务的 /search，统计：
  - 城市命中率 Hit@k（期望城市出现在 top-k 结果中）
  - Hit@1（期望城市排第一）
  - MRR（命中用例的平均倒数排名）
  - 无地名兜底用例的非空返回率

用法：
  python rag_eval.py                      # 默认 http://127.0.0.1:8010
  python rag_eval.py --mode keyword --limit 5
  python rag_eval.py --output baseline.json
"""
import argparse
import json
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8010")


def search(query, limit, search_type):
    resp = requests.post(
        f"{API_URL}/search",
        json={"query": query, "search_type": search_type, "limit": limit},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def judge(results, q):
    """返回 (是否命中, 排名)；兜底类用例返回 (None, None)"""
    codes = [str(r.get("city_code", "")) for r in results]
    expected = q.get("expected_city_codes") or []
    if expected:
        for i, c in enumerate(codes):
            if c in expected:
                return True, i + 1
        return False, None
    if q.get("expected_prefix"):
        p = q["expected_prefix"]
        for i, c in enumerate(codes):
            if c.startswith(p):
                return True, i + 1
        return False, None
    prefixes = q.get("expected_prefix_any") or []
    if prefixes:
        for i, c in enumerate(codes):
            if any(c.startswith(p) for p in prefixes):
                return True, i + 1
        return False, None
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=os.path.join(HERE, "test_queries.json"))
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--mode", default="hybrid", choices=["vector", "keyword", "hybrid"])
    ap.add_argument("--output", default="", help="结果保存为 JSON 的路径")
    ap.add_argument("--show", type=int, default=5, help="打印失败样例条数")
    args = ap.parse_args()

    with open(args.queries, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    hit_k = hit_1 = total_eval = 0
    rr_sum = 0.0
    fallback_total = fallback_ok = 0
    strategy_total = strategy_pass = 0
    chunk_total = chunk_pass = 0
    section_total = section_pass = 0
    shown = 0

    for q in data["queries"]:
        row = {"id": q["id"], "category": q["category"], "query": q["query"]}
        try:
            resp = search(q["query"], args.limit, args.mode)
            results = resp.get("results") or []
            ok, rank = judge(results, q)
            row.update(
                ok=ok,
                rank=rank,
                strategy=resp.get("search_strategy", ""),
                top_cities=[r.get("city_name") or r.get("source") or "" for r in results[:3]],
            )

            # 块级指标：命中块的标题是否符合期望小节
            titles = [str((r.get("matched_chunk") or {}).get("title", "")) for r in results]
            row["chunk_titles"] = titles[:3]
            et = q.get("expected_chunk_titles")
            if et:
                chunk_total += 1
                if any(any(k in t for t in titles) for k in et):
                    chunk_pass += 1
                else:
                    row["chunk_miss"] = True

            # 段落覆盖：返回内容是否含期望小节标记
            es = q.get("expected_sections")
            if es and results:
                content0 = results[0].get("content", "")
                section_total += 1
                if all(s in content0 for s in es):
                    section_pass += 1
                else:
                    row["section_miss"] = True

            if q.get("expect_strategy"):
                good = row["strategy"] == q["expect_strategy"] and bool(results)
                strategy_total += 1
                strategy_pass += 1 if good else 0
                row["ok"] = bool(good)
            elif ok is None:
                fallback_total += 1
                if results:
                    fallback_ok += 1
            else:
                total_eval += 1
                if ok:
                    hit_k += 1
                    rr_sum += 1.0 / rank
                    if rank == 1:
                        hit_1 += 1
        except Exception as e:
            row.update(ok=False, rank=None, error=str(e)[:120], top_cities=[])
            total_eval += 1

        rows.append(row)
        if row.get("ok") is False and shown < args.show:
            shown += 1
            print(f"[失败] #{row['id']} {row['category']} | {row['query']} | 返回: {row.get('top_cities')}")

    print("\n===== RAG 评测结果 =====")
    print(f"服务: {API_URL} | 模式: {args.mode} | top-k: {args.limit}")
    print(f"有效用例: {total_eval}（另有无地名兜底 {fallback_total} 条）")
    if total_eval:
        print(f"Hit@{args.limit}:  {hit_k}/{total_eval} ({100.0 * hit_k / total_eval:.1f}%)")
        print(f"Hit@1:   {hit_1}/{total_eval} ({100.0 * hit_1 / total_eval:.1f}%)")
        print(f"MRR:     {rr_sum / total_eval:.3f}")
    if fallback_total:
        print(f"兜底非空率: {fallback_ok}/{fallback_total}")
    if strategy_total:
        print(f"策略用例（网络兜底等）: {strategy_pass}/{strategy_total} 通过")
    if chunk_total:
        print(f"块级命中（期望小节）: {chunk_pass}/{chunk_total} ({100.0 * chunk_pass / chunk_total:.1f}%)")
    if section_total:
        print(f"段落覆盖（返回内容含期望小节）: {section_pass}/{section_total}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "api": API_URL,
                    "mode": args.mode,
                    "limit": args.limit,
                    "hit_k": hit_k,
                    "hit_1": hit_1,
                    "total_eval": total_eval,
                    "mrr": rr_sum / total_eval if total_eval else 0,
                    "fallback_ok": fallback_ok,
                    "fallback_total": fallback_total,
                    "strategy_pass": strategy_pass,
                    "strategy_total": strategy_total,
                    "chunk_pass": chunk_pass,
                    "chunk_total": chunk_total,
                    "section_pass": section_pass,
                    "section_total": section_total,
                    "rows": rows,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"明细已保存: {args.output}")


if __name__ == "__main__":
    main()
