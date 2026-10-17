# -*- coding: utf-8 -*-
"""RAG 三套件测试：

A) 语料城市覆盖：731 个语料城市逐一查询"{城市}有什么好玩的"，
   检查 top-1 / top-3 是否返回该城市自己的攻略。
B1) 语料内城市（含别名形式 长安→西安 / 星城→长沙 / 金城→兰州）：
   期望走本地检索（location_priority）且命中该城市自身。
   ——这是 2026-09-26 语料从 361 城补全到 731 城后的回归用例：
   补全前这些城市全在 uncovered_places 里、走网络兜底；补全后应全部本地化。
B2) 语料外地点（景点级：九寨沟/稻城亚丁/少林寺/黄果树/凤凰古城）：
   期望走网络兜底（web_fallback）且有结果。

注：境外城市不在测试范围内 —— 本项目是**国内**旅行助手，语料与映射表都只覆盖中国，
出境游属于设计外场景（Agent 侧由"非旅行问题礼貌拒答"那类规则处理）。

输出：汇总 + city_coverage_result.json
"""
import io
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR  # noqa: E402

API = "http://127.0.0.1:8010"


def search(q, limit=3):
    r = requests.post(API + "/search",
                      json={"query": q, "search_type": "hybrid", "limit": limit}, timeout=180)
    r.raise_for_status()
    return r.json()


def main():
    with io.open(os.path.join(BASE_DIR, "city_code_mapping.json"), encoding="utf-8") as f:
        mapping = json.load(f)["城市编码映射"]
    cities = [(str(c), info["name"]) for c, info in mapping.items() if info.get("name")]

    print(f"== A. 语料城市覆盖（{len(cities)} 城）==", flush=True)
    top1 = top3 = err = 0
    failures = []
    rows_a = []
    t_start = time.time()
    for i, (code, name) in enumerate(cities):
        q = f"{name}有什么好玩的"
        try:
            d = search(q)
            rs = d.get("results", [])
            got_codes = [str(r.get("city_code") or "") for r in rs]
            ok1 = bool(got_codes) and got_codes[0] == code
            ok3 = code in got_codes
            top1 += ok1
            top3 += ok3
            rows_a.append({"city": name, "code": code, "strategy": d.get("search_strategy"),
                           "top1": ok1, "top3": ok3,
                           "got": [r.get("city_name") or r.get("source") for r in rs[:3]]})
            if not ok3:
                failures.append((name, code, d.get("search_strategy"),
                                 [r.get("city_name") or r.get("source") for r in rs[:3]]))
        except Exception as e:
            err += 1
            failures.append((name, code, "ERROR", str(e)[:60]))
        if (i + 1) % 20 == 0:
            print(f"  进度 {i + 1}/{len(cities)} | Top1={top1} Top3={top3} 失败={len(failures)} "
                  f"({time.time() - t_start:.0f}s)", flush=True)

    n = len(cities)
    print(f"\n[A 组汇总] Top1 {top1}/{n} ({100.0 * top1 / n:.1f}%) | "
          f"Top3 {top3}/{n} ({100.0 * top3 / n:.1f}%) | 异常 {err}")
    if failures:
        print("  失败清单（前 15）:")
        for f in failures[:15]:
            print("   ", f)

    print(f"\n== B1. 语料内城市（应走本地检索，含别名形式）==", flush=True)
    probes_b1 = [("西安有什么好玩的", "西安"), ("兰州有什么好玩的", "兰州"),
                 ("洛阳有什么好玩的", "洛阳"), ("郑州有什么好玩的", "郑州"),
                 ("长沙有什么好玩的", "长沙"), ("拉萨有什么好玩的", "拉萨"),
                 ("三亚有什么好玩的", "三亚"), ("桂林有什么好玩的", "桂林"),
                 ("长安怎么玩", "西安"), ("星城有什么好玩的", "长沙"),
                 ("金城有什么好玩的", "兰州")]
    rows_b = []
    b1_ok = 0
    for q, expect_city in probes_b1:
        try:
            d = search(q)
            rs = d.get("results", [])
            strategy = d.get("search_strategy", "")
            got_city = (rs[0].get("city_name") if rs else "") or ""
            ok = strategy == "location_priority" and expect_city in got_city
            b1_ok += ok
            rows_b.append({"suite": "B1", "query": q, "expect": expect_city,
                           "strategy": strategy, "got": got_city, "ok": ok})
            print(f"  {q:<14} → {strategy:<18} | {got_city or '-':<8} | {'OK' if ok else 'FAIL'}")
        except Exception as e:
            rows_b.append({"suite": "B1", "query": q, "expect": expect_city,
                           "strategy": "ERROR", "ok": False})
            print(f"  {q:<14} → ERROR {str(e)[:50]}")
    print(f"\n[B1 汇总] {b1_ok}/{len(probes_b1)} 通过（语料补全生效的回归用例）")

    # B2：真正无城市级覆盖的地点（在 uncovered_places 名单内）→ 期望网络兜底
    print(f"\n== B2. 语料外地点（应走网络兜底）==", flush=True)
    probes_b2 = ["九寨沟怎么玩", "稻城亚丁值得去吗", "少林寺有什么好玩的",
                 "黄果树怎么玩", "凤凰古城值得去吗"]
    b2_ok = 0
    for q in probes_b2:
        try:
            d = search(q)
            rs = d.get("results", [])
            strategy = d.get("search_strategy", "")
            ok = strategy == "web_fallback" and bool(rs)
            b2_ok += ok
            rows_b.append({"suite": "B2", "query": q, "strategy": strategy,
                           "results": len(rs), "ok": ok})
            print(f"  {q:<16} → {strategy:<18} | 结果 {len(rs)} | {'OK' if ok else 'FAIL'}")
        except Exception as e:
            rows_b.append({"suite": "B2", "query": q, "strategy": "ERROR", "ok": False})
            print(f"  {q:<16} → ERROR {str(e)[:50]}")
    print(f"\n[B2 汇总] {b2_ok}/{len(probes_b2)} 通过")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "city_coverage_result.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump({"suite_a": {"top1": top1, "top3": top3, "total": n, "failures": failures, "rows": rows_a},
                   "suite_b1": {"passed": b1_ok, "total": len(probes_b1)},
                   "suite_b2": {"passed": b2_ok, "total": len(probes_b2)},
                   "rows_b": rows_b},
                  f, ensure_ascii=False, indent=2)
    print(f"明细已保存: {out}")


if __name__ == "__main__":
    main()
