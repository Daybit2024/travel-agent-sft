# -*- coding: utf-8 -*-
"""请求流水日志 → 指标报告

读 logs/<name>/*.jsonl（由 agent/tools/request_log.py 落盘），输出：
  - 请求数、每次请求的 LLM 调用数与耗时构成
  - LLM：成功率 / 延迟 P50·P95 / tokens / 云端本地占比 / 工具调用名分布
  - 高德：按接口分组的调用数与延迟 / 成功率 / 业务失败（status != 1，如配额或 key 问题）/ 重试率
  - 护栏：触发次数与触发率
  - 失败明细（error 非空的事件）

用法：
    python scripts/request_log_report.py                  # 读 logs/agent 下全部
    python scripts/request_log_report.py --date 2026-09-28
    python scripts/request_log_report.py --dir logs/rag --name rag
"""
import argparse
import glob
import io
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_rows(paths):
    rows = []
    for p in paths:
        try:
            with io.open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except ValueError:
                            pass          # 单行损坏不影响整体
        except OSError:
            pass
    return rows


def pct(values, q):
    """最近秩百分位（样本少时也稳定）"""
    if not values:
        return 0
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def amap_api(url):
    """从高德 URL 取接口路径，便于分组（如 /v3/geocode/geo）"""
    try:
        return "/" + url.split("restapi.amap.com/", 1)[1].split("?", 1)[0]
    except IndexError:
        return url.split("?", 1)[0]


def section(title):
    print(f"\n{'=' * 6} {title} {'=' * 6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "logs", "agent"), help="日志目录")
    ap.add_argument("--date", default="", help="只统计某天（YYYY-MM-DD），默认全部")
    ap.add_argument("--json", action="store_true", help="额外输出紧凑 JSON（便于二次消费）")
    args = ap.parse_args()

    pattern = os.path.join(args.dir, (args.date or "*") + ".jsonl")
    rows = load_rows(glob.glob(pattern))
    if not rows:
        print(f"没有读到日志：{pattern}")
        print("提示：跑一次对话后重试（示例 `python -m agent_graph.smoke_agent \"从北京南站到颐和园\"`）")
        return 0

    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r.get("kind")].append(r)

    turns = by_kind.get("turn", [])
    llms = by_kind.get("llm", [])
    amaps = by_kind.get("amap", [])
    guards = by_kind.get("guard", [])

    # 按 trace 聚合成"一次请求"
    traces = defaultdict(list)
    for r in rows:
        if r.get("trace"):
            traces[r["trace"]].append(r)

    section("请求 / 会话")
    print(f"请求数（turn 事件）: {len(turns)}   有 trace 的请求: {len(traces)}")
    if traces:
        per_req_llm = [sum(1 for x in v if x.get("kind") == "llm") for v in traces.values()]
        per_req_ms = [sum((x.get("ms") or 0) for x in v if x.get("kind") in ("llm", "amap"))
                      for v in traces.values()]
        print(f"每次请求的 LLM 调用数: 平均 {sum(per_req_llm) / len(per_req_llm):.1f} "
              f"最大 {max(per_req_llm)}")
        print(f"每次请求的 LLM+高德耗时合计: 平均 {sum(per_req_ms) / len(per_req_ms):.0f}ms "
              f"P95 {pct(per_req_ms, 0.95)}ms（不含框架开销）")

    section("LLM")
    if llms:
        ok = [x for x in llms if x.get("ok")]
        ms = [x.get("ms") or 0 for x in ok]
        print(f"调用 {len(llms)} 次 | 成功 {len(ok)}（{100 * len(ok) / len(llms):.0f}%）"
              f" | 延迟 P50 {pct(ms, 0.5)}ms / P95 {pct(ms, 0.95)}ms")
        tin = sum((x.get("tokens_in") or 0) for x in ok)
        tout = sum((x.get("tokens_out") or 0) for x in ok)
        print(f"tokens：入 {tin} / 出 {tout}")
        print(f"后端分布: {dict(Counter(x.get('backend') for x in ok))}")
        print(f"产生工具调用的 LLM 调用占比: "
              f"{100 * sum(1 for x in ok if x.get('call_names')) / len(ok):.0f}%"
              f"（**结构性指标**：一次工具回合 ≈ 调工具轮 + 终答轮，故单工具场景约 50%——"
              f"这是正常值；接近 100% 反而说明模型收不了口。"
              f"能力指标见 eval/：需要工具的样本里成功调用的比例为 92.9%）")
        names = Counter(n for x in ok for n in (x.get("call_names") or []))
        if names:
            print(f"工具调用名分布: {dict(names)}")
    else:
        print("（无）")

    section("高德（按接口分组）")
    if amaps:
        ok = [x for x in amaps if x.get("ok")]
        print(f"调用 {len(amaps)} 次 | 网络层成功 {len(ok)}（{100 * len(ok) / len(amaps):.0f}%）"
              f" | 重试率 {100 * sum(1 for x in amaps if (x.get('attempts') or 1) > 1) / len(amaps):.0f}%")
        bad = [x for x in amaps if x.get("status") not in (None, "1")]
        if bad:
            print(f"⚠️ 高德业务失败 {len(bad)} 次（status != 1）—— 常见于配额用尽或 key 问题：")
            for x in bad[:5]:
                print(f"   {amap_api(x.get('url', ''))}  status={x.get('status')} info={x.get('info')}")
        print()
        print(f"  {'接口':<34}{'调用':>6}{'P50(ms)':>9}{'P95(ms)':>9}")
        grouped = defaultdict(list)
        for x in amaps:
            grouped[amap_api(x.get("url", ""))].append(x.get("ms") or 0)
        for api, vals in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
            print(f"  {api:<34}{len(vals):>6}{pct(vals, 0.5):>9}{pct(vals, 0.95):>9}")
    else:
        print("（无——本段没有任何高德调用）")

    rags = by_kind.get("rag", [])
    section("RAG 检索")
    if rags:
        ok = [x for x in rags if x.get("ok")]
        ms = [x.get("ms") or 0 for x in ok]
        print(f"检索 {len(rags)} 次 | 成功 {len(ok)}（{100 * len(ok) / len(rags):.0f}%）"
              f" | 延迟 P50 {pct(ms, 0.5)}ms / P95 {pct(ms, 0.95)}ms")
        print(f"策略分布: {dict(Counter(x.get('strategy') for x in ok))}")
        web = sum(1 for x in ok if x.get("web"))
        print(f"联网兜底: {web} 次（{100 * web / max(len(ok), 1):.0f}%）"
              f" | 平均返回 {sum((x.get('n') or 0) for x in ok) / len(ok):.1f} 块")
        print("  最近返回的块（chunk_id / 标题 / 命中名次）:")
        for x in ok[-4:]:
            for t in (x.get("top") or [])[:2]:
                print(f"    {t.get('chunk_id')}  {str(t.get('title'))[:16]:<18}"
                      f" rrf={t.get('rrf')} vr={t.get('vr')} br={t.get('br')}")
    else:
        print("（无——本段没有检索调用；读 RAG 日志请加 --dir logs/rag）")

    section("护栏")
    if guards:
        last = guards[-1]                    # hits/turns 是累计值，取最后一条
        h, t = last.get("hits"), last.get("turns")
        rate = f"{100 * h / t:.0f}%" if h is not None and t else "-"
        print(f"触发 {h} 次 / 受检 {t} 次 → 触发率 {rate}")
        for g in guards[:3]:
            print(f"   不一致片段: {g.get('mismatches')}")
    else:
        print("（未触发——终答与工具结果一致）")

    errs = [x for x in rows if x.get("error")]
    section("失败明细")
    if errs:
        for x in errs[:10]:
            print(f"  [{x.get('kind')}] {x.get('error')}")
    else:
        print("（无）")

    if args.json:
        print()
        print(json.dumps({
            "requests": len(turns), "traces": len(traces),
            "llm": {"n": len(llms), "ok": sum(1 for x in llms if x.get("ok")),
                    "p50_ms": pct([x.get("ms") or 0 for x in llms if x.get("ok")], 0.5),
                    "p95_ms": pct([x.get("ms") or 0 for x in llms if x.get("ok")], 0.95)},
            "amap": {"n": len(amaps), "ok": sum(1 for x in amaps if x.get("ok")),
                     "p50_ms": pct([x.get("ms") or 0 for x in amaps], 0.5),
                     "p95_ms": pct([x.get("ms") or 0 for x in amaps], 0.95),
                     "by_api": {k: len(v) for k, v in grouped.items()} if amaps else {}},
            "guard": {"hits": guards[-1].get("hits") if guards else 0,
                      "turns": guards[-1].get("turns") if guards else 0},
            "errors": len(errs),
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
