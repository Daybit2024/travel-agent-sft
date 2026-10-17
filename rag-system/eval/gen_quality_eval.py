# -*- coding: utf-8 -*-
"""生成质量评测（LLM-as-judge）：

对一组端到端查询跑 agent 全流程（攻略/酒店/问路/拒答等），收集最终答复，
再用 qwen-plus 当裁判按四个维度打分（各 1-5 分）：
  completeness 完整性 / no_leak 无泄漏 / attribution 来源标注 / relevance 相关性

输出：逐条得分 + 汇总 + gen_quality_result.json
用法（在 rag-system 目录）：python eval/gen_quality_eval.py
"""
import io
import json
import os
import sys
import time

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "agent"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from agent_graph.graph import build_graph  # noqa: E402

QUERIES = [
    ("9月25日去青岛玩三天，有什么推荐", "行程规划-语料内"),
    ("推荐北京的商务酒店，并看看评价", "酒店"),
    ("从北京南站到天安门怎么走", "问路"),
    ("9月25日去三亚玩三天怎么安排", "行程规划-语料外(网络)"),
    ("推荐一个适合秋天去的地方", "无地名"),
    ("帮我写一段python代码", "拒答"),
]

JUDGE_PROMPT = """你是严格的旅行助手回复评审。根据用户问题和助手回复打分（每项 1-5 分，3 分为及格线）：

- completeness 完整性：是否真正回答了问题。行程类要有逐日安排且覆盖景点；酒店类要有具体酒店和评价；问路要有路线方案；拒答类礼貌拒绝即满分。
- no_leak 无泄漏：回复中出现 <tool_call>、原始 JSON 工具调用文本等泄漏 → 1 分；否则 5 分。
- attribution 来源标注：涉及酒店/评价时是否标注来源（如"数据来源：高德"、"评价为模拟生成"）；网络检索内容是否标注"未核实"；不涉及则默认 5 分。
- relevance 相关性：内容贴合问题，无大量无关内容。

只输出 JSON，不要任何解释：
{"completeness": n, "no_leak": n, "attribution": n, "relevance": n, "comment": "一句话点评"}"""


def judge(q, answer):
    resp = requests.post(
        DASHSCOPE_BASE_URL + "/chat/completions",
        headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"},
        json={"model": "qwen-plus",
              "messages": [{"role": "user", "content": JUDGE_PROMPT
                            + f"\n\n用户问题：{q}\n助手回复：{answer[:3000]}"}],
              "temperature": 0, "max_tokens": 300},
        timeout=120,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return {"completeness": 0, "no_leak": 0, "attribution": 0, "relevance": 0,
                "comment": f"裁判解析失败: {text[:80]}"}


def main():
    rows = []
    for q, tag in QUERIES:
        t0 = time.time()
        try:
            graph = build_graph()  # 每条用全新会话
            result = graph.invoke({"messages": [HumanMessage(content=q)]})
            answer = result["messages"][-1].content
            error = ""
        except Exception as e:
            answer, error = "", str(e)[:100]
        secs = time.time() - t0
        score = judge(q, answer) if answer else {}
        row = {"query": q, "tag": tag, "secs": round(secs, 1), "error": error,
               "answer_head": answer[:120].replace("\n", " "), "score": score}
        rows.append(row)
        s = score
        print(f"[{tag}] {q}")
        print(f"  完整性 {s.get('completeness', '-')} | 无泄漏 {s.get('no_leak', '-')} | "
              f"来源 {s.get('attribution', '-')} | 相关 {s.get('relevance', '-')} | {secs:.0f}s")
        print(f"  点评: {s.get('comment', error)[:80]}")
        print(f"  回复头: {row['answer_head'][:80]}")
        print()

    dims = ["completeness", "no_leak", "attribution", "relevance"]
    print("===== 汇总 =====")
    for d in dims:
        vals = [r["score"].get(d, 0) for r in rows if r["score"]]
        avg = sum(vals) / len(vals) if vals else 0
        print(f"{d:<14}: 平均 {avg:.2f} / 5  （{vals}）")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_quality_result.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"明细已保存: {out}")


if __name__ == "__main__":
    main()
