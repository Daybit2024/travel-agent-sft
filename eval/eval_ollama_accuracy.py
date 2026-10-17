# -*- coding: utf-8 -*-
"""评测部署形态模型（Ollama / GGUF）的工具调用能力。

指标与 eval_toolcall_accuracy.py 分层一致：
  mode=toolcall：工具调用产出率 / 工具名匹配 / 参数键匹配 / 参数值匹配
  mode=final   ：最终答复中无意外工具调用（无泄漏）的比例

用法：
  python eval_ollama_accuracy.py --model qwen3-travel --limit 50
"""
import argparse
import json
import os
import re
import time

from openai import OpenAI

TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

_DATASETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")


def _norm_args(args):
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    return args if isinstance(args, dict) else {}


def parse_text_toolcall(text):
    m = TOOLCALL_RE.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return None
    if not isinstance(obj.get("name"), str):
        return None
    return {"name": obj["name"], "arguments": _norm_args(obj.get("arguments"))}


def extract_call(message):
    """优先取结构化 tool_calls；退化到解析文本里的 <tool_call>"""
    tcs = getattr(message, "tool_calls", None)
    if tcs:
        fn = tcs[0].function
        return {"name": fn.name, "arguments": _norm_args(fn.arguments)}
    return parse_text_toolcall(message.content)


def clean_messages(msgs):
    out = []
    for m in msgs:
        mm = {"role": m.get("role"), "content": m.get("content") or ""}
        if m.get("tool_calls"):
            mm["tool_calls"] = m["tool_calls"]
        if m.get("tool_call_id"):
            mm["tool_call_id"] = m["tool_call_id"]
        out.append(mm)
    return out


def reference_call(msg):
    tcs = msg.get("tool_calls")
    if isinstance(tcs, list) and tcs:
        fn = tcs[0].get("function", {})
        return {"name": fn.get("name"), "arguments": _norm_args(fn.get("arguments"))}
    return None


def pct(a, b):
    return f"{a}/{b} ({100.0*a/b:.1f}%)" if b else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-travel")
    ap.add_argument("--base_url", default="http://localhost:11434/v1")
    ap.add_argument("--test_file", default=os.path.join(_DATASETS, "merged_test_final.json"))
    ap.add_argument("--tools_file", default=os.path.join(_DATASETS, "all_tools.json"))
    ap.add_argument("--mode", choices=["final", "toolcall", "both"], default="both")
    ap.add_argument("--limit", type=int, default=50, help="0 = 全部")
    ap.add_argument("--final_limit", type=int, default=20)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    client = OpenAI(api_key="ollama", base_url=args.base_url, timeout=600)
    tools = json.load(open(args.tools_file, encoding="utf-8"))
    data = json.load(open(args.test_file, encoding="utf-8"))

    print(f"model={args.model} base_url={args.base_url}", flush=True)

    if args.mode in ("toolcall", "both"):
        cases = []
        for i, d in enumerate(data):
            msgs = d.get("conversation") or d.get("messages")
            for j, m in enumerate(msgs):
                if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list) and m["tool_calls"]:
                    cases.append((i, j, msgs))
                    break
            if args.limit and len(cases) >= args.limit:
                break

        stats = dict(total=0, parsed=0, name=0, keys=0, vals=0)
        shown = 0
        t0 = time.time()
        for k, (i, j, msgs) in enumerate(cases):
            ref = reference_call(msgs[j])
            resp = client.chat.completions.create(
                model=args.model,
                messages=clean_messages(msgs[:j]),
                tools=tools,
                tool_choice="auto",
                temperature=0,
                max_tokens=args.max_tokens,
            )
            got = extract_call(resp.choices[0].message)
            stats["total"] += 1
            if got:
                stats["parsed"] += 1
                if got["name"] == ref["name"]:
                    stats["name"] += 1
                    rk, pk = set(ref["arguments"]), set(got["arguments"])
                    if rk == pk:
                        stats["keys"] += 1
                        if all(str(ref["arguments"][x]) == str(got["arguments"][x]) for x in rk):
                            stats["vals"] += 1
            if shown < args.show:
                shown += 1
                print(f"[case {k}] ref={json.dumps(ref, ensure_ascii=False)[:180]}", flush=True)
                print(f"[case {k}] got={json.dumps(got, ensure_ascii=False)[:180]}", flush=True)
            if (k + 1) % 5 == 0:
                print(f"progress {k+1}/{len(cases)} ({time.time()-t0:.0f}s)", flush=True)

        print("\n=== TOOLCALL RESULT (ollama) ===")
        print("cases                            :", stats["total"])
        print("generated tool_call              :", pct(stats["parsed"], stats["total"]))
        print("tool name match                  :", pct(stats["name"], stats["total"]))
        print("argument keys match              :", pct(stats["keys"], stats["total"]))
        print("argument values exact match      :", pct(stats["vals"], stats["total"]))

    if args.mode in ("final", "both"):
        n = args.final_limit or len(data)
        total = leaked = 0
        shown = 0
        for i, d in enumerate(data[:n]):
            msgs = d.get("conversation") or d.get("messages")
            resp = client.chat.completions.create(
                model=args.model,
                messages=clean_messages(msgs[:-1]),
                tools=tools,
                tool_choice="auto",
                temperature=0,
                max_tokens=args.max_tokens,
            )
            msg = resp.choices[0].message
            total += 1
            if extract_call(msg):
                leaked += 1
            if shown < args.show:
                shown += 1
                print(f"[final {i}] {(msg.content or '')[:150]}", flush=True)

        print("\n=== FINAL RESULT (ollama) ===")
        print("cases                            :", total)
        print("no spurious tool_call in answer  :", f"{total-leaked}/{total} ({100.0*(total-leaked)/total:.1f}%)")


if __name__ == "__main__":
    main()
