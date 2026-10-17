#!/usr/bin/env python3
# coding: utf-8
"""评测：基座+LoRA adapter 在测试集上的两种能力。

mode=final    : 取最后一条 assistant 之前的对话，生成最终答复（文本）
mode=toolcall : 取每条第一个人工工具调用之前的对话，生成工具调用，对比参考

指标分层：<tool_call> JSON 合法性 / 工具名 / 参数键 / 参数值。
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

_DATASETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")


def load_tools(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(tokenizer, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, tools=tools, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tools=tools, **kwargs)


def _norm_args(args: Any) -> Dict[str, Any]:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    return args if isinstance(args, dict) else {}


def parse_toolcall(text: str) -> Optional[Dict[str, Any]]:
    m = TOOLCALL_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return None
    if not isinstance(obj.get("name"), str):
        return None
    return {"name": obj["name"], "arguments": _norm_args(obj.get("arguments"))}


def reference_call(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tcs = msg.get("tool_calls")
    if isinstance(tcs, list) and tcs:
        fn = tcs[0].get("function", {})
        return {"name": fn.get("name"), "arguments": _norm_args(fn.get("arguments"))}
    return None


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_toolcall_mode(model, tokenizer, data, tools, limit, max_new_tokens, show):
    cases = []
    for i, d in enumerate(data):
        msgs = d.get("conversation") or d.get("messages")
        for j, m in enumerate(msgs):
            if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list) and m["tool_calls"]:
                cases.append((i, j, msgs))
                break
        if limit and len(cases) >= limit:
            break

    stats = dict(total=0, parsed=0, name=0, keys=0, vals=0)
    shown = 0
    for k, (i, j, msgs) in enumerate(cases):
        ref = reference_call(msgs[j])
        prompt = build_prompt(tokenizer, msgs[:j], tools)
        gen = generate(model, tokenizer, prompt, max_new_tokens)
        parsed = parse_toolcall(gen)
        stats["total"] += 1
        if parsed:
            stats["parsed"] += 1
            if parsed["name"] == ref["name"]:
                stats["name"] += 1
                rk, pk = set(ref["arguments"]), set(parsed["arguments"])
                if rk == pk:
                    stats["keys"] += 1
                    if all(str(ref["arguments"][x]) == str(parsed["arguments"][x]) for x in rk):
                        stats["vals"] += 1
        if shown < show:
            shown += 1
            print(f"[case {k} sample {i} step {j}] ref={json.dumps(ref, ensure_ascii=False)[:200]}")
            print(f"[case {k}] gen={gen.strip().replace(chr(10), ' ')[:240]}")
        if (k + 1) % 20 == 0:
            print(f"progress {k+1}/{len(cases)}", flush=True)
    return stats, len(cases)


def run_final_mode(model, tokenizer, data, tools, limit, max_new_tokens, show):
    stats = dict(total=0, spurious=0)
    shown = 0
    for i, d in enumerate(data[: limit or len(data)]):
        msgs = d.get("conversation") or d.get("messages")
        prompt = build_prompt(tokenizer, msgs[:-1], tools)
        gen = generate(model, tokenizer, prompt, max_new_tokens)
        stats["total"] += 1
        if parse_toolcall(gen) is None:
            stats["spurious"] += 1
        if shown < show:
            shown += 1
            print(f"[final {i}] gen={gen.strip().replace(chr(10), ' ')[:240]}")
    return stats, limit or len(data)


def pct(a, b):
    return f"{a}/{b} ({100.0*a/b:.1f}%)" if b else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_path", default="", help="LoRA adapter 目录；留空或 none 表示只评测基座")
    ap.add_argument("--test_file", default=os.path.join(_DATASETS, "merged_test_final.json"))
    ap.add_argument("--tools_file", default=os.path.join(_DATASETS, "all_tools.json"))
    ap.add_argument("--mode", choices=["final", "toolcall", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--final_limit", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--show", type=int, default=2)
    args = ap.parse_args()

    tools = load_tools(args.tools_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, local_files_only=True, dtype=dtype
    ).to(device)
    if args.adapter_path and args.adapter_path.lower() not in ("none", ""):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    print(f"device={device} dtype={dtype}", flush=True)

    with open(args.test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"model={args.base_model} adapter={args.adapter_path}", flush=True)
    if args.mode in ("toolcall", "both"):
        print("----- MODE: toolcall -----", flush=True)
        st, tot = run_toolcall_mode(model, tokenizer, data, tools, args.limit, args.max_new_tokens, args.show)
        print("\n=== TOOLCALL RESULT ===")
        print("cases                            :", tot)
        print("generated valid <tool_call> JSON :", pct(st["parsed"], st["total"]))
        print("tool name match                  :", pct(st["name"], st["total"]))
        print("argument keys match              :", pct(st["keys"], st["total"]))
        print("argument values exact match      :", pct(st["vals"], st["total"]), flush=True)
    if args.mode in ("final", "both"):
        print("\n----- MODE: final -----", flush=True)
        st2, tot2 = run_final_mode(model, tokenizer, data, tools, args.final_limit, args.max_new_tokens, args.show)
        print("\n=== FINAL RESULT ===")
        print("cases                            :", tot2)
        print("no spurious tool_call in answer  :", pct(st2["spurious"], st2["total"]))


if __name__ == "__main__":
    main()
