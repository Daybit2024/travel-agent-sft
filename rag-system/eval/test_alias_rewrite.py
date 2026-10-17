# -*- coding: utf-8 -*-
"""实验：模型能否完成"地名归一化"（别名/景点 → 城市）？

对比三组：
  1. 本地 4B（qwen3-travel，零样本）
  2. 本地 4B（少样本，给 2 个示例）
  3. 云端 qwen-plus（零样本）

输出每条的原始回答 + 正确率 + 每次调用耗时。
"""
import io
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL  # noqa: E402  (会加载 .env)

OLLAMA_BASE = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen3-travel"

CASES = [
    ("帝都", "北京"), ("魔都", "上海"), ("冰城", "哈尔滨"), ("泉城", "济南"),
    ("姑苏", "苏州"), ("鹭岛", "厦门"), ("山城", "重庆"),
    ("泰山", "泰安"), ("乌镇", "嘉兴"), ("婺源篁岭", "上饶"),
    ("鼓浪屿", "厦门"), ("庐山", "九江"), ("想去帝都玩几天", "北京"),
    ("爬泰山住哪里", "泰安"),
]

PROMPT_ZS = "你是地名标准化助手。把用户说法中的地点转换为对应的地级市名称，只输出城市名。\n用户：{q}\n城市："

PROMPT_FS = (
    "你是地名标准化助手。把用户说法中的地点转换为对应的地级市名称，只输出城市名。\n"
    "示例：\n用户：泉城\n城市：济南市\n用户：篁岭\n城市：上饶市\n"
    "用户：{q}\n城市："
)


def call(base_url, model, api_key, prompt, timeout=600):
    t0 = time.time()
    r = requests.post(
        base_url + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0, "max_tokens": 32},
        timeout=timeout,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"].strip()
    return content, time.time() - t0


def hit(output, expected):
    return expected in output.replace("市", "") or (expected + "市") in output


def main():
    results = []
    for q, expected in CASES:
        row = {"query": q, "expected": expected}
        for tag, base, model, key, tmpl in (
            ("local_zs", OLLAMA_BASE, OLLAMA_MODEL, "ollama", PROMPT_ZS),
            ("local_fs", OLLAMA_BASE, OLLAMA_MODEL, "ollama", PROMPT_FS),
            ("cloud_zs", DASHSCOPE_BASE_URL, "qwen-plus", DASHSCOPE_API_KEY, PROMPT_ZS),
        ):
            try:
                out, secs = call(base, model, key, tmpl.format(q=q))
                row[tag] = {"out": out[:40], "hit": hit(out, expected), "secs": round(secs, 1)}
            except Exception as e:
                row[tag] = {"out": f"ERROR {str(e)[:60]}", "hit": False, "secs": None}
        results.append(row)
        print(f"[{q}] 本地零样本: {row['local_zs']['out']!r} | 本地少样本: {row['local_fs']['out']!r} | 云端: {row['cloud_zs']['out']!r}", flush=True)

    print("\n===== 汇总 =====")
    for tag in ("local_zs", "local_fs", "cloud_zs"):
        hits = sum(1 for r in results if r[tag]["hit"])
        secs = [r[tag]["secs"] for r in results if r[tag]["secs"] is not None]
        avg = sum(secs) / len(secs) if secs else 0
        print(f"{tag}: 正确 {hits}/{len(results)} | 平均耗时 {avg:.1f}s")

    with io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "alias_rewrite_result.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("明细已保存: alias_rewrite_result.json")


if __name__ == "__main__":
    main()
