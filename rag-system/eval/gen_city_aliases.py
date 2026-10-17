# -*- coding: utf-8 -*-
"""为语料内 361 城批量生成别称表（按省分批，云端模型生成 + 自动校验）。

流程：
  1) 按省分组，逐省让 qwen-plus 生成「城市 → 别称」（只列无歧义的，没有就留空）
  2) 程序校验：别称不得与任何城市名冲突、不得跨城市重复
  3) 自我核查：批量让模型回答"别称X指哪个城市"，与生成结果不一致的剔除
输出：city_aliases_generated.json（供人工过目后合并进 location_aliases.json）
"""
import io
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "city_aliases_generated.json")

PROVINCES = ["北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
             "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南",
             "湖北", "湖南", "广东", "广西", "海南", "重庆", "四川", "贵州",
             "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆"]

GEN_PROMPT = """请列出{province}下所有地级行政区（地级市、自治州、地区、盟），并为每个城市给出【广为人知且不会与其他城市混淆】的别称/雅称（例如"泉城"是济南）。

要求：
1. 只列确实常用、不会被当成其他城市的别称；没有公认别称的城市给空数组；
2. 严禁编造，不确定就不要写；
3. 城市名不带"市"字；
4. 只输出 JSON 对象，格式：{{"城市名": ["别称1"], ...}}，不要输出任何其他文字。"""

CHECK_PROMPT = """下面是一些中国城市别称，请回答每个别称通常指哪个城市（不确定就写"不确定"）。
只输出 JSON 对象，格式：{{"别称": "城市名", ...}}
待查询：
{items}"""


def call_llm(prompt, max_tokens=2000):
    resp = requests.post(
        DASHSCOPE_BASE_URL + "/chat/completions",
        headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"},
        json={"model": "qwen-plus", "messages": [{"role": "user", "content": prompt}],
              "temperature": 0, "max_tokens": max_tokens},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def parse_json_block(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return {}


def main():
    with io.open(os.path.join(BASE_DIR, "city_code_mapping.json"), encoding="utf-8") as f:
        mapping = json.load(f)["城市编码映射"]
    corpus_names = {info.get("name", "").replace("市", "") for info in mapping.values() if info.get("name")}
    all_city_names = set(corpus_names)

    print(f"语料城市 {len(corpus_names)} 城；全国生成 {len(PROVINCES)} 个省区")

    generated = {}
    for prov in PROVINCES:
        try:
            raw = call_llm(GEN_PROMPT.format(province=prov), max_tokens=3000)
            data = parse_json_block(raw)
            n = 0
            for city, aliases in data.items():
                city = (city or "").strip().replace("市", "")
                if not city or not isinstance(aliases, list):
                    continue
                all_city_names.add(city)
                cleaned = []
                for a in aliases:
                    a = (a or "").strip()
                    if not a or len(a) > 4 or "之" in a:
                        continue
                    if a == city or a in city or city in a:   # 名称回声
                        continue
                    cleaned.append(a)
                if cleaned:
                    generated[city] = cleaned
                    n += 1
            print(f"[生成] {prov}: {len(data)} 城 → {n} 城有别称", flush=True)
        except Exception as e:
            print(f"[生成] {prov} 失败: {str(e)[:80]}")

    # 程序校验：别称不得与任何城市名冲突 / 不得跨城市重复
    alias_owner = {}
    conflicts = []
    for city, aliases in list(generated.items()):
        kept = []
        for a in aliases:
            if a in all_city_names and a != city:
                conflicts.append((a, city, "与城市名冲突"))
                continue
            if a in alias_owner and alias_owner[a] != city:
                conflicts.append((a, city, f"跨城市重复（已属 {alias_owner[a]}）"))
                continue
            alias_owner[a] = city
            kept.append(a)
        if kept:
            generated[city] = kept
        else:
            generated.pop(city, None)
    print(f"\n[校验] 剔除冲突 {len(conflicts)} 条: {conflicts[:8]}")

    # 自我核查：批量问"别称指哪个城市"
    pairs = [(a, c) for c, al in generated.items() for a in al]
    print(f"[核查] 待核查别称 {len(pairs)} 条")
    verified = {}
    BATCH = 25
    for i in range(0, len(pairs), BATCH):
        batch = pairs[i:i + BATCH]
        items = "\n".join(f"- {a}" for a, _ in batch)
        try:
            raw = call_llm(CHECK_PROMPT.format(items=items), max_tokens=1500)
            answer = parse_json_block(raw)
        except Exception as e:
            print(f"  核查批次 {i} 失败: {str(e)[:60]}")
            answer = {}
        for alias, city in batch:
            said = (answer.get(alias) or "").replace("市", "").strip()
            if said and said != "不确定" and said == city.replace("市", ""):
                verified.setdefault(city, []).append(alias)
        time.sleep(0.3)

    print(f"[核查] 通过 {sum(len(v) for v in verified.values())} 条")
    with io.open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(verified, f, ensure_ascii=False, indent=2)
    print(f"已输出: {OUT_PATH}（{len(verified)} 城）")


if __name__ == "__main__":
    main()
