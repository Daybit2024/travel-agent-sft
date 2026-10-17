# -*- coding: utf-8 -*-
"""合并全国生成的别称表与人工维护的映射表，写回 location_aliases.json。

规则：
  1. 人工维护的条目优先（逐城市覆盖）；
  2. 同一别称被多个城市声称时：人工的城市优先，其余丢弃；
  3. 仍存在歧义的别称（多人声称且无法裁决）一律丢弃——宁缺勿错。
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR

ALIASES_PATH = os.path.join(BASE_DIR, "location_aliases.json")
GEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "city_aliases_generated.json")


def main():
    with io.open(ALIASES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    with io.open(GEN_PATH, encoding="utf-8") as f:
        generated = json.load(f)
    manual = data.get("city_alias_map", {})

    manual_claims = {a: c for c, al in manual.items() for a in al}

    merged = {}
    for city, aliases in generated.items():
        c = city.replace("市", "")
        kept = [a for a in aliases if a and (manual_claims.get(a, c) == c)]
        if kept:
            merged[c] = kept
    for city, aliases in manual.items():
        merged[city.replace("市", "")] = [a for a in aliases if a]

    # 生成结果之间的重复：先到先得，后到丢弃
    seen = {}
    dropped = []
    for city, aliases in list(merged.items()):
        kept = []
        for a in aliases:
            owner = seen.get(a)
            if owner is None or owner == city:
                seen[a] = city
                kept.append(a)
            else:
                dropped.append((a, city, owner))
        if kept:
            merged[city] = kept
        else:
            merged.pop(city, None)

    total_alias = sum(len(v) for v in merged.values())
    print(f"合并后：{len(merged)} 城 / {total_alias} 个别称；丢弃歧义 {len(dropped)}: {dropped[:6]}")
    for probe in ("铜城", "鹰城", "帝都", "蓉城", "金陵"):
        owners = [c for c, al in merged.items() if probe in al]
        print(f"  抽查 {probe} → {owners}")

    data["city_alias_map"] = dict(sorted(merged.items()))
    with io.open(ALIASES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("已写回 location_aliases.json")


if __name__ == "__main__":
    main()
