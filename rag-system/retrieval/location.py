# -*- coding: utf-8 -*-
"""地名识别：query → 城市主键（adcode）/ 语料外地点。

映射来源（location_aliases.json）：
  - city_alias_map: 城市 → 别名（含语料内与语料外城市）
  - attractions:    景点 → 城市
  - regions:        区域合称 → 省前缀
  - uncovered_places: 语料外地点名单
另提供 canonicalize()：把别名替换为标准城市名（检索/联网前标准化用）。
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR

_ALIASES_PATH = os.path.join(BASE_DIR, "location_aliases.json")
_MAPPING_PATH = os.path.join(BASE_DIR, "city_code_mapping.json")

_city_names = {}        # 城市名（含去"市"短名）→ adcode
_all_codes = set()      # 全部城市码
_provinces = {}         # 省名（含去后缀短名）→ 2 位前缀
_attractions, _regions = {}, {}
_uncovered = []         # 语料外地点名单
_alias_to_city = {}     # 别名 → 标准城市名
_map_city_names = set() # 映射表中的城市键


def _strip_suffix(name):
    for suf in ("省", "市", "自治区", "壮族", "回族", "维吾尔"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    return name


def _load():
    global _city_names, _all_codes, _provinces, _attractions, _regions, _uncovered
    global _alias_to_city, _map_city_names
    with io.open(_MAPPING_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for adcode, info in (data.get("城市编码映射") or {}).items():
        if not (isinstance(info, dict) and info.get("name")):
            continue
        code = str(adcode)
        name = info["name"]
        _city_names[name] = code
        if len(name) > 2 and name.endswith("市"):
            _city_names.setdefault(name[:-1], code)
        _all_codes.add(code)
        prov = info.get("province") or ""
        if prov:
            _provinces.setdefault(prov, code[:2])
            _provinces.setdefault(_strip_suffix(prov), code[:2])
    with io.open(_ALIASES_PATH, encoding="utf-8") as f:
        d = json.load(f)
    _attractions = d.get("attractions", {})
    _regions = d.get("regions", {})
    _uncovered = d.get("uncovered_places", [])
    for city, aliases in (d.get("city_alias_map") or {}).items():
        _map_city_names.add(city)
        for a in aliases:
            _alias_to_city[a] = city


_load()
_city_names_sorted = sorted(_city_names, key=len, reverse=True)
_provinces_sorted = sorted(_provinces, key=len, reverse=True)
_canon_pairs = [(a, _alias_to_city[a]) for a in sorted(_alias_to_city, key=len, reverse=True)]
_map_cities_sorted = sorted(_map_city_names, key=len, reverse=True)


def resolve_city_name(name: str):
    """规范城市名 → adcode（用于校验模型输出/映射表）；未收录返回 None"""
    n = (name or "").strip()
    if not n:
        return None
    if n in _city_names:
        return _city_names[n]
    if (n + "市") in _city_names:
        return _city_names[n + "市"]
    return None


def known_city_names():
    """语料内城市名列表（含去"市"短名；供地名解析时优先匹配）"""
    return list(_city_names.keys())


def _find_all(text, sub):
    out = []
    i = text.find(sub)
    while i >= 0:
        out.append((i, i + len(sub)))
        i = text.find(sub, i + 1)
    return out


def _protected_spans(query):
    """城市名/映射表城市/景点占用的区间（别名匹配时避让，防子串误伤，
    如"共青城市"里的"青城"、城市"临安"与杭州古称"临安"同名）"""
    spans = []
    for name in _city_names_sorted:
        spans.extend(_find_all(query, name))
    for city in _map_cities_sorted:
        spans.extend(_find_all(query, city))
    for kw in _attractions:
        spans.extend(_find_all(query, kw))
    return spans


def _is_covered(span, spans):
    s, e = span
    return any(a <= s and e <= b for a, b in spans)


def canonicalize(query: str) -> str:
    """把 query 中的别名替换为标准城市名（长词优先；避让城市名/景点占用的区间）"""
    q = query or ""
    spans = _protected_spans(q)
    for alias, city in _canon_pairs:
        occ = _find_all(q, alias)
        if occ and any(not _is_covered(o, spans) for o in occ):
            q = q.replace(alias, city)
    return q


def extract_candidates(query: str):
    """返回：city_codes（语料内城市）/ prefixes（省、区域）/ terms（命中的说法）/ uncovered（语料外地点）"""
    query = query or ""
    city_codes = set()
    prefixes = set()
    terms = []
    unresolved = []

    for kw, ps in _regions.items():
        if kw in query:
            prefixes.update(ps)
            terms.append(kw)
    for kw, code in _attractions.items():
        if kw in query:
            city_codes.add(code)
            terms.append(kw)

    def _add_city(city, hit_term):
        terms.append(hit_term)
        code = resolve_city_name(city)
        if code:
            city_codes.add(code)
        elif city not in unresolved:
            unresolved.append(city)

    protected = _protected_spans(query)
    for alias, city in _canon_pairs:          # 别名（长词优先；避让已被城市名/景点占用的区间）
        occ = _find_all(query, alias)
        if occ and any(not _is_covered(o, protected) for o in occ):
            _add_city(city, alias)
    for city in _map_cities_sorted:           # 映射表中的城市名
        if city in query:
            _add_city(city, city)
    for name in _city_names_sorted:           # 语料城市名
        if name in query:
            city_codes.add(_city_names[name])
            terms.append(name)
    for name in _provinces_sorted:            # 省名
        if len(name) >= 2 and name in query:
            prefixes.add(_provinces[name])
            terms.append(name)

    if prefixes:
        for code in _all_codes:
            if code[:2] in prefixes:
                city_codes.add(code)

    uncovered = [kw for kw in _uncovered if kw in query]
    for n in unresolved:
        if n not in uncovered:
            uncovered.append(n)

    return {"city_codes": sorted(city_codes), "prefixes": sorted(prefixes),
            "terms": sorted(set(terms)), "uncovered": uncovered}
