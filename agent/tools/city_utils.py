# -*- coding: utf-8 -*-
"""城市工具：城市名 → 高德 adcode 的统一解析（供酒店/天气/agent 共用）"""
import json
import os
from typing import Dict, Optional, Tuple

from tools.get_route import PROJECT_ROOT

_CITY_MAP_PATH = os.path.join(PROJECT_ROOT, "rag-system", "city_code_mapping.json")
_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"


def load_city_map() -> Dict[str, str]:
    """城市名（含去掉'市'后缀的短名）→ 高德 adcode

    映射表结构：城市编码映射 {adcode: {name, level, province}}（361 城）
    + 反向映射 {城市名: adcode}（部分城市）
    """
    try:
        with open(_CITY_MAP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        out: Dict[str, str] = {}
        for adcode, info in (data.get("城市编码映射") or {}).items():
            if isinstance(info, dict) and info.get("name"):
                out[info["name"]] = adcode
        for name, adcode in (data.get("反向映射") or {}).items():
            out.setdefault(name, adcode)
        for name in list(out.keys()):
            if len(name) > 2 and name.endswith("市"):
                out.setdefault(name[:-1], out[name])
        return out
    except Exception:
        return {}


def city_to_adcode(text: str) -> Optional[str]:
    """城市名/县名/景区名 → 高德 adcode；解析不了返回 None"""
    t = (text or "").strip()
    if not t:
        return None
    if t.isdigit() and len(t) == 6:
        return t
    cmap = load_city_map()
    if t in cmap:
        return cmap[t]
    # 兜底：高德地理编码（支持县/景区名）
    try:
        import requests
        from tools.get_route import AMAP_KEY
        resp = requests.get(_GEOCODE_URL, params={"address": t, "key": AMAP_KEY}, timeout=10).json()
        geos = resp.get("geocodes") or []
        if geos and geos[0].get("adcode"):
            return geos[0]["adcode"]
    except Exception:
        pass
    return None


def resolve_city(text: str) -> Tuple[str, str]:
    """城市输入 → (城市名, 高德adcode)

    **解析失败返回 ("", "")，不返回默认城市**：调用方（agent 的 system prompt）
    对空值已有成熟处理——"未提供 → 需要时反问用户所在城市"。而静默返回一个默认城市
    等于**伪造一条"事实"**喂给模型：prompt 里写着"以用户信息为准"，
    模型会当真并按错城市查询，且没有任何报错。宁可说"不知道"，也不要猜。
    """
    t = (text or "").strip()
    if not t:
        return "", ""
    adcode = city_to_adcode(t)
    if adcode:
        # 用映射表里的标准名（如 郑州 -> 郑州市）
        name = t
        for n, code in load_city_map().items():
            if code == adcode and len(n) <= 3 and n.endswith("市"):
                name = n
                break
        return name, adcode
    print(f"[地名] 城市无法解析：{t!r}（字典未命中，且高德地理编码也失败）→ 交给反问")
    return "", ""
