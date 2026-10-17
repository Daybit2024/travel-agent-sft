# -*- coding: utf-8 -*-
"""酒店工具

- 推荐：高德 POI 搜索（真实酒店数据：名称/地址/评分）
- 评价：LLM 生成评价池（严格 2:1 正负比，6 条），按酒店名缓存到
  hotel_review_cache.json；查询时从池中随机抽 3 条
"""
import sys
import os
import re
import json
import random
import requests
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_client import call_llm
from tools.get_route import AMAP_KEY, PROJECT_ROOT

_CITY_MAP_PATH = os.path.join(PROJECT_ROOT, "rag-system", "city_code_mapping.json")
_REVIEW_CACHE_PATH = os.path.join(PROJECT_ROOT, "datasets", "hotel_review_cache.json")

AMAP_PLACE_URL = "https://restapi.amap.com/v3/place/text"
AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"

_REVIEW_SYSTEM_PROMPT = """你是酒店评论生成专家。为给定酒店生成 6 条真实自然的用户评价，严格按 2:1 的正负比例：
- 4 条正面评价（rating 填 4 或 5）
- 2 条负面评价（rating 填 2 或 3）
每条 3-5 句话、口语化，结合酒店的位置和类型信息（如有提供），正面和负面都要给出具体细节。
评论日期取 2025 年内的随机日期。
只返回 JSON 数组，格式如下，不要输出任何其他文字：
[
  {"reviewer_name": "评论者昵称", "rating": 4, "review_date": "2025-08-12", "review_content": "评价正文"}
]"""


def _load_city_map() -> Dict[str, str]:
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


def _extract_city(requirements: str) -> Optional[str]:
    """从需求文本里提取城市名（最长匹配优先）"""
    if not requirements:
        return None
    cmap = _load_city_map()
    best = None
    for name in cmap:
        if name in requirements and (best is None or len(name) > len(best)):
            best = name
    return best


def _extract_location_coords(requirements: str) -> Optional[str]:
    """兜底：取需求第一个地名片段做高德地理编码（支持县/景区名，如"婺源""婺源篁岭"）"""
    req = (requirements or "").strip()
    if not req:
        return None
    seg = re.split(r"[，,。；;、\s]+", req)[0].strip()
    if not (1 < len(seg) <= 12):
        return None
    try:
        resp = requests.get(AMAP_GEOCODE_URL,
                            params={"address": seg, "key": AMAP_KEY},
                            timeout=10).json()
        geos = resp.get("geocodes") or []
        if geos and geos[0].get("location"):
            return geos[0]["location"]  # "lng,lat"
    except Exception:
        pass
    return None


def _extract_keywords(requirements: str) -> str:
    """从需求文本提取搜索关键词（类型/星级偏好）"""
    req = requirements or ""
    if "商务" in req:
        return "商务酒店"
    if "五星" in req or "豪华" in req or "高端" in req:
        return "五星级酒店"
    if "经济" in req or "便宜" in req or "实惠" in req:
        return "经济型酒店"
    if "民宿" in req:
        return "民宿"
    if "青年旅舍" in req or "青旅" in req:
        return "青年旅舍"
    return "酒店"


def get_hotel_recommendations(user_requirements: str, model_name: str = "qwen-plus") -> List[Dict[str, Any]]:
    """高德 POI 搜索真实酒店。

    Args:
        user_requirements: 用户需求描述，如"北京商务酒店，近地铁站"
        model_name: 保留参数（兼容原接口，不再使用）

    Returns:
        [{hotel_name, location, rating, type}, ...]；失败或无结果返回 []
    """
    try:
        city = _extract_city(user_requirements or "")
        params: Dict[str, Any] = {
            "keywords": _extract_keywords(user_requirements),
            "offset": 5,
            "page": 1,
            "extensions": "all",
            "key": AMAP_KEY,
        }
        if city:
            params["city"] = _load_city_map().get(city) or city
            params["citylimit"] = "true"
        else:
            # 城市映射表未命中（县/景区等）→ 地名转坐标，按周边 30km 搜索
            loc = _extract_location_coords(user_requirements or "")
            if loc:
                params["location"] = loc
                params["radius"] = 30000
        resp = requests.get(AMAP_PLACE_URL, params=params, timeout=15).json()
        if resp.get("status") != "1":
            print(f"高德酒店搜索失败: {resp.get('info')}")
            return []
        hotels: List[Dict[str, Any]] = []
        for poi in resp.get("pois", []) or []:
            if not isinstance(poi, dict) or not poi.get("name"):
                continue
            addr = poi.get("address")
            if not isinstance(addr, str):
                addr = ""
            biz = poi.get("biz_ext") if isinstance(poi.get("biz_ext"), dict) else {}
            hotels.append({
                "hotel_name": poi.get("name"),
                "location": addr or f"{poi.get('pname', '')}{poi.get('cityname', '')}{poi.get('adname', '')}",
                "rating": biz.get("rating") or "暂无",
                "type": poi.get("type", ""),
            })
        return hotels
    except Exception as e:
        print(f"高德酒店搜索出错: {e}")
        return []


def _load_review_cache() -> Dict[str, List[Dict[str, Any]]]:
    try:
        with open(_REVIEW_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_review_cache(cache: Dict[str, List[Dict[str, Any]]]) -> None:
    try:
        with open(_REVIEW_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"评价缓存写入失败: {e}")


def _generate_review_pool(hotel_name: str, hotel_meta: Optional[Dict[str, Any]],
                          model_name: str) -> List[Dict[str, Any]]:
    """一次生成 6 条评价（4 正 2 负）"""
    meta_txt = ""
    if hotel_meta:
        meta_txt = (f"酒店信息：位置 {hotel_meta.get('location', '未知')}；"
                    f"类型 {hotel_meta.get('type', '未知')}；评分 {hotel_meta.get('rating', '未知')}")
    user_prompt = f"酒店名称：{hotel_name}\n{meta_txt}\n\n请按 2:1 正负比例生成 6 条评价（4 条正面、2 条负面）。"
    try:
        response = call_llm(model_name, user_prompt, _REVIEW_SYSTEM_PROMPT, stream=False, show_thinking=False)
        s = response.find('[')
        e = response.rfind(']') + 1
        if s != -1 and e > s:
            pool = json.loads(response[s:e])
            if isinstance(pool, list) and pool:
                return pool
    except Exception as ex:
        print(f"评价生成出错: {ex}")
    return []


def get_hotel_reviews(hotel_name: str, model_name: str = "qwen-plus",
                      hotel_meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """获取酒店评价：优先读缓存池，没有则为该酒店生成 6 条入池，再随机抽 3 条返回。

    Args:
        hotel_name: 酒店名称
        model_name: LLM 模型名
        hotel_meta: 可选的酒店元信息（位置/类型/评分），用于让评价内容更贴合

    Returns:
        [{reviewer_name, rating, review_date, review_content}, ...]（最多 3 条）
    """
    if not hotel_name:
        return []
    cache = _load_review_cache()
    pool = cache.get(hotel_name)
    if not pool:
        pool = _generate_review_pool(hotel_name, hotel_meta, model_name)
        if pool:
            cache[hotel_name] = pool
            _save_review_cache(cache)
    if not pool:
        return []
    k = min(3, len(pool))
    return random.sample(pool, k)


# 示例用法
if __name__ == "__main__":
    print("=== 酒店推荐（高德） ===")
    hotels = get_hotel_recommendations("北京商务酒店，近地铁站")
    for i, hotel in enumerate(hotels[:3], 1):
        print(f"{i}. {hotel['hotel_name']} | {hotel['location'][:30]} | 评分:{hotel['rating']}")

    print("\n=== 酒店评价（LLM 生成 + 缓存） ===")
    if hotels:
        name = hotels[0]["hotel_name"]
        for round_i in (1, 2):
            reviews = get_hotel_reviews(name, hotel_meta=hotels[0])
            print(f"-- 第 {round_i} 次查询 {name}：{len(reviews)} 条")
            for r in reviews:
                print(f"   [{r.get('rating')}] {r.get('review_content', '')[:50]}")
