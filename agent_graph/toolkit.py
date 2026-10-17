# -*- coding: utf-8 -*-
"""工具层与终答护栏(LangGraph 版自包含)。

从早期单文件实现中抽离:攻略检索 / 天气 / 路线 / 酒店 + 保真护栏 + 策略检测。
图版节点直接调用这里的模块级函数,不依赖任何旧 Agent 类。
"""
import json
import re
import sys
from pathlib import Path

import requests
from langchain_core.messages import HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from tools.city_utils import resolve_city  # noqa: E402,F401
from tools.get_hotel import get_hotel_recommendations, get_hotel_reviews  # noqa: E402
from tools.get_route import geocode, geocode_place, query_routes, regeo  # noqa: E402
from tools.get_weather import get_weather_by_date_range  # noqa: E402
from tools.request_log import get_trace_id, record, set_trace_id  # noqa: E402,F401
from tools.transit_strategy import (  # noqa: E402
    detect_transit_strategy,
    llm_transit_strategy,
    strategy_name,
)

RAG_API_URL = "http://127.0.0.1:8010"

TOOLCALL_TEXT_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)

# 面向模型的注记：降级渲染给用户看时要去掉
_PLAN_NOTE_RE = re.compile(r"^[ \t]*共\d+条方案，列出\d+条[:：]?[ \t]*\n?", re.MULTILINE)
_STRATEGY_NOTE_RE = re.compile(r"[ \t]*\(\s*策略[:：][^)]*\)")

# 线路名前缀必须写死枚举：中文没有词边界，用贪婪前缀会把"乘坐332路"的"乘坐"吞进来
_LINE_NAME_RE = re.compile(r"(?:专|快|夜|特|临|观光|机场|BRT|地铁|[A-Za-z])?\d+(?:号线|路)")
_NUM_UNIT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:分钟|元|米)")
_STATION_ALIGHT_RE = re.compile(r"([^，。！？、；,.!?;\s]{1,14})\s*(?:上车|下车)")
_STATION_ARRIVE_RE = re.compile(r"(?:到达|到)\s*([^，。！？、；,.!?;\s]{1,14})\s*站")
_GENERIC_STATIONS = {"终点", "起点", "换乘", "目的", "目的地", "车站", "该站", "此站", "本站", "一站", "几站"}

_ADMIN_MARKERS = ("省", "市", "区", "县", "自治州", "自治区", "特别行政区")
_ADMIN_SUFFIXES = ("维吾尔自治区", "回族自治区", "壮族自治区", "特别行政区", "自治区",
                   "自治州", "地区", "省", "市", "盟", "区", "县")
_COORD_RE = re.compile(r"^-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?$")

_GUIDE_META_RE = re.compile(r"^(?:城市编码|生成时间|生成用时|生成模型)\s*[:：].*$", re.M)
_GUIDE_SEP_RE = re.compile(r"^=+\s*$", re.M)

_TAG_PRIORITY = ("时间最短", "少换乘", "步行最少")

# 酒店评价需要上一轮推荐结果（hotel_name → meta）
_last_hotels: dict = {}


# ── 地名补全 ─────────────────────────────────────────────

def _city_core(name: str) -> str:
    """"北京市" → "北京"、"新疆维吾尔自治区" → "新疆"（用于地名里认城市）"""
    s = (name or "").strip()
    for suf in _ADMIN_SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf):
            return s[:-len(suf)]
    return s


def _qualify_place(name: str, city_name: str, lookup=None) -> str:
    """给裸地名补城市前缀，避免高德地理编码跨城误配（如"颐和园"落到别的城市）。"""
    n = (name or "").strip()
    c = (city_name or "").strip()
    if not n or not c:
        return n
    if _COORD_RE.match(n):
        return n
    if any(m in n for m in _ADMIN_MARKERS):
        return n
    if _city_core(c) in n:
        return n

    probe = lookup or geocode_place
    try:
        info = probe(n) or {}
    except Exception:
        return c + n
    for key in ("city", "province"):
        v = info.get(key) or ""
        if isinstance(v, list):
            v = "".join(str(x) for x in v)
        v = str(v).strip()
        if v and (v in n or _city_core(v) in n):
            return n
    return c + n


# ── 终答保真护栏 ─────────────────────────────────────────

def _fallback_text(last_tool_results) -> str:
    """降级答复：直接给用户看工具结果原文（去掉面向模型的注记）"""
    header = "（以下为查询到的原始结果，未经汇总整理）"
    parts = []
    for r in (last_tool_results or []):
        t = _STRATEGY_NOTE_RE.sub("", str(r or ""))
        t = _PLAN_NOTE_RE.sub("", t)
        if t.strip():
            parts.append(t.strip())
    if not parts:
        return header + "\n暂时没有可展示的查询结果。"
    return header + "\n\n" + "\n\n".join(parts[:3])


def _station_in_facts(tok: str, facts_clean: str) -> bool:
    """站名核对：容忍动词前缀（取后缀）与尾部"站"字；通用说法不算失真"""
    t = re.sub(r"\s+", "", tok or "")
    if len(t) < 2 or t in _GENERIC_STATIONS:
        return True
    for i in range(len(t) - 1):
        sub = t[i:]
        if sub in facts_clean or (sub.endswith("站") and sub[:-1] in facts_clean):
            return True
    return False


def _answer_mismatches(answer: str, tool_results) -> list:
    """抽终答里的线路名 / 带单位数字 / 站名，核对是否出现在工具结果中

    只查"事实是否存在"，不查归属与顺序。
    """
    if isinstance(tool_results, str):
        results = [tool_results]
    else:
        results = [str(r or "") for r in (tool_results or [])]
    results = [r for r in results if r.strip()]
    if not results:
        return []

    # 模型常用 markdown 加粗站名/数字(**天桥站**);强调符不是事实的一部分,先剥离再核对
    facts_clean = re.sub(r"[*_`~]+", "", "".join(results))
    facts_clean = re.sub(r"\s+", "", facts_clean)
    ans = re.sub(r"[*_`~]+", "", answer or "")
    bad = []

    for m in _LINE_NAME_RE.finditer(ans):
        t = re.sub(r"\s+", "", m.group(0))
        if t not in facts_clean:
            bad.append(t)
    for m in _NUM_UNIT_RE.finditer(ans):
        t = re.sub(r"\s+", "", m.group(0))
        if t not in facts_clean:
            bad.append(t)
    for m in _STATION_ALIGHT_RE.finditer(ans):
        if not _station_in_facts(m.group(1), facts_clean):
            bad.append(re.sub(r"\s+", "", m.group(1)))
    for m in _STATION_ARRIVE_RE.finditer(ans):
        if not _station_in_facts(m.group(1), facts_clean):
            bad.append(re.sub(r"\s+", "", m.group(1)))
    return bad


# ── 攻略 / 天气 ──────────────────────────────────────────

def _strip_guide_meta(text: str) -> str:
    t = _GUIDE_META_RE.sub("", text or "")
    t = _GUIDE_SEP_RE.sub("", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def search_travel_guide(location: str, search_mode: str = "hybrid") -> str:
    try:
        response = requests.post(
            f"{RAG_API_URL}/search",
            json={"query": location, "search_type": search_mode, "limit": 3},
            timeout=45,
            headers={"X-Trace-Id": get_trace_id()},
        )
        if response.status_code == 200:
            result = response.json()
            if result.get("results"):
                guides = [
                    _strip_guide_meta(item["content"])
                    for item in result["results"]
                    if "content" in item
                ]
                return "\n\n".join(guides)
            return "未找到相关旅行攻略信息"
        return "旅行攻略查询服务暂时不可用"
    except Exception as e:
        return f"旅行攻略查询出错: {e}"


def get_weather_info(location: str, start_date: str, num_days: int = 1) -> str:
    try:
        weather_list = get_weather_by_date_range(location, start_date, num_days)
        if weather_list:
            weather_text = f"天气信息({location} - {start_date}起{num_days}天):\n"
            for i, weather in enumerate(weather_list, 1):
                weather_text += f"\n第{i}天 ({weather['日期']}):\n"
                weather_text += f"  天气: {weather.get('天气', weather.get('白天天气', '未知'))}\n"
                weather_text += f"  温度: {weather['最低温']} ~ {weather['最高温']}\n"
            return weather_text
        return f"无法获取{location}在{start_date}起{num_days}天的天气信息"
    except Exception as e:
        return f"天气查询出错: {e}"


# ── 路线 ─────────────────────────────────────────────────

def pick_plans(plans, limit: int = 3):
    """按标签优先级取 ≤limit 条；某条方案独占多维标签时按原顺序补足"""
    picked = []
    seen = set()
    for tag in _TAG_PRIORITY:
        if len(picked) >= limit:
            break
        for p in plans:
            if id(p) in seen:
                continue
            if tag in (p.get("标签") or []):
                picked.append(p)
                seen.add(id(p))
                break
    for p in plans:
        if len(picked) >= limit:
            break
        if id(p) not in seen:
            picked.append(p)
            seen.add(id(p))
    return picked


def format_routes(start_location, end_location, routes, strategy=None) -> str:
    """路线结果 → 文本。公交给多方案（不截断段），步行/驾车保持原样。"""
    route_text = f"从 {start_location} 到 {end_location} 的路线:\n\n"

    for mode, info in routes.items():
        route_text += f"【{mode}】"
        if mode == "公交" and strategy is not None:
            route_text += f"(策略: {strategy_name(strategy)})"
        route_text += "\n"

        if mode == "步行":
            if info:
                route_text += f"总时间: {info['总时间(分钟)']}分钟\n"
                route_text += f"总距离: {info['总距离(米)']}米\n"
            else:
                route_text += "暂无路线信息\n"
        elif mode == "公交":
            plans = info or []
            if not plans:
                route_text += "暂无路线信息\n"
            else:
                picked = pick_plans(plans)
                route_text += f"共{len(plans)}条方案，列出{len(picked)}条:\n"
                for i, plan in enumerate(picked, 1):
                    tags = "|".join(plan.get("标签") or [])
                    head = f"- 方案{i}" + (f"【{tags}】" if tags else "")
                    route_text += (f"{head}全程{plan['总时间(分钟)']}分钟 | 票价{plan['票价(元)']}元"
                                   f" | 换乘{plan['换乘次数']}次 | 步行{plan['步行距离(米)']}米\n")
                    for j, seg in enumerate(plan.get("详细路线") or [], 1):
                        route_text += f"    {j}) {seg}\n"
        elif mode == "驾车/打车":
            if info:
                route_text += f"总时间: {info['总时间(分钟)']}分钟\n"
                route_text += f"总距离: {info['总距离(米)']}米\n"
                if info['过路费(元)'] != "0":
                    route_text += f"过路费: {info['过路费(元)']}元\n"
            else:
                route_text += "暂无路线信息\n"
        else:
            route_text += "暂无信息\n"

    return route_text


def query_route(start_location: str, end_location: str, city_code: str = "110000",
                strategy=None) -> str:
    """查询路线（strategy 由代码注入，不进 tool schema）"""
    try:
        display_start = start_location
        if "," in start_location and len(start_location.split(",")) == 2:
            start_coords = start_location
            display_start = regeo(start_location) or start_location
        elif start_location == "当前位置":
            return "未提供当前位置坐标,请说明出发地"
        else:
            start_coords = geocode(_qualify_place(start_location, ""))

        destination = _qualify_place(end_location, "")
        routes = query_routes(start_coords, destination, city_code, strategy=strategy)
        return format_routes(display_start, end_location, routes, strategy=strategy)
    except Exception as e:
        return f"路线查询出错: {e}"


# ── 酒店 ─────────────────────────────────────────────────

def recommend_hotels(requirements: str) -> str:
    global _last_hotels
    try:
        hotels = get_hotel_recommendations(requirements)
        _last_hotels = {h.get("hotel_name"): h for h in (hotels or [])}
        if not hotels:
            return "暂时没有找到符合您需求的酒店推荐"
        hotel_text = "根据您的需求，推荐以下酒店（数据来源：高德）:\n\n"
        for i, hotel in enumerate(hotels[:3], 1):
            hotel_text += f"酒店{i}: {hotel['hotel_name']}\n"
            hotel_text += f"位置: {hotel['location']}\n"
            if hotel.get("price_range"):
                hotel_text += f"价格: {hotel['price_range']}\n"
            if hotel.get("rating"):
                hotel_text += f"评分: {hotel['rating']}/5\n"
            if hotel.get("amenities"):
                hotel_text += f"设施: {', '.join(hotel['amenities'][:4])}\n"
            if hotel.get("distance_to_transport"):
                hotel_text += f"交通: {hotel['distance_to_transport']}\n"
            hotel_text += "\n"
        return hotel_text
    except Exception as e:
        return f"酒店推荐出错: {e}"


def get_hotel_reviews_func(hotel_name: str) -> str:
    try:
        reviews = get_hotel_reviews(hotel_name, hotel_meta=_last_hotels.get(hotel_name))
        if reviews:
            review_text = f"{hotel_name}的用户评价（模拟生成，仅供参考）:\n\n"
            for i, review in enumerate(reviews[:3], 1):
                review_text += f"评价{i}:\n"
                review_text += f"评分: {review.get('rating', '未知')}/5\n"
                review_text += f"评论: {review.get('review_content', '')}\n\n"
            return review_text
        return f"暂时没有找到{hotel_name}的用户评价信息"
    except Exception as e:
        return f"酒店评价查询出错: {e}"


# ── 工具调度 ─────────────────────────────────────────────

def should_continue_tool_chain(function_name: str, result: str) -> bool:
    if function_name == "recommend_hotels":
        try:
            if "酒店" in result and ("位置" in result or "价格" in result):
                return True
            if "没有找到" in result or "暂时没有" in result or len(result.strip()) < 20:
                return False
        except Exception:
            pass
        return False
    if function_name == "search_travel_guide":
        try:
            if "未找到" in result or "不可用" in result or len(result.strip()) < 50:
                return False
            return True
        except Exception:
            pass
        return False
    return False


def call_function(function_name: str, arguments: dict) -> str:
    if function_name == "search_travel_guide":
        return search_travel_guide(arguments.get("location"), arguments.get("search_mode", "hybrid"))
    if function_name == "get_weather_info":
        return get_weather_info(
            arguments.get("location"),
            arguments.get("start_date"),
            arguments.get("num_days", 1),
        )
    if function_name == "query_route":
        return query_route(
            arguments.get("start_location"),
            arguments.get("end_location"),
            arguments.get("city_code", "110000"),
        )
    if function_name == "recommend_hotels":
        return recommend_hotels(arguments.get("requirements"))
    if function_name == "get_hotel_reviews":
        return get_hotel_reviews_func(arguments.get("hotel_name"))
    return f"未知函数: {function_name}"


def load_tool_schemas() -> list:
    """训练/评测/运行时共用同一份 all_tools.json,模型侧 schema 零漂移。"""
    with open(PROJECT_ROOT / "datasets" / "all_tools.json", encoding="utf-8") as f:
        return json.load(f)


def detect_route_strategy(messages, n: int = 3):
    """只取最近 n 条用户消息做公交偏好识别。"""
    recent = []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            recent.append(msg.content)
            if len(recent) >= n:
                break
    recent.reverse()
    hit = detect_transit_strategy(recent, llm_fn=llm_transit_strategy)
    return hit.code if hit else None
