# -*- coding: utf-8 -*-
"""天气工具：Open-Meteo（免费、免 key，最多 16 天预报）

接口结构与原实现保持兼容：
get_weather_by_date_range() 返回 [{日期, 天气, 白天天气, 夜间天气, 最高温, 最低温}, ...]
（Open-Meteo 每日一个天气代码，白天/夜间字段填同一描述）
"""
import os
import sys
import requests
from datetime import datetime, timedelta

sys_path_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(sys_path_root)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

MAX_FORECAST_DAYS = 16

# WMO 天气代码 → 中文描述
_WMO_TEXT = {
    0: "晴", 1: "基本晴", 2: "多云", 3: "阴",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨", 56: "冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨", 66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "强阵雪",
    95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "强雷阵雨伴冰雹",
}

# 和风风格城市 ID → 城市名（兼容老接口传参）
_QWEATHER_ID_MAP = {
    "101010100": "北京", "101020100": "上海", "101280101": "广州",
    "101280601": "深圳", "101210101": "杭州", "101190101": "南京",
}


def _resolve_coords(location: str):
    """城市名 / "lng,lat" / 和风城市ID → (latitude, longitude)"""
    loc = (location or "").strip()
    if not loc:
        return None, None
    if "," in loc:
        try:
            lng, lat = [float(x.strip()) for x in loc.split(",")[:2]]
            return lat, lng
        except Exception:
            return None, None
    if loc.isdigit() and len(loc) == 9 and loc in _QWEATHER_ID_MAP:
        loc = _QWEATHER_ID_MAP[loc]
    # 1) Open-Meteo 地理编码（城市/地名）
    try:
        resp = requests.get(OPEN_METEO_GEOCODE_URL,
                            params={"name": loc, "count": 1, "language": "zh"},
                            timeout=15).json()
        results = resp.get("results") or []
        if results:
            return results[0].get("latitude"), results[0].get("longitude")
    except Exception:
        pass
    # 2) 高德地理编码兜底（能解析景区/POI 名，如"婺源篁岭"）
    try:
        from tools.get_route import geocode
        g = geocode(loc)  # 返回 "lng,lat"
        if isinstance(g, str) and "," in g:
            lng, lat = [float(x) for x in g.split(",")[:2]]
            return lat, lng
    except Exception:
        pass
    return None, None


def get_weather_by_date_range(location: str, start_date: str, num_days: int, query_range: str = "30d"):
    """查询指定地点和时间段的天气（Open-Meteo，最多 16 天）。

    :param location: 城市名（'杭州'）、'lng,lat' 坐标，或和风城市 ID
    :param start_date: 开始日期 'YYYY-MM-DD'（早于今天则自动改用今天）
    :param num_days: 查询天数（超出 16 天上限自动截断）
    :param query_range: 保留参数（兼容原接口，不再使用）
    :return: [{日期, 天气, 白天天气, 夜间天气, 最高温, 最低温}, ...]
    """
    try:
        lat, lon = _resolve_coords(location)
        if lat is None or lon is None:
            print(f"[警告] 无法解析地点: {location}")
            return []

        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        except Exception:
            start_dt = datetime.now()
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if start_dt < today:
            print(f"[警告] 请求日期 {start_date} 已过去，改用今天 {today.strftime('%Y-%m-%d')}")
            start_dt = today

        num_days = max(1, int(num_days or 1))
        horizon = today + timedelta(days=MAX_FORECAST_DAYS - 1)
        if start_dt > horizon:
            print(f"[警告] 超出预报范围（最多 {MAX_FORECAST_DAYS} 天）")
            return []
        end_dt = min(start_dt + timedelta(days=num_days - 1), horizon)

        resp = requests.get(OPEN_METEO_FORECAST_URL, params={
            "latitude": lat, "longitude": lon,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min",
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "timezone": "Asia/Shanghai",
        }, timeout=20).json()
        daily = resp.get("daily") or {}
        dates = daily.get("time") or []
        codes = daily.get("weather_code") or []
        tmax = daily.get("temperature_2m_max") or []
        tmin = daily.get("temperature_2m_min") or []

        weather_list = []
        for i, d in enumerate(dates):
            code = codes[i] if i < len(codes) else None
            desc = _WMO_TEXT.get(code, "未知")
            weather_list.append({
                "日期": d,
                "天气": desc,
                "白天天气": desc,
                "夜间天气": desc,
                "最高温": f"{tmax[i]}℃" if i < len(tmax) and tmax[i] is not None else "-",
                "最低温": f"{tmin[i]}℃" if i < len(tmin) and tmin[i] is not None else "-",
            })
        weather_list.sort(key=lambda x: x["日期"])
        return weather_list
    except Exception as e:
        print(f"天气查询出错: {e}")
        return []


def get_weather_by_date(location: str, date: str, days: str = "30d"):
    """查询指定地点和日期的天气（兼容原有接口）"""
    weather_list = get_weather_by_date_range(location, date, 1, days)
    return weather_list[0] if weather_list else None


if __name__ == "__main__":
    result = get_weather_by_date("杭州", datetime.now().strftime("%Y-%m-%d"))
    if result:
        for k, v in result.items():
            print(f"{k}: {v}")
    else:
        print("未找到对应日期的天气数据")
