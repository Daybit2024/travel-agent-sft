import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

from tools.request_log import elapsed_ms, record

def _find_project_root(start_dir: str) -> str:
    """向上定位项目根（以 .env.example / rag-system 目录为标记）"""
    d = os.path.abspath(start_dir)
    while True:
        if os.path.exists(os.path.join(d, ".env.example")) or os.path.isdir(os.path.join(d, "rag-system")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.abspath(start_dir)
        d = parent


PROJECT_ROOT = _find_project_root(os.path.dirname(os.path.abspath(__file__)))

# 密钥从项目根目录 .env 读取（系统环境变量优先；参考 .env.example）
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

AMAP_KEY = os.environ.get("AMAP_KEY", "")
if not AMAP_KEY:
    raise RuntimeError("缺少 AMAP_KEY：请在项目根目录 .env 中配置（参考 .env.example）")


def _get_json(url: str, retries: int = 2, timeout: int = 10):
    """GET 并解析 JSON；网络类异常自动重试 retries 次

    全部高德接口都走这里，所以请求流水只需在这一处埋点。
    记录脱敏后的 URL（key 会被 request_log 兜底替换成 ***）、耗时、重试次数，
    以及高德的 status/info（quota 用尽、key 无效这类问题就靠它暴露）。
    """
    last = None
    t0 = time.perf_counter()
    for i in range(retries + 1):
        try:
            resp = requests.get(url, timeout=timeout).json()
            record("amap", url=url, ok=True, ms=elapsed_ms(t0), attempts=i + 1,
                   status=resp.get("status"), info=resp.get("info"))
            return resp
        except Exception as e:
            last = e
            if i < retries:
                time.sleep(0.5 * (i + 1))
    record("amap", url=url, ok=False, ms=elapsed_ms(t0), attempts=retries + 1,
           error=f"{type(last).__name__}: {last}")
    raise last


def _as_text(v) -> str:
    """高德偶把 city/province 返回成空列表，统一成字符串"""
    if isinstance(v, list):
        return "".join(str(x) for x in v)
    return str(v or "")


def _to_int(v, default: int = 0) -> int:
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return default


def geocode(address):
    """通过地名获取经纬度"""
    url = f"https://restapi.amap.com/v3/geocode/geo?address={address}&key={AMAP_KEY}"
    resp = _get_json(url)
    if resp["status"] == "1" and resp["geocodes"]:
        return resp["geocodes"][0]["location"]  # "lng,lat"
    else:
        raise ValueError(f"无法找到地址: {address}")


def geocode_place(address: str) -> dict:
    """地名 → 坐标 + 所属城市/省份（跨城判据用）；解析不到抛 ValueError"""
    url = f"https://restapi.amap.com/v3/geocode/geo?address={address}&key={AMAP_KEY}"
    resp = _get_json(url)
    geos = (resp.get("geocodes") or []) if resp.get("status") == "1" else []
    if not geos:
        raise ValueError(f"无法找到地址: {address}")
    g = geos[0]
    return {
        "location": g.get("location", ""),
        "city": _as_text(g.get("city")),
        "province": _as_text(g.get("province")),
    }


def regeo(location: str) -> Optional[str]:
    """坐标 → 可读地名；失败返回 None（不抛异常）"""
    try:
        url = f"https://restapi.amap.com/v3/geocode/regeo?location={location}&key={AMAP_KEY}"
        resp = _get_json(url)
        if resp.get("status") == "1":
            name = (resp.get("regeocode") or {}).get("formatted_address") or ""
            return name.strip() or None
    except Exception:
        pass
    return None


def get_walking(origin, destination):
    url = f"https://restapi.amap.com/v3/direction/walking?origin={origin}&destination={destination}&key={AMAP_KEY}"
    resp = _get_json(url)
    if resp["status"] == "1" and resp["route"]["paths"]:
        path = resp["route"]["paths"][0]
        steps = [s["instruction"] for s in path["steps"]]
        return {
            "总时间(分钟)": int(path["duration"]) // 60,
            "总距离(米)": int(path["distance"]),
            # "详细路线": steps
        }
    return None


def _iter_walk_distances(walking):
    """步行段距离（walking 有 dict / list 两种形态）"""
    if isinstance(walking, dict):
        if "distance" in walking:
            yield walking["distance"]
    elif isinstance(walking, list):
        for w in walking:
            if isinstance(w, dict) and "distance" in w:
                yield w["distance"]


def _normalize_cost(cost) -> str:
    """票价归一："507.000000" → "507"；空 / None → "未知" """
    if cost is None:
        return "未知"
    s = str(cost).strip()
    if not s or s == "未知":
        return "未知"
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if f == int(f) else f"{f:g}"


def _railway_text(railway: dict) -> str:
    """高德跨城公交方案里的铁路段 → 可读文本；无车次信息的空壳段返回 ""。"""
    name = railway.get("name") or railway.get("trip") or ""
    if not name:
        return ""
    dep = (railway.get("departure_stop") or {}).get("name", "")
    arr = (railway.get("arrival_stop") or {}).get("name", "")
    text = f"{name}: {dep} 上车 → {arr} 下车"
    seconds = _to_int(railway.get("time"))
    if seconds > 0:
        text += f"（约{seconds // 60}分钟）"
    return text


def _bus_text(busline: dict) -> str:
    """单个公交段 → 可读文本（含站数；途经站过多只给站数）"""
    name = busline.get("name", "")
    dep = (busline.get("departure_stop") or {}).get("name", "")
    arr = (busline.get("arrival_stop") or {}).get("name", "")
    text = f"{name}: {dep} 上车 → {arr} 下车"

    via = busline.get("via_stops") or []
    via_num = busline.get("via_num")
    n = None
    if via:
        n = len(via)
        if str(via_num or "").strip().isdigit():
            n = int(via_num)
    elif str(via_num or "").strip().isdigit() and int(via_num) > 0:
        n = int(via_num)          # 途经站列表缺失时仍可给站数

    if n:
        # via_stops 是中间途经站、不含首末站，所以实际站数 = via_num + 1
        text += f"（共{n + 1}站"
        if via and len(via) <= 6:
            text += "，途经" + "、".join((s or {}).get("name", "") for s in via)
        text += "）"
    return text


def _parse_transit_plan(t: dict) -> dict:
    """单条高德 transits 方案 → 结构化 dict（含换乘次数、步行距离、逐段文本）"""
    segments = []
    rides = 0
    walk_sum = 0
    for seg in (t.get("segments") or []):
        if not isinstance(seg, dict):
            continue
        buslines = ((seg.get("bus") or {}).get("buslines")) or []
        if buslines:
            rides += 1
            segments.append(_bus_text(buslines[0]))
        railway = seg.get("railway")
        if isinstance(railway, dict):
            railway_text = _railway_text(railway)
            if railway_text:
                segments.append(railway_text)
        if "walking" in seg:
            for d in _iter_walk_distances(seg["walking"]):
                segments.append(f"步行 {d} 米")
                walk_sum += _to_int(d)

    walking = t.get("walking_distance")
    if (
        walking is None
        or isinstance(walking, (list, tuple))
        or str(walking).strip() in ("", "[]")
    ):
        walking = walk_sum          # 顶层缺失/空列表时累加各段步行

    return {
        "总时间(分钟)": _to_int(t.get("duration")) // 60,
        "票价(元)": _normalize_cost(t.get("cost")),
        "步行距离(米)": _to_int(walking),
        "换乘次数": max(rides - 1, 0),
        "标签": [],
        "详细路线": segments,
    }


_TAG_DIMENSIONS = (
    ("时间最短", lambda p: (p["总时间(分钟)"], p["换乘次数"], p["步行距离(米)"])),
    ("少换乘", lambda p: (p["换乘次数"], p["总时间(分钟)"], p["步行距离(米)"])),
    ("步行最少", lambda p: (p["步行距离(米)"], p["总时间(分钟)"])),
)


def _annotate_plans(plans) -> None:
    """按三维度给方案打标签（原地写入 "标签"）

    每个维度各标一条：排序键带平局链，同一条方案可同时挂多个标签，
    但同一维度不会重复发给多条（完全平局时归第一条）。
    """
    if not plans:
        return
    for label, key in _TAG_DIMENSIONS:
        best = min(plans, key=key)
        best.setdefault("标签", []).append(label)


def get_transit(origin, destination, city_code, strategy=None):
    """公交方案（返回全部方案，不再只取 transits[0]）

    strategy 是高德侧候选集筛选器（1最经济 2最少换乘 3最少步行 4最舒适 5不乘地铁）；
    0 / None 表示不筛选，此时不追加参数、返回默认候选集。
    """
    url = (f"https://restapi.amap.com/v3/direction/transit/integrated"
           f"?origin={origin}&destination={destination}&city={city_code}&key={AMAP_KEY}")
    try:
        s = int(strategy) if strategy is not None else 0
    except (TypeError, ValueError):
        s = 0
    if s in (1, 2, 3, 4, 5):
        url += f"&strategy={s}"

    resp = _get_json(url)
    if resp.get("status") == "1" and (resp.get("route") or {}).get("transits"):
        plans = [_parse_transit_plan(t) for t in resp["route"]["transits"]]
        _annotate_plans(plans)
        return plans
    return []


def get_driving(origin, destination):
    url = f"https://restapi.amap.com/v3/direction/driving?origin={origin}&destination={destination}&extensions=all&key={AMAP_KEY}"
    resp = _get_json(url)
    if resp["status"] == "1" and resp["route"]["paths"]:
        path = resp["route"]["paths"][0]
        steps = [s["instruction"] for s in path["steps"]]
        return {
            "总时间(分钟)": int(path["duration"]) // 60,
            "总距离(米)": int(path["distance"]),
            "过路费(元)": path.get("tolls", "0"),
            # "详细路线": steps
        }
    return None


def _is_coord(value) -> bool:
    parts = str(value or "").strip().split(",")
    if len(parts) != 2:
        return False
    try:
        float(parts[0])
        float(parts[1])
        return True
    except ValueError:
        return False


def query_routes(start_coords, end_place, city_code="110000", strategy=None):
    destination = str(end_place) if _is_coord(end_place) else geocode(end_place)

    result = {
        "步行": get_walking(start_coords, destination),
        "公交": get_transit(start_coords, destination, city_code, strategy=strategy),
        "驾车/打车": get_driving(start_coords, destination)
    }
    return result


# 使用示例
if __name__ == "__main__":
    start = "116.481028,39.989643"   # 起点经纬度（望京SOHO）
    end = "天坛"
    routes = query_routes(start, end, city_code="110000")

    for mode, info in routes.items():
        print(f"\n【{mode}】")
        if mode == "公交":
            for i, plan in enumerate(info or [], 1):
                print(f"方案{i} 标签={plan['标签']}")
                for k, v in plan.items():
                    if k == "标签":
                        continue
                    print(f"  {k}: {v}")
        elif info:
            for k, v in info.items():
                print(f"{k}: {v}")
        else:
            print("未查询到结果")
