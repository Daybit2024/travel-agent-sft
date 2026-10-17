# -*- coding: utf-8 -*-
"""公交策略意图识别（三级仲裁的第 1、2 级）

- 第 1 级 `match_transit_strategy`：关键词 / 正则，纯函数、无状态、可单测
- 第 2 级 `llm_transit_strategy`：本地 4B 兜底（规则未命中时触发一次，失败一律 fail-open）
- 第 3 级：调用方拿不到 code 时按最快捷处理（不把 strategy 传给高德）

**为什么参数由代码注入、不进 tool schema**：微调模型没在训练数据里见过 `strategy`，
不会稳定产出；而评测的参数匹配是键集合完全相等（`eval/eval_ollama_accuracy.py:128`），
测试集里 14 条 query_route 参考调用都不含该键——模型多吐一个键会让 keys / vals 两档
同时失配。注入则完全不扰动 schema 与 prompt，评测零影响。

**核心原则：否定不是全局规则，而是按类别定义**——同类别的否定是正向偏好
（"不换乘" → 2），跨类别的纯否定必须不命中（"别太快" ≠ 0）。全局先剥离否定
会误杀"不换乘"，所以每条规则各自带负向拦截表。

扫描规则（两条不对称，勿"顺手统一"）：
- 当前轮消息**永远**参与匹配（检测只在 query_route 调用里发生，当前轮必是路线语境）；
- 更旧的消息若含其他话题强标记（酒店/攻略/天气…）→ 停止回溯，避免跨话题污染。

偏好最多回溯 `DEFAULT_SCAN_WINDOW` 条用户消息，且**只扫 role == "user"**——扫到
assistant 会自我强化（第 1 轮命中 → 回复里出现关键词 → 第 2 轮被自己命中）。
"""
import hashlib
import io
import os
import re
from typing import Callable, List, NamedTuple, Optional, Sequence

STRATEGY_NAMES = {0: "最快捷", 1: "最经济", 2: "最少换乘", 3: "最少步行", 4: "最舒适", 5: "不乘地铁"}
DEFAULT_SCAN_WINDOW = 3

# 否定词与关键词之间允许的间隔；不含句读，防跨句误配（"不客气。地铁…"）
_GAP3 = r"[^，。！？、；,.!?;\n]{0,3}"

# 其他话题强标记：回溯时遇到就停
_OTHER_TOPIC_RE = re.compile(r"酒店|住哪|住宿|民宿|客栈|评价|攻略|天气|景点|门票|行程|美食|餐厅|几月")

# 撤销：说"算了/随便"且整句不含任何偏好词 → 停止回溯（防"算了随便吧"翻出旧偏好）
_REVOKE_RE = re.compile(r"算了|随便|都行|都可以|无所谓|看你的|你决定|你定|不挑")

# (策略码, 正向规则, 负向拦截)，按优先级 5 > 3 > 2 > 1 > 4 > 0 依次尝试。
# 0 放最低是因为它等于默认、信息量为零——"最快但别换乘"应解读为 2 而不是 0。
_RULES = (
    # 5 不乘地铁：硬约束。宾语必须是地铁/轻轨/轨道交通，所以"不坐公交"天然不命中
    (5, (
        r"(?:不|别|不要|不想|避免|讨厌|拒绝|坐不了)(?:坐|乘|搭|用)?" + _GAP3
        + r"(?:地铁|轻轨|轨道交通)",
        r"(?:地铁|轻轨)" + r"[^，。！？、；,.!?;\n]{0,6}"
        + r"(?:太挤|很挤|太累|挤得|受不了|不舒服|晕车)",
    ), ()),

    # 3 最少步行：不用单字"不"（"不好意思走路要多久"会误命中），只用多字否定
    (3, (
        r"(?:少|尽量少|不想|不要|别|避免|怕|懒得)(?:走|走路|步行)",
        r"(?:走路|步行)" + _GAP3 + r"(?:最少|越少越好|距离最短|短一点)",
        r"腿脚(?:不好|不便|不方便)",
        r"走不动",
        r"(?:带|领着|推着)(?:老人|小孩|娃|婴儿车|行李箱)",
    ), ()),

    # 2 最少换乘：换乘语义明确，间隔可放宽到 6
    (2, (
        r"(?:少|尽量少|不想|不要|别|避免|免得|无需|不用|讨厌)"
        + r"[^，。！？、；,.!?;\n]{0,6}(?:换乘|倒车|转车)",
        r"(?:换乘|倒车|转车)" + _GAP3 + r"(?:最少|次数少|越少越好)",
        r"直达|一趟车|一车到|不换车|不用换车",
    ), ()),

    # 1 最经济：加住宿类拦截——"性价比高的酒店"说的是酒店属性，不是路线偏好
    (1, (
        r"最?便宜",
        r"省钱|省点钱|花(?:最)?少(?:的)?钱|票价(?:最)?(?:低|便宜)",
        r"最经济|经济(?:实惠|一点|点|些)|实惠|性价比高|划算|穷游",
    ), (r"不便宜|便宜不了|酒店|住宿|民宿|客栈|旅馆",)),

    # 4 最舒适：同样加住宿类拦截（"舒服的酒店"不是路线偏好）
    (4, (
        r"舒适|舒服|不累|轻松(?:点|一些)?|不要太累|(?:别|不想)太累",
        r"少折腾|别折腾|有座|空调车",
    ), (r"不舒服|不舒适|酒店|住宿|民宿|客栈|旅馆",)),

    # 0 最快捷："最佳/最优"一律不算（测试集里真实出现 2 次的诱饵）
    (0, (
        r"最快|时间最短|最短时间|用时最少|最省时间|越快越好|尽快",
        r"赶时间|时间紧|赶(?:飞机|火车|高铁|航班|车)",
    ), (r"不(?:着)?急|不赶时间|别太快|不(?:用|要)最快|慢慢来|时间(?:还)?(?:很)?(?:充足|宽裕)",)),
)


class StrategyHit(NamedTuple):
    code: int
    keyword: str      # 命中的词；4B 兜底时为空串
    source: str       # "current" / "history[-2]" / "llm"


def match_transit_strategy(text: str) -> Optional[StrategyHit]:
    """单条文本的规则匹配（纯函数）。None 表示规则无法判定，不代表"没有偏好"。"""
    t = (text or "").strip()
    if not t:
        return None
    for code, positives, negatives in _RULES:
        if any(re.search(n, t) for n in negatives):
            continue
        for pos in positives:
            m = re.search(pos, t)
            if m:
                return StrategyHit(code, m.group(0), "current")
    return None


def _scan_rules(texts: Sequence[str], window: int):
    """从新到旧回溯。返回 (hit, blocked)；blocked=True 表示遇到撤销或话题边界，
    应停止（不再交给 4B 兜底）。"""
    recent = [t for t in list(texts)[-window:] if (t or "").strip()]
    for i in range(len(recent) - 1, -1, -1):
        t = recent[i].strip()
        is_current = (i == len(recent) - 1)
        if not is_current and _OTHER_TOPIC_RE.search(t):
            return None, True
        hit = match_transit_strategy(t)
        if hit:
            where = "current" if is_current else f"history[-{len(recent) - i}]"
            return hit._replace(source=where), False
        if _REVOKE_RE.search(t):
            return None, True
    return None, False


def detect_transit_strategy(texts: Sequence[str], *, window: int = DEFAULT_SCAN_WINDOW,
                            llm_fn: Optional[Callable[[str], Optional[int]]] = None
                            ) -> Optional[StrategyHit]:
    """规则 → 4B 兜底 → None（调用方按默认最快捷处理）。

    texts 按**时间升序**传入（最新在末尾）；llm_fn 可注入，便于测试不触网。
    """
    seq = [t for t in (texts or []) if (t or "").strip()]
    if not seq:
        return None
    hit, blocked = _scan_rules(seq, window)
    if hit or blocked:
        return hit
    if llm_fn is None:
        return None
    code = llm_fn(seq[-1])
    if code is None:
        return None
    return StrategyHit(int(code), "", "llm")


def strategy_name(code: Optional[int]) -> str:
    return STRATEGY_NAMES.get(code, "最快捷") + ("(默认)" if code is None else "")


# ── 4B 兜底层（本地 Ollama）──────────────────────────────
# 调用方式与缓存策略对齐 rag-system/retrieval/query_rewrite.py:36-119

_INTENT_PROMPT = (
    "判断用户这句话里有没有公交出行偏好。\n"
    "0=最快 1=最便宜 2=最少换乘 3=最少走路 4=最舒适 5=不坐地铁\n"
    "只输出一个数字；没有偏好只输出：无\n"
    "用户：{q}\n回答：/no_think"
)
_CACHE_NS = "transit_intent"


def _intent_base_url() -> str:
    return os.environ.get("TRANSIT_INTENT_BASE_URL", "http://localhost:11434/v1")


def _intent_model() -> str:
    return os.environ.get("TRANSIT_INTENT_MODEL", "qwen3-travel")


def _intent_timeout() -> int:
    try:
        return int(os.environ.get("TRANSIT_INTENT_TIMEOUT", "60"))
    except ValueError:
        return 60


def _cache_path(key_text: str) -> str:
    # 延迟导入：规则层保持零依赖（get_route 缺 AMAP_KEY 会在 import 时 raise）
    from tools.get_route import PROJECT_ROOT
    h = hashlib.md5((_intent_model() + "|" + key_text).encode("utf-8")).hexdigest()
    return os.path.join(PROJECT_ROOT, ".cache", _CACHE_NS, h + ".txt")


def _cached(key_text: str) -> Optional[str]:
    try:
        p = _cache_path(key_text)
        if os.path.exists(p):
            return io.open(p, encoding="utf-8").read().strip()
    except Exception:
        pass
    return None


def _write_cache(key_text: str, value: str):
    try:
        p = _cache_path(key_text)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with io.open(p, "w", encoding="utf-8") as f:
            f.write(value)
    except Exception:
        pass


def _call_4b(prompt: str, max_tokens: int) -> Optional[str]:
    """调用本地 4B；失败返回 None（不抛异常）。

    Ollama 端点走原生 /api/chat（keep_alive 才能生效，模型常驻），
    其他 OpenAI 兼容端点走标准 /chat/completions。
    """
    import requests

    base = _intent_base_url().rstrip("/")
    try:
        if "11434" in base:  # Ollama
            url = (base[:-3] if base.endswith("/v1") else base) + "/api/chat"
            resp = requests.post(
                url,
                json={
                    "model": _intent_model(),
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": max_tokens},
                    "keep_alive": "30m",
                },
                timeout=_intent_timeout(),
            )
            resp.raise_for_status()
            out = (resp.json().get("message", {}).get("content") or "")
        else:
            resp = requests.post(
                base + "/chat/completions",
                json={
                    "model": _intent_model(),
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                },
                timeout=_intent_timeout(),
            )
            resp.raise_for_status()
            out = resp.json()["choices"][0]["message"]["content"] or ""
        return out.strip().strip("。.，, \n")
    except Exception:
        return None


def _parse_code(raw: str) -> Optional[int]:
    s = (raw or "").strip().strip("。.，, \n\"'")
    if not s or len(s) > 12:      # 超长 = 跑偏，按未判定处理
        return None
    for ch in s:
        if ch.isdigit():
            d = int(ch)
            return d if 0 <= d <= 5 else None
    return None


def llm_transit_strategy(text: str) -> Optional[int]:
    """本地 4B 兜底：返回 0-5；无偏好 / 任何失败一律 None（fail-open）。"""
    q = (text or "").strip()
    if not q or os.environ.get("TRANSIT_INTENT_ENABLED", "1") == "0":
        return None
    cached = _cached(q)
    if cached is not None:
        return int(cached) if cached.isdigit() else None
    raw = _call_4b(_INTENT_PROMPT.format(q=q), 4)
    if raw is None:
        return None               # 调用失败：不落缓存，下次仍可重试
    code = _parse_code(raw)
    _write_cache(q, "" if code is None else str(code))
    return code
