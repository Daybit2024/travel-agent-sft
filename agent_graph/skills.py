"""Claude Agent Skills 风格:SKILL.md 加载、规则路由、按技能绑定工具子集。

default 技能 = 原完整 system prompt(字节级),4B 始终走 default;
云端才启用技能路由,新技能不改动 default 路径。
"""
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langchain_core.messages import HumanMessage

from .prompts import WORKFLOW_PROMPT, build_system_prompt, build_user_info_text

# 仅云端启用:本地 4B 的 default prompt 必须保持与训练时字节级一致
CLOUD_CROSS_CITY_RULE = """
## 跨城出行补充规则
用户询问从一个城市到另一个城市(跨城长途)的路线时:
1. 先调用 query_route(该工具支持跨城公交方案,结果中包含铁路段);
2. 方案含铁路段时按车次转述,如 G370 郑州东 → 北京西;
3. 工具查询失败时,再给出高铁/航班/自驾建议并说明原因。
"""

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

# None = 不限制(default);否则只绑定列出的工具
SKILL_TOOLS = {
    "default": None,
    "itinerary-review": {"search_travel_guide", "get_weather_info", "query_route"},
}

REVIEW_INTENT_RE = re.compile(
    r"合理(吗|不|么)|优化|帮我看看|有问题|检查|审查|顺不顺|赶不赶|"
    r"安排得怎么样|规划得怎么样"
)
ITINERARY_CONTEXT_RE = re.compile(
    r"行程|安排|计划|第[一二三四五六七八九十\d]+天|每天|Day\s*\d|日游"
)


@dataclass
class Skill:
    name: str
    description: str
    body: str


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, parts[2]


@lru_cache(maxsize=1)
def load_skills() -> dict:
    skills = {}
    for path in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        name = meta.get("name") or path.parent.name
        skills[name] = Skill(
            name=name,
            description=meta.get("description", ""),
            body=body.strip(),
        )
    return skills


def is_local_backend() -> bool:
    return "11434" in os.environ.get("TRAVEL_LLM_BASE_URL", "")


def route_skill(messages) -> str:
    """审查意图 + 行程上下文双命中才加载审查技能;4B(本地后端)恒走 default。"""
    if is_local_backend():
        return "default"
    text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            text = msg.content or ""
            break
    if REVIEW_INTENT_RE.search(text) and ITINERARY_CONTEXT_RE.search(text):
        return "itinerary-review"
    return "default"


def tool_schemas_for(schemas: list, skill_name: str) -> list:
    names = SKILL_TOOLS.get(skill_name)
    if names is None:
        return schemas
    return [s for s in schemas if (s.get("function") or {}).get("name") in names]


def build_skill_system_prompt(skill_name: str, user_info: dict | None = None) -> str:
    """default → 原完整 prompt(本地字节级;云端追加跨城规则);其他 → 技能模式。"""
    if skill_name == "default":
        if is_local_backend():
            return build_system_prompt(user_info)
        return build_user_info_text(user_info) + CLOUD_CROSS_CITY_RULE + WORKFLOW_PROMPT
    skill = load_skills().get(skill_name)
    if skill is None:
        return build_system_prompt(user_info)
    listing = "\n".join(
        f"- {s.name}: {s.description}" for s in load_skills().values()
    )
    return (
        build_user_info_text(user_info)
        + "\n你是一个专业的旅行助手。\n\n## 可用技能\n"
        + listing
        + f"\n\n## 当前技能:{skill.name}\n"
        + skill.body
    )
