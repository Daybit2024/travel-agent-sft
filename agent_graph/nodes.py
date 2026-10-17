import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .callbacks import RequestLogHandler
from .skills import build_skill_system_prompt, tool_schemas_for
from .state import AgentState
from .toolkit import (
    TOOLCALL_TEXT_RE,
    _answer_mismatches,
    _fallback_text,
    call_function,
    detect_route_strategy,
    query_route as run_query_route,
    record,
    should_continue_tool_chain,
)

STRUCTURED_TOOLS = {"query_route", "recommend_hotels", "get_hotel_reviews"}

_ROUTE_FROM_TO_RE = re.compile(
    r"(?:从)?\s*([^,，。！？?、；;:：\s][^,，。！？?、；;:：]*?)\s*"
    r"(?:到|去|往|至)\s*([^,，。！？?、；;:：]+?)\s*"
    r"(?:怎么走|怎么去|如何走|如何到达|怎么坐车|的路线|路线|$)"
)


def _is_coord_like(value) -> bool:
    parts = str(value or "").strip().split(",")
    if len(parts) != 2:
        return False
    try:
        float(parts[0])
        float(parts[1])
        return True
    except ValueError:
        return False


def extract_from_to(text: str):
    """从用户原话里抽取「从X到Y」的地名对;没匹配到返回 None。"""
    match = _ROUTE_FROM_TO_RE.search(text or "")
    if not match:
        return None
    start, end = match.group(1).strip(" ,，"), match.group(2).strip(" ,，")
    if not start or not end:
        return None
    return start, end


def _fix_route_args(args: dict, messages) -> dict:
    """模型偶尔把地名转成(幻觉)坐标:若用户原话里有「从X到Y」,用地名覆盖坐标。"""
    if not (_is_coord_like(args.get("start_location")) or _is_coord_like(args.get("end_location"))):
        return args
    text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            text = msg.content or ""
            break
    pair = extract_from_to(text)
    if not pair:
        return args
    start, end = pair
    fixed = dict(args)
    if _is_coord_like(fixed.get("start_location")):
        fixed["start_location"] = start
    if _is_coord_like(fixed.get("end_location")):
        fixed["end_location"] = end
    return fixed


WRAP_UP_HINT = (
    "（提示：以上信息已足够，请直接基于这些内容生成给用户的最终答复；"
    "不要再调用工具，也不要输出任何工具调用格式。"
    "站名、线路名、分钟数、票价、距离必须从工具结果原样引用，不得改写、四舍五入或换算。）"
)


def _with_system(state: AgentState) -> list:
    skill = state.get("active_skill", "default")
    return [
        SystemMessage(content=build_skill_system_prompt(skill, state.get("user_info")))
    ] + list(state["messages"])


def agent_node(state: AgentState, llm, tool_schemas) -> dict:
    """模型节点:有工具调用 → 入历史进 tools;自然收口 → 只产出待清洗的终答。"""
    schemas = tool_schemas_for(tool_schemas, state.get("active_skill", "default"))
    bound = llm.bind_tools(schemas) if schemas else llm
    response = bound.invoke(_with_system(state), config={"callbacks": [RequestLogHandler()]})
    if getattr(response, "tool_calls", None):
        return {"messages": [response]}
    return {"final_answer": response.content}


def tools_node(state: AgentState) -> dict:
    """工具节点:执行 tool_calls;query_route 前由代码注入公交偏好策略。"""
    last = state["messages"][-1]
    tool_messages = []
    results = []
    names = []
    continues = []
    strategy = state.get("route_strategy")
    for call in last.tool_calls:
        name = call["name"]
        args = call.get("args") or {}
        if name == "query_route":
            args = _fix_route_args(args, state["messages"])
            strategy = detect_route_strategy(state["messages"])
            result = str(
                run_query_route(
                    args.get("start_location"),
                    args.get("end_location"),
                    args.get("city_code", "110000"),
                    strategy=strategy,
                )
            )
        else:
            result = str(call_function(name, args))
        results.append(result)
        names.append(name)
        continues.append(should_continue_tool_chain(name, result))
        tool_messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return {
        "messages": tool_messages,
        "tool_rounds": state.get("tool_rounds", 0) + 1,
        "last_tool_results": results,
        "last_tool_names": names,
        "last_round_continue": any(continues),
        "route_strategy": strategy,
    }


def wrap_up_node(state: AgentState, llm) -> dict:
    """收尾轮:不绑工具,要求模型只输出文本;尾部提示追加到最后一条工具结果。"""
    messages = _with_system(state)
    last = messages[-1]
    if isinstance(last, ToolMessage):
        messages[-1] = ToolMessage(
            content=str(last.content) + "\n\n" + WRAP_UP_HINT,
            tool_call_id=last.tool_call_id,
        )
    response = llm.invoke(messages, config={"callbacks": [RequestLogHandler()]})
    return {"final_answer": response.content}


def finalize_node(state: AgentState) -> dict:
    """出口:清洗工具调用文本泄漏 → 终答保真核对 → 不一致降级为原始结果。"""
    raw = state.get("final_answer") or ""
    cleaned = TOOLCALL_TEXT_RE.sub("", raw).strip()
    last_results = state.get("last_tool_results") or []
    tool_names = set(state.get("last_tool_names") or [])
    guard_turns = state.get("guard_turns", 0)
    guard_hits = state.get("guard_hits", 0)

    def done(answer: str) -> dict:
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
            "guard_turns": guard_turns,
            "guard_hits": guard_hits,
        }

    if cleaned:
        if last_results and tool_names & STRUCTURED_TOOLS:
            guard_turns += 1
            mismatches = _answer_mismatches(cleaned, last_results)
            if mismatches:
                guard_hits += 1
                record("guard", hit=True, mismatches=mismatches[:5],
                       hits=guard_hits, turns=guard_turns)
                return done(_fallback_text(last_results))
        return done(cleaned)

    fallback = (
        _fallback_text(last_results)
        if last_results
        else "抱歉，我暂时无法处理这个请求，请换个说法再试试。"
    )
    return done(fallback)
