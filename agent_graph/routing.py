MAX_TOOL_ROUNDS = 3


def after_agent(state) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "finalize"


def after_tools(state) -> str:
    if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
        return "wrap_up"
    return "agent" if state.get("last_round_continue") else "wrap_up"
