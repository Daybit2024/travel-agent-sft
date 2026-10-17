"""历史会话列表:直接读 SQLite checkpoints,供 Gradio 启动时选择。"""
import sqlite3
from pathlib import Path

from langchain_core.messages import HumanMessage

from .memory import DB_PATH, make_checkpointer


def list_sessions(db_path: str | Path = DB_PATH, limit: int = 20) -> list[dict]:
    """按最近检查点倒序列出会话,标题取该会话第一条用户消息。"""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT thread_id, MAX(checkpoint_id) "
        "FROM checkpoints GROUP BY thread_id "
        "ORDER BY MAX(checkpoint_id) DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    saver = make_checkpointer(db_path)
    sessions = []
    for thread_id, _ in rows:
        tup = saver.get_tuple({"configurable": {"thread_id": thread_id}})
        messages = (
            tup.checkpoint.get("channel_values", {}).get("messages", [])
            if tup else []
        )
        title = next(
            (m.content for m in messages if isinstance(m, HumanMessage)),
            "(空会话)",
        )
        sessions.append({"thread_id": thread_id, "title": str(title)[:30]})
    return sessions
