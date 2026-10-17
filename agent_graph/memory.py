import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = Path(__file__).resolve().parent.parent / "agent_memory.sqlite"


def make_checkpointer(db_path: str | Path = DB_PATH) -> SqliteSaver:
    """同步 SQLite checkpointer;check_same_thread=False 供 Gradio 多线程使用。"""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)
