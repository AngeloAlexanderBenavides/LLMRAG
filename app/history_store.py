import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HISTORY_DB_PATH = os.path.join(PROJECT_ROOT, "chat_history.db")


def _get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(HISTORY_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_history_store() -> None:
    with _get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                chat_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                question_type TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(chat_id) REFERENCES chats(chat_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                query TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(chat_id) REFERENCES chats(chat_id)
            )
            """
        )


def ensure_chat(chat_id: str, title: Optional[str] = None) -> None:
    initialize_history_store()
    now = datetime.utcnow().isoformat()
    chat_title = title or "Nuevo chat"
    with _get_connection() as connection:
        connection.execute(
            """
            INSERT INTO chats (chat_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                title = COALESCE(excluded.title, chats.title),
                updated_at = excluded.updated_at
            """,
            (chat_id, chat_title, now, now),
        )


def touch_chat(chat_id: str) -> None:
    initialize_history_store()
    now = datetime.utcnow().isoformat()
    with _get_connection() as connection:
        connection.execute(
            "UPDATE chats SET updated_at = ? WHERE chat_id = ?",
            (now, chat_id),
        )


def set_chat_title(chat_id: str, title: str) -> None:
    initialize_history_store()
    now = datetime.utcnow().isoformat()
    with _get_connection() as connection:
        connection.execute(
            """
            INSERT INTO chats (chat_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                title = excluded.title,
                updated_at = excluded.updated_at
            """,
            (chat_id, title, now, now),
        )


def save_message(
    chat_id: str,
    role: str,
    content: str,
    source: Optional[str] = None,
    question_type: Optional[str] = None,
) -> None:
    initialize_history_store()
    ensure_chat(chat_id)
    now = datetime.utcnow().isoformat()
    with _get_connection() as connection:
        connection.execute(
            """
            INSERT INTO messages (chat_id, role, content, source, question_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (chat_id, role, content, source, question_type, now),
        )
        connection.execute(
            "UPDATE chats SET updated_at = ? WHERE chat_id = ?",
            (now, chat_id),
        )


def save_memory_entry(chat_id: str, query: str, content: str, source: str) -> None:
    initialize_history_store()
    ensure_chat(chat_id)
    now = datetime.utcnow().isoformat()
    with _get_connection() as connection:
        connection.execute(
            """
            INSERT INTO memory_entries (chat_id, query, content, source, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (chat_id, query, content, source, now),
        )


def list_chats() -> List[Dict[str, Any]]:
    initialize_history_store()
    with _get_connection() as connection:
        rows = connection.execute(
            """
            SELECT chat_id, title, created_at, updated_at
            FROM chats
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_chat_messages(chat_id: str) -> List[Dict[str, Any]]:
    initialize_history_store()
    with _get_connection() as connection:
        rows = connection.execute(
            """
            SELECT role, content, source, question_type, created_at
            FROM messages
            WHERE chat_id = ?
            ORDER BY id ASC
            """,
            (chat_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_memory_entries(chat_id: str) -> List[Dict[str, Any]]:
    initialize_history_store()
    with _get_connection() as connection:
        rows = connection.execute(
            """
            SELECT query, content, source, created_at
            FROM memory_entries
            WHERE chat_id = ?
            ORDER BY id DESC
            """,
            (chat_id,),
        ).fetchall()
    return [dict(row) for row in rows]
