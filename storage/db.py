"""Тонкая обёртка над SQLite: WAL, единый lock, безопасная работа из потоков."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    chat_title TEXT NOT NULL DEFAULT '',
    message_id INTEGER NOT NULL,
    thread_id INTEGER NOT NULL DEFAULT 0,
    user_id INTEGER NOT NULL DEFAULT 0,
    author TEXT NOT NULL DEFAULT '',
    username TEXT NOT NULL DEFAULT '',
    ts INTEGER NOT NULL,
    dt TEXT NOT NULL,
    text TEXT NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'text',
    permalink TEXT NOT NULL DEFAULT '',
    reply_to_message_id INTEGER NOT NULL DEFAULT 0,
    session_id TEXT,
    indexed INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (chat_id, message_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_pending ON messages (indexed, id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, ts, id);
CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages (chat_id, ts);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    chat_title TEXT NOT NULL DEFAULT '',
    started_by INTEGER NOT NULL DEFAULT 0,
    started_by_name TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    message_count INTEGER NOT NULL DEFAULT 0
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_one_active
    ON sessions (chat_id) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS summaries (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    doc_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_summaries_chat ON summaries (chat_id, created_at);
"""


class Database:
    """Единственное соединение с SQLite под общим RLock.

    Бот многопоточный (polling + индексатор + watchdog), поэтому все операции
    сериализуются. Для нагрузки «один чат компании» этого более чем достаточно.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock, self._conn:
            return self._conn.execute(sql, params)

    def executemany(self, sql: str, seq_params: Iterable[Sequence[Any]]) -> sqlite3.Cursor:
        with self._lock, self._conn:
            return self._conn.executemany(sql, seq_params)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
