"""Репозиторий поверх SQLite: сообщения, сессии, итоги."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Sequence

from storage.db import Database
from storage.models import MessageRecord, SessionRecord, SummaryRecord

_MESSAGE_COLUMNS = """
    id, chat_id, chat_title, message_id, thread_id, user_id, author, username,
    ts, dt, text, content_type, permalink, reply_to_message_id, session_id,
    indexed, attempts
"""

_SESSION_COLUMNS = """
    id, chat_id, chat_title, started_by, started_by_name, started_at,
    ended_at, status, message_count
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_message(row: sqlite3.Row) -> MessageRecord:
    return MessageRecord(
        id=row["id"],
        chat_id=row["chat_id"],
        chat_title=row["chat_title"],
        message_id=row["message_id"],
        thread_id=row["thread_id"],
        user_id=row["user_id"],
        author=row["author"],
        username=row["username"],
        ts=row["ts"],
        dt=row["dt"],
        text=row["text"],
        content_type=row["content_type"],
        permalink=row["permalink"],
        reply_to_message_id=row["reply_to_message_id"],
        session_id=row["session_id"],
        indexed=bool(row["indexed"]),
        attempts=row["attempts"],
    )


def _to_session(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        id=row["id"],
        chat_id=row["chat_id"],
        chat_title=row["chat_title"],
        started_by=row["started_by"],
        started_by_name=row["started_by_name"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        status=row["status"],
        message_count=row["message_count"],
    )


class Repository:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------ messages

    def add_message(
        self,
        *,
        chat_id: int,
        chat_title: str,
        message_id: int,
        thread_id: int,
        user_id: int,
        author: str,
        username: str,
        ts: int,
        dt: str,
        text: str,
        content_type: str,
        permalink: str,
        reply_to_message_id: int,
        session_id: str | None,
    ) -> bool:
        cursor = self.db.execute(
            """
            INSERT OR IGNORE INTO messages (
                chat_id, chat_title, message_id, thread_id, user_id, author, username,
                ts, dt, text, content_type, permalink, reply_to_message_id, session_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chat_id,
                chat_title,
                message_id,
                thread_id,
                user_id,
                author,
                username,
                ts,
                dt,
                text,
                content_type,
                permalink,
                reply_to_message_id,
                session_id,
            ),
        )
        return cursor.rowcount > 0

    def get_pending(self, limit: int, max_attempts: int) -> list[MessageRecord]:
        rows = self.db.query(
            f"""
            SELECT {_MESSAGE_COLUMNS}
            FROM messages
            WHERE indexed = 0 AND attempts < ?
            ORDER BY id
            LIMIT ?
            """,
            (max_attempts, limit),
        )
        return [_to_message(row) for row in rows]

    def count_pending(self, max_attempts: int) -> int:
        row = self.db.query_one(
            "SELECT COUNT(*) AS n FROM messages WHERE indexed = 0 AND attempts < ?",
            (max_attempts,),
        )
        return int(row["n"]) if row else 0

    def mark_indexed(self, message_row_ids: Sequence[int]) -> None:
        if not message_row_ids:
            return
        placeholders = ", ".join("?" for _ in message_row_ids)
        self.db.execute(
            f"UPDATE messages SET indexed = 1, last_error = NULL WHERE id IN ({placeholders})",
            tuple(message_row_ids),
        )

    def mark_index_failed(self, message_row_ids: Sequence[int], error: str) -> None:
        if not message_row_ids:
            return
        placeholders = ", ".join("?" for _ in message_row_ids)
        self.db.execute(
            f"""
            UPDATE messages
            SET attempts = attempts + 1, last_error = ?
            WHERE id IN ({placeholders})
            """,
            (error[:1000], *message_row_ids),
        )

    def get_message(self, chat_id: int, message_id: int) -> MessageRecord | None:
        row = self.db.query_one(
            f"SELECT {_MESSAGE_COLUMNS} FROM messages WHERE chat_id = ? AND message_id = ?",
            (chat_id, message_id),
        )
        return _to_message(row) if row else None

    def count_messages(self, chat_id: int) -> int:
        row = self.db.query_one(
            "SELECT COUNT(*) AS n FROM messages WHERE chat_id = ?", (chat_id,)
        )
        return int(row["n"]) if row else 0

    def count_indexed(self, chat_id: int) -> int:
        row = self.db.query_one(
            "SELECT COUNT(*) AS n FROM messages WHERE chat_id = ? AND indexed = 1",
            (chat_id,),
        )
        return int(row["n"]) if row else 0

    def get_recent_messages(self, chat_id: int, limit: int) -> list[MessageRecord]:
        rows = self.db.query(
            f"""
            SELECT {_MESSAGE_COLUMNS}
            FROM messages
            WHERE chat_id = ?
            ORDER BY ts DESC, id DESC
            LIMIT ?
            """,
            (chat_id, limit),
        )
        return [_to_message(row) for row in reversed(rows)]

    # ------------------------------------------------------------------ sessions

    def start_session(
        self,
        *,
        chat_id: int,
        chat_title: str,
        user_id: int,
        user_name: str,
    ) -> tuple[SessionRecord, bool]:
        active = self.get_active_session(chat_id)
        if active:
            return active, False

        session_id = uuid.uuid4().hex[:12]
        try:
            self.db.execute(
                """
                INSERT INTO sessions (
                    id, chat_id, chat_title, started_by, started_by_name, started_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, 'active')
                """,
                (session_id, chat_id, chat_title, user_id, user_name, now_iso()),
            )
        except sqlite3.IntegrityError:
            # Гонка двух одновременных /start — возвращаем уже созданную сессию.
            active = self.get_active_session(chat_id)
            if active:
                return active, False
            raise

        session = self.get_session(session_id)
        if session is None:  # pragma: no cover - защита от неверного состояния БД
            raise RuntimeError("Не удалось создать сессию")
        return session, True

    def get_session(self, session_id: str) -> SessionRecord | None:
        row = self.db.query_one(
            f"SELECT {_SESSION_COLUMNS} FROM sessions WHERE id = ?",
            (session_id,),
        )
        return _to_session(row) if row else None

    def get_active_session(self, chat_id: int) -> SessionRecord | None:
        row = self.db.query_one(
            f"SELECT {_SESSION_COLUMNS} FROM sessions WHERE chat_id = ? AND status = 'active'",
            (chat_id,),
        )
        return _to_session(row) if row else None

    def get_active_sessions(self) -> list[SessionRecord]:
        rows = self.db.query(
            f"SELECT {_SESSION_COLUMNS} FROM sessions WHERE status = 'active'"
        )
        return [_to_session(row) for row in rows]

    def set_session_status(self, session_id: str, status: str) -> None:
        self.db.execute("UPDATE sessions SET status = ? WHERE id = ?", (status, session_id))

    def close_session(self, session_id: str, status: str = "completed") -> bool:
        """Переводит активную сессию в новый статус.

        Возвращает False, если сессия уже не активна (например, её параллельно
        закрыл watchdog или второй /stop).
        """

        cursor = self.db.execute(
            """
            UPDATE sessions
            SET status = ?,
                ended_at = ?,
                message_count = (
                    SELECT COUNT(*) FROM messages WHERE session_id = sessions.id
                )
            WHERE id = ? AND status = 'active'
            """,
            (status, now_iso(), session_id),
        )
        return cursor.rowcount > 0

    def get_session_messages(self, session_id: str) -> list[MessageRecord]:
        rows = self.db.query(
            f"""
            SELECT {_MESSAGE_COLUMNS}
            FROM messages
            WHERE session_id = ?
            ORDER BY ts, id
            """,
            (session_id,),
        )
        return [_to_message(row) for row in rows]

    def count_session_messages(self, session_id: str) -> int:
        row = self.db.query_one(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?",
            (session_id,),
        )
        return int(row["n"]) if row else 0

    def get_last_session_message_ts(self, session_id: str) -> int | None:
        row = self.db.query_one(
            "SELECT MAX(ts) AS ts FROM messages WHERE session_id = ?",
            (session_id,),
        )
        if row is None or row["ts"] is None:
            return None
        return int(row["ts"])

    # ------------------------------------------------------------------ summaries

    def save_summary(self, record: SummaryRecord) -> None:
        self.db.execute(
            """
            INSERT OR REPLACE INTO summaries (
                id, session_id, chat_id, text, payload_json, doc_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.session_id,
                record.chat_id,
                record.text,
                record.payload_json,
                record.doc_id,
                record.created_at,
            ),
        )

    def get_summary(self, session_id: str) -> SummaryRecord | None:
        row = self.db.query_one(
            "SELECT * FROM summaries WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        )
        if row is None:
            return None
        return SummaryRecord(
            id=row["id"],
            session_id=row["session_id"],
            chat_id=row["chat_id"],
            text=row["text"],
            payload_json=row["payload_json"],
            doc_id=row["doc_id"],
            created_at=row["created_at"],
        )
