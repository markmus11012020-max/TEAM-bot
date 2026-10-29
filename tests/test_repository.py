"""Тесты репозитория: сообщения, сессии, итоги."""

from __future__ import annotations

import pytest

from storage.db import Database
from storage.models import SummaryRecord
from storage.repository import Repository


@pytest.fixture()
def repo(tmp_path):
    database = Database(tmp_path / "test.sqlite3")
    yield Repository(database)
    database.close()


def add_message(repo: Repository, message_id: int, **overrides):
    params = dict(
        chat_id=-1001,
        chat_title="Тестовый чат",
        message_id=message_id,
        thread_id=0,
        user_id=7,
        author="Иван",
        username="ivan",
        ts=1_700_000_000 + message_id,
        dt="14.11.2023 22:13",
        text=f"сообщение {message_id}",
        content_type="text",
        permalink="",
        reply_to_message_id=0,
        session_id=None,
    )
    params.update(overrides)
    return repo.add_message(**params)


def test_add_message_deduplicates(repo):
    assert add_message(repo, 1) is True
    assert add_message(repo, 1) is False
    assert repo.count_messages(-1001) == 1


def test_pending_and_attempts(repo):
    add_message(repo, 1)
    add_message(repo, 2)

    pending = repo.get_pending(limit=10, max_attempts=3)
    assert [record.message_id for record in pending] == [1, 2]

    repo.mark_indexed([pending[0].id])
    assert repo.count_indexed(-1001) == 1
    assert repo.count_pending(max_attempts=3) == 1

    repo.mark_index_failed([pending[1].id], "boom")
    assert repo.count_pending(max_attempts=1) == 0
    assert repo.count_pending(max_attempts=2) == 1


def test_recent_messages_sorted_ascending(repo):
    add_message(repo, 1, ts=100)
    add_message(repo, 2, ts=300)
    add_message(repo, 3, ts=200)

    recent = repo.get_recent_messages(-1001, limit=2)
    assert [record.message_id for record in recent] == [3, 2]


def test_session_lifecycle(repo):
    session, created = repo.start_session(
        chat_id=-1001,
        chat_title="Тестовый чат",
        user_id=7,
        user_name="Иван",
    )
    assert created is True
    assert session.status == "active"

    same, created_again = repo.start_session(
        chat_id=-1001,
        chat_title="Тестовый чат",
        user_id=8,
        user_name="Пётр",
    )
    assert created_again is False
    assert same.id == session.id

    add_message(repo, 1, session_id=session.id)
    add_message(repo, 2, session_id=session.id)
    add_message(repo, 3)  # вне сессии

    assert repo.count_session_messages(session.id) == 2
    assert repo.get_last_session_message_ts(session.id) == 1_700_000_002

    assert repo.close_session(session.id, "summarizing") is True
    assert repo.close_session(session.id, "summarizing") is False

    closed = repo.get_session(session.id)
    assert closed is not None
    assert closed.status == "summarizing"
    assert closed.ended_at is not None
    assert closed.message_count == 2

    repo.set_session_status(session.id, "completed")
    assert repo.get_session(session.id).status == "completed"
    assert repo.get_active_session(-1001) is None

    # После закрытия можно начать новую сессию.
    new_session, created_new = repo.start_session(
        chat_id=-1001,
        chat_title="Тестовый чат",
        user_id=7,
        user_name="Иван",
    )
    assert created_new is True
    assert new_session.id != session.id


def test_session_messages_ordered(repo):
    session, _ = repo.start_session(
        chat_id=-1001, chat_title="Чат", user_id=7, user_name="Иван"
    )
    add_message(repo, 1, ts=300, session_id=session.id)
    add_message(repo, 2, ts=100, session_id=session.id)
    add_message(repo, 3, ts=200, session_id=session.id)

    messages = repo.get_session_messages(session.id)
    assert [record.message_id for record in messages] == [2, 3, 1]


def test_summary_roundtrip(repo):
    record = SummaryRecord(
        id="s1",
        session_id="session-1",
        chat_id=-1001,
        text="Итоги",
        payload_json='{"summary": "ok"}',
        doc_id="summary:session-1",
        created_at="2026-10-29T10:00:00+00:00",
    )
    repo.save_summary(record)

    loaded = repo.get_summary("session-1")
    assert loaded is not None
    assert loaded.text == "Итоги"
    assert loaded.doc_id == "summary:session-1"
    assert repo.get_summary("unknown") is None
