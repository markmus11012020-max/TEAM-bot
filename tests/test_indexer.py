"""Тесты фонового индексатора: батч, fallback по одному, счётчик попыток."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bot.indexer import MessageIndexer
from storage.db import Database
from storage.repository import Repository


class FakeMemory:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []
        self.poison: set[int] = set()

    def index_messages(self, records):
        ids = [record.message_id for record in records]
        self.calls.append(ids)
        if len(records) > 1:
            raise RuntimeError("batch failed")
        if ids[0] in self.poison:
            raise RuntimeError("poison message")
        return len(records)


def make_settings():
    return SimpleNamespace(
        index_batch_size=10,
        index_max_attempts=2,
        index_poll_seconds=0.01,
    )


def add_message(repo: Repository, message_id: int) -> None:
    repo.add_message(
        chat_id=-1001,
        chat_title="Чат",
        message_id=message_id,
        thread_id=0,
        user_id=1,
        author="Иван",
        username="ivan",
        ts=1_700_000_000 + message_id,
        dt="14.11.2023 22:13",
        text=f"текст {message_id}",
        content_type="text",
        permalink="",
        reply_to_message_id=0,
        session_id=None,
    )


@pytest.fixture()
def repo(tmp_path):
    database = Database(tmp_path / "indexer.sqlite3")
    yield Repository(database)
    database.close()


def test_batch_failure_falls_back_to_single_messages(repo):
    for message_id in (1, 2, 3):
        add_message(repo, message_id)

    memory = FakeMemory()
    indexer = MessageIndexer(make_settings(), repo, memory)

    assert indexer.drain() == 3
    assert memory.calls[0] == [1, 2, 3]
    assert memory.calls[1:] == [[1], [2], [3]]
    assert repo.count_pending(max_attempts=2) == 0
    assert repo.count_indexed(-1001) == 3


def test_poison_message_is_retried_until_max_attempts(repo):
    add_message(repo, 1)
    add_message(repo, 99)

    memory = FakeMemory()
    memory.poison.add(99)
    indexer = MessageIndexer(make_settings(), repo, memory)

    indexer.drain()
    assert repo.count_indexed(-1001) == 1
    assert repo.count_pending(max_attempts=2) == 1

    indexer.drain()
    # после двух попыток «отравленное» сообщение больше не выбирается
    assert repo.count_pending(max_attempts=2) == 0
    assert repo.count_indexed(-1001) == 1
