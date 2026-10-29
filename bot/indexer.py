"""Фоновый индексатор: забирает сообщения из SQLite и пишет их в Pinecone."""

from __future__ import annotations

import logging
import threading

from config import Settings
from rag.service import TeamMemory
from storage.models import MessageRecord
from storage.repository import Repository

logger = logging.getLogger(__name__)


class MessageIndexer(threading.Thread):
    """Один воркер: батчи, при ошибке — повтор по одному сообщению."""

    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        memory: TeamMemory,
    ) -> None:
        super().__init__(name="message-indexer", daemon=True)
        self.settings = settings
        self.repository = repository
        self.memory = memory
        self._stop = threading.Event()
        self._had_failures = False

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        logger.info("Индексатор запущен (batch=%s)", self.settings.index_batch_size)
        while not self._stop.is_set():
            try:
                processed = self.drain()
            except Exception:  # noqa: BLE001 - воркер не должен умирать
                logger.exception("Сбой индексатора")
                processed = 0

            if processed == 0 or self._had_failures:
                self._stop.wait(self.settings.index_poll_seconds)

        logger.info("Индексатор остановлен")

    def drain(self) -> int:
        """Индексирует всё, что накопилось, и возвращает число сообщений.

        Сообщение, на котором упал батч, повторяется по одному, но в рамках
        одного прохода — не более одного раза: иначе «отравленное» сообщение
        съест все попытки в tight-loop без паузы.
        """

        total = 0
        failures = False
        seen: set[int] = set()
        while not self._stop.is_set():
            candidates = self.repository.get_pending(
                self.settings.index_batch_size * 2,
                self.settings.index_max_attempts,
            )
            records = [
                record
                for record in candidates
                if record.id not in seen
            ][: self.settings.index_batch_size]
            if not records:
                break
            seen.update(record.id for record in records)
            indexed, failed = self._index_batch(records)
            total += indexed
            failures = failures or failed > 0

        self._had_failures = failures
        return total

    def _index_batch(self, records: list[MessageRecord]) -> tuple[int, int]:
        """Возвращает (сколько проиндексировано, сколько упало)."""

        try:
            self.memory.index_messages(records)
        except Exception as exc:  # noqa: BLE001 - падение батча не должно терять сообщения
            logger.warning(
                "Батч из %s сообщений не проиндексирован (%s), повторяю по одному",
                len(records),
                exc,
            )
            indexed = 0
            for record in records:
                if self._stop.is_set():
                    break
                if self._index_single(record):
                    indexed += 1
            return indexed, len(records) - indexed

        self.repository.mark_indexed([record.id for record in records])
        logger.debug("Проиндексирован батч: %s сообщений", len(records))
        return len(records), 0

    def _index_single(self, record: MessageRecord) -> bool:
        try:
            self.memory.index_messages([record])
        except Exception as exc:  # noqa: BLE001 - фиксируем попытку и идём дальше
            logger.warning(
                "Сообщение %s/%s не проиндексировано (попытка %s): %s",
                record.chat_id,
                record.message_id,
                record.attempts + 1,
                exc,
            )
            self.repository.mark_index_failed([record.id], str(exc))
            return False

        self.repository.mark_indexed([record.id])
        return True
