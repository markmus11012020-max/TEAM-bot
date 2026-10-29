"""Сервис «памяти команды»: индексация, RAG-ответы, итоги сессий."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from config import Settings
from rag import pipelines
from rag.text import (
    build_plain_summary,
    build_transcript,
    normalize_payload,
    parse_summary_json,
    participants as collect_participants,
    truncate,
)
from storage.models import MessageRecord, SessionRecord
from storage.repository import Repository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SummaryResult:
    payload: dict[str, Any]
    plain_text: str
    transcript_chars: int
    parts: int


class TeamMemory:
    def __init__(
        self,
        settings: Settings,
        store: PineconeDocumentStore,
        repository: Repository,
    ) -> None:
        self.settings = settings
        self.store = store
        self.repository = repository

        self.index_pipeline = pipelines.build_index_pipeline(settings, store)
        self.retrieval_pipeline = pipelines.build_retrieval_pipeline(settings, store)
        self.qa_pipeline = pipelines.build_qa_pipeline(settings)
        self.notes_pipeline = pipelines.build_notes_pipeline(settings)
        self.summary_pipeline, self.summary_plain_pipeline = pipelines.build_summary_pipelines(
            settings
        )

        self._splitter = DocumentSplitter(split_by="word", split_length=300, split_overlap=0)
        self._api_semaphore = threading.BoundedSemaphore(2)

    # --------------------------------------------------------------- indexing

    def index_messages(self, records: Sequence[MessageRecord]) -> int:
        documents: list[Document] = []
        for record in records:
            documents.extend(self._records_to_documents(record))

        if not documents:
            return 0

        with self._api_semaphore:
            self.index_pipeline.run({"embedder": {"documents": documents}})

        logger.debug(
            "Проиндексировано сообщений=%s, чанков=%s", len(records), len(documents)
        )
        return len(documents)

    def _records_to_documents(self, record: MessageRecord) -> list[Document]:
        text = (record.text or "").strip()
        if not text:
            return []

        meta = self._message_meta(record)

        if len(text) <= self.settings.message_chunk_chars:
            return [
                Document(
                    id=f"tg:{record.chat_id}:{record.message_id}",
                    content=text,
                    meta=meta,
                )
            ]

        split_result = self._splitter.run(documents=[Document(content=text, meta=meta)])
        documents: list[Document] = []
        for chunk in split_result["documents"]:
            split_id = int(chunk.meta.get("split_id", 0) or 0)
            chunk_meta = {**chunk.meta, "split_id": split_id}
            documents.append(
                Document(
                    id=f"tg:{record.chat_id}:{record.message_id}:{split_id}",
                    content=chunk.content or "",
                    meta=chunk_meta,
                )
            )
        return documents

    @staticmethod
    def _message_meta(record: MessageRecord) -> dict[str, Any]:
        return {
            "kind": "message",
            "chat_id": record.chat_id,
            "chat_title": record.chat_title,
            "message_id": record.message_id,
            "thread_id": record.thread_id,
            "user_id": record.user_id,
            "author": record.author,
            "username": record.username,
            "ts": record.ts,
            "dt": record.dt,
            "content_type": record.content_type,
            "permalink": record.permalink,
            "reply_to_message_id": record.reply_to_message_id,
            "session_id": record.session_id or "",
        }

    def _recent_document(self, record: MessageRecord) -> Document:
        meta = {**self._message_meta(record), "source": "recent"}
        return Document(
            id=f"recent:{record.chat_id}:{record.message_id}",
            content=truncate(record.text, 4000),
            meta=meta,
        )

    # ------------------------------------------------------------------- ask

    def ask(self, question: str, chat_id: int, chat_title: str) -> str:
        question = (question or "").strip()
        if not question:
            return "Задайте вопрос: например, «что решили по релизу?»."

        retrieved: list[Document] = []
        try:
            with self._api_semaphore:
                result = self.retrieval_pipeline.run(
                    {
                        "text_embedder": {"text": question},
                        "retriever": {
                            "filters": self._filters(chat_id),
                            "top_k": self.settings.retrieval_top_k,
                        },
                    }
                )
            retrieved = result["retriever"]["documents"]
        except Exception:
            logger.exception("Не удалось выполнить векторный поиск, использую последние сообщения")

        recent_records = self.repository.get_recent_messages(
            chat_id, self.settings.recent_messages
        )
        recent = [self._recent_document(record) for record in recent_records]
        documents = self._merge_documents(retrieved, recent)

        if not documents:
            return (
                "В памяти пока нет сообщений по этому чату. "
                "Напишите что-нибудь в чат и повторите вопрос."
            )

        answer = self._run_chat(
            self.qa_pipeline,
            {"documents": documents, "question": question},
        )
        return answer or "Не удалось сформулировать ответ."

    def _filters(self, chat_id: int) -> dict[str, Any]:
        conditions: list[dict[str, Any]] = [
            {"field": "meta.kind", "operator": "in", "value": ["message", "summary"]}
        ]
        if self.settings.search_scope == "chat":
            conditions.insert(
                0,
                {"field": "meta.chat_id", "operator": "==", "value": chat_id},
            )
        return {"operator": "AND", "conditions": conditions}

    @staticmethod
    def _doc_key(document: Document) -> tuple:
        meta = document.meta or {}
        if meta.get("kind") == "summary":
            return ("summary", meta.get("session_id") or document.id)
        return (
            "message",
            meta.get("chat_id"),
            meta.get("message_id"),
            meta.get("split_id", 0),
        )

    def _merge_documents(
        self, retrieved: Iterable[Document], recent: Iterable[Document]
    ) -> list[Document]:
        merged: list[Document] = []
        seen: set[tuple] = set()
        for document in [*retrieved, *recent]:
            key = self._doc_key(document)
            if key in seen:
                continue
            seen.add(key)
            merged.append(document)
        return merged

    # --------------------------------------------------------------- summary

    def summarize(
        self, session: SessionRecord, messages: Sequence[MessageRecord]
    ) -> SummaryResult:
        if not messages:
            raise ValueError("В сессии нет сообщений")

        transcript = build_transcript(
            messages,
            max_chars_per_message=self.settings.summary_message_max_chars,
        )
        transcript_chars = len(transcript)
        parts = 1

        if transcript_chars > self.settings.summary_chunk_chars:
            transcript, parts = self._summarize_in_parts(transcript, session.chat_title)

        transcript = truncate(transcript, self.settings.summary_max_chars)

        participant_names = collect_participants(messages)
        period = self.period(session)
        variables = {
            "chat_title": session.chat_title or "рабочий чат",
            "period": period,
            "message_count": len(messages),
            "participants": ", ".join(participant_names) or "—",
            "transcript": transcript,
        }

        payload: dict[str, Any] | None = None
        if self.settings.structured_summary:
            try:
                raw = self._run_chat(self.summary_pipeline, variables)
                payload = parse_summary_json(raw)
            except Exception as exc:  # noqa: BLE001 - любая ошибка API/парсинга -> fallback
                logger.warning(
                    "Структурированный ответ не получен (%s), повторяю обычным текстом",
                    exc,
                )

        if payload is None:
            raw = self._run_chat(self.summary_plain_pipeline, variables)
            payload = normalize_payload({"summary": raw, "discussion_type": "other"})

        plain_text = build_plain_summary(
            payload,
            chat_title=session.chat_title or "рабочий чат",
            period=period,
            message_count=len(messages),
            participant_names=participant_names,
        )

        return SummaryResult(
            payload=payload,
            plain_text=plain_text,
            transcript_chars=transcript_chars,
            parts=parts,
        )

    def _summarize_in_parts(self, transcript: str, chat_title: str) -> tuple[str, int]:
        parts = self._split_by_lines(transcript, self.settings.summary_chunk_chars)
        logger.info("Стенограмма длинная, сжимаю по частям: %s", len(parts))

        notes: list[str] = []
        for index, part in enumerate(parts, start=1):
            note = self._run_chat(
                self.notes_pipeline,
                {
                    "chat_title": chat_title or "рабочий чат",
                    "transcript": part,
                    "part_index": index,
                    "part_count": len(parts),
                },
            )
            notes.append(f"--- Часть {index} ---\n{note}")

        combined = "\n\n".join(notes)
        return truncate(combined, self.settings.summary_max_chars), len(parts)

    @staticmethod
    def _split_by_lines(text: str, chunk_chars: int) -> list[str]:
        parts: list[str] = []
        current: list[str] = []
        size = 0
        for line in text.splitlines():
            if current and size + len(line) + 1 > chunk_chars:
                parts.append("\n".join(current))
                current = []
                size = 0
            current.append(line)
            size += len(line) + 1
        if current:
            parts.append("\n".join(current))
        return parts or [text]

    def period(self, session: SessionRecord) -> str:
        start = self._format_iso(session.started_at)
        end = self._format_iso(session.ended_at) if session.ended_at else self._now_local()
        return f"{start} — {end}"

    def _format_iso(self, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return value
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(self.settings.tz).strftime("%d.%m.%Y %H:%M")

    def _now_local(self) -> str:
        return datetime.now(self.settings.tz).strftime("%d.%m.%Y %H:%M")

    def _run_chat(self, pipeline, variables: dict[str, Any]) -> str:
        with self._api_semaphore:
            result = pipeline.run({"prompt_builder": variables})
        replies = result.get("llm", {}).get("replies") or []
        if not replies:
            raise RuntimeError("Модель не вернула ответ")
        return (replies[0].text or "").strip()

    # ---------------------------------------------------------- summary index

    def index_summary(
        self,
        session: SessionRecord,
        result: SummaryResult,
        participant_names: Sequence[str],
        message_count: int,
    ) -> str:
        ts = int(time.time())
        dt = datetime.fromtimestamp(ts, self.settings.tz).strftime("%d.%m.%Y %H:%M")
        document = Document(
            id=f"summary:{session.id}",
            content=truncate(result.plain_text, 20000),
            meta={
                "kind": "summary",
                "chat_id": session.chat_id,
                "chat_title": session.chat_title,
                "message_id": 0,
                "thread_id": 0,
                "user_id": session.started_by,
                "author": "Итоги сессии",
                "username": "",
                "ts": ts,
                "dt": dt,
                "content_type": "summary",
                "permalink": "",
                "reply_to_message_id": 0,
                "session_id": session.id,
                "discussion_type": result.payload.get("discussion_type", "other"),
                "participants": list(participant_names),
                "message_count": message_count,
            },
        )
        with self._api_semaphore:
            self.index_pipeline.run({"embedder": {"documents": [document]}})
        return document.id
