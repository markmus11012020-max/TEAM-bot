"""Доменные модели хранилища."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MessageRecord:
    id: int
    chat_id: int
    chat_title: str
    message_id: int
    thread_id: int
    user_id: int
    author: str
    username: str
    ts: int
    dt: str
    text: str
    content_type: str
    permalink: str
    reply_to_message_id: int
    session_id: str | None
    indexed: bool
    attempts: int

    @property
    def is_indexed(self) -> bool:
        return self.indexed


@dataclass(frozen=True, slots=True)
class SessionRecord:
    id: str
    chat_id: int
    chat_title: str
    started_by: int
    started_by_name: str
    started_at: str
    ended_at: str | None
    status: str
    message_count: int


@dataclass(frozen=True, slots=True)
class SummaryRecord:
    id: str
    session_id: str
    chat_id: int
    text: str
    payload_json: str
    doc_id: str
    created_at: str
