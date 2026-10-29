"""Проверка шаблонов промптов через настоящий ChatPromptBuilder Haystack."""

from __future__ import annotations

import pytest
from haystack import Document

from config import load_settings
from rag.pipelines import build_notes_pipeline, build_qa_pipeline, build_summary_pipelines


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "db.sqlite3"))
    return load_settings()


def test_qa_prompt_renders_documents(settings):
    pipeline = build_qa_pipeline(settings)
    builder = pipeline.get_component("prompt_builder")
    document = Document(
        content="Релиз переносим на вторник",
        meta={
            "kind": "message",
            "chat_id": -1001,
            "chat_title": "Команда",
            "message_id": 1,
            "dt": "29.10.2026 14:05",
            "author": "Иван",
            "username": "ivan",
            "permalink": "https://t.me/c/1/1",
        },
    )
    result = builder.run(documents=[document], question="Когда релиз?")
    messages = result["prompt"]

    assert len(messages) == 2
    user_text = messages[1].text
    assert "29.10.2026 14:05" in user_text
    assert "Иван (@ivan)" in user_text
    assert "Релиз переносим на вторник" in user_text
    assert "Когда релиз?" in user_text


def test_qa_prompt_without_username(settings):
    pipeline = build_qa_pipeline(settings)
    builder = pipeline.get_component("prompt_builder")
    document = Document(
        content="Итоги сессии",
        meta={
            "kind": "summary",
            "chat_title": "Команда",
            "dt": "29.10.2026 15:00",
            "author": "Итоги сессии",
            "username": "",
            "permalink": "",
        },
    )
    result = builder.run(documents=[document], question="Что решили?")
    assert "(@)" not in result["prompt"][1].text


def test_summary_prompt_renders(settings):
    structured, plain = build_summary_pipelines(settings)
    variables = {
        "chat_title": "Команда",
        "period": "29.10.2026 10:00 — 29.10.2026 11:00",
        "message_count": 3,
        "participants": "Иван, Пётр",
        "transcript": "[29.10.2026 10:05] Иван: привет",
    }
    result = structured.get_component("prompt_builder").run(**variables)
    text = result["prompt"][1].text
    assert "Команда" in text
    assert "Сообщений: 3" in text
    assert "Иван, Пётр" in text

    plain_result = plain.get_component("prompt_builder").run(**variables)
    assert "Кратко" in plain_result["prompt"][0].text


def test_notes_prompt_renders(settings):
    pipeline = build_notes_pipeline(settings)
    result = pipeline.get_component("prompt_builder").run(
        chat_title="Команда",
        transcript="[10:00] Иван: привет",
        part_index=1,
        part_count=2,
    )
    text = result["prompt"][1].text
    assert "Фрагмент 1 из 2" in text
    assert "[10:00] Иван: привет" in text
