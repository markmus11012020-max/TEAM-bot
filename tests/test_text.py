"""Тесты обработки текста: транскрипт, JSON итогов, plain-версия."""

from __future__ import annotations

import pytest

from rag.text import (
    SummaryParseError,
    build_plain_summary,
    build_transcript,
    normalize_payload,
    parse_summary_json,
    participants,
    truncate,
)
from storage.models import MessageRecord


def make_record(**overrides) -> MessageRecord:
    params = dict(
        id=1,
        chat_id=-1001,
        chat_title="Чат",
        message_id=10,
        thread_id=0,
        user_id=1,
        author="Иван",
        username="ivan",
        ts=1_700_000_000,
        dt="14.11.2023 22:13",
        text="Привет",
        content_type="text",
        permalink="",
        reply_to_message_id=0,
        session_id=None,
        indexed=True,
        attempts=0,
    )
    params.update(overrides)
    return MessageRecord(**params)


def test_truncate():
    assert truncate("abc", 10) == "abc"
    assert truncate("abcdef", 4) == "abc…"
    assert truncate("", 5) == ""


def test_parse_summary_json_plain():
    payload = parse_summary_json(
        '{"summary": "ok", "discussion_type": "decision", '
        '"key_points": ["a", 1, ""], "decisions": [{"decision": "d"}]}'
    )
    assert payload["summary"] == "ok"
    assert payload["discussion_type"] == "decision"
    assert payload["key_points"] == ["a", "1"]
    assert payload["decisions"] == [
        {"decision": "d", "rationale": "", "owner": "", "deadline": ""}
    ]


def test_parse_summary_json_code_fence_and_text():
    raw = 'Вот итоги:\n```json\n{"summary": "ok"}\n```\nГотово.'
    assert parse_summary_json(raw)["summary"] == "ok"


def test_parse_summary_json_invalid():
    with pytest.raises(SummaryParseError):
        parse_summary_json("совсем не json")
    with pytest.raises(SummaryParseError):
        parse_summary_json("")


def test_normalize_payload_unknown_type_and_shapes():
    payload = normalize_payload(
        {
            "discussion_type": "что-то",
            "disputes": [
                {"topic": "Тема", "positions": [{"author": "Иван", "position": "за"}]},
                "мусор",
            ],
            "action_items": [{"task": "сделать", "owner": None, "deadline": 5}],
            "open_questions": "не список",
        }
    )
    assert payload["discussion_type"] == "other"
    assert payload["disputes"][0]["topic"] == "Тема"
    assert payload["action_items"] == [
        {"task": "сделать", "owner": "", "deadline": "5"}
    ]
    assert payload["open_questions"] == []


def test_build_transcript_with_reply_quote():
    first = make_record(message_id=1, text="Релиз в пятницу", author="Иван")
    second = make_record(
        message_id=2,
        text="Не успеем",
        author="Пётр",
        username="petr",
        reply_to_message_id=1,
    )
    transcript = build_transcript([first, second], max_chars_per_message=100)

    assert "[14.11.2023 22:13] Иван (@ivan):" in transcript
    assert "Релиз в пятницу" in transcript
    assert "Пётр (@petr):" in transcript
    assert "↳ ответ на «Релиз в пятницу» (Иван)" in transcript


def test_build_transcript_truncates_long_messages():
    record = make_record(text="я" * 50)
    transcript = build_transcript([record], max_chars_per_message=10)
    assert "я" * 10 not in transcript
    assert "…" in transcript


def test_participants_unique():
    records = [
        make_record(author="Иван"),
        make_record(author="Пётр"),
        make_record(author="Иван"),
        make_record(author="", username="anna"),
    ]
    assert participants(records) == ["Иван", "Пётр", "@anna"]


def test_build_plain_summary_contains_sections():
    payload = normalize_payload(
        {
            "summary": "Обсудили релиз",
            "discussion_type": "dispute",
            "key_points": ["сроки под угрозой"],
            "disputes": [
                {
                    "topic": "Дата релиза",
                    "positions": [{"author": "Иван", "position": "пятница"}],
                    "assessment": "рискованно",
                    "recommended_resolution": "перенести на вторник",
                }
            ],
            "decisions": [
                {
                    "decision": "перенести релиз",
                    "rationale": "не готовы тесты",
                    "owner": "Пётр",
                    "deadline": "вторник",
                }
            ],
            "action_items": [{"task": "дописать тесты", "owner": "Иван", "deadline": "пн"}],
            "open_questions": ["кто дежурит?"],
            "risks": ["срыв демо"],
            "assistant_opinion": "перенос оправдан",
        }
    )
    text = build_plain_summary(
        payload,
        chat_title="Чат",
        period="01.01 — 02.01",
        message_count=5,
        participant_names=["Иван", "Пётр"],
    )
    for expected in [
        "Итоги обсуждения «Чат»",
        "Сообщений: 5",
        "Споры и позиции",
        "перенести релиз",
        "Ответственный: Пётр",
        "Задачи",
        "Открытые вопросы",
        "Риски",
        "Мнение ассистента",
    ]:
        assert expected in text
