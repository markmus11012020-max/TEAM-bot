"""Интеграционные тесты обработчиков без Telegram и внешних API."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from bot.handlers import BotApp
from config import load_settings
from rag.service import SummaryResult
from storage.db import Database
from storage.repository import Repository


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str, dict]] = []

    def send_message(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text, kwargs))
        return SimpleNamespace(message_id=len(self.sent))

    def send_chat_action(self, *args, **kwargs):
        return True

    def register_message_handler(self, *args, **kwargs):
        return None


class FakeMemory:
    def period(self, session):
        return "29.10.2026 10:00 — 29.10.2026 11:00"

    def ask(self, question, chat_id, chat_title):
        return "Ответ по истории чата"

    def summarize(self, session, messages):
        payload = {
            "summary": "Обсудили релиз",
            "discussion_type": "decision",
            "key_points": ["релиз в пятницу"],
            "disputes": [],
            "decisions": [
                {
                    "decision": "выпускаем в пятницу",
                    "rationale": "тесты зелёные",
                    "owner": "Пётр",
                    "deadline": "пятница",
                }
            ],
            "action_items": [],
            "open_questions": [],
            "risks": [],
            "assistant_opinion": "план реалистичен",
        }
        return SummaryResult(
            payload=payload,
            plain_text="Итоги: релиз в пятницу",
            transcript_chars=100,
            parts=1,
        )

    def index_summary(self, session, result, participant_names, message_count):
        return f"summary:{session.id}"


def make_message(
    *,
    text: str,
    message_id: int = 1,
    user_id: int = 7,
    chat_id: int = -100,
    chat_type: str = "supergroup",
    is_bot: bool = False,
    entities=None,
    reply_to=None,
    content_type: str = "text",
    date: int | None = None,
):
    return SimpleNamespace(
        text=text,
        caption=None,
        entities=entities or [],
        reply_to_message=reply_to,
        content_type=content_type,
        message_id=message_id,
        message_thread_id=0,
        date=int(time.time()) if date is None else date,
        chat=SimpleNamespace(id=chat_id, type=chat_type, title="Команда", username="team"),
        from_user=SimpleNamespace(
            id=user_id,
            first_name="Иван",
            last_name="",
            username=f"user{user_id}",
            is_bot=is_bot,
        ),
    )


def build_app(tmp_path, monkeypatch, **env):
    defaults = {
        "TELEGRAM_BOT_TOKEN": "123:test",
        "OPENAI_API_KEY": "sk-test",
        "PINECONE_API_KEY": "pc-test",
        "DB_PATH": str(tmp_path / "db.sqlite3"),
        "COMMAND_ACCESS": "admins",
        "ADMIN_USER_IDS": "7",
        "ANSWER_COOLDOWN_SECONDS": "0",
        "MAX_ANSWER_AGE_SECONDS": "0",
    }
    defaults.update(env)
    for key, value in defaults.items():
        monkeypatch.setenv(key, value)

    settings = load_settings()
    database = Database(settings.db_path)
    repository = Repository(database)
    bot = FakeBot()
    app = BotApp(settings, bot, repository, FakeMemory(), indexer=None)
    app.set_identity(SimpleNamespace(id=999, username="team_bot"))
    return app, database


@pytest.fixture()
def app(tmp_path, monkeypatch):
    instance, database = build_app(tmp_path, monkeypatch)
    yield instance
    instance.shutdown()
    database.close()


def test_start_records_messages_and_stop_summarizes(app):
    app.cmd_start(make_message(text="/start"))
    session = app.repository.get_active_session(-100)
    assert session is not None
    assert "Запись обсуждения начата" in app.bot.sent[-1][1]

    app.on_message(make_message(text="Релиз в пятницу", message_id=11))
    record = app.repository.get_message(-100, 11)
    assert record is not None
    assert record.session_id == session.id

    app.cmd_stop(make_message(text="/stop", message_id=12))
    assert app.repository.get_session(session.id).status == "summarizing"

    app._executor.shutdown(wait=True)

    assert app.repository.get_session(session.id).status == "completed"
    summary = app.repository.get_summary(session.id)
    assert summary is not None
    assert any("Итоги обсуждения" in text for _, text, _ in app.bot.sent)


def test_stop_without_active_session(app):
    app.cmd_stop(make_message(text="/stop"))
    assert "Активной записи нет" in app.bot.sent[-1][1]


def test_start_denied_for_non_admin(tmp_path, monkeypatch):
    instance, database = build_app(tmp_path, monkeypatch, ADMIN_USER_IDS="7")
    try:
        instance.cmd_start(make_message(text="/start", user_id=8))
        assert instance.repository.get_active_session(-100) is None
        assert "только администраторам" in instance.bot.sent[-1][1]
    finally:
        instance.shutdown()
        database.close()


def test_management_disabled_when_admins_empty(tmp_path, monkeypatch):
    instance, database = build_app(tmp_path, monkeypatch, ADMIN_USER_IDS="")
    try:
        instance.cmd_start(make_message(text="/start", user_id=8))
        assert instance.repository.get_active_session(-100) is None
        assert "ADMIN_USER_IDS" in instance.bot.sent[-1][1]
    finally:
        instance.shutdown()
        database.close()


def test_commands_for_other_bot_are_ignored(app):
    app.cmd_start(make_message(text="/start@other_bot"))
    assert app.repository.get_active_session(-100) is None
    assert app.bot.sent == []


def test_private_chat_gets_help_instead_of_session(app):
    app.cmd_start(make_message(text="/start", chat_type="private"))
    assert app.repository.get_active_session(-100) is None
    assert "работаю в рабочих группах" in app.bot.sent[-1][1].lower()


def test_bot_messages_are_not_indexed(app):
    app.on_message(make_message(text="я бот", is_bot=True))
    assert app.repository.count_messages(-100) == 0


def test_mention_triggers_answer(app):
    message = make_message(
        text="@team_bot что решили по релизу?",
        message_id=21,
        entities=[SimpleNamespace(type="mention", offset=0, length=len("@team_bot"))],
    )
    app.on_message(message)
    app._executor.shutdown(wait=True)

    assert app.repository.get_message(-100, 21) is not None
    answers = [item for item in app.bot.sent if "Ответ по истории чата" in item[1]]
    assert answers
    chat_id, _, kwargs = answers[0]
    assert chat_id == -100
    assert kwargs["reply_to_message_id"] == 21


def test_ask_command_answers(app):
    app.cmd_ask(make_message(text="/ask что по срокам?", message_id=31))
    app._executor.shutdown(wait=True)
    assert any("Ответ по истории чата" in text for _, text, _ in app.bot.sent)


def test_unknown_command_hint(app):
    app.on_message(make_message(text="/unknown"))
    assert "Не знаю такую команду" in app.bot.sent[-1][1]
    assert app.repository.count_messages(-100) == 0
