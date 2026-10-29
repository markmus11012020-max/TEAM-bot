"""Тесты триггеров: команды, упоминания, ответы, текст медиа."""

from __future__ import annotations

from types import SimpleNamespace

from bot.triggers import (
    build_permalink,
    display_name,
    extract_question,
    is_command,
    message_text,
)


def make_message(**kwargs):
    defaults = dict(
        text="",
        caption=None,
        entities=[],
        reply_to_message=None,
        content_type="text",
        message_id=42,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def entity(kind, offset, length, **extra):
    return SimpleNamespace(type=kind, offset=offset, length=length, **extra)


def test_is_command():
    assert is_command(make_message(text="/start")) is True
    assert is_command(make_message(text="/start@team_bot")) is True
    assert is_command(make_message(text=" /help")) is True
    assert is_command(make_message(text="привет /start")) is False
    assert is_command(make_message(text="")) is False


def test_display_name():
    user = SimpleNamespace(first_name="Иван", last_name="Петров", username="ivan", id=1)
    assert display_name(user) == "Иван Петров"

    only_username = SimpleNamespace(first_name="", last_name="", username="ivan", id=1)
    assert display_name(only_username) == "@ivan"

    anonymous = SimpleNamespace(first_name="", last_name="", username=None, id=99)
    assert display_name(anonymous) == "id99"
    assert display_name(None) == "Неизвестный"


def test_message_text_for_media():
    photo = make_message(content_type="photo", caption="схема релиза")
    assert message_text(photo) == "[фото] схема релиза"

    poll = make_message(
        content_type="poll",
        poll=SimpleNamespace(
            question="Где обедаем?",
            options=[SimpleNamespace(text="Пицца"), SimpleNamespace(text="Суши")],
        ),
    )
    assert "опрос" in message_text(poll)
    assert "Пицца" in message_text(poll)

    location = make_message(
        content_type="location",
        location=SimpleNamespace(latitude=55.75, longitude=37.61),
    )
    assert message_text(location) == "[локация] 55.75, 37.61"


def test_build_permalink():
    public = make_message(
        chat=SimpleNamespace(type="supergroup", username="teamchat", id=-100123),
        message_id=5,
    )
    assert build_permalink(public) == "https://t.me/teamchat/5"

    private_group = make_message(
        chat=SimpleNamespace(type="supergroup", username=None, id=-1001234567890),
        message_id=7,
    )
    assert build_permalink(private_group) == "https://t.me/c/1234567890/7"

    private_chat = make_message(
        chat=SimpleNamespace(type="private", username=None, id=777),
        message_id=1,
    )
    assert build_permalink(private_chat) == ""


def test_mention_trigger_and_strip():
    text = "привет @team_bot как дела?"
    message = make_message(
        text=text,
        entities=[entity("mention", offset=7, length=len("@team_bot"))],
    )
    assert extract_question(message, "team_bot", bot_id=1) == "привет как дела?"


def test_mention_of_other_bot_ignored():
    message = make_message(
        text="@other_bot привет",
        entities=[entity("mention", offset=0, length=len("@other_bot"))],
    )
    assert extract_question(message, "team_bot", bot_id=1) is None


def test_text_mention_with_utf16_offsets():
    prefix = "🙂 "
    mention = "@Админ"
    text = f"{prefix}{mention} что решили?"
    offset = len(prefix.encode("utf-16-le")) // 2
    message = make_message(
        text=text,
        entities=[
            entity(
                "text_mention",
                offset=offset,
                length=len(mention.encode("utf-16-le")) // 2,
                user=SimpleNamespace(id=55),
            )
        ],
    )
    assert extract_question(message, "team_bot", bot_id=55) == "🙂 что решили?"


def test_reply_to_bot_triggers():
    reply = make_message(from_user=SimpleNamespace(id=55), text="ответ бота")
    message = make_message(text="а что по срокам?", reply_to_message=reply)
    assert extract_question(message, "team_bot", bot_id=55) == "а что по срокам?"

    human_reply = make_message(from_user=SimpleNamespace(id=56), text="реплика")
    message = make_message(text="а что по срокам?", reply_to_message=human_reply)
    assert extract_question(message, "team_bot", bot_id=55) is None


def test_plain_username_fallback():
    message = make_message(text="@team_bot подведи итоги")
    assert extract_question(message, "team_bot", bot_id=1) == "подведи итоги"

    empty = make_message(text="@team_bot")
    assert extract_question(empty, "team_bot", bot_id=1) == ""
