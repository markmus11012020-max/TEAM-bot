"""Определение триггеров и извлечение текста из сообщений Telegram."""

from __future__ import annotations

import re
from typing import Any

_COMMAND_RE = re.compile(r"^/([A-Za-z0-9_]+)(?:@([A-Za-z0-9_]+))?\b")

_MEDIA_LABELS = {
    "photo": "фото",
    "video": "видео",
    "document": "документ",
    "audio": "аудио",
    "voice": "голосовое",
    "animation": "гиф",
    "video_note": "видеосообщение",
    "sticker": "стикер",
    "dice": "кубик",
}


def is_command(message: Any) -> bool:
    text = (getattr(message, "text", None) or "").strip()
    return bool(_COMMAND_RE.match(text))


def command_name(message: Any) -> str | None:
    match = _COMMAND_RE.match((getattr(message, "text", None) or "").strip())
    return match.group(1).lower() if match else None


def display_name(user: Any) -> str:
    if user is None:
        return "Неизвестный"
    parts = [
        getattr(user, "first_name", None) or "",
        getattr(user, "last_name", None) or "",
    ]
    name = " ".join(part for part in parts if part).strip()
    if name:
        return name
    username = getattr(user, "username", None)
    if username:
        return f"@{username}"
    return f"id{getattr(user, 'id', 0)}"


def message_text(message: Any) -> str:
    """Текстовое представление любого типа сообщения для индексации."""

    content_type = getattr(message, "content_type", "text")
    if content_type == "text":
        return (getattr(message, "text", None) or "").strip()

    caption = (getattr(message, "caption", None) or "").strip()
    label = _MEDIA_LABELS.get(content_type)

    if content_type == "poll" and getattr(message, "poll", None):
        poll = message.poll
        options = ", ".join(getattr(option, "text", "") for option in poll.options)
        return f"[опрос] {poll.question}" + (f" — варианты: {options}" if options else "")

    if content_type == "location" and getattr(message, "location", None):
        location = message.location
        return f"[локация] {location.latitude}, {location.longitude}"

    if content_type == "contact" and getattr(message, "contact", None):
        contact = message.contact
        name = " ".join(
            part
            for part in [
                getattr(contact, "first_name", "") or "",
                getattr(contact, "last_name", "") or "",
            ]
            if part
        )
        return f"[контакт] {name} {getattr(contact, 'phone_number', '')}".strip()

    if label:
        return f"[{label}] {caption}".strip()
    return f"[{content_type}] {caption}".strip()


def build_permalink(message: Any) -> str:
    chat = getattr(message, "chat", None)
    if chat is None:
        return ""
    chat_type = getattr(chat, "type", "")
    if chat_type not in {"group", "supergroup"}:
        return ""
    username = getattr(chat, "username", None)
    if username:
        return f"https://t.me/{username}/{message.message_id}"
    chat_id = str(getattr(chat, "id", ""))
    if chat_id.startswith("-100"):
        chat_id = chat_id[4:]
    return f"https://t.me/c/{chat_id}/{message.message_id}"


def _utf16_slice(text: str, offset: int, length: int) -> str:
    """Telegram считает offset/length в UTF-16 code units."""

    encoded = text.encode("utf-16-le")
    chunk = encoded[offset * 2 : (offset + length) * 2]
    try:
        return chunk.decode("utf-16-le")
    except UnicodeDecodeError:
        return ""


def extract_question(
    message: Any,
    bot_username: str | None,
    bot_id: int | None,
) -> str | None:
    """Возвращает текст вопроса, если бота упомянули или ответили на его сообщение.

    `None` — сообщение не адресовано боту. Пустая строка — бота позвали без вопроса.
    """

    text = getattr(message, "text", None) or getattr(message, "caption", None) or ""
    triggered = False

    reply = getattr(message, "reply_to_message", None)
    if reply is not None:
        reply_author = getattr(reply, "from_user", None)
        if reply_author is not None and bot_id is not None and reply_author.id == bot_id:
            triggered = True

    for entity in getattr(message, "entities", None) or []:
        entity_type = getattr(entity, "type", "")
        if entity_type == "text_mention":
            user = getattr(entity, "user", None)
            if user is not None and bot_id is not None and user.id == bot_id:
                triggered = True
                mention = _utf16_slice(text, entity.offset, entity.length)
                if mention:
                    text = text.replace(mention, " ", 1)
        elif entity_type == "mention" and bot_username:
            mention = _utf16_slice(text, entity.offset, entity.length)
            if mention.lower() == f"@{bot_username}".lower():
                triggered = True
                text = text.replace(mention, " ", 1)

    if bot_username:
        pattern = re.compile(rf"@{re.escape(bot_username)}\b", re.IGNORECASE)
        if pattern.search(text):
            triggered = True
            text = pattern.sub(" ", text)

    if not triggered:
        return None

    cleaned = re.sub(r"\s+", " ", text)
    return cleaned.strip(" \t\r\n,.:;—-")
