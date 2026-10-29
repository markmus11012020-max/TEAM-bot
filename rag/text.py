"""Чистые функции обработки текста: транскрипт, JSON итогов, plain-версия."""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from storage.models import MessageRecord

ALLOWED_DISCUSSION_TYPES = {"dispute", "decision", "planning", "status", "other"}


class SummaryParseError(ValueError):
    """Не удалось разобрать ответ модели как JSON итогов."""


def truncate(text: str, limit: int) -> str:
    text = text or ""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _as_str(item)
        if text:
            result.append(text)
    return result


def _as_dict_list(value: Any, keys: Sequence[str]) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        normalized = {key: _as_str(item.get(key)) for key in keys}
        if any(normalized.values()):
            result.append(normalized)
    return result


def _as_positions(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        author = _as_str(item.get("author"))
        position = _as_str(item.get("position"))
        if author or position:
            result.append({"author": author, "position": position})
    return result


def normalize_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Приводит произвольный JSON модели к ожидаемой структуре."""

    discussion_type = _as_str(data.get("discussion_type")).lower()
    if discussion_type not in ALLOWED_DISCUSSION_TYPES:
        discussion_type = "other"

    disputes: list[dict[str, Any]] = []
    raw_disputes = data.get("disputes")
    if isinstance(raw_disputes, list):
        for item in raw_disputes:
            if not isinstance(item, dict):
                continue
            topic = _as_str(item.get("topic"))
            positions = _as_positions(item.get("positions"))
            assessment = _as_str(item.get("assessment"))
            resolution = _as_str(item.get("recommended_resolution"))
            if topic or positions or assessment or resolution:
                disputes.append(
                    {
                        "topic": topic,
                        "positions": positions,
                        "assessment": assessment,
                        "recommended_resolution": resolution,
                    }
                )

    return {
        "summary": _as_str(data.get("summary")),
        "discussion_type": discussion_type,
        "key_points": _as_str_list(data.get("key_points")),
        "disputes": disputes,
        "decisions": _as_dict_list(
            data.get("decisions"), ("decision", "rationale", "owner", "deadline")
        ),
        "action_items": _as_dict_list(
            data.get("action_items"), ("task", "owner", "deadline")
        ),
        "open_questions": _as_str_list(data.get("open_questions")),
        "risks": _as_str_list(data.get("risks")),
        "assistant_opinion": _as_str(data.get("assistant_opinion")),
    }


def parse_summary_json(raw: str) -> dict[str, Any]:
    """Парсит JSON, допуская ```json-обёртку и текст вокруг объекта."""

    text = (raw or "").strip()
    if not text:
        raise SummaryParseError("Пустой ответ модели")

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise SummaryParseError("В ответе модели нет JSON-объекта") from None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise SummaryParseError(f"Некорректный JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise SummaryParseError("Ожидался JSON-объект")
    return normalize_payload(data)


def build_transcript(
    messages: Iterable[MessageRecord],
    max_chars_per_message: int = 2000,
) -> str:
    """Собирает читаемую стенограмму с авторами, временем и цитатами ответов."""

    records = list(messages)
    by_message_id = {record.message_id: record for record in records}
    lines: list[str] = []

    for record in records:
        author = record.author or "Неизвестный"
        username = f" (@{record.username})" if record.username else ""
        text = truncate(record.text.strip(), max_chars_per_message)
        header = f"[{record.dt}] {author}{username}:"
        if text:
            lines.append(f"{header} {text}")
        else:
            lines.append(f"{header} [пустое сообщение]")

        if record.reply_to_message_id:
            quoted = by_message_id.get(record.reply_to_message_id)
            if quoted is not None and quoted.text:
                snippet = truncate(quoted.text.replace("\n", " ").strip(), 200)
                lines.append(f"    ↳ ответ на «{snippet}» ({quoted.author})")

    return "\n".join(lines)


def participants(messages: Iterable[MessageRecord]) -> list[str]:
    seen: list[str] = []
    for record in messages:
        name = record.author or (f"@{record.username}" if record.username else "")
        if name and name not in seen:
            seen.append(name)
    return seen


def build_plain_summary(
    payload: dict[str, Any],
    *,
    chat_title: str,
    period: str,
    message_count: int,
    participant_names: Sequence[str],
) -> str:
    """Текстовая версия итогов — именно она уходит в Pinecone."""

    parts: list[str] = [
        f"Итоги обсуждения «{chat_title}» за {period}.",
        f"Сообщений: {message_count}. Участники: {', '.join(participant_names) or '—'}.",
        f"Тип обсуждения: {payload.get('discussion_type', 'other')}.",
    ]

    if payload.get("summary"):
        parts.append(f"Кратко: {payload['summary']}")

    if payload.get("key_points"):
        parts.append("Ключевые пункты:\n" + "\n".join(f"- {item}" for item in payload["key_points"]))

    if payload.get("disputes"):
        dispute_lines = ["Споры и позиции:"]
        for dispute in payload["disputes"]:
            positions = "; ".join(
                f"{item.get('author') or '—'}: {item.get('position') or '—'}"
                for item in dispute.get("positions", [])
            )
            line = f"- {dispute.get('topic') or 'Без темы'}"
            if positions:
                line += f". Позиции: {positions}"
            if dispute.get("assessment"):
                line += f". Оценка: {dispute['assessment']}"
            if dispute.get("recommended_resolution"):
                line += f". Предложение: {dispute['recommended_resolution']}"
            dispute_lines.append(line)
        parts.append("\n".join(dispute_lines))

    if payload.get("decisions"):
        decision_lines = ["Решения:"]
        for decision in payload["decisions"]:
            line = f"- {decision.get('decision') or '—'}"
            if decision.get("rationale"):
                line += f" (обоснование: {decision['rationale']})"
            if decision.get("owner"):
                line += f". Ответственный: {decision['owner']}"
            if decision.get("deadline"):
                line += f". Срок: {decision['deadline']}"
            decision_lines.append(line)
        parts.append("\n".join(decision_lines))

    if payload.get("action_items"):
        action_lines = ["Задачи:"]
        for item in payload["action_items"]:
            line = f"- {item.get('task') or '—'}"
            if item.get("owner"):
                line += f" — {item['owner']}"
            if item.get("deadline"):
                line += f", срок: {item['deadline']}"
            action_lines.append(line)
        parts.append("\n".join(action_lines))

    if payload.get("open_questions"):
        parts.append(
            "Открытые вопросы:\n" + "\n".join(f"- {item}" for item in payload["open_questions"])
        )

    if payload.get("risks"):
        parts.append("Риски:\n" + "\n".join(f"- {item}" for item in payload["risks"]))

    if payload.get("assistant_opinion"):
        parts.append(f"Мнение ассистента: {payload['assistant_opinion']}")

    return "\n\n".join(parts)
