"""Форматирование ответов для Telegram: HTML, разбиение, рендер итогов."""

from __future__ import annotations

import html
import re
from typing import Any, Sequence

_TAG_RE = re.compile(r"(<[^>]{1,300}>|&[a-zA-Z#0-9]{2,10};)")
_VOID_TAGS = {"br", "hr", "img"}

DISCUSSION_TYPE_LABELS = {
    "dispute": "спор",
    "decision": "решение",
    "planning": "планирование",
    "status": "статус",
    "other": "обсуждение",
}


def escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=False)


def _tag_name(tag: str) -> str:
    inner = tag[1:-1].strip()
    if inner.startswith("/"):
        inner = inner[1:].strip()
    name = inner.split(maxsplit=1)[0] if inner else ""
    return name.rstrip("/").lower()


def split_html(text: str, limit: int = 3900) -> list[str]:
    """Режет HTML на части, закрывая и заново открывая теги на границах."""

    if not text:
        return [""]

    chunks: list[str] = []
    current: list[str] = []
    open_tags: list[str] = []
    visible = 0

    def close_tags() -> str:
        return "".join(f"</{tag}>" for tag in reversed(open_tags))

    def reopen_tags() -> str:
        return "".join(f"<{tag}>" for tag in open_tags)

    def flush() -> None:
        nonlocal current, visible
        chunks.append("".join(current) + close_tags())
        current = [reopen_tags()] if open_tags else []
        visible = 0

    for part in _TAG_RE.split(text):
        if not part:
            continue

        if part.startswith("<") and part.endswith(">"):
            name = _tag_name(part)
            if name in _VOID_TAGS:
                current.append(part)
            elif part.startswith("</"):
                if name in open_tags:
                    while open_tags:
                        popped = open_tags.pop()
                        if popped == name:
                            break
                current.append(part)
            else:
                open_tags.append(name)
                current.append(part)
            continue

        if part.startswith("&") and part.endswith(";"):
            # HTML-сущность считаем одним видимым символом и не разрываем.
            if visible + 1 > limit:
                flush()
            current.append(part)
            visible += 1
            continue

        remaining = part
        while remaining:
            space = limit - visible
            if space <= 0:
                flush()
                space = limit

            if len(remaining) <= space:
                current.append(remaining)
                visible += len(remaining)
                remaining = ""
                continue

            cut = remaining.rfind(" ", 0, space)
            if cut <= 0:
                cut = space
            current.append(remaining[:cut])
            flush()
            remaining = remaining[cut:].lstrip()

    tail = "".join(current) + close_tags()
    if tail.strip():
        chunks.append(tail)

    return chunks or [""]


def strip_markdown(text: str) -> str:
    """Мягко убирает markdown-разметку из ответа модели."""

    text = text or ""
    text = re.sub(r"```(?:[a-zA-Z0-9_+-]+)?\n?", "", text).replace("```", "")
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*]\s+", "• ", text)
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    return text.strip()


def _bullet_block(title: str, items: Sequence[str]) -> str:
    body = "\n".join(f"• {escape(item)}" for item in items if str(item).strip())
    return f"<b>{escape(title)}</b>\n{body}" if body else ""


def render_summary(
    payload: dict[str, Any],
    *,
    chat_title: str,
    period: str,
    message_count: int,
    participant_names: Sequence[str],
) -> str:
    """Превращает JSON итогов в аккуратный HTML для Telegram."""

    discussion_type = DISCUSSION_TYPE_LABELS.get(
        str(payload.get("discussion_type", "other")), "обсуждение"
    )
    parts: list[str] = [
        f"🧾 <b>Итоги обсуждения</b> «{escape(chat_title)}»",
        f"<i>{escape(period)} · сообщений: {message_count} · "
        f"участников: {len(participant_names)} · тип: {escape(discussion_type)}</i>",
    ]

    if participant_names:
        parts.append(f"👥 {escape(', '.join(participant_names))}")

    if payload.get("summary"):
        parts.append(f"📌 <b>Кратко</b>\n{escape(payload['summary'])}")

    if payload.get("key_points"):
        parts.append(_bullet_block("Ключевые пункты", payload["key_points"]))

    disputes = payload.get("disputes") or []
    if disputes:
        lines = ["🗣 <b>Споры и позиции</b>"]
        for index, dispute in enumerate(disputes, start=1):
            topic = escape(dispute.get("topic") or f"Спор {index}")
            lines.append(f"\n<b>{index}. {topic}</b>")
            for position in dispute.get("positions") or []:
                author = escape(position.get("author") or "—")
                text = escape(position.get("position") or "—")
                lines.append(f"• <b>{author}</b>: {text}")
            if dispute.get("assessment"):
                lines.append(f"<i>Оценка:</i> {escape(dispute['assessment'])}")
            if dispute.get("recommended_resolution"):
                lines.append(
                    f"<i>Рекомендация:</i> {escape(dispute['recommended_resolution'])}"
                )
        parts.append("\n".join(lines))

    decisions = payload.get("decisions") or []
    if decisions:
        lines = ["✅ <b>Решения</b>"]
        for decision in decisions:
            text = escape(decision.get("decision") or "—")
            lines.append(f"\n• <b>{text}</b>")
            if decision.get("rationale"):
                lines.append(f"  Обоснование: {escape(decision['rationale'])}")
            details = []
            if decision.get("owner"):
                details.append(f"ответственный: {decision['owner']}")
            if decision.get("deadline"):
                details.append(f"срок: {decision['deadline']}")
            if details:
                lines.append(f"  {escape(' · '.join(details))}")
        parts.append("\n".join(lines))

    actions = payload.get("action_items") or []
    if actions:
        lines = ["📋 <b>Задачи</b>"]
        for item in actions:
            task = escape(item.get("task") or "—")
            suffix = []
            if item.get("owner"):
                suffix.append(escape(item["owner"]))
            if item.get("deadline"):
                suffix.append(f"до {escape(item['deadline'])}")
            tail = f" — {', '.join(suffix)}" if suffix else ""
            lines.append(f"• {task}{tail}")
        parts.append("\n".join(lines))

    if payload.get("open_questions"):
        parts.append(_bullet_block("Открытые вопросы", payload["open_questions"]))

    if payload.get("risks"):
        parts.append(_bullet_block("Риски", payload["risks"]))

    if payload.get("assistant_opinion"):
        parts.append(f"🧠 <b>Мнение ассистента</b>\n{escape(payload['assistant_opinion'])}")

    return "\n\n".join(part for part in parts if part)
