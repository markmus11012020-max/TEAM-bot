"""Обработчики Telegram: команды, запись сообщений и RAG-ответы."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from bot import formatting
from bot.triggers import (
    build_permalink,
    display_name,
    extract_question,
    is_command,
    message_text,
)
from config import Settings
from rag.service import TeamMemory
from rag.text import participants as collect_participants
from storage.models import SessionRecord, SummaryRecord
from storage.repository import Repository, now_iso

logger = logging.getLogger(__name__)

INDEXABLE_CONTENT_TYPES = {
    "text",
    "photo",
    "video",
    "document",
    "audio",
    "voice",
    "animation",
    "video_note",
    "sticker",
    "poll",
    "location",
    "contact",
    "dice",
}

# Telegram: служебный аккаунт анонимных администраторов групп.
ANONYMOUS_ADMIN_ID = 1087968824

GROUP_HELP = """\
Я — ассистент команды и помню историю этого чата.

Как пользоваться:
• /start — начать запись обсуждения (потом /stop — получить итоги).
• /stop — завершить запись: пришлю итоги, решения, задачи и мнение по спорам.
• /status — что сейчас в памяти и идёт ли запись.
• /ask вопрос — ответ по истории чата.
• Упомяните меня (@бот вопрос) или ответьте на моё сообщение — отвечу со ссылками на авторов и время.

Чтобы я видел все сообщения, отключите privacy mode в @BotFather (/setprivacy → Disable) или сделайте меня администратором чата."""

PRIVATE_HELP = """\
Я работаю в рабочих группах: индексирую переписку и отвечаю по истории.

Добавьте меня в группу и:
1. Отключите privacy mode в @BotFather (/setprivacy → Disable) либо сделайте администратором — иначе я не увижу обычные сообщения.
2. /start — начать запись обсуждения, /stop — получить итоги.
3. Упомяните меня в чате или ответьте на моё сообщение, чтобы задать вопрос.

Команды: /start, /stop, /status, /ask, /help."""


def _chat_title(message) -> str:
    chat = message.chat
    return (
        getattr(chat, "title", None)
        or getattr(chat, "username", None)
        or str(getattr(chat, "id", ""))
    )


def _thread_id(message) -> int:
    return int(getattr(message, "message_thread_id", 0) or 0)


class SessionWatchdog(threading.Thread):
    """Закрывает сессии, в которых давно не было сообщений."""

    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        on_idle,
        interval: float = 60.0,
    ) -> None:
        super().__init__(name="session-watchdog", daemon=True)
        self.settings = settings
        self.repository = repository
        self.on_idle = on_idle
        self.interval = interval
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        timeout_minutes = self.settings.session_idle_timeout_minutes
        if timeout_minutes <= 0:
            logger.info("Автозакрытие сессий отключено (SESSION_IDLE_TIMEOUT_MINUTES=0)")
            return

        logger.info("Watchdog сессий запущен: простой > %s мин", timeout_minutes)
        while not self._stop.wait(self.interval):
            try:
                self._tick(timeout_minutes * 60)
            except Exception:  # noqa: BLE001 - поток не должен умирать
                logger.exception("Сбой watchdog сессий")

    def _tick(self, timeout_seconds: int) -> None:
        now = time.time()
        for session in self.repository.get_active_sessions():
            last = self.repository.get_last_session_message_ts(session.id)
            if last is None:
                last = self._started_ts(session.started_at)
            if last is None or now - last < timeout_seconds:
                continue
            logger.info("Сессия %s простаивает, закрываю", session.id)
            self.on_idle(session)

    @staticmethod
    def _started_ts(value: str) -> float | None:
        try:
            parsed = datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()


class BotApp:
    def __init__(
        self,
        settings: Settings,
        bot,
        repository: Repository,
        memory: TeamMemory,
        indexer,
    ) -> None:
        self.settings = settings
        self.bot = bot
        self.repository = repository
        self.memory = memory
        self.indexer = indexer

        self.bot_id: int | None = None
        self.bot_username: str | None = None

        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm")
        self._cooldowns: dict[tuple[int, int], float] = {}
        self._cooldown_lock = threading.Lock()
        self._warned_no_admins = False
        self.watchdog = SessionWatchdog(settings, repository, self._auto_finish)

    # ------------------------------------------------------------------ setup

    def set_identity(self, me) -> None:
        self.bot_id = getattr(me, "id", None)
        self.bot_username = getattr(me, "username", None)
        logger.info("Бот @%s (id=%s) готов", self.bot_username, self.bot_id)

    def register(self) -> None:
        # Команды регистрируются раньше catch-all: telebot берёт первый
        # подошедший обработчик.
        self.bot.register_message_handler(self.cmd_start, commands=["start", "listen"])
        self.bot.register_message_handler(
            self.cmd_stop, commands=["stop", "stop_listening", "finish"]
        )
        self.bot.register_message_handler(self.cmd_status, commands=["status", "session"])
        self.bot.register_message_handler(self.cmd_help, commands=["help"])
        self.bot.register_message_handler(self.cmd_ask, commands=["ask", "q"])
        self.bot.register_message_handler(self.on_message, content_types=None)

    def start_background(self) -> None:
        self.indexer.start()
        self.watchdog.start()

    def shutdown(self) -> None:
        self.watchdog.stop()
        if self.indexer is not None:
            self.indexer.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self.watchdog.is_alive():
            self.watchdog.join(timeout=5)
        if self.indexer is not None and self.indexer.is_alive():
            self.indexer.join(timeout=10)

    # ----------------------------------------------------------------- guards

    def _chat_allowed(self, chat_id: int) -> bool:
        return not self.settings.allowed_chat_ids or chat_id in self.settings.allowed_chat_ids

    def _is_admin(self, user_id: int) -> bool:
        if self.settings.command_access == "everyone":
            return True
        if not self.settings.admin_user_ids:
            if not self._warned_no_admins:
                logger.warning(
                    "COMMAND_ACCESS=admins, но ADMIN_USER_IDS пуст — "
                    "управление записью отключено, заполните ADMIN_USER_IDS"
                )
                self._warned_no_admins = True
            return False
        return user_id in self.settings.admin_user_ids

    def _command_for_me(self, message) -> bool:
        """Отсекает /command@другой_бот, адресованные не нам."""

        text = (getattr(message, "text", None) or "").strip()
        if not text.startswith("/"):
            return False
        head = text.split(maxsplit=1)[0][1:]
        if "@" not in head:
            return True
        _, _, target = head.partition("@")
        return bool(self.bot_username) and target.lower() == self.bot_username.lower()

    @staticmethod
    def _targets_other_bot(message) -> bool:
        """True, если команда явно адресована другому боту (/cmd@other_bot)."""

        text = (getattr(message, "text", None) or "").strip()
        if not text.startswith("/"):
            return False
        head = text.split(maxsplit=1)[0]
        return "@" in head

    def _require_admin(self, message) -> bool:
        user = getattr(message, "from_user", None)
        user_id = getattr(user, "id", 0)
        if self._is_admin(user_id):
            return True
        if not self.settings.admin_user_ids and self.settings.command_access == "admins":
            self._reply(
                message,
                "⛔ Управление записью не настроено: добавьте ADMIN_USER_IDS в .env "
                "или включите COMMAND_ACCESS=everyone.",
            )
            return False
        self._reply(message, "⛔ Эта команда доступна только администраторам бота.")
        return False

    # --------------------------------------------------------------- sending

    def _send(
        self,
        chat_id: int,
        text: str,
        *,
        thread_id: int = 0,
        reply_to: int | None = None,
        parse_mode: str | None = None,
    ) -> None:
        try:
            self.bot.send_message(
                chat_id,
                text,
                parse_mode=parse_mode,
                reply_to_message_id=reply_to,
                allow_sending_without_reply=True,
                message_thread_id=thread_id or None,
                disable_web_page_preview=True,
            )
        except Exception as exc:  # noqa: BLE001 - сеть/лимиты Telegram
            logger.warning("Не удалось отправить сообщение в чат %s: %s", chat_id, exc)

    def _send_html(
        self,
        chat_id: int,
        html_text: str,
        *,
        thread_id: int = 0,
        reply_to: int | None = None,
    ) -> None:
        for index, chunk in enumerate(formatting.split_html(html_text, 3500)):
            self._send(
                chat_id,
                chunk,
                thread_id=thread_id,
                reply_to=reply_to if index == 0 else None,
                parse_mode="HTML",
            )

    def _reply(self, message, text: str, *, parse_mode: str | None = None) -> None:
        self._send(
            message.chat.id,
            text,
            thread_id=_thread_id(message),
            reply_to=message.message_id,
            parse_mode=parse_mode,
        )

    # -------------------------------------------------------------- commands

    def cmd_start(self, message) -> None:
        if not self._command_for_me(message) or not self._chat_allowed(message.chat.id):
            return
        if getattr(message.chat, "type", "") == "private":
            self._reply(message, PRIVATE_HELP)
            return
        if not self._require_admin(message):
            return

        user = getattr(message, "from_user", None)
        session, created = self.repository.start_session(
            chat_id=message.chat.id,
            chat_title=_chat_title(message),
            user_id=getattr(user, "id", 0),
            user_name=display_name(user),
        )
        if created:
            self._reply(
                message,
                "🔴 Запись обсуждения начата. Я сохраняю все сообщения этого чата.\n"
                "Когда закончите — /stop, и я пришлю итоги: решения, задачи, "
                "споры и своё мнение.",
            )
        else:
            self._reply(
                message,
                f"Запись уже идёт с {self.memory.period(session)}.\n"
                "Пришлите /stop, чтобы завершить её и получить итоги.",
            )

    def cmd_stop(self, message) -> None:
        if not self._command_for_me(message) or not self._chat_allowed(message.chat.id):
            return
        if getattr(message.chat, "type", "") == "private":
            self._reply(message, PRIVATE_HELP)
            return
        if not self._require_admin(message):
            return

        session = self.repository.get_active_session(message.chat.id)
        if session is None:
            self._reply(message, "Активной записи нет. /start — начать новую.")
            return

        if not self.repository.close_session(session.id, "summarizing"):
            self._reply(message, "Запись уже завершается, итоги скоро придут.")
            return

        fresh = self.repository.get_session(session.id) or session
        count = self.repository.count_session_messages(fresh.id)
        if count == 0:
            self.repository.set_session_status(fresh.id, "empty")
            self._reply(message, "Запись остановлена: в ней не было сообщений.")
            return

        self._reply(
            message,
            f"⏳ Запись остановлена. Готовлю итоги по {count} сообщениям — пришлю сюда же.",
        )
        self._executor.submit(self._summary_job, fresh, _thread_id(message))

    def cmd_status(self, message) -> None:
        if not self._command_for_me(message) or not self._chat_allowed(message.chat.id):
            return

        chat_id = message.chat.id
        total = self.repository.count_messages(chat_id)
        indexed = self.repository.count_indexed(chat_id)
        pending = self.repository.count_pending(self.settings.index_max_attempts)
        session = self.repository.get_active_session(chat_id)

        lines = [
            "📊 Статус",
            f"Чат: {_chat_title(message)}",
            f"Сообщений в памяти: {total} "
            f"(в Pinecone: {indexed}, в очереди: {pending})",
        ]
        if session is not None:
            in_session = self.repository.count_session_messages(session.id)
            lines.append(
                f"🔴 Запись идёт с {self.memory.period(session)}; "
                f"сообщений в сессии: {in_session}"
            )
        else:
            lines.append("Запись выключена. /start — начать.")

        lines.append(
            f"Модель: {self.settings.chat_model} · поиск: {self.settings.search_scope} · "
            f"индекс: {self.settings.pinecone_index}"
        )
        self._reply(message, "\n".join(lines))

    def cmd_help(self, message) -> None:
        if not self._command_for_me(message) or not self._chat_allowed(message.chat.id):
            return
        if getattr(message.chat, "type", "") == "private":
            self._reply(message, PRIVATE_HELP)
        else:
            self._reply(message, GROUP_HELP)

    def cmd_ask(self, message) -> None:
        if not self._command_for_me(message) or not self._chat_allowed(message.chat.id):
            return
        if getattr(message.chat, "type", "") == "private":
            self._reply(message, PRIVATE_HELP)
            return

        parts = (getattr(message, "text", None) or "").split(maxsplit=1)
        question = parts[1].strip() if len(parts) > 1 else ""
        if not question:
            self._reply(message, "Напишите вопрос после команды: /ask что решили по релизу?")
            return
        self._submit_answer(message, question)

    # -------------------------------------------------------------- messages

    def on_message(self, message) -> None:
        try:
            self._handle_message(message)
        except Exception:  # noqa: BLE001 - один апдейт не должен ронять бота
            logger.exception("Ошибка обработки сообщения")

    def _handle_message(self, message) -> None:
        chat = message.chat
        if not self._chat_allowed(chat.id):
            return

        content_type = getattr(message, "content_type", "text")

        if content_type == "new_chat_members":
            if self._bot_added(message):
                self._send(chat.id, GROUP_HELP, thread_id=_thread_id(message))
            return

        if is_command(message):
            if self._targets_other_bot(message) and not self._command_for_me(message):
                return
            self._reply(message, "Не знаю такую команду. Список возможностей: /help")
            return

        if getattr(chat, "type", "") not in {"group", "supergroup"}:
            question = extract_question(message, self.bot_username, self.bot_id)
            if question is not None:
                self._reply(message, PRIVATE_HELP)
            return

        user = getattr(message, "from_user", None)
        user_id = getattr(user, "id", 0)
        if (
            self.settings.ignore_bots
            and getattr(user, "is_bot", False)
            and user_id != ANONYMOUS_ADMIN_ID
        ):
            return
        if content_type not in INDEXABLE_CONTENT_TYPES:
            return

        text = message_text(message)
        if not text:
            return

        session = self.repository.get_active_session(chat.id)
        reply_to = getattr(message, "reply_to_message", None)
        inserted = self.repository.add_message(
            chat_id=chat.id,
            chat_title=_chat_title(message),
            message_id=message.message_id,
            thread_id=_thread_id(message),
            user_id=getattr(user, "id", 0),
            author=display_name(user),
            username=getattr(user, "username", None) or "",
            ts=int(getattr(message, "date", time.time())),
            dt=datetime.fromtimestamp(
                int(getattr(message, "date", time.time())), self.settings.tz
            ).strftime("%d.%m.%Y %H:%M"),
            text=text,
            content_type=content_type,
            permalink=build_permalink(message),
            reply_to_message_id=getattr(reply_to, "message_id", 0) or 0,
            session_id=session.id if session is not None else None,
        )
        if not inserted:
            logger.debug(
                "Сообщение %s/%s уже есть в базе", chat.id, message.message_id
            )

        question = extract_question(message, self.bot_username, self.bot_id)
        if question is not None:
            self._maybe_answer(message, question)

    def _bot_added(self, message) -> bool:
        if self.bot_id is None:
            return False
        members = getattr(message, "new_chat_members", None) or []
        return any(getattr(member, "id", None) == self.bot_id for member in members)

    # ----------------------------------------------------------------- answers

    def _maybe_answer(self, message, question: str) -> None:
        question = (question or "").strip()
        if not question:
            self._reply(message, "Я здесь. Напишите вопрос — отвечу по истории чата.")
            return

        user = getattr(message, "from_user", None)
        key = (message.chat.id, getattr(user, "id", 0))
        now = time.monotonic()

        age = time.time() - int(getattr(message, "date", time.time()))
        if self.settings.max_answer_age_seconds > 0 and age > self.settings.max_answer_age_seconds:
            logger.debug("Пропускаю устаревшее упоминание (%.0f c)", age)
            return

        with self._cooldown_lock:
            last = self._cooldowns.get(key)
            if last is not None and now - last < self.settings.answer_cooldown_seconds:
                logger.debug("Ответ пропущен из-за кулдауна: %s", key)
                return
            self._cooldowns[key] = now
            if len(self._cooldowns) > 5000:
                cutoff = now - 3600
                self._cooldowns = {
                    key_: ts for key_, ts in self._cooldowns.items() if ts >= cutoff
                }

        self._submit_answer(message, question)

    def _submit_answer(self, message, question: str) -> None:
        user = getattr(message, "from_user", None)
        self._executor.submit(
            self._answer_job,
            message.chat.id,
            message.message_id,
            _thread_id(message),
            question,
            _chat_title(message),
            display_name(user),
        )

    def _answer_job(
        self,
        chat_id: int,
        message_id: int,
        thread_id: int,
        question: str,
        chat_title: str,
        user_name: str,
    ) -> None:
        logger.info("Вопрос от %s в чате %s: %s", user_name, chat_title, question)
        try:
            self.bot.send_chat_action(
                chat_id, "typing", message_thread_id=thread_id or None
            )
        except Exception:  # noqa: BLE001 - индикатор набора не критичен
            pass

        try:
            answer = self.memory.ask(question, chat_id, chat_title)
        except Exception:  # noqa: BLE001 - детали уходят в лог, не в чат
            logger.exception("Не удалось ответить на вопрос")
            answer = (
                "Не получилось ответить: сервис модели временно недоступен. "
                "Попробуйте чуть позже."
            )

        text = formatting.strip_markdown(answer)
        if not text:
            text = "Не удалось сформулировать ответ, попробуйте переформулировать вопрос."
        self._send_html(chat_id, formatting.escape(text), thread_id=thread_id, reply_to=message_id)

    # ---------------------------------------------------------------- summary

    def _summary_job(self, session: SessionRecord, thread_id: int) -> None:
        chat_id = session.chat_id
        try:
            messages = self.repository.get_session_messages(session.id)
            if not messages:
                self.repository.set_session_status(session.id, "empty")
                self._send(
                    chat_id,
                    "Запись завершена: в ней не было сообщений.",
                    thread_id=thread_id,
                )
                return

            result = self.memory.summarize(session, messages)
            names = collect_participants(messages)

            doc_id = ""
            try:
                doc_id = self.memory.index_summary(
                    session, result, names, len(messages)
                )
            except Exception:  # noqa: BLE001 - итоги всё равно отдадим в чат
                logger.exception("Итоги не удалось записать в Pinecone")

            self.repository.save_summary(
                SummaryRecord(
                    id=uuid.uuid4().hex,
                    session_id=session.id,
                    chat_id=chat_id,
                    text=result.plain_text,
                    payload_json=json.dumps(result.payload, ensure_ascii=False),
                    doc_id=doc_id,
                    created_at=now_iso(),
                )
            )
            self.repository.set_session_status(session.id, "completed")

            html_text = formatting.render_summary(
                result.payload,
                chat_title=session.chat_title or "рабочий чат",
                period=self.memory.period(session),
                message_count=len(messages),
                participant_names=names,
            )
            self._send_html(chat_id, html_text, thread_id=thread_id)
            logger.info(
                "Итоги сессии %s отправлены (сообщений=%s, частей=%s)",
                session.id,
                len(messages),
                result.parts,
            )
        except Exception:  # noqa: BLE001 - фиксируем провал сессии
            logger.exception("Не удалось собрать итоги сессии %s", session.id)
            self.repository.set_session_status(session.id, "failed")
            self._send(
                chat_id,
                "⚠️ Не удалось собрать итоги. Подробности — в логах бота; "
                "попробуйте /start и /stop ещё раз.",
                thread_id=thread_id,
            )

    def _auto_finish(self, session: SessionRecord) -> None:
        fresh = self.repository.get_session(session.id)
        if fresh is None or fresh.status != "active":
            return
        if not self.repository.close_session(fresh.id, "summarizing"):
            return

        fresh = self.repository.get_session(fresh.id) or fresh
        count = self.repository.count_session_messages(fresh.id)
        if count == 0:
            self.repository.set_session_status(fresh.id, "empty")
            return

        timeout = self.settings.session_idle_timeout_minutes
        self._send(
            fresh.chat_id,
            f"⏳ Запись остановлена автоматически: сообщений не было {timeout} мин. "
            f"Готовлю итоги по {count} сообщениям.",
        )
        self._executor.submit(self._summary_job, fresh, 0)
