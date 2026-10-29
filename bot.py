"""Точка входа Telegram-бота: конфигурация, Pinecone, polling."""

from __future__ import annotations

import logging
import sys

import telebot
from telebot.types import BotCommand

from bot.handlers import BotApp
from bot.indexer import MessageIndexer
from config import ConfigError, Settings, load_settings
from rag.service import TeamMemory
from rag.store import build_document_store, ensure_ready
from storage.db import Database
from storage.repository import Repository

BOT_COMMANDS = [
    BotCommand("start", "Начать запись обсуждения"),
    BotCommand("stop", "Завершить запись и получить итоги"),
    BotCommand("status", "Статус записи и памяти"),
    BotCommand("ask", "Задать вопрос по истории чата"),
    BotCommand("help", "Как пользоваться ботом"),
]


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    for noisy in ("httpx", "httpcore", "urllib3", "pinecone", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_app(settings: Settings) -> tuple[BotApp, Database]:
    database = Database(settings.db_path)
    repository = Repository(database)

    store = build_document_store(settings)
    try:
        ensure_ready(store, settings)
    except Exception as exc:  # noqa: BLE001 - хотим понятное сообщение при старте
        database.close()
        raise RuntimeError(
            "Не удалось подключиться к Pinecone. Проверьте PINECONE_API_KEY, "
            "PINECONE_INDEX и доступ к сети."
        ) from exc

    memory = TeamMemory(settings, store, repository)
    indexer = MessageIndexer(settings, repository, memory)

    bot = telebot.TeleBot(
        settings.telegram_bot_token,
        parse_mode=None,
        threaded=True,
        num_threads=4,
    )
    telebot.logger.setLevel(logging.WARNING)

    app = BotApp(settings, bot, repository, memory, indexer)
    app.register()
    return app, database


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 2

    setup_logging(settings.log_level)
    logger = logging.getLogger("team-bot")
    logger.info(
        "Запуск: модель=%s, эмбеддинги=%s, индекс=%s, база=%s",
        settings.chat_model,
        settings.embedding_model,
        settings.pinecone_index,
        settings.db_path,
    )

    app: BotApp | None = None
    database: Database | None = None
    try:
        app, database = build_app(settings)
        app.set_identity(app.bot.get_me())
        try:
            app.bot.set_my_commands(BOT_COMMANDS)
        except Exception:  # noqa: BLE001 - не критично для работы
            logger.warning("Не удалось обновить список команд", exc_info=True)

        app.start_background()
        logger.info("Начинаю приём сообщений (Ctrl+C — остановка)")
        app.bot.infinity_polling(
            timeout=30,
            long_polling_timeout=25,
            allowed_updates=["message"],
            skip_pending=False,
        )
        return 0
    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl+C")
        return 0
    except Exception:  # noqa: BLE001 - логируем и выходим с кодом ошибки
        logger.exception("Фатальная ошибка")
        return 1
    finally:
        if app is not None:
            app.shutdown()
        if database is not None:
            database.close()
            logger.info("База закрыта")


if __name__ == "__main__":
    raise SystemExit(main())
