"""Диагностика окружения бота.

Запуск:
    python scripts/smoke_test.py                # версии, импорты, .env
    python scripts/smoke_test.py --llm          # + запрос к LLM и эмбеддингам
    python scripts/smoke_test.py --pinecone     # + подключение к Pinecone
    python scripts/smoke_test.py --all
"""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PACKAGES = [
    "haystack-ai",
    "pinecone-haystack",
    "pinecone",
    "pyTelegramBotAPI",
    "openai",
    "python-dotenv",
]


def check_versions() -> bool:
    ok = True
    print(f"Python: {sys.version.split()[0]}")
    for package in PACKAGES:
        try:
            print(f"  {package}: {version(package)}")
        except PackageNotFoundError:
            print(f"  {package}: НЕ УСТАНОВЛЕН")
            ok = False
    return ok


def check_imports() -> bool:
    try:
        import haystack  # noqa: F401
        import telebot  # noqa: F401
        from haystack_integrations.document_stores.pinecone import (  # noqa: F401
            PineconeDocumentStore,
        )

        from bot.handlers import BotApp  # noqa: F401
        from rag.service import TeamMemory  # noqa: F401
        from storage.db import Database  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"Импорт модулей не удался: {exc}")
        return False
    print("Импорты: OK")
    return True


def check_config():
    try:
        from config import ConfigError, load_settings
    except Exception as exc:  # noqa: BLE001
        print(f"Конфигурация не импортируется: {exc}")
        return None
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Конфигурация: {exc}")
        return None
    print(
        f"Конфигурация: OK (модель={settings.chat_model}, "
        f"эмбеддинги={settings.embedding_model}, индекс={settings.pinecone_index}, "
        f"база={settings.db_path})"
    )
    return settings


def check_llm(settings) -> bool:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    ok = True

    try:
        response = client.chat.completions.create(
            model=settings.chat_model,
            messages=[{"role": "user", "content": "Ответь одним словом: работает?"}],
            max_tokens=20,
        )
        answer = (response.choices[0].message.content or "").strip()
        print(f"LLM: OK ({settings.chat_model}) -> {answer[:80]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"LLM: ОШИБКА ({settings.chat_model}): {exc}")
        ok = False

    try:
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=["проверка эмбеддингов"],
        )
        vector = response.data[0].embedding
        print(
            f"Эмбеддинги: OK ({settings.embedding_model}), "
            f"размерность={len(vector)} (ожидается {settings.embedding_dimensions})"
        )
        if len(vector) != settings.embedding_dimensions:
            print("  ВНИМАНИЕ: размерность не совпадает с EMBEDDING_DIMENSIONS")
            ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"Эмбеддинги: ОШИБКА ({settings.embedding_model}): {exc}")
        ok = False

    return ok


def check_pinecone(settings) -> bool:
    try:
        from rag.store import build_document_store, ensure_ready

        store = build_document_store(settings)
        count = ensure_ready(store, settings)
        print(f"Pinecone: OK, документов в namespace: {count}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Pinecone: ОШИБКА: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка окружения team-bot")
    parser.add_argument("--llm", action="store_true", help="проверить LLM и эмбеддинги")
    parser.add_argument("--pinecone", action="store_true", help="проверить Pinecone")
    parser.add_argument("--all", action="store_true", help="проверить всё")
    args = parser.parse_args()

    ok = check_versions() and check_imports()
    settings = check_config()
    if settings is None:
        ok = False

    if settings is not None and (args.llm or args.all):
        ok = check_llm(settings) and ok
    if settings is not None and (args.pinecone or args.all):
        ok = check_pinecone(settings) and ok

    print("ИТОГ:", "OK" if ok else "ЕСТЬ ПРОБЛЕМЫ")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
