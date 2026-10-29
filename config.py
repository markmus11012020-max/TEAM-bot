"""Конфигурация приложения: читается из переменных окружения / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# .env рядом с bot.py; существующие переменные окружения имеют приоритет.
load_dotenv(BASE_DIR / ".env", override=False)


class ConfigError(RuntimeError):
    """Неверная или неполная конфигурация."""


def _env(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip()


def _env_required(name: str, hint: str = "") -> str:
    value = _env(name)
    if not value:
        suffix = f" {hint}" if hint else ""
        raise ConfigError(f"Не задана обязательная переменная окружения {name}.{suffix}")
    return value


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} должно быть целым числом, получено: {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} должно быть числом, получено: {raw!r}") from exc


def _env_float_opt(name: str) -> float | None:
    raw = _env(name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} должно быть числом, получено: {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigError(f"{name} должно быть true/false, получено: {raw!r}")


def _env_int_set(name: str) -> frozenset[int]:
    raw = _env(name)
    if not raw:
        return frozenset()
    result: set[int] = set()
    for chunk in raw.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            result.add(int(chunk))
        except ValueError as exc:
            raise ConfigError(f"{name}: {chunk!r} не является числом") from exc
    return frozenset(result)


def _env_str_tuple(name: str) -> tuple[str, ...]:
    raw = _env(name)
    if not raw:
        return ()
    return tuple(chunk.strip() for chunk in raw.replace(";", ",").split(",") if chunk.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    # Telegram
    telegram_bot_token: str
    command_access: str
    admin_user_ids: frozenset[int]
    allowed_chat_ids: frozenset[int]
    ignore_bots: bool
    max_answer_age_seconds: int
    answer_cooldown_seconds: int

    # LLM
    openai_api_key: str
    openai_base_url: str
    chat_model: str
    llm_max_tokens: int
    llm_max_tokens_param: str
    llm_temperature: float | None
    llm_timeout: float
    llm_max_retries: int
    structured_summary: bool

    # Embeddings
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    embedding_dimensions: int
    embedding_send_dimensions: bool

    # Pinecone
    pinecone_api_key: str
    pinecone_index: str
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str

    # Retrieval / memory
    retrieval_top_k: int
    recent_messages: int
    search_scope: str
    message_chunk_chars: int
    summary_chunk_chars: int
    summary_max_chars: int
    summary_message_max_chars: int

    # Indexer
    index_batch_size: int
    index_poll_seconds: float
    index_max_attempts: int

    # Sessions
    session_idle_timeout_minutes: int

    # Infra
    db_path: Path
    timezone_name: str
    tz: ZoneInfo
    log_level: str


def load_settings() -> Settings:
    """Собирает и валидирует настройки из окружения."""

    command_access = _env("COMMAND_ACCESS", "admins").lower()
    if command_access not in {"admins", "everyone"}:
        raise ConfigError("COMMAND_ACCESS должен быть 'admins' или 'everyone'")

    search_scope = _env("SEARCH_SCOPE", "chat").lower()
    if search_scope not in {"chat", "all"}:
        raise ConfigError("SEARCH_SCOPE должен быть 'chat' или 'all'")

    max_tokens_param = _env("LLM_MAX_TOKENS_PARAM", "max_tokens").lower()
    if max_tokens_param not in {"max_tokens", "max_completion_tokens", "none"}:
        raise ConfigError(
            "LLM_MAX_TOKENS_PARAM должен быть 'max_tokens', 'max_completion_tokens' или 'none'"
        )

    timezone_name = _env("TIMEZONE", "Europe/Moscow")
    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ConfigError(
            f"Неизвестная таймзона TIMEZONE={timezone_name!r}. "
            "Убедитесь, что установлен пакет tzdata."
        ) from exc

    db_path_raw = _env("DB_PATH", "data/team_bot.sqlite3")
    db_path = Path(db_path_raw)
    if not db_path.is_absolute():
        db_path = BASE_DIR / db_path

    openai_api_key = _env_required("OPENAI_API_KEY", "Ключ AITUNNEL/OpenAI.")
    openai_base_url = _env("OPENAI_BASE_URL", "https://api.aitunnel.ru/v1").rstrip("/")

    settings = Settings(
        telegram_bot_token=_env_required("TELEGRAM_BOT_TOKEN", "Токен от @BotFather."),
        command_access=command_access,
        admin_user_ids=_env_int_set("ADMIN_USER_IDS"),
        allowed_chat_ids=_env_int_set("ALLOWED_CHAT_IDS"),
        ignore_bots=_env_bool("IGNORE_BOTS", True),
        max_answer_age_seconds=_env_int("MAX_ANSWER_AGE_SECONDS", 120),
        answer_cooldown_seconds=_env_int("ANSWER_COOLDOWN_SECONDS", 5),
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        chat_model=_env("CHAT_MODEL", "deepseek-v4.1-flash-1017"),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 2000),
        llm_max_tokens_param=max_tokens_param,
        llm_temperature=_env_float_opt("LLM_TEMPERATURE"),
        llm_timeout=_env_float("LLM_TIMEOUT", 120.0),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 3),
        structured_summary=_env_bool("STRUCTURED_SUMMARY", True),
        embedding_api_key=_env("EMBEDDING_API_KEY") or openai_api_key,
        embedding_base_url=(_env("EMBEDDING_BASE_URL") or openai_base_url).rstrip("/"),
        embedding_model=_env("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dimensions=_env_int("EMBEDDING_DIMENSIONS", 1536),
        embedding_send_dimensions=_env_bool("EMBEDDING_SEND_DIMENSIONS", False),
        pinecone_api_key=_env_required("PINECONE_API_KEY", "Ключ из https://app.pinecone.io"),
        pinecone_index=_env("PINECONE_INDEX", "team-bot"),
        pinecone_namespace=_env("PINECONE_NAMESPACE", "team-chat"),
        pinecone_cloud=_env("PINECONE_CLOUD", "aws"),
        pinecone_region=_env("PINECONE_REGION", "us-east-1"),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 12),
        recent_messages=_env_int("RECENT_MESSAGES", 25),
        search_scope=search_scope,
        message_chunk_chars=_env_int("MESSAGE_CHUNK_CHARS", 1800),
        summary_chunk_chars=_env_int("SUMMARY_CHUNK_CHARS", 24000),
        summary_max_chars=_env_int("SUMMARY_MAX_CHARS", 120000),
        summary_message_max_chars=_env_int("SUMMARY_MESSAGE_MAX_CHARS", 2000),
        index_batch_size=_env_int("INDEX_BATCH_SIZE", 12),
        index_poll_seconds=_env_float("INDEX_POLL_SECONDS", 2.0),
        index_max_attempts=_env_int("INDEX_MAX_ATTEMPTS", 5),
        session_idle_timeout_minutes=_env_int("SESSION_IDLE_TIMEOUT_MINUTES", 0),
        db_path=db_path,
        timezone_name=timezone_name,
        tz=tz,
        log_level=_env("LOG_LEVEL", "INFO").upper(),
    )

    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    checks = {
        "RETRIEVAL_TOP_K": settings.retrieval_top_k,
        "RECENT_MESSAGES": settings.recent_messages,
        "INDEX_BATCH_SIZE": settings.index_batch_size,
        "INDEX_MAX_ATTEMPTS": settings.index_max_attempts,
        "LLM_MAX_TOKENS": settings.llm_max_tokens,
        "SUMMARY_CHUNK_CHARS": settings.summary_chunk_chars,
        "SUMMARY_MAX_CHARS": settings.summary_max_chars,
    }
    for name, value in checks.items():
        if value <= 0:
            raise ConfigError(f"{name} должно быть больше нуля")

    if settings.embedding_dimensions <= 0:
        raise ConfigError("EMBEDDING_DIMENSIONS должно быть больше нуля")

    if settings.summary_chunk_chars > settings.summary_max_chars:
        raise ConfigError("SUMMARY_CHUNK_CHARS не может быть больше SUMMARY_MAX_CHARS")
