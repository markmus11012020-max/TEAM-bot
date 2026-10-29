"""Pinecone Document Store (официальная интеграция Haystack)."""

from __future__ import annotations

import logging

from haystack.utils import Secret
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from config import Settings

logger = logging.getLogger(__name__)


def build_document_store(settings: Settings) -> PineconeDocumentStore:
    """Создаёт store; индекс создастся автоматически при первом обращении."""

    return PineconeDocumentStore(
        api_key=Secret.from_token(settings.pinecone_api_key),
        index=settings.pinecone_index,
        namespace=settings.pinecone_namespace,
        dimension=settings.embedding_dimensions,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": settings.pinecone_cloud,
                "region": settings.pinecone_region,
            }
        },
        show_progress=False,
    )


def ensure_ready(store: PineconeDocumentStore, settings: Settings) -> int:
    """Форсирует подключение/создание индекса и проверяет размерность."""

    count = store.count_documents()
    actual_dimension = getattr(store, "dimension", settings.embedding_dimensions)
    if actual_dimension != settings.embedding_dimensions:
        logger.warning(
            "Размерность индекса Pinecone (%s) не совпадает с EMBEDDING_DIMENSIONS (%s). "
            "Индекс уже существует, поэтому используется его размерность. "
            "Для смены модели эмбеддингов создайте новый индекс.",
            actual_dimension,
            settings.embedding_dimensions,
        )
    logger.info(
        "Pinecone готов: index=%s namespace=%s dimension=%s документов=%s",
        settings.pinecone_index,
        settings.pinecone_namespace,
        actual_dimension,
        count,
    )
    return count
