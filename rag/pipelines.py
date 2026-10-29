"""Haystack-пайплайны: индексация, поиск, ответы и итоги."""

from __future__ import annotations

from typing import Any

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy, FilterPolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from config import Settings
from rag import prompts


def _embedding_kwargs(settings: Settings) -> dict[str, Any]:
    return {
        "api_key": Secret.from_token(settings.embedding_api_key),
        "model": settings.embedding_model,
        "api_base_url": settings.embedding_base_url,
        "dimensions": (
            settings.embedding_dimensions if settings.embedding_send_dimensions else None
        ),
        "timeout": settings.llm_timeout,
        "max_retries": settings.llm_max_retries,
    }


def _generation_kwargs(
    settings: Settings, response_format: dict[str, Any] | None = None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if settings.llm_max_tokens_param != "none":
        kwargs[settings.llm_max_tokens_param] = settings.llm_max_tokens
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    if response_format is not None:
        kwargs["response_format"] = response_format
    return kwargs


def build_index_pipeline(
    settings: Settings, store: PineconeDocumentStore
) -> Pipeline:
    """embedder -> writer. Документы приходят уже нарезанными с детерминированными id."""

    embedder = OpenAIDocumentEmbedder(
        **_embedding_kwargs(settings),
        batch_size=16,
        progress_bar=False,
        raise_on_failure=True,
    )
    writer = DocumentWriter(document_store=store, policy=DuplicatePolicy.OVERWRITE)

    return (
        Pipeline()
        .add_components({"embedder": embedder, "writer": writer})
        .connect("embedder.documents", "writer.documents")
    )


def build_retrieval_pipeline(
    settings: Settings, store: PineconeDocumentStore
) -> Pipeline:
    """text_embedder -> retriever (Pinecone)."""

    text_embedder = OpenAITextEmbedder(**_embedding_kwargs(settings))
    retriever = PineconeEmbeddingRetriever(
        document_store=store,
        top_k=settings.retrieval_top_k,
        filter_policy=FilterPolicy.REPLACE,
    )

    return (
        Pipeline()
        .add_components({"text_embedder": text_embedder, "retriever": retriever})
        .connect("text_embedder.embedding", "retriever.query_embedding")
    )


def build_chat_pipeline(
    settings: Settings,
    template: list,
    response_format: dict[str, Any] | None = None,
) -> Pipeline:
    """prompt_builder -> llm."""

    prompt_builder = ChatPromptBuilder(template=template, required_variables="*")
    llm = OpenAIChatGenerator(
        api_key=Secret.from_token(settings.openai_api_key),
        model=settings.chat_model,
        api_base_url=settings.openai_base_url,
        generation_kwargs=_generation_kwargs(settings, response_format),
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )

    return (
        Pipeline()
        .add_components({"prompt_builder": prompt_builder, "llm": llm})
        .connect("prompt_builder.prompt", "llm.messages")
    )


def build_qa_pipeline(settings: Settings) -> Pipeline:
    return build_chat_pipeline(settings, prompts.QA_TEMPLATE)


def build_notes_pipeline(settings: Settings) -> Pipeline:
    return build_chat_pipeline(settings, prompts.NOTES_TEMPLATE)


def build_summary_pipelines(settings: Settings) -> tuple[Pipeline, Pipeline]:
    structured = build_chat_pipeline(
        settings,
        prompts.SUMMARY_TEMPLATE_JSON,
        prompts.SUMMARY_RESPONSE_FORMAT,
    )
    plain = build_chat_pipeline(settings, prompts.SUMMARY_TEMPLATE_PLAIN)
    return structured, plain
