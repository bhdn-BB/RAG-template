"""Dependency injection (dishka): the composition root.

This is the ONLY module that knows both the ports (`app.domain.ports`) and their adapters
(`app.infra`). The application layer receives ports; the API layer receives application services.

Scopes:
- APP : one instance per process (clients with connection pools, BM25 cache, services, graphs).

Infra adapters are provided as async generators so dishka closes them on shutdown.
Tests swap adapters by passing extra providers whose factories use `override=True`.
"""

from collections.abc import AsyncIterator

from dishka import AsyncContainer, Provider, Scope, from_context, make_async_container, provide

from app.application.rag.graph import RAGPipeline
from app.application.services.chat_service import ChatService
from app.application.services.collection_service import CollectionService
from app.application.services.document_service import DocumentService
from app.application.services.retrieval_service import RetrievalService
from app.application.services.search_service import SearchService
from app.application.services.system_service import PipelineConfig, SystemService
from app.core.config import Settings
from app.domain.ports import (
    DocumentParser,
    Embedder,
    LexicalIndex,
    LLMClientFactory,
    Reranker,
    UrlResolver,
    VectorStore,
)
from app.infra.bm25 import BM25Index
from app.infra.embedder import VLLMEmbedder
from app.infra.llm.factory import LLMFactory
from app.infra.parsing import LangChainParser
from app.infra.reranker import VLLMReranker
from app.infra.url_resolver import SafeUrlResolver
from app.infra.vectorstores.factory import VECTOR_STORES, create_vector_store


class InfraProvider(Provider):
    """Adapters for external services: vector DB, vLLM servers, LLM APIs, web pages, LangChain."""

    scope = Scope.APP

    settings = from_context(provides=Settings, scope=Scope.APP)

    @provide
    async def vector_store(self, settings: Settings) -> AsyncIterator[VectorStore]:
        store = create_vector_store(settings)
        yield store
        await store.close()

    @provide
    async def embedder(self, settings: Settings) -> AsyncIterator[Embedder]:
        embedder = VLLMEmbedder(
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            query_instruction=settings.embedding_query_instruction,
            batch_size=settings.embedding_batch_size,
        )
        yield embedder
        await embedder.close()

    @provide
    async def reranker(self, settings: Settings) -> AsyncIterator[Reranker]:
        reranker = VLLMReranker(
            base_url=settings.reranker_base_url,
            model=settings.reranker_model,
            instruction=settings.reranker_instruction,
        )
        yield reranker
        await reranker.close()

    @provide
    def lexical_index(self, vector_store: VectorStore) -> LexicalIndex:
        return BM25Index(vector_store)

    @provide
    def url_resolver(self, settings: Settings) -> UrlResolver:
        return SafeUrlResolver(allow_private=settings.allow_private_urls)

    @provide
    def parser(self, settings: Settings) -> DocumentParser:
        return LangChainParser(settings.chunk_size, settings.chunk_overlap)

    @provide
    def llm_factory(self, settings: Settings) -> LLMClientFactory:
        return LLMFactory(settings)


class ApplicationProvider(Provider):
    """Use cases. Config values are unpacked from Settings here, so services stay config-free."""

    scope = Scope.APP

    @provide
    def pipeline_config(self, settings: Settings) -> PipelineConfig:
        return PipelineConfig(
            vector_store=settings.vector_store,
            available_vector_stores=list(VECTOR_STORES),
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
            reranker_model=settings.reranker_model,
            retrieval_top_k=settings.retrieval_top_k,
            rerank_top_n=settings.rerank_top_n,
        )

    @provide
    def retrieval(
        self,
        settings: Settings,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Reranker,
        lexical_index: LexicalIndex,
    ) -> RetrievalService:
        return RetrievalService(
            vector_store,
            embedder,
            reranker,
            lexical_index,
            top_k=settings.retrieval_top_k,
            rerank_top_n=settings.rerank_top_n,
            rrf_k=settings.rrf_k,
            dense_weight=settings.dense_weight,
            bm25_weight=settings.bm25_weight,
        )

    @provide
    def pipeline(self, settings: Settings, retrieval: RetrievalService) -> RAGPipeline:
        return RAGPipeline(retrieval, settings.max_context_chars_per_chunk)

    @provide
    def collections(
        self, settings: Settings, vector_store: VectorStore, lexical_index: LexicalIndex
    ) -> CollectionService:
        return CollectionService(vector_store, lexical_index, settings.embedding_dim)

    @provide
    def documents(
        self,
        settings: Settings,
        vector_store: VectorStore,
        embedder: Embedder,
        parser: DocumentParser,
        lexical_index: LexicalIndex,
        collections: CollectionService,
        url_resolver: UrlResolver,
    ) -> DocumentService:
        return DocumentService(
            vector_store,
            embedder,
            parser,
            lexical_index,
            collections,
            url_resolver,
            max_upload_bytes=settings.max_upload_mb * 2**20,
        )

    search = provide(SearchService)

    @provide
    def chat(
        self,
        settings: Settings,
        collections: CollectionService,
        pipeline: RAGPipeline,
        llm_factory: LLMClientFactory,
    ) -> ChatService:
        return ChatService(
            collections,
            pipeline,
            llm_factory,
            max_question_chars=settings.max_question_chars,
            max_history_messages=settings.max_history_messages,
            max_chars_per_chunk=settings.max_context_chars_per_chunk,
        )

    system = provide(SystemService)


def create_container(settings: Settings, *extra: Provider) -> AsyncContainer:
    """`extra` providers (e.g. test fakes declared with `override=True`) are applied last."""
    return make_async_container(InfraProvider(), ApplicationProvider(), *extra, context={Settings: settings})
