"""Use cases: system information (providers, models, pipeline settings) and health checks."""

import asyncio
from dataclasses import dataclass

from app.application.dto.system import HealthDTO, SystemInfoDTO
from app.domain.ports import DocumentParser, Embedder, LLMClientFactory, Reranker, VectorStore


@dataclass(frozen=True)
class PipelineConfig:
    """Static configuration the system reports about itself (filled in by the composition root)."""

    vector_store: str
    available_vector_stores: list[str]
    embedding_model: str
    embedding_dim: int
    reranker_model: str
    retrieval_top_k: int
    rerank_top_n: int


class SystemService:
    def __init__(
        self,
        config: PipelineConfig,
        llm_factory: LLMClientFactory,
        parser: DocumentParser,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Reranker,
    ) -> None:
        self.config = config
        self.llm_factory = llm_factory
        self.parser = parser
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker

    def info(self) -> SystemInfoDTO:
        c = self.config
        return SystemInfoDTO(
            default_llm_provider=self.llm_factory.default_provider(),
            llm_providers=self.llm_factory.providers(),
            vector_store=c.vector_store,
            available_vector_stores=c.available_vector_stores,
            embedding_model=c.embedding_model,
            embedding_dim=c.embedding_dim,
            reranker_model=c.reranker_model,
            retrieval_top_k=c.retrieval_top_k,
            rerank_top_n=c.rerank_top_n,
            supported_file_types=self.parser.supported_extensions(),
        )

    async def health(self) -> HealthDTO:
        vector_store, embedding, reranker = await asyncio.gather(
            self.vector_store.healthcheck(), self.embedder.healthcheck(), self.reranker.healthcheck()
        )
        checks = {"vector_store": vector_store, "embedding": embedding, "reranker": reranker}
        return HealthDTO(
            status="ok" if all(checks.values()) else "degraded",
            vector_store=self.config.vector_store,
            checks=checks,
        )
