from dataclasses import dataclass, field

from app.domain.value_objects import LLMProviderInfo


@dataclass(frozen=True)
class SystemInfoDTO:
    default_llm_provider: str
    llm_providers: list[LLMProviderInfo]
    vector_store: str
    available_vector_stores: list[str]
    embedding_model: str
    embedding_dim: int
    reranker_model: str
    retrieval_top_k: int
    rerank_top_n: int
    supported_file_types: list[str]


@dataclass(frozen=True)
class HealthDTO:
    status: str  # "ok" | "degraded"
    vector_store: str
    checks: dict[str, bool] = field(default_factory=dict)
