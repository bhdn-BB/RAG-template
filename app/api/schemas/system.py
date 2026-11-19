from pydantic import BaseModel

from app.application.dto.system import HealthDTO, SystemInfoDTO


class ProviderSchema(BaseModel):
    configured: bool
    default_model: str
    key_env: str | None = None


class LLMInfoSchema(BaseModel):
    default: str
    providers: dict[str, ProviderSchema]


class VectorStoreInfoSchema(BaseModel):
    active: str
    available: list[str]


class EmbeddingInfoSchema(BaseModel):
    model: str
    dim: int


class RerankerInfoSchema(BaseModel):
    model: str


class RetrievalInfoSchema(BaseModel):
    top_k: int
    rerank_top_n: int


class SystemInfoResponse(BaseModel):
    llm: LLMInfoSchema
    vector_store: VectorStoreInfoSchema
    embedding: EmbeddingInfoSchema
    reranker: RerankerInfoSchema
    retrieval: RetrievalInfoSchema
    supported_file_types: list[str]

    @classmethod
    def from_dto(cls, dto: SystemInfoDTO) -> "SystemInfoResponse":
        return cls(
            llm=LLMInfoSchema(
                default=dto.default_llm_provider,
                providers={
                    p.name: ProviderSchema(
                        configured=p.configured, default_model=p.default_model, key_env=p.key_env
                    )
                    for p in dto.llm_providers
                },
            ),
            vector_store=VectorStoreInfoSchema(
                active=dto.vector_store, available=dto.available_vector_stores
            ),
            embedding=EmbeddingInfoSchema(model=dto.embedding_model, dim=dto.embedding_dim),
            reranker=RerankerInfoSchema(model=dto.reranker_model),
            retrieval=RetrievalInfoSchema(top_k=dto.retrieval_top_k, rerank_top_n=dto.rerank_top_n),
            supported_file_types=dto.supported_file_types,
        )


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    checks: dict[str, bool]

    @classmethod
    def from_dto(cls, dto: HealthDTO) -> "HealthResponse":
        return cls(status=dto.status, vector_store=dto.vector_store, checks=dto.checks)
