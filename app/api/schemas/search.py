from uuid import UUID

from pydantic import BaseModel, Field

from app.application.dto.search import SearchHitDTO, SearchQuery


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(None, ge=1, le=100)
    rerank: bool = True
    rerank_top_n: int | None = Field(None, ge=1, le=50)

    def to_query(self, collection: str) -> SearchQuery:
        return SearchQuery(
            collection=collection,
            query=self.query,
            top_k=self.top_k,
            rerank=self.rerank,
            rerank_top_n=self.rerank_top_n,
        )


class SearchHit(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    index: int
    text: str
    dense_score: float | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None

    @classmethod
    def from_dto(cls, dto: SearchHitDTO) -> "SearchHit":
        return cls(**dto.__dict__)


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
