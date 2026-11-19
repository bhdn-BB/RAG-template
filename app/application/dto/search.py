from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class SearchQuery:
    collection: str
    query: str
    top_k: int | None = None
    rerank: bool = True
    rerank_top_n: int | None = None


@dataclass(frozen=True)
class SearchHitDTO:
    chunk_id: UUID
    document_id: UUID
    filename: str
    index: int
    text: str
    dense_score: float | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
