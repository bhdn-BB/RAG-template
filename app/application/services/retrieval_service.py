"""Hybrid retrieval: dense (vector) + BM25, fused with Reciprocal Rank Fusion, then cross-encoder rerank."""

import asyncio
from uuid import UUID

from app.domain.entities import Chunk
from app.domain.ports import Embedder, LexicalIndex, Reranker, VectorStore
from app.domain.value_objects import ScoredChunk


def reciprocal_rank_fusion(
    dense: list[tuple[Chunk, float]],
    lexical: list[Chunk],
    k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[ScoredChunk]:
    """Fuse two ranked lists: score(d) = sum(w / (k + rank)). Rank-based, so score scales don't matter."""
    fused: dict[UUID, dict] = {}

    def add(chunk: Chunk, rank: int, weight: float) -> dict:
        entry = fused.setdefault(chunk.id, {"chunk": chunk, "rrf_score": 0.0})
        entry["rrf_score"] += weight / (k + rank)
        return entry

    for rank, (chunk, score) in enumerate(dense, start=1):
        add(chunk, rank, dense_weight)["dense_score"] = score
    for rank, chunk in enumerate(lexical, start=1):
        add(chunk, rank, bm25_weight)["bm25_rank"] = rank
    ranked = sorted(fused.values(), key=lambda e: e["rrf_score"], reverse=True)
    return [ScoredChunk(**entry) for entry in ranked]


class RetrievalService:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Reranker,
        bm25: LexicalIndex,
        top_k: int = 10,
        rerank_top_n: int = 5,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.bm25 = bm25
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    async def hybrid_search(self, collection: str, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        """Uses the raw user question only (no query rewriting)."""
        k = top_k or self.top_k

        async def dense() -> list[tuple[Chunk, float]]:
            vector = await self.embedder.embed_query(query)
            return await self.vector_store.search(collection, vector, k)

        dense_results, bm25_results = await asyncio.gather(dense(), self.bm25.search(collection, query, k))
        fused = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            k=self.rrf_k,
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
        )
        return fused[:k]

    async def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int | None = None
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        scores = await self.reranker.rerank(query, [c.chunk.text for c in candidates])
        rescored = [c.with_scores(rerank_score=s) for c, s in zip(candidates, scores, strict=True)]
        rescored.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        return rescored[: top_n or self.rerank_top_n]
