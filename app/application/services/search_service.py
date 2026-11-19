"""Use case: debug search. Hybrid retrieval with every score exposed, optionally reranked."""

from app.application.dto.search import SearchHitDTO, SearchQuery
from app.application.mappers import scored_to_hit
from app.application.services.collection_service import CollectionService
from app.application.services.retrieval_service import RetrievalService


class SearchService:
    def __init__(self, collections: CollectionService, retrieval: RetrievalService) -> None:
        self.collections = collections
        self.retrieval = retrieval

    async def search(self, query: SearchQuery) -> list[SearchHitDTO]:
        await self.collections.require(query.collection)
        hits = await self.retrieval.hybrid_search(query.collection, query.query, query.top_k)
        if query.rerank:
            hits = await self.retrieval.rerank(query.query, hits, query.rerank_top_n)
        return [scored_to_hit(h) for h in hits]
