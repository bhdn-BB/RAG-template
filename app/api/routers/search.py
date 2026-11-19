from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter

from app.api.schemas.search import SearchHit, SearchRequest, SearchResponse
from app.application.services.search_service import SearchService

router = APIRouter(prefix="/collections/{collection}/search", tags=["search"], route_class=DishkaRoute)


@router.post("", response_model=SearchResponse)
async def search(collection: str, body: SearchRequest, service: FromDishka[SearchService]) -> SearchResponse:
    """Debug endpoint: hybrid search (dense + BM25 -> RRF), optionally reranked. Shows every score."""
    hits = await service.search(body.to_query(collection))
    return SearchResponse(query=body.query, hits=[SearchHit.from_dto(h) for h in hits])
