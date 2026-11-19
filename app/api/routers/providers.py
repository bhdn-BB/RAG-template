from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter

from app.api.schemas.system import SystemInfoResponse
from app.application.services.system_service import SystemService

router = APIRouter(prefix="/providers", tags=["providers"], route_class=DishkaRoute)


@router.get("", response_model=SystemInfoResponse)
async def providers(service: FromDishka[SystemService]) -> SystemInfoResponse:
    """LLM providers (with key status), vector store, models and retrieval settings."""
    return SystemInfoResponse.from_dto(service.info())
