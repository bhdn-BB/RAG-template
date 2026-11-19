from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter

from app.api.schemas.system import HealthResponse
from app.application.services.system_service import SystemService

router = APIRouter(tags=["health"], route_class=DishkaRoute)


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/health/ready", response_model=HealthResponse)
async def ready(service: FromDishka[SystemService]) -> HealthResponse:
    return HealthResponse.from_dto(await service.health())
