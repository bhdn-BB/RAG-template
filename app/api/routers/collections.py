from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter, status

from app.api.schemas.collections import CollectionCreate, CollectionRead
from app.application.services.collection_service import CollectionService

router = APIRouter(prefix="/collections", tags=["collections"], route_class=DishkaRoute)


@router.post("", response_model=CollectionRead, status_code=status.HTTP_201_CREATED)
async def create_collection(body: CollectionCreate, service: FromDishka[CollectionService]) -> CollectionRead:
    return CollectionRead.from_dto(await service.create(body.name))


@router.get("", response_model=list[CollectionRead])
async def list_collections(service: FromDishka[CollectionService]) -> list[CollectionRead]:
    return [CollectionRead.from_dto(c) for c in await service.list_all()]


@router.get("/{name}", response_model=CollectionRead)
async def get_collection(name: str, service: FromDishka[CollectionService]) -> CollectionRead:
    return CollectionRead.from_dto(await service.get(name))


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(name: str, service: FromDishka[CollectionService]) -> None:
    await service.delete(name)
