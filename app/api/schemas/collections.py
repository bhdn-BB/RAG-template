from pydantic import BaseModel, Field

from app.application.dto.collections import CollectionDTO


class CollectionCreate(BaseModel):
    name: str = Field(..., examples=["cyber_docs"], description="^[a-z][a-z0-9_]{0,62}$")


class CollectionRead(BaseModel):
    name: str
    chunk_count: int = 0
    document_count: int | None = None

    @classmethod
    def from_dto(cls, dto: CollectionDTO) -> "CollectionRead":
        return cls(name=dto.name, chunk_count=dto.chunk_count, document_count=dto.document_count)
