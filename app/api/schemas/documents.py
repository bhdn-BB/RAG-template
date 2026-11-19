from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.application.dto.documents import (
    ChunkDTO,
    DocumentDTO,
    IngestTextCommand,
    IngestUrlCommand,
    UploadResultDTO,
)


class TextDocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=512)
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_command(self, collection: str) -> IngestTextCommand:
        return IngestTextCommand(
            collection=collection, title=self.title, text=self.text, metadata=self.metadata
        )


class UrlDocumentCreate(BaseModel):
    url: str = Field(..., examples=["https://pro.kyiv.ua/vartist-proizdu-kyiv/"], max_length=2048)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_command(self, collection: str) -> IngestUrlCommand:
        return IngestUrlCommand(collection=collection, url=self.url, metadata=self.metadata)


class ChunkUpdate(BaseModel):
    text: str = Field(..., min_length=1, max_length=20_000)


class DocumentRead(BaseModel):
    id: UUID
    collection: str
    filename: str
    chunk_count: int
    content_type: str | None = None
    uploaded_at: datetime | None = None
    source_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dto(cls, dto: DocumentDTO) -> "DocumentRead":
        return cls(
            id=dto.id,
            collection=dto.collection,
            filename=dto.filename,
            chunk_count=dto.chunk_count,
            content_type=dto.content_type,
            uploaded_at=dto.uploaded_at,
            source_url=dto.source_url,
            metadata=dto.metadata,
        )


class ChunkRead(BaseModel):
    id: UUID
    document_id: UUID
    index: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # The record as stored in the vector DB (point id + payload), shown in the UI's DB inspector.
    stored_payload: dict[str, Any]

    @classmethod
    def from_dto(cls, dto: ChunkDTO) -> "ChunkRead":
        return cls(
            id=dto.id,
            document_id=dto.document_id,
            index=dto.index,
            text=dto.text,
            metadata=dto.metadata,
            stored_payload={
                "chunk_id": str(dto.id),
                "document_id": str(dto.document_id),
                "collection_id": dto.collection,
                "chunk_index": dto.index,
                "text": dto.text,
                "metadata": dto.metadata,
            },
        )


class UploadResult(BaseModel):
    documents: list[DocumentRead]
    errors: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_dto(cls, dto: UploadResultDTO) -> "UploadResult":
        return cls(documents=[DocumentRead.from_dto(d) for d in dto.documents], errors=dto.errors)
