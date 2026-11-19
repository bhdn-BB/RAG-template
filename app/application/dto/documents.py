from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

# --- commands (input) ---------------------------------------------------------


@dataclass(frozen=True)
class UploadedFile:
    filename: str
    data: bytes
    content_type: str | None = None


@dataclass(frozen=True)
class IngestTextCommand:
    collection: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestUrlCommand:
    collection: str
    url: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UpdateChunkCommand:
    collection: str
    document_id: UUID
    chunk_id: UUID
    text: str


# --- results (output) ---------------------------------------------------------


@dataclass(frozen=True)
class DocumentDTO:
    id: UUID
    collection: str
    filename: str
    chunk_count: int
    content_type: str | None = None
    uploaded_at: datetime | None = None
    source_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkDTO:
    """A chunk exactly as it is stored in the vector DB (text + payload)."""

    id: UUID
    document_id: UUID
    collection: str
    index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UploadResultDTO:
    documents: list[DocumentDTO]
    errors: dict[str, str] = field(default_factory=dict)  # filename -> error message
