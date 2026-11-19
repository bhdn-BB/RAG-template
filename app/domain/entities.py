"""Domain entities: objects with an identity (an ID) that persists over time.

Everything is stored in the vector database:
- a `Collection` is a vector-DB collection (its name is its ID);
- a `Chunk` is one point / row, its payload carries document info;
- a `Document` is an aggregate view over the chunks that share `document_id`.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from app.core.exceptions import ValidationError

# A safe subset every vector DB accepts as (part of) a collection/class name.
COLLECTION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


@dataclass
class Collection:
    id: str  # collection name, unique within the vector store
    chunk_count: int = 0
    document_count: int = 0

    @staticmethod
    def validate_name(name: str) -> str:
        if not COLLECTION_NAME_RE.match(name):
            raise ValidationError(
                "Collection name must match ^[a-z][a-z0-9_]{0,62}$ (lowercase letters, digits, underscores)."
            )
        return name


@dataclass
class Document:
    id: UUID
    collection_id: str
    filename: str
    chunk_count: int
    content_type: str | None = None
    uploaded_at: datetime | None = None
    source_url: str | None = None  # set for web pages
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: UUID
    document_id: UUID
    collection_id: str
    index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        return str(self.metadata.get("filename", ""))

    @property
    def source_url(self) -> str | None:
        return self.metadata.get("source_url")

    @property
    def source(self) -> str:
        """Human-readable origin: the page URL for web sources, otherwise the file name (+ page)."""
        if self.source_url:
            return self.source_url
        page = self.metadata.get("page_label") or self.metadata.get("page")
        return f"{self.filename}, p. {page}" if page not in (None, "") else self.filename
