"""pgvector schema. Managed by Alembic (see `alembic/`).

This is NOT a relational "source of truth" layer: Postgres here is used purely as a
vector database, with the same shape as the Qdrant / Weaviate payloads.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class VectorCollectionORM(Base):
    __tablename__ = "vector_collections"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    dim: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class VectorChunkORM(Base):
    __tablename__ = "vector_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    collection: Mapped[str] = mapped_column(
        String(64), ForeignKey("vector_collections.name", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    # Dimension is fixed by the migration (EMBEDDING_DIM), HNSW index requires it.
    embedding: Mapped[list[float]] = mapped_column(Vector())
