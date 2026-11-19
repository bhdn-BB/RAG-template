"""pgvector schema: collections + chunks with an HNSW cosine index.

The embedding dimension is taken from EMBEDDING_DIM at migration time. Changing it later
requires a new migration (or `alembic downgrade base && alembic upgrade head` on an empty DB).

Revision ID: 0001
Revises:
Create Date: 2026-11-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op
from app.core.config import get_settings

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    dim = get_settings().embedding_dim
    if dim > 2000:
        raise RuntimeError(f"EMBEDDING_DIM={dim}: pgvector HNSW indexes support at most 2000 dimensions.")

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "vector_collections",
        sa.Column("name", sa.String(64), primary_key=True),
        sa.Column("dim", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "vector_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "collection",
            sa.String(64),
            sa.ForeignKey("vector_collections.name", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("embedding", Vector(dim), nullable=False),
    )
    op.create_index("ix_vector_chunks_collection", "vector_chunks", ["collection"])
    op.create_index("ix_vector_chunks_document_id", "vector_chunks", ["document_id"])
    op.create_index(
        "ix_vector_chunks_embedding_hnsw",
        "vector_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_table("vector_chunks")
    op.drop_table("vector_collections")
