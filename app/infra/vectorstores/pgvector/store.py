"""pgvector vector store (async SQLAlchemy + asyncpg). Schema comes from Alembic migrations."""

from uuid import UUID

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.exceptions import ValidationError
from app.domain.entities import Chunk, Collection
from app.infra.vectorstores.pgvector.models import VectorChunkORM, VectorCollectionORM


def _to_chunk(row: VectorChunkORM) -> Chunk:
    return Chunk(
        id=row.id,
        document_id=row.document_id,
        collection_id=row.collection,
        index=row.chunk_index,
        text=row.text,
        metadata=row.meta or {},
    )


class PgVectorStore:
    name = "pgvector"

    def __init__(self, url: str) -> None:
        self.engine = create_async_engine(url, pool_pre_ping=True)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def create_collection(self, collection: str, dim: int) -> None:
        async with self.sessions.begin() as s:
            s.add(VectorCollectionORM(name=collection, dim=dim))

    async def delete_collection(self, collection: str) -> None:
        async with self.sessions.begin() as s:
            await s.execute(delete(VectorCollectionORM).where(VectorCollectionORM.name == collection))

    async def collection_exists(self, collection: str) -> bool:
        async with self.sessions() as s:
            return await s.get(VectorCollectionORM, collection) is not None

    async def list_collections(self) -> list[Collection]:
        stmt = (
            select(VectorCollectionORM.name, func.count(VectorChunkORM.id))
            .outerjoin(VectorChunkORM, VectorChunkORM.collection == VectorCollectionORM.name)
            .group_by(VectorCollectionORM.name)
            .order_by(VectorCollectionORM.name)
        )
        async with self.sessions() as s:
            return [Collection(id=name, chunk_count=n) for name, n in (await s.execute(stmt)).all()]

    async def count(self, collection: str) -> int:
        stmt = select(func.count()).select_from(VectorChunkORM).where(VectorChunkORM.collection == collection)
        async with self.sessions() as s:
            return int((await s.execute(stmt)).scalar_one())

    async def upsert(self, collection: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        rows = [
            {
                "id": c.id,
                "collection": collection,
                "document_id": c.document_id,
                "chunk_index": c.index,
                "text": c.text,
                "metadata": c.metadata,
                "embedding": v,
            }
            for c, v in zip(chunks, vectors, strict=True)
        ]
        stmt = insert(VectorChunkORM.__table__).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "text": stmt.excluded.text,
                "metadata": stmt.excluded.metadata,
                "embedding": stmt.excluded.embedding,
            },
        )
        try:
            async with self.sessions.begin() as s:
                await s.execute(stmt)
        except Exception as exc:  # dimension mismatch is the most common misconfiguration
            if "dimensions" in str(exc):
                raise ValidationError(
                    "Embedding dimension does not match the pgvector column. "
                    "EMBEDDING_DIM must equal the value used when running migrations."
                ) from exc
            raise

    async def search(self, collection: str, vector: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        distance = VectorChunkORM.embedding.cosine_distance(vector)
        stmt = (
            select(VectorChunkORM, (1 - distance).label("score"))
            .where(VectorChunkORM.collection == collection)
            .order_by(distance)
            .limit(top_k)
        )
        async with self.sessions() as s:
            # pgvector >= 0.8: keep scanning the HNSW index until enough rows pass the filter.
            await s.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            return [(_to_chunk(row), float(score)) for row, score in (await s.execute(stmt)).all()]

    async def list_chunks(self, collection: str, document_id: UUID | None = None) -> list[Chunk]:
        stmt = select(VectorChunkORM).where(VectorChunkORM.collection == collection)
        if document_id is not None:
            stmt = stmt.where(VectorChunkORM.document_id == document_id)
        async with self.sessions() as s:
            return [
                _to_chunk(r) for r in (await s.execute(stmt.order_by(VectorChunkORM.chunk_index))).scalars()
            ]

    async def delete_document(self, collection: str, document_id: UUID) -> None:
        async with self.sessions.begin() as s:
            await s.execute(
                delete(VectorChunkORM).where(
                    VectorChunkORM.collection == collection, VectorChunkORM.document_id == document_id
                )
            )

    async def delete_chunks(self, collection: str, chunk_ids: list[UUID]) -> None:
        async with self.sessions.begin() as s:
            await s.execute(
                delete(VectorChunkORM).where(
                    VectorChunkORM.collection == collection, VectorChunkORM.id.in_(chunk_ids)
                )
            )

    async def healthcheck(self) -> bool:
        try:
            async with self.sessions() as s:
                await s.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self.engine.dispose()
