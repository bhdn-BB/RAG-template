"""Qdrant vector store (one Qdrant collection per RAG collection)."""

from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from app.domain.entities import Chunk, Collection
from app.infra.vectorstores.payload import chunk_to_payload, payload_to_chunk

_PREFIX = "rag_"
_SCROLL_PAGE = 512


def _qname(collection: str) -> str:
    return f"{_PREFIX}{collection}"


def _doc_filter(document_id: UUID) -> qm.Filter:
    return qm.Filter(must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=str(document_id)))])


class QdrantVectorStore:
    name = "qdrant"

    def __init__(self, url: str, api_key: str | None = None) -> None:
        self.client = AsyncQdrantClient(url=url, api_key=api_key)

    async def create_collection(self, collection: str, dim: int) -> None:
        await self.client.create_collection(
            collection_name=_qname(collection),
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        await self.client.create_payload_index(
            collection_name=_qname(collection),
            field_name="document_id",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )

    async def delete_collection(self, collection: str) -> None:
        await self.client.delete_collection(collection_name=_qname(collection))

    async def collection_exists(self, collection: str) -> bool:
        return await self.client.collection_exists(collection_name=_qname(collection))

    async def list_collections(self) -> list[Collection]:
        response = await self.client.get_collections()
        names = sorted(c.name[len(_PREFIX) :] for c in response.collections if c.name.startswith(_PREFIX))
        return [Collection(id=n, chunk_count=await self.count(n)) for n in names]

    async def count(self, collection: str) -> int:
        return (await self.client.count(collection_name=_qname(collection), exact=True)).count

    async def upsert(self, collection: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            qm.PointStruct(id=str(c.id), vector=v, payload=chunk_to_payload(c))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        await self.client.upsert(collection_name=_qname(collection), points=points, wait=True)

    async def search(self, collection: str, vector: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        response = await self.client.query_points(
            collection_name=_qname(collection), query=vector, limit=top_k, with_payload=True
        )
        return [(payload_to_chunk(p.payload or {}), float(p.score)) for p in response.points]

    async def list_chunks(self, collection: str, document_id: UUID | None = None) -> list[Chunk]:
        chunks: list[Chunk] = []
        offset = None
        while True:
            records, offset = await self.client.scroll(
                collection_name=_qname(collection),
                scroll_filter=_doc_filter(document_id) if document_id else None,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(payload_to_chunk(r.payload or {}) for r in records)
            if offset is None:
                return chunks

    async def delete_document(self, collection: str, document_id: UUID) -> None:
        await self.client.delete(
            collection_name=_qname(collection),
            points_selector=qm.FilterSelector(filter=_doc_filter(document_id)),
            wait=True,
        )

    async def delete_chunks(self, collection: str, chunk_ids: list[UUID]) -> None:
        await self.client.delete(
            collection_name=_qname(collection),
            points_selector=qm.PointIdsList(points=[str(c) for c in chunk_ids]),
            wait=True,
        )

    async def healthcheck(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self.client.close()
