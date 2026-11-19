"""Weaviate vector store (v4 async client, self-provided vectors)."""

from uuid import UUID

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter, MetadataQuery

from app.domain.entities import Chunk, Collection
from app.infra.vectorstores.payload import chunk_to_payload, payload_to_chunk

# Weaviate class names must start with an uppercase letter.
_PREFIX = "Rag_"

_PROPERTIES = [
    Property(name="chunk_id", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="document_id", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="collection_id", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="chunk_index", data_type=DataType.INT),
    Property(name="text", data_type=DataType.TEXT),
    Property(name="metadata_json", data_type=DataType.TEXT, index_searchable=False, index_filterable=False),
]


def _wname(collection: str) -> str:
    return f"{_PREFIX}{collection}"


class WeaviateVectorStore:
    name = "weaviate"

    def __init__(self, host: str, http_port: int, grpc_port: int) -> None:
        self.client = weaviate.use_async_with_custom(
            http_host=host,
            http_port=http_port,
            http_secure=False,
            grpc_host=host,
            grpc_port=grpc_port,
            grpc_secure=False,
        )

    async def _connected(self) -> weaviate.WeaviateAsyncClient:
        if not self.client.is_connected():
            await self.client.connect()
        return self.client

    async def create_collection(self, collection: str, dim: int) -> None:
        # `dim` is implied by the first inserted vector in Weaviate.
        client = await self._connected()
        await client.collections.create(
            name=_wname(collection),
            properties=_PROPERTIES,
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)
            ),
        )

    async def delete_collection(self, collection: str) -> None:
        client = await self._connected()
        await client.collections.delete(_wname(collection))

    async def collection_exists(self, collection: str) -> bool:
        client = await self._connected()
        return await client.collections.exists(_wname(collection))

    async def list_collections(self) -> list[Collection]:
        client = await self._connected()
        names = await client.collections.list_all(simple=True)
        result = [n[len(_PREFIX) :] for n in names if n.startswith(_PREFIX)]
        return [Collection(id=n, chunk_count=await self.count(n)) for n in sorted(result)]

    async def count(self, collection: str) -> int:
        client = await self._connected()
        agg = await client.collections.get(_wname(collection)).aggregate.over_all(total_count=True)
        return int(agg.total_count or 0)

    async def upsert(self, collection: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        client = await self._connected()
        objects = [
            DataObject(properties=chunk_to_payload(c), uuid=c.id, vector=v)
            for c, v in zip(chunks, vectors, strict=True)
        ]
        result = await client.collections.get(_wname(collection)).data.insert_many(objects)
        if result.has_errors:
            first = next(iter(result.errors.values()))
            raise RuntimeError(f"Weaviate insert failed: {first.message}")

    async def search(self, collection: str, vector: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        client = await self._connected()
        response = await client.collections.get(_wname(collection)).query.near_vector(
            near_vector=vector, limit=top_k, return_metadata=MetadataQuery(distance=True)
        )
        # Cosine distance -> similarity.
        return [
            (payload_to_chunk(o.properties), 1.0 - float(o.metadata.distance or 0.0))
            for o in response.objects
        ]

    async def list_chunks(self, collection: str, document_id: UUID | None = None) -> list[Chunk]:
        client = await self._connected()
        col = client.collections.get(_wname(collection))
        if document_id is None:
            return [payload_to_chunk(o.properties) async for o in col.iterator()]
        response = await col.query.fetch_objects(
            filters=Filter.by_property("document_id").equal(str(document_id)), limit=10_000
        )
        return [payload_to_chunk(o.properties) for o in response.objects]

    async def delete_document(self, collection: str, document_id: UUID) -> None:
        client = await self._connected()
        await client.collections.get(_wname(collection)).data.delete_many(
            where=Filter.by_property("document_id").equal(str(document_id))
        )

    async def delete_chunks(self, collection: str, chunk_ids: list[UUID]) -> None:
        client = await self._connected()
        await client.collections.get(_wname(collection)).data.delete_many(
            where=Filter.by_id().contains_any(list(chunk_ids))
        )

    async def healthcheck(self) -> bool:
        try:
            client = await self._connected()
            return await client.is_ready()
        except Exception:
            return False

    async def close(self) -> None:
        await self.client.close()
