"""Use cases: manage collections."""

from app.application.dto.collections import CollectionDTO
from app.application.mappers import collection_to_dto
from app.core.exceptions import ConflictError, NotFoundError
from app.domain.entities import Collection
from app.domain.ports import LexicalIndex, VectorStore


class CollectionService:
    def __init__(self, vector_store: VectorStore, lexical_index: LexicalIndex, embedding_dim: int) -> None:
        self.vector_store = vector_store
        self.lexical_index = lexical_index
        self.embedding_dim = embedding_dim

    async def create(self, name: str) -> CollectionDTO:
        Collection.validate_name(name)
        if await self.vector_store.collection_exists(name):
            raise ConflictError(f"Collection '{name}' already exists.")
        await self.vector_store.create_collection(name, self.embedding_dim)
        return collection_to_dto(Collection(id=name))

    async def list_all(self) -> list[CollectionDTO]:
        return [collection_to_dto(c) for c in await self.vector_store.list_collections()]

    async def require(self, name: str) -> None:
        """Raise 404 if the collection does not exist (used by other use cases too)."""
        if not await self.vector_store.collection_exists(name):
            raise NotFoundError(f"Collection '{name}' not found.")

    async def get(self, name: str) -> CollectionDTO:
        await self.require(name)
        chunks = await self.vector_store.list_chunks(name)
        collection = Collection(
            id=name, chunk_count=len(chunks), document_count=len({c.document_id for c in chunks})
        )
        return collection_to_dto(collection, with_documents=True)

    async def delete(self, name: str) -> None:
        await self.require(name)
        await self.vector_store.delete_collection(name)
        self.lexical_index.invalidate(name)
