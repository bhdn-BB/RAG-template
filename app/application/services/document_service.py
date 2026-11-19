"""Use cases: ingest sources (files, text, web pages) and manage documents and chunks.

A document is an aggregate of chunks sharing a `document_id`. Chunks can be inspected, edited
(re-embedded in place) and deleted individually.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from app.application.dto.documents import (
    ChunkDTO,
    DocumentDTO,
    IngestTextCommand,
    IngestUrlCommand,
    UpdateChunkCommand,
    UploadedFile,
    UploadResultDTO,
)
from app.application.mappers import chunk_to_dto, document_to_dto
from app.application.services.collection_service import CollectionService
from app.core.exceptions import AppError, NotFoundError, ValidationError
from app.domain.entities import Chunk, Document
from app.domain.ports import DocumentParser, Embedder, LexicalIndex, UrlResolver, VectorStore

logger = logging.getLogger(__name__)


def _aggregate(collection: str, document_id: UUID, chunks: list[Chunk]) -> Document:
    meta = chunks[0].metadata
    uploaded = meta.get("uploaded_at")
    return Document(
        id=document_id,
        collection_id=collection,
        filename=meta.get("filename", ""),
        chunk_count=len(chunks),
        content_type=meta.get("content_type"),
        uploaded_at=datetime.fromisoformat(uploaded) if uploaded else None,
        source_url=meta.get("source_url"),
        metadata=meta.get("user_metadata", {}),
    )


class DocumentService:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        parser: DocumentParser,
        lexical_index: LexicalIndex,
        collections: CollectionService,
        url_resolver: UrlResolver,
        max_upload_bytes: int,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.parser = parser
        self.lexical_index = lexical_index
        self.collections = collections
        self.url_resolver = url_resolver
        self.max_upload_bytes = max_upload_bytes

    # --- ingestion ----------------------------------------------------------------

    async def _index(
        self,
        collection: str,
        pieces: list[tuple[str, dict]],
        filename: str,
        content_type: str | None,
        user_metadata: dict[str, Any],
        source_url: str | None = None,
    ) -> Document:
        document_id = uuid4()
        base_meta = {
            "filename": filename,
            "content_type": content_type,
            "uploaded_at": datetime.now(UTC).isoformat(),
            "user_metadata": user_metadata,
        }
        if source_url:
            base_meta["source_url"] = source_url
        chunks = [
            Chunk(
                id=uuid4(),
                document_id=document_id,
                collection_id=collection,
                index=i,
                text=text,
                metadata={**base_meta, **meta},
            )
            for i, (text, meta) in enumerate(pieces)
        ]
        vectors = await self.embedder.embed_documents([c.text for c in chunks])
        await self.vector_store.upsert(collection, chunks, vectors)
        self.lexical_index.invalidate(collection)
        logger.info("Indexed %s (%d chunks) into '%s'", filename, len(chunks), collection)
        return _aggregate(collection, document_id, chunks)

    async def _ingest_file(self, collection: str, file: UploadedFile) -> Document:
        if len(file.data) > self.max_upload_bytes:
            raise ValidationError(
                f"File '{file.filename}' exceeds the {self.max_upload_bytes // 2**20} MB limit."
            )
        pieces = self.parser.parse(file.filename, file.data)
        return await self._index(collection, pieces, file.filename, file.content_type, {})

    async def ingest_files(self, collection: str, files: list[UploadedFile]) -> UploadResultDTO:
        """Index several files; a failing file is reported in `errors` and does not stop the others."""
        await self.collections.require(collection)
        documents, errors = [], {}
        for file in files:
            try:
                documents.append(document_to_dto(await self._ingest_file(collection, file)))
            except AppError as exc:
                errors[file.filename] = exc.message
        return UploadResultDTO(documents=documents, errors=errors)

    async def ingest_text(self, command: IngestTextCommand) -> DocumentDTO:
        await self.collections.require(command.collection)
        pieces = self.parser.split_text(command.text, {})
        if not pieces:
            raise ValidationError("Text is empty.")
        document = await self._index(
            command.collection, pieces, command.title, "text/plain", command.metadata
        )
        return document_to_dto(document)

    async def ingest_url(self, command: IngestUrlCommand) -> DocumentDTO:
        """SSRF-check the URL, load & parse the page (LangChain WebBaseLoader), index it.

        Re-ingesting the same URL replaces the previous version of that page.
        """
        await self.collections.require(command.collection)
        final_url = await self.url_resolver.resolve(command.url)
        title, pieces = await asyncio.to_thread(self.parser.parse_url, final_url)
        for doc in await self._documents(command.collection):
            if doc.source_url in (command.url, final_url):
                await self.vector_store.delete_document(command.collection, doc.id)
        document = await self._index(
            command.collection, pieces, title, "text/html", command.metadata, source_url=final_url
        )
        return document_to_dto(document)

    # --- reading --------------------------------------------------------------------

    async def _documents(self, collection: str) -> list[Document]:
        grouped: dict[UUID, list[Chunk]] = defaultdict(list)
        for chunk in await self.vector_store.list_chunks(collection):
            grouped[chunk.document_id].append(chunk)
        documents = [_aggregate(collection, doc_id, chunks) for doc_id, chunks in grouped.items()]
        return sorted(
            documents, key=lambda d: d.uploaded_at or datetime.min.replace(tzinfo=UTC), reverse=True
        )

    async def _chunks(self, collection: str, document_id: UUID) -> list[Chunk]:
        await self.collections.require(collection)
        chunks = await self.vector_store.list_chunks(collection, document_id)
        if not chunks:
            raise NotFoundError(f"Document '{document_id}' not found in '{collection}'.")
        return sorted(chunks, key=lambda c: c.index)

    async def list_all(self, collection: str) -> list[DocumentDTO]:
        await self.collections.require(collection)
        return [document_to_dto(d) for d in await self._documents(collection)]

    async def get(self, collection: str, document_id: UUID) -> DocumentDTO:
        return document_to_dto(
            _aggregate(collection, document_id, await self._chunks(collection, document_id))
        )

    async def chunks(self, collection: str, document_id: UUID) -> list[ChunkDTO]:
        return [chunk_to_dto(c) for c in await self._chunks(collection, document_id)]

    # --- changes --------------------------------------------------------------------

    async def delete(self, collection: str, document_id: UUID) -> None:
        await self._chunks(collection, document_id)  # 404 if missing
        await self.vector_store.delete_document(collection, document_id)
        self.lexical_index.invalidate(collection)

    async def _find_chunk(self, collection: str, document_id: UUID, chunk_id: UUID) -> Chunk:
        chunk = next((c for c in await self._chunks(collection, document_id) if c.id == chunk_id), None)
        if chunk is None:
            raise NotFoundError(f"Chunk '{chunk_id}' not found in document '{document_id}'.")
        return chunk

    async def update_chunk(self, command: UpdateChunkCommand) -> ChunkDTO:
        """Edit a chunk's text in place and re-embed it (same id, so the vector is replaced)."""
        if not command.text.strip():
            raise ValidationError("Chunk text is empty.")
        chunk = await self._find_chunk(command.collection, command.document_id, command.chunk_id)
        updated = replace(
            chunk, text=command.text, metadata={**chunk.metadata, "edited_at": datetime.now(UTC).isoformat()}
        )
        vector = (await self.embedder.embed_documents([updated.text]))[0]
        await self.vector_store.upsert(command.collection, [updated], [vector])
        self.lexical_index.invalidate(command.collection)
        return chunk_to_dto(updated)

    async def delete_chunk(self, collection: str, document_id: UUID, chunk_id: UUID) -> None:
        await self._find_chunk(collection, document_id, chunk_id)
        await self.vector_store.delete_chunks(collection, [chunk_id])
        self.lexical_index.invalidate(collection)

    async def reindex(self, collection: str, document_id: UUID) -> DocumentDTO:
        """Re-embed stored chunk texts (e.g. after changing the embedding model, same dimension)."""
        chunks = await self._chunks(collection, document_id)
        vectors = await self.embedder.embed_documents([c.text for c in chunks])
        await self.vector_store.upsert(collection, chunks, vectors)
        self.lexical_index.invalidate(collection)
        return document_to_dto(_aggregate(collection, document_id, chunks))
