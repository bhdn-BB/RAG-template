"""Mapping between domain objects and application DTOs (kept in one place)."""

from uuid import UUID

from app.application.dto.chat import SourceDTO
from app.application.dto.collections import CollectionDTO
from app.application.dto.documents import ChunkDTO, DocumentDTO
from app.application.dto.search import SearchHitDTO
from app.domain.entities import Chunk, Collection, Document
from app.domain.value_objects import ScoredChunk


def collection_to_dto(c: Collection, with_documents: bool = False) -> CollectionDTO:
    return CollectionDTO(
        name=c.id, chunk_count=c.chunk_count, document_count=c.document_count if with_documents else None
    )


def document_to_dto(d: Document) -> DocumentDTO:
    return DocumentDTO(
        id=d.id,
        collection=d.collection_id,
        filename=d.filename,
        chunk_count=d.chunk_count,
        content_type=d.content_type,
        uploaded_at=d.uploaded_at,
        source_url=d.source_url,
        metadata=d.metadata,
    )


def chunk_to_dto(c: Chunk) -> ChunkDTO:
    return ChunkDTO(
        id=c.id,
        document_id=c.document_id,
        collection=c.collection_id,
        index=c.index,
        text=c.text,
        metadata=c.metadata,
    )


def scored_to_hit(s: ScoredChunk) -> SearchHitDTO:
    return SearchHitDTO(
        chunk_id=s.chunk.id,
        document_id=s.chunk.document_id,
        filename=s.chunk.filename,
        index=s.chunk.index,
        text=s.chunk.text,
        dense_score=s.dense_score,
        bm25_rank=s.bm25_rank,
        rrf_score=s.rrf_score,
        rerank_score=s.rerank_score,
    )


def scored_to_source(n: int, s: ScoredChunk) -> SourceDTO:
    c = s.chunk
    return SourceDTO(
        n=n,
        chunk_id=c.id,
        document_id=c.document_id,
        collection=c.collection_id,
        filename=c.filename,
        index=c.index,
        text=c.text,
        source=c.source,
        source_url=c.source_url,
        metadata=c.metadata,
        rerank_score=s.rerank_score,
        rrf_score=s.rrf_score,
    )


def source_to_scored(s: SourceDTO) -> ScoredChunk:
    metadata = {**s.metadata, "filename": s.filename}
    if s.source_url:
        metadata["source_url"] = s.source_url
    chunk = Chunk(
        id=UUID(str(s.chunk_id)),
        document_id=UUID(str(s.document_id)),
        collection_id=s.collection,
        index=s.index,
        text=s.text,
        metadata=metadata,
    )
    return ScoredChunk(chunk=chunk, rerank_score=s.rerank_score, rrf_score=s.rrf_score)
