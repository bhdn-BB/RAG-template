"""Shared helpers to (de)serialize a `Chunk` to a flat payload, so every backend stores the same shape."""

import json
from typing import Any
from uuid import UUID

from app.domain.entities import Chunk


def chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": str(chunk.id),
        "document_id": str(chunk.document_id),
        "collection_id": chunk.collection_id,
        "chunk_index": chunk.index,
        "text": chunk.text,
        "metadata_json": json.dumps(chunk.metadata, ensure_ascii=False, default=str),
    }


def payload_to_chunk(payload: dict[str, Any]) -> Chunk:
    return Chunk(
        id=UUID(str(payload["chunk_id"])),
        document_id=UUID(str(payload["document_id"])),
        collection_id=str(payload["collection_id"]),
        index=int(payload["chunk_index"]),
        text=str(payload["text"]),
        metadata=json.loads(payload.get("metadata_json") or "{}"),
    )
