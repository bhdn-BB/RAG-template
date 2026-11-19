from uuid import UUID

from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter, File, UploadFile, status

from app.api.schemas.documents import (
    ChunkRead,
    ChunkUpdate,
    DocumentRead,
    TextDocumentCreate,
    UploadResult,
    UrlDocumentCreate,
)
from app.application.dto.documents import UpdateChunkCommand, UploadedFile
from app.application.services.document_service import DocumentService

router = APIRouter(prefix="/collections/{collection}/documents", tags=["documents"], route_class=DishkaRoute)


@router.post("", response_model=UploadResult, status_code=status.HTTP_201_CREATED)
async def upload_documents(
    collection: str,
    service: FromDishka[DocumentService],
    files: list[UploadFile] = File(..., description="PDF, DOCX, TXT, MD, HTML"),
) -> UploadResult:
    """Parse (LangChain), chunk, embed and index files. Per-file errors are reported, not fatal."""
    uploads = [UploadedFile(f.filename or "upload", await f.read(), f.content_type) for f in files]
    return UploadResult.from_dto(await service.ingest_files(collection, uploads))


@router.post("/url", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def ingest_url(
    collection: str, body: UrlDocumentCreate, service: FromDishka[DocumentService]
) -> DocumentRead:
    """Fetch a web page (SSRF-protected), extract the main content with LangChain and index it.
    Re-adding the same URL replaces the old version."""
    return DocumentRead.from_dto(await service.ingest_url(body.to_command(collection)))


@router.post("/text", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def ingest_text(
    collection: str, body: TextDocumentCreate, service: FromDishka[DocumentService]
) -> DocumentRead:
    return DocumentRead.from_dto(await service.ingest_text(body.to_command(collection)))


@router.get("", response_model=list[DocumentRead])
async def list_documents(collection: str, service: FromDishka[DocumentService]) -> list[DocumentRead]:
    return [DocumentRead.from_dto(d) for d in await service.list_all(collection)]


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(
    collection: str, document_id: UUID, service: FromDishka[DocumentService]
) -> DocumentRead:
    return DocumentRead.from_dto(await service.get(collection, document_id))


@router.get("/{document_id}/chunks", response_model=list[ChunkRead])
async def get_chunks(
    collection: str, document_id: UUID, service: FromDishka[DocumentService]
) -> list[ChunkRead]:
    return [ChunkRead.from_dto(c) for c in await service.chunks(collection, document_id)]


@router.patch("/{document_id}/chunks/{chunk_id}", response_model=ChunkRead)
async def update_chunk(
    collection: str,
    document_id: UUID,
    chunk_id: UUID,
    body: ChunkUpdate,
    service: FromDishka[DocumentService],
) -> ChunkRead:
    """Edit a chunk's text; it is re-embedded and its vector replaced in place."""
    command = UpdateChunkCommand(
        collection=collection, document_id=document_id, chunk_id=chunk_id, text=body.text
    )
    return ChunkRead.from_dto(await service.update_chunk(command))


@router.delete("/{document_id}/chunks/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chunk(
    collection: str, document_id: UUID, chunk_id: UUID, service: FromDishka[DocumentService]
) -> None:
    await service.delete_chunk(collection, document_id, chunk_id)


@router.post("/{document_id}/reindex", response_model=DocumentRead)
async def reindex_document(
    collection: str, document_id: UUID, service: FromDishka[DocumentService]
) -> DocumentRead:
    return DocumentRead.from_dto(await service.reindex(collection, document_id))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(collection: str, document_id: UUID, service: FromDishka[DocumentService]) -> None:
    await service.delete(collection, document_id)
