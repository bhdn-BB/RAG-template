import json
import logging
from collections.abc import AsyncIterator

from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.api.schemas.chat import AnswerMetaSchema, ChatRequest, ChatResponse, RegenerateRequest
from app.application.dto.chat import ChatStreamDTO
from app.application.services.chat_service import ChatService
from app.core.exceptions import AppError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"], route_class=DishkaRoute)


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, service: FromDishka[ChatService]) -> ChatResponse:
    """Ask a question: hybrid retrieve (3) -> rerank (1) -> LLM. Send previous turns in `history`."""
    return ChatResponse.from_answer(await service.ask(body.to_command()))


@router.post("/regenerate", response_model=ChatResponse)
async def regenerate(body: RegenerateRequest, service: FromDishka[ChatService]) -> ChatResponse:
    """Re-call the LLM with the full previous context (`sources`) and history. No retrieval."""
    return ChatResponse.from_answer(await service.regenerate(body.to_command()))


def _sse(stream: ChatStreamDTO) -> EventSourceResponse:
    """Events: `sources` (once, with metadata) -> `token` (many) -> `done`, or `error`."""

    async def events() -> AsyncIterator[dict]:
        meta = AnswerMetaSchema.from_dto(stream.meta).model_dump(mode="json")
        yield {"event": "sources", "data": json.dumps(meta, ensure_ascii=False)}
        try:
            async for token in stream.tokens:
                yield {"event": "token", "data": json.dumps({"text": token}, ensure_ascii=False)}
        except AppError as exc:
            logger.exception("Streaming failed")
            yield {"event": "error", "data": json.dumps({"detail": exc.message})}
            return
        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(events())


@router.post("/stream")
async def chat_stream(body: ChatRequest, service: FromDishka[ChatService]) -> EventSourceResponse:
    return _sse(await service.ask_stream(body.to_command()))


@router.post("/regenerate/stream")
async def regenerate_stream(body: RegenerateRequest, service: FromDishka[ChatService]) -> EventSourceResponse:
    return _sse(await service.regenerate_stream(body.to_command()))
