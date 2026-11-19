"""FastAPI application entrypoint."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dishka import AsyncContainer
from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routers import chat, collections, documents, health, providers, search
from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.di.providers import create_container

logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")


def create_app(settings: Settings | None = None, container: AsyncContainer | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    container = container or create_container(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger.info("Starting with vector store '%s'", settings.vector_store)
        yield
        await container.close()  # finalizes APP-scoped clients (vector DB, httpx pools)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Ukrainian RAG: hybrid search (dense + BM25), Qwen3 embeddings & reranker on vLLM, LangGraph."
        ),
        lifespan=lifespan,
        docs_url=f"{settings.api_prefix}/docs",
        openapi_url=f"{settings.api_prefix}/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("%s: %s", type(exc).__name__, exc.message)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    for module in (health, providers, collections, documents, search, chat):
        app.include_router(module.router, prefix=settings.api_prefix)

    setup_dishka(container, app)
    return app


app = create_app()
