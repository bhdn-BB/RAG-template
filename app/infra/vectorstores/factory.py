"""Vector store factory: a registry of creator functions keyed by provider name.

To add a backend: implement the `VectorStore` protocol and register a creator here.
Backends are imported lazily so unused client libraries are never loaded.
"""

from collections.abc import Callable

from app.core.config import Settings
from app.core.exceptions import ProviderNotConfiguredError
from app.domain.ports import VectorStore

VectorStoreCreator = Callable[[Settings], VectorStore]


def _qdrant(s: Settings) -> VectorStore:
    from app.infra.vectorstores.qdrant_store import QdrantVectorStore

    return QdrantVectorStore(url=s.qdrant_url, api_key=s.qdrant_api_key)


def _pgvector(s: Settings) -> VectorStore:
    from app.infra.vectorstores.pgvector.store import PgVectorStore

    return PgVectorStore(url=s.pgvector_url)


def _weaviate(s: Settings) -> VectorStore:
    from app.infra.vectorstores.weaviate_store import WeaviateVectorStore

    return WeaviateVectorStore(
        host=s.weaviate_host, http_port=s.weaviate_http_port, grpc_port=s.weaviate_grpc_port
    )


VECTOR_STORES: dict[str, VectorStoreCreator] = {
    "qdrant": _qdrant,
    "pgvector": _pgvector,
    "weaviate": _weaviate,
}


def create_vector_store(settings: Settings, name: str | None = None) -> VectorStore:
    key = name or settings.vector_store
    try:
        creator = VECTOR_STORES[key]
    except KeyError as exc:
        raise ProviderNotConfiguredError(
            f"Unknown vector store '{key}'. Available: {', '.join(VECTOR_STORES)}"
        ) from exc
    return creator(settings)
