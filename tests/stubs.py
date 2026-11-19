"""Deterministic stand-ins for the GPU / network dependencies."""

import hashlib
import math
from collections.abc import AsyncIterator
from uuid import UUID

from app.domain.entities import Chunk, Collection
from app.domain.value_objects import ChatMessage, LLMProviderInfo
from app.infra.bm25 import tokenize

DIM = 64


def _bag_of_words(text: str) -> list[float]:
    vec = [0.0] * DIM
    for token in tokenize(text):
        vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class FakeEmbedder:
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_bag_of_words(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return _bag_of_words(text)

    async def healthcheck(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class FakeReranker:
    """Scores by lemma overlap with the query and records how many docs it received."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append(len(documents))
        q = set(tokenize(query))
        return [len(q & set(tokenize(d))) / (len(q) or 1) for d in documents]

    async def healthcheck(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class FakeLLM:
    provider = "fake"
    model = "fake-1"

    def __init__(self, answer: str = "Відповідь [1]") -> None:
        self.answer = answer
        self.last_system: str | None = None
        self.last_messages: list[ChatMessage] = []

    async def generate(self, system: str, messages: list[ChatMessage]) -> str:
        self.last_system, self.last_messages = system, messages
        return self.answer

    async def stream(self, system: str, messages: list[ChatMessage]) -> AsyncIterator[str]:
        self.last_system, self.last_messages = system, messages
        for token in self.answer.split(" "):
            yield token + " "


class FakeLLMFactory:
    def __init__(self, llm: FakeLLM) -> None:
        self.llm = llm

    def create(self, provider: str | None = None, model: str | None = None) -> FakeLLM:
        return self.llm

    def default_provider(self) -> str:
        return "fake"

    def providers(self) -> list[LLMProviderInfo]:
        return [LLMProviderInfo(name="fake", configured=True, default_model="fake-1")]


class FakeUrlResolver:
    async def resolve(self, url: str) -> str:
        return url


class FakeWebLoader:
    """Mimics LangChain's WebBaseLoader output for a page with a <main> block."""

    def __init__(self, url: str, main_only: bool = True) -> None:
        self.url = url

    def load(self):
        from langchain_core.documents import Document

        text = (
            "Вартість проїзду в Києві - pro.kyiv.ua\n"
            "Вартість проїзду\n"
            "Вартість проїзду в Києві становить 30 гривень з 15 липня 2026 року. "
            "Ставка однакова для метро, автобуса, трамвая, тролейбуса та фунікулера.\n"
            "Пересадковий квиток коштуватиме 60 грн і діятиме 90 хвилин між усіма видами транспорту."
        )
        return [
            Document(
                page_content=text,
                metadata={"source": self.url, "title": "Вартість проїзду в Києві - pro.kyiv.ua"},
            )
        ]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class InMemoryVectorStore:
    """Test double implementing the VectorStore protocol in memory."""

    name = "memory"

    def __init__(self) -> None:
        self._data: dict[str, dict[UUID, tuple[Chunk, list[float]]]] = {}

    async def create_collection(self, collection: str, dim: int) -> None:
        self._data.setdefault(collection, {})

    async def delete_collection(self, collection: str) -> None:
        self._data.pop(collection, None)

    async def collection_exists(self, collection: str) -> bool:
        return collection in self._data

    async def list_collections(self) -> list[Collection]:
        return [Collection(id=name, chunk_count=len(points)) for name, points in sorted(self._data.items())]

    async def count(self, collection: str) -> int:
        return len(self._data.get(collection, {}))

    async def upsert(self, collection: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = self._data.setdefault(collection, {})
        for chunk, vector in zip(chunks, vectors, strict=True):
            points[chunk.id] = (chunk, vector)

    async def search(self, collection: str, vector: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        scored = [(c, _cosine(vector, v)) for c, v in self._data.get(collection, {}).values()]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

    async def list_chunks(self, collection: str, document_id: UUID | None = None) -> list[Chunk]:
        chunks = [c for c, _ in self._data.get(collection, {}).values()]
        if document_id is not None:
            chunks = [c for c in chunks if c.document_id == document_id]
        return sorted(chunks, key=lambda c: (str(c.document_id), c.index))

    async def delete_document(self, collection: str, document_id: UUID) -> None:
        points = self._data.get(collection, {})
        for chunk_id in [cid for cid, (c, _) in points.items() if c.document_id == document_id]:
            del points[chunk_id]

    async def delete_chunks(self, collection: str, chunk_ids: list[UUID]) -> None:
        points = self._data.get(collection, {})
        for chunk_id in chunk_ids:
            points.pop(chunk_id, None)

    async def healthcheck(self) -> bool:
        return True

    async def close(self) -> None:
        return None
