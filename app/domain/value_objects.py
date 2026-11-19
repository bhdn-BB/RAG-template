"""Value objects: immutable, defined by their values, no identity of their own."""

from dataclasses import dataclass, replace
from enum import StrEnum

from app.domain.entities import Chunk


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk together with the scores it collected along the retrieval pipeline."""

    chunk: Chunk
    dense_score: float | None = None
    bm25_rank: int | None = None  # 1 = best, from LangChain's BM25Retriever
    rrf_score: float | None = None
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        """Best available score: rerank > rrf > dense."""
        for value in (self.rerank_score, self.rrf_score, self.dense_score):
            if value is not None:
                return value
        return 0.0

    def with_scores(self, **scores: float | None) -> "ScoredChunk":
        return replace(self, **scores)


@dataclass(frozen=True)
class GuardResult:
    """Outcome of the prompt-injection heuristics for a user question."""

    flagged: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMProviderInfo:
    """What the system knows about an LLM provider (no client is created)."""

    name: str
    configured: bool
    default_model: str
    key_env: str | None = None  # env var holding the API key; None = self-hosted server
