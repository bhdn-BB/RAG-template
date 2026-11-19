from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.domain.value_objects import ChatMessage


@dataclass(frozen=True)
class LLMSelection:
    provider: str | None = None  # None = configured default
    model: str | None = None  # None = provider's default model


@dataclass(frozen=True)
class SourceDTO:
    """A context chunk used for an answer. `n` is its citation number ([n])."""

    n: int
    chunk_id: UUID
    document_id: UUID
    collection: str
    filename: str
    index: int
    text: str
    source: str = ""  # page URL for web sources, otherwise file name (+ page)
    source_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rerank_score: float | None = None
    rrf_score: float | None = None


# --- commands (input) ---------------------------------------------------------


@dataclass(frozen=True)
class AskCommand:
    collection: str
    question: str
    history: tuple[ChatMessage, ...] = ()
    llm: LLMSelection = LLMSelection()
    include_llm_input: bool = False


@dataclass(frozen=True)
class RegenerateCommand:
    """Re-call the LLM with the same context (`sources`) and history; no retrieval."""

    question: str
    sources: tuple[SourceDTO, ...]
    history: tuple[ChatMessage, ...] = ()
    llm: LLMSelection = LLMSelection()
    include_llm_input: bool = False


# --- results (output) ---------------------------------------------------------


@dataclass(frozen=True)
class LLMInputDTO:
    """The exact prompt sent to the LLM."""

    system_prompt: str
    messages: tuple[ChatMessage, ...]


@dataclass(frozen=True)
class AnswerMeta:
    sources: tuple[SourceDTO, ...]
    provider: str
    model: str
    flagged: bool = False
    flag_reasons: tuple[str, ...] = ()
    llm_input: LLMInputDTO | None = None


@dataclass(frozen=True)
class ChatAnswerDTO:
    answer: str
    meta: AnswerMeta


@dataclass(frozen=True)
class ChatStreamDTO:
    """Metadata is known before generation starts; tokens are produced lazily."""

    meta: AnswerMeta
    tokens: AsyncIterator[str]
