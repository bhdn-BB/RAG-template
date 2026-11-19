"""LangGraph state for the RAG pipeline."""

from typing import TypedDict

from app.domain.ports import LLMClient
from app.domain.value_objects import ChatMessage, GuardResult, ScoredChunk


class RAGState(TypedDict, total=False):
    # inputs
    collection: str
    question: str
    history: list[ChatMessage]
    llm: LLMClient
    # intermediate
    guard: GuardResult
    candidates: list[ScoredChunk]  # hybrid search results (top_k, default 3)
    reranked: list[ScoredChunk]  # after cross-encoder (top_n, default 1)
    # outputs
    system_prompt: str
    messages: list[ChatMessage]
    answer: str
