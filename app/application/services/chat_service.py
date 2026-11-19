"""Use cases: chat with the knowledge base. Stateless: the client keeps the conversation.

- `ask`        : full RAG pipeline (LangGraph) -> answer + sources.
- `ask_stream` : same pipeline; LLM tokens are streamed.
- `regenerate` : re-call the LLM with the SAME context (sources returned earlier) and the full
                 history, without retrieving again. Optionally with a different provider/model.
"""

from dataclasses import dataclass

from app.application.dto.chat import (
    AnswerMeta,
    AskCommand,
    ChatAnswerDTO,
    ChatStreamDTO,
    LLMInputDTO,
    RegenerateCommand,
)
from app.application.mappers import scored_to_source, source_to_scored
from app.application.rag.graph import RAGPipeline
from app.application.services.collection_service import CollectionService
from app.application.services.prompt_service import build_messages, build_system_prompt, detect_injection
from app.core.exceptions import ValidationError
from app.domain.ports import LLMClient, LLMClientFactory
from app.domain.value_objects import ChatMessage, GuardResult, ScoredChunk


@dataclass(frozen=True)
class _Prepared:
    """Everything needed to call the LLM (internal to this use case)."""

    system_prompt: str
    messages: list[ChatMessage]
    sources: list[ScoredChunk]
    guard: GuardResult


class ChatService:
    def __init__(
        self,
        collections: CollectionService,
        pipeline: RAGPipeline,
        llm_factory: LLMClientFactory,
        max_question_chars: int = 4000,
        max_history_messages: int = 20,
        max_chars_per_chunk: int = 4000,
    ) -> None:
        self.collections = collections
        self.pipeline = pipeline
        self.llm_factory = llm_factory
        self.max_question_chars = max_question_chars
        self.max_history_messages = max_history_messages
        self.max_chars_per_chunk = max_chars_per_chunk

    # --- helpers -----------------------------------------------------------------

    def _history(self, question: str, history: tuple[ChatMessage, ...]) -> list[ChatMessage]:
        if not question.strip():
            raise ValidationError("Question is empty.")
        if len(question) > self.max_question_chars:
            raise ValidationError(f"Question exceeds {self.max_question_chars} characters.")
        return list(history[-self.max_history_messages :]) if self.max_history_messages else []

    @staticmethod
    def _meta(prepared: _Prepared, llm: LLMClient, include_llm_input: bool) -> AnswerMeta:
        return AnswerMeta(
            sources=tuple(scored_to_source(n, s) for n, s in enumerate(prepared.sources, start=1)),
            provider=llm.provider,
            model=llm.model,
            flagged=prepared.guard.flagged,
            flag_reasons=prepared.guard.reasons,
            llm_input=LLMInputDTO(prepared.system_prompt, tuple(prepared.messages))
            if include_llm_input
            else None,
        )

    async def _prepare_ask(self, command: AskCommand) -> _Prepared:
        """guard -> hybrid retrieval -> rerank -> prompt (LangGraph, without the LLM call)."""
        history = self._history(command.question, command.history)
        await self.collections.require(command.collection)
        state = await self.pipeline.prepare.ainvoke(
            {"collection": command.collection, "question": command.question, "history": history}
        )
        return _Prepared(state["system_prompt"], state["messages"], state["reranked"], state["guard"])

    def _prepare_regenerate(self, command: RegenerateCommand) -> _Prepared:
        history = self._history(command.question, command.history)
        sources = [source_to_scored(s) for s in sorted(command.sources, key=lambda s: s.n)]
        guard = detect_injection(command.question)
        return _Prepared(
            build_system_prompt(guard),
            build_messages(command.question, history, sources, self.max_chars_per_chunk),
            sources,
            guard,
        )

    # --- use cases ---------------------------------------------------------------

    async def ask(self, command: AskCommand) -> ChatAnswerDTO:
        llm = self.llm_factory.create(command.llm.provider, command.llm.model)
        history = self._history(command.question, command.history)
        await self.collections.require(command.collection)
        state = await self.pipeline.full.ainvoke(
            {"collection": command.collection, "question": command.question, "history": history, "llm": llm}
        )
        prepared = _Prepared(state["system_prompt"], state["messages"], state["reranked"], state["guard"])
        return ChatAnswerDTO(state["answer"], self._meta(prepared, llm, command.include_llm_input))

    async def ask_stream(self, command: AskCommand) -> ChatStreamDTO:
        llm = self.llm_factory.create(command.llm.provider, command.llm.model)
        prepared = await self._prepare_ask(command)
        return ChatStreamDTO(
            self._meta(prepared, llm, command.include_llm_input),
            llm.stream(prepared.system_prompt, prepared.messages),
        )

    async def regenerate(self, command: RegenerateCommand) -> ChatAnswerDTO:
        llm = self.llm_factory.create(command.llm.provider, command.llm.model)
        prepared = self._prepare_regenerate(command)
        answer = await llm.generate(prepared.system_prompt, prepared.messages)
        return ChatAnswerDTO(answer, self._meta(prepared, llm, command.include_llm_input))

    async def regenerate_stream(self, command: RegenerateCommand) -> ChatStreamDTO:
        llm = self.llm_factory.create(command.llm.provider, command.llm.model)
        prepared = self._prepare_regenerate(command)
        return ChatStreamDTO(
            self._meta(prepared, llm, command.include_llm_input),
            llm.stream(prepared.system_prompt, prepared.messages),
        )
