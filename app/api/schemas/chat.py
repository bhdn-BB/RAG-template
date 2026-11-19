"""Chat HTTP schemas. The API is stateless: clients send the conversation history (and, for
`regenerate`, the sources returned by the previous answer)."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.application.dto.chat import (
    AnswerMeta,
    AskCommand,
    ChatAnswerDTO,
    LLMSelection,
    RegenerateCommand,
    SourceDTO,
)
from app.domain.value_objects import ChatMessage, Role


class MessageSchema(BaseModel):
    role: Role
    content: str = Field(..., max_length=60_000)

    def to_domain(self) -> ChatMessage:
        return ChatMessage(role=self.role, content=self.content)


class SourceSchema(BaseModel):
    """A context chunk used for an answer. `n` is the citation number ([n]) in the answer."""

    n: int
    chunk_id: UUID
    document_id: UUID
    collection: str
    filename: str
    source: str = ""  # page URL for web sources, otherwise file name (+ page)
    source_url: str | None = None
    index: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    rerank_score: float | None = None
    rrf_score: float | None = None

    @classmethod
    def from_dto(cls, dto: SourceDTO) -> "SourceSchema":
        return cls(**dto.__dict__)

    def to_dto(self) -> SourceDTO:
        return SourceDTO(**self.model_dump())


class LLMInputSchema(BaseModel):
    """The exact prompt sent to the LLM: system prompt + conversation (context is in the last turn)."""

    system_prompt: str
    messages: list[MessageSchema]


class _LLMChoice(BaseModel):
    llm_provider: str | None = Field(
        None, examples=["groq", "anthropic", "openai", "gemini", "openrouter", "vllm", "ollama"]
    )
    model: str | None = None
    include_llm_input: bool = Field(
        False, description="Return the exact prompt sent to the LLM (debug/demo)."
    )

    def _selection(self) -> LLMSelection:
        return LLMSelection(provider=self.llm_provider, model=self.model)


class ChatRequest(_LLMChoice):
    collection: str
    question: str = Field(..., min_length=1)
    history: list[MessageSchema] = Field(default_factory=list)

    def to_command(self) -> AskCommand:
        return AskCommand(
            collection=self.collection,
            question=self.question,
            history=tuple(m.to_domain() for m in self.history),
            llm=self._selection(),
            include_llm_input=self.include_llm_input,
        )


class RegenerateRequest(_LLMChoice):
    """Re-call the LLM with the same context: no new retrieval happens."""

    question: str = Field(..., min_length=1)
    history: list[MessageSchema] = Field(default_factory=list)
    sources: list[SourceSchema] = Field(..., max_length=50)

    def to_command(self) -> RegenerateCommand:
        return RegenerateCommand(
            question=self.question,
            sources=tuple(s.to_dto() for s in self.sources),
            history=tuple(m.to_domain() for m in self.history),
            llm=self._selection(),
            include_llm_input=self.include_llm_input,
        )


class AnswerMetaSchema(BaseModel):
    sources: list[SourceSchema]
    provider: str
    model: str
    flagged: bool = False
    flag_reasons: list[str] = Field(default_factory=list)
    llm_input: LLMInputSchema | None = None

    @classmethod
    def from_dto(cls, meta: AnswerMeta) -> "AnswerMetaSchema":
        llm_input = None
        if meta.llm_input is not None:
            llm_input = LLMInputSchema(
                system_prompt=meta.llm_input.system_prompt,
                messages=[MessageSchema(role=m.role, content=m.content) for m in meta.llm_input.messages],
            )
        return cls(
            sources=[SourceSchema.from_dto(s) for s in meta.sources],
            provider=meta.provider,
            model=meta.model,
            flagged=meta.flagged,
            flag_reasons=list(meta.flag_reasons),
            llm_input=llm_input,
        )


class ChatResponse(AnswerMetaSchema):
    answer: str

    @classmethod
    def from_answer(cls, dto: ChatAnswerDTO) -> "ChatResponse":
        return cls(answer=dto.answer, **AnswerMetaSchema.from_dto(dto.meta).model_dump())
