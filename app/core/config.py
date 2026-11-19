"""Application settings, loaded from environment variables / `.env`."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

VectorStoreName = Literal["qdrant", "pgvector", "weaviate"]
LLMProviderName = Literal["auto", "anthropic", "openai", "gemini", "openrouter", "groq", "vllm", "ollama"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "Ukrainian RAG"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- Vector store (the only persistence layer) ---
    vector_store: VectorStoreName = "qdrant"
    pgvector_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    weaviate_host: str = "localhost"
    weaviate_http_port: int = 8080
    weaviate_grpc_port: int = 50051

    # --- Embeddings: any OpenAI-compatible /v1/embeddings server (vLLM by default, Ollama works too) ---
    embedding_base_url: str = "http://localhost:8001/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    # Qwen3-Embedding supports Matryoshka truncation; 1024 keeps pgvector HNSW (<=2000 dims) happy.
    embedding_dim: int = 1024
    embedding_batch_size: int = 32
    embedding_query_instruction: str = (
        "Given a user question, retrieve relevant passages that answer the question"
    )

    # --- Reranker (vLLM /rerank) ---
    reranker_base_url: str = "http://localhost:8002"
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_instruction: str = (
        "Given a user question, judge whether the passage contains information that answers it"
    )

    # --- LLM ---
    llm_provider: LLMProviderName = "auto"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 512  # Groq free tier allows 1000 output tokens/min
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5-5"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    openrouter_api_key: str | None = None
    openrouter_model: str = "qwen/qwen3-235b-a22b"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # GroqCloud (OpenAI-compatible). Env var: GROQ_CLOUD_API
    groq_cloud_api: str | None = None
    groq_model: str = "qwen/qwen3.8-27b"  # alternatives: openai/gpt-oss-120b, openai/gpt-oss-20b
    groq_base_url: str = "https://api.groq.com/openai/v1"
    vllm_llm_base_url: str = "http://localhost:8003/v1"
    vllm_llm_model: str = "Qwen/Qwen3-8B-AWQ"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen3:8b"

    # --- Retrieval ---
    retrieval_top_k: int = 3  # candidates after hybrid fusion (dense embedder + BM25)
    rerank_top_n: int = 1  # chunks the cross-encoder passes to the LLM
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
    dense_weight: float = 1.0
    bm25_weight: float = 1.0

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 150
    max_upload_mb: int = 25
    # URL ingestion: allow fetching private/loopback hosts (SSRF protection off). Keep False.
    allow_private_urls: bool = False

    # --- Safety ---
    max_question_chars: int = 4000
    max_history_messages: int = 20
    max_context_chars_per_chunk: int = 4000


@lru_cache
def get_settings() -> Settings:
    return Settings()
