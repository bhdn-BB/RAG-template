import pytest

from app.core.config import Settings
from app.core.exceptions import ProviderNotConfiguredError
from app.infra.llm.factory import LLM_PROVIDERS, create_llm, resolve_provider
from app.infra.vectorstores.factory import VECTOR_STORES, create_vector_store


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_vector_store_registry_contains_all_backends():
    assert set(VECTOR_STORES) == {"qdrant", "pgvector", "weaviate"}


def test_create_vector_store_uses_registry(monkeypatch):
    sentinel = object()
    monkeypatch.setitem(VECTOR_STORES, "weaviate", lambda settings: sentinel)
    assert create_vector_store(_settings(vector_store="weaviate")) is sentinel


def test_unknown_vector_store_raises():
    with pytest.raises(ProviderNotConfiguredError):
        create_vector_store(_settings(), name="nope")


def test_llm_registry_contains_all_providers():
    assert {"anthropic", "openai", "gemini", "openrouter", "groq", "vllm", "ollama"} <= set(LLM_PROVIDERS)


def test_auto_falls_back_to_vllm_without_keys():
    assert resolve_provider(_settings(llm_provider="auto")) == "vllm"


def test_auto_picks_first_configured_provider():
    s = _settings(llm_provider="auto", gemini_api_key="g", openrouter_api_key="o")
    assert resolve_provider(s) == "gemini"
    s = _settings(llm_provider="auto", anthropic_api_key="a", openai_api_key="o")
    assert resolve_provider(s) == "anthropic"


def test_explicit_provider_without_key_raises():
    with pytest.raises(ProviderNotConfiguredError):
        create_llm(_settings(), "openai")


def test_unknown_llm_provider_raises():
    with pytest.raises(ProviderNotConfiguredError):
        create_llm(_settings(), "nope")


def test_create_llm_with_model_override():
    llm = create_llm(_settings(openai_api_key="sk-test"), "openai", "gpt-custom")
    assert llm.provider == "openai" and llm.model == "gpt-custom"


def test_groq_reads_groq_cloud_api_env(monkeypatch):
    monkeypatch.setenv("GROQ_CLOUD_API", "gsk-test")
    s = _settings(llm_provider="auto")
    assert resolve_provider(s) == "groq"
    llm = create_llm(s)
    assert llm.provider == "groq" and llm.model == "qwen/qwen3.8-27b"
    assert str(llm.client.base_url).startswith("https://api.groq.com/openai/v1")
