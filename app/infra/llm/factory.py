"""LLM factory: a registry of creator functions keyed by provider name.

`auto` picks the first provider with an API key (Anthropic -> OpenAI -> Gemini -> OpenRouter -> Groq)
and falls back to the self-hosted Qwen model served by vLLM (Ollama is available explicitly).
"""

from collections.abc import Callable
from dataclasses import dataclass

from app.core.config import Settings
from app.core.exceptions import ProviderNotConfiguredError
from app.domain.ports import LLMClient
from app.domain.value_objects import LLMProviderInfo


@dataclass(frozen=True)
class ProviderSpec:
    create: Callable[[Settings, str | None], LLMClient]
    is_configured: Callable[[Settings], bool]
    default_model: Callable[[Settings], str]
    key_env: str | None = None  # env var holding the API key (None = self-hosted server, no key)


def _anthropic(s: Settings, model: str | None) -> LLMClient:
    from app.infra.llm.anthropic_client import AnthropicClient

    return AnthropicClient(
        model=model or s.anthropic_model, api_key=s.anthropic_api_key or "", max_tokens=s.llm_max_tokens
    )


def _openai(s: Settings, model: str | None) -> LLMClient:
    from app.infra.llm.openai_compatible import OpenAICompatibleClient

    return OpenAICompatibleClient(
        provider="openai",
        model=model or s.openai_model,
        api_key=s.openai_api_key or "",
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
        max_tokens_param="max_completion_tokens",
    )


def _gemini(s: Settings, model: str | None) -> LLMClient:
    from app.infra.llm.gemini_client import GeminiClient

    return GeminiClient(
        model=model or s.gemini_model,
        api_key=s.gemini_api_key or "",
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
    )


def _openrouter(s: Settings, model: str | None) -> LLMClient:
    from app.infra.llm.openai_compatible import OpenAICompatibleClient

    return OpenAICompatibleClient(
        provider="openrouter",
        model=model or s.openrouter_model,
        api_key=s.openrouter_api_key or "",
        base_url=s.openrouter_base_url,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
    )


def _groq(s: Settings, model: str | None) -> LLMClient:
    from app.infra.llm.openai_compatible import OpenAICompatibleClient

    return OpenAICompatibleClient(
        provider="groq",
        model=model or s.groq_model,
        api_key=s.groq_cloud_api or "",
        base_url=s.groq_base_url,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
    )


def _vllm(s: Settings, model: str | None) -> LLMClient:
    from app.infra.llm.openai_compatible import OpenAICompatibleClient

    return OpenAICompatibleClient(
        provider="vllm",
        model=model or s.vllm_llm_model,
        api_key="EMPTY",  # vLLM does not require a key by default
        base_url=s.vllm_llm_base_url,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
    )


def _ollama(s: Settings, model: str | None) -> LLMClient:
    from app.infra.llm.openai_compatible import OpenAICompatibleClient

    return OpenAICompatibleClient(
        provider="ollama",
        model=model or s.ollama_model,
        api_key="ollama",  # Ollama ignores the key but the SDK requires one
        base_url=s.ollama_base_url,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
    )


LLM_PROVIDERS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        _anthropic, lambda s: bool(s.anthropic_api_key), lambda s: s.anthropic_model, "ANTHROPIC_API_KEY"
    ),
    "openai": ProviderSpec(
        _openai, lambda s: bool(s.openai_api_key), lambda s: s.openai_model, "OPENAI_API_KEY"
    ),
    "gemini": ProviderSpec(
        _gemini, lambda s: bool(s.gemini_api_key), lambda s: s.gemini_model, "GEMINI_API_KEY"
    ),
    "openrouter": ProviderSpec(
        _openrouter, lambda s: bool(s.openrouter_api_key), lambda s: s.openrouter_model, "OPENROUTER_API_KEY"
    ),
    "groq": ProviderSpec(_groq, lambda s: bool(s.groq_cloud_api), lambda s: s.groq_model, "GROQ_CLOUD_API"),
    "vllm": ProviderSpec(_vllm, lambda s: True, lambda s: s.vllm_llm_model),
    "ollama": ProviderSpec(_ollama, lambda s: True, lambda s: s.ollama_model),
}

AUTO_ORDER = ("anthropic", "openai", "gemini", "openrouter", "groq")


def resolve_provider(settings: Settings, provider: str | None = None) -> str:
    name = provider or settings.llm_provider
    if name != "auto":
        return name
    return next((p for p in AUTO_ORDER if LLM_PROVIDERS[p].is_configured(settings)), "vllm")


def create_llm(settings: Settings, provider: str | None = None, model: str | None = None) -> LLMClient:
    name = resolve_provider(settings, provider)
    spec = LLM_PROVIDERS.get(name)
    if spec is None:
        raise ProviderNotConfiguredError(
            f"Unknown LLM provider '{name}'. Available: {', '.join(LLM_PROVIDERS)}"
        )
    if not spec.is_configured(settings):
        raise ProviderNotConfiguredError(f"LLM provider '{name}' has no API key configured.")
    return spec.create(settings, model)


def available_providers(settings: Settings) -> list[LLMProviderInfo]:
    return [
        LLMProviderInfo(
            name=name,
            configured=spec.is_configured(settings),
            default_model=spec.default_model(settings),
            key_env=spec.key_env,
        )
        for name, spec in LLM_PROVIDERS.items()
    ]


class LLMFactory:
    """`LLMClientFactory` adapter: clients are created per request (provider/model can be overridden)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def create(self, provider: str | None = None, model: str | None = None) -> LLMClient:
        return create_llm(self.settings, provider, model)

    def default_provider(self) -> str:
        return resolve_provider(self.settings)

    def providers(self) -> list[LLMProviderInfo]:
        return available_providers(self.settings)
