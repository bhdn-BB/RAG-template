"""OpenAI-compatible chat client. Used for OpenAI, OpenRouter, Groq, vLLM and Ollama."""

from collections.abc import AsyncIterator

from openai import AsyncOpenAI, OpenAIError

from app.core.exceptions import UpstreamError
from app.domain.value_objects import ChatMessage


class OpenAICompatibleClient:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_tokens_param: str = "max_tokens",
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # OpenAI's newer models want `max_completion_tokens`; vLLM/OpenRouter use `max_tokens`.
        self.max_tokens_param = max_tokens_param
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _request(self, system: str, messages: list[ChatMessage]) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": self.temperature,
            self.max_tokens_param: self.max_tokens,
        }

    async def generate(self, system: str, messages: list[ChatMessage]) -> str:
        try:
            response = await self.client.chat.completions.create(**self._request(system, messages))
        except OpenAIError as exc:
            raise UpstreamError(f"{self.provider} error: {exc}") from exc
        return response.choices[0].message.content or ""

    async def stream(self, system: str, messages: list[ChatMessage]) -> AsyncIterator[str]:
        try:
            response = await self.client.chat.completions.create(
                **self._request(system, messages), stream=True
            )
            async for event in response:
                if event.choices and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
        except OpenAIError as exc:
            raise UpstreamError(f"{self.provider} error: {exc}") from exc
