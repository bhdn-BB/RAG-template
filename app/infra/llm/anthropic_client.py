"""Anthropic (Claude) chat client."""

from collections.abc import AsyncIterator

from anthropic import AnthropicError, AsyncAnthropic

from app.core.exceptions import UpstreamError
from app.domain.value_objects import ChatMessage


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, model: str, api_key: str, max_tokens: int = 1024) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.client = AsyncAnthropic(api_key=api_key)

    def _request(self, system: str, messages: list[ChatMessage]) -> dict:
        # Sampling params (temperature) are intentionally not sent: recent Claude models
        # manage sampling themselves and may reject them.
        return {
            "model": self.model,
            "system": system,
            "max_tokens": self.max_tokens,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
        }

    async def generate(self, system: str, messages: list[ChatMessage]) -> str:
        try:
            response = await self.client.messages.create(**self._request(system, messages))
        except AnthropicError as exc:
            raise UpstreamError(f"anthropic error: {exc}") from exc
        return "".join(block.text for block in response.content if block.type == "text")

    async def stream(self, system: str, messages: list[ChatMessage]) -> AsyncIterator[str]:
        try:
            async with self.client.messages.stream(**self._request(system, messages)) as stream:
                async for text in stream.text_stream:
                    yield text
        except AnthropicError as exc:
            raise UpstreamError(f"anthropic error: {exc}") from exc
