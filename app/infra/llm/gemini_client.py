"""Google Gemini chat client (google-genai SDK)."""

from collections.abc import AsyncIterator

from google import genai
from google.genai import errors, types

from app.core.exceptions import UpstreamError
from app.domain.value_objects import ChatMessage, Role


class GeminiClient:
    provider = "gemini"

    def __init__(self, model: str, api_key: str, temperature: float = 0.2, max_tokens: int = 1024) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = genai.Client(api_key=api_key)

    def _contents(self, messages: list[ChatMessage]) -> list[types.Content]:
        # Gemini calls the assistant role "model".
        return [
            types.Content(
                role="model" if m.role == Role.ASSISTANT else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
        ]

    def _config(self, system: str) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system, temperature=self.temperature, max_output_tokens=self.max_tokens
        )

    async def generate(self, system: str, messages: list[ChatMessage]) -> str:
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model, contents=self._contents(messages), config=self._config(system)
            )
        except errors.APIError as exc:
            raise UpstreamError(f"gemini error: {exc}") from exc
        return response.text or ""

    async def stream(self, system: str, messages: list[ChatMessage]) -> AsyncIterator[str]:
        try:
            response = await self.client.aio.models.generate_content_stream(
                model=self.model, contents=self._contents(messages), config=self._config(system)
            )
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        except errors.APIError as exc:
            raise UpstreamError(f"gemini error: {exc}") from exc
