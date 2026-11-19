"""Qwen3-Embedding served by vLLM through the OpenAI-compatible `/v1/embeddings` endpoint."""

import math

import httpx

from app.core.exceptions import UpstreamError


def truncate_and_normalize(vector: list[float], dim: int) -> list[float]:
    """Matryoshka (MRL) truncation: keep the first `dim` values and L2-renormalize.

    Qwen3-Embedding is trained with MRL, so this equals asking the model for a smaller dimension.
    """
    head = vector[:dim]
    norm = math.sqrt(sum(x * x for x in head)) or 1.0
    return [x / norm for x in head]


class VLLMEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        dim: int,
        query_instruction: str,
        batch_size: int = 32,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.dim = dim
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self.client.post("/embeddings", json={"model": self.model, "input": texts})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise UpstreamError(f"Embedding service error: {exc}") from exc
        data = sorted(response.json()["data"], key=lambda d: d["index"])
        return [truncate_and_normalize(d["embedding"], self.dim) for d in data]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(await self._embed(texts[start : start + self.batch_size]))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        # Qwen3-Embedding is instruction-aware on the query side only.
        return (await self._embed([f"Instruct: {self.query_instruction}\nQuery: {text}"]))[0]

    async def healthcheck(self) -> bool:
        # vLLM exposes /health at the server root, next to /v1.
        root = str(self.client.base_url).rstrip("/").removesuffix("/v1")
        try:
            return (await self.client.get(f"{root}/health", timeout=3)).status_code < 500
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        await self.client.aclose()
