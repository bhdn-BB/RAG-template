"""Qwen3-Reranker (cross-encoder) served by vLLM through the `/rerank` endpoint.

vLLM must run the model as a sequence classifier, see `deploy/docker-compose.yml`:
  --task score --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],
                                "classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
Qwen3-Reranker expects its chat-style template, so we format query/documents here.
"""

import httpx

from app.core.exceptions import UpstreamError

# Prompt template expected by Qwen3-Reranker.
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_query(instruction: str, query: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n"


def format_document(document: str) -> str:
    return f"<Document>: {document}"


class VLLMReranker:
    def __init__(self, base_url: str, model: str, instruction: str, timeout: float = 60.0) -> None:
        self.model = model
        self.instruction = instruction
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        payload = {
            "model": self.model,
            "query": PREFIX + format_query(self.instruction, query),
            "documents": [format_document(d) + SUFFIX for d in documents],
        }
        try:
            response = await self.client.post("/rerank", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise UpstreamError(f"Reranker service error: {exc}") from exc
        scores = [0.0] * len(documents)
        for result in response.json()["results"]:
            scores[result["index"]] = float(result["relevance_score"])
        return scores

    async def healthcheck(self) -> bool:
        try:
            return (await self.client.get("/health", timeout=3)).status_code < 500
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        await self.client.aclose()
