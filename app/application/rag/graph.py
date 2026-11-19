"""RAG pipeline as a LangGraph state machine (`RAGPipeline` holds the compiled graphs):

    guard -> retrieve (hybrid, top 3) -> rerank (cross-encoder, top 1) -> build_prompt -> generate

`build_rag_graph(..., with_generation=False)` stops after `build_prompt`, which is used for
streaming (the LLM tokens are streamed outside the graph).
"""

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.application.rag.state import RAGState
from app.application.services.prompt_service import build_messages, build_system_prompt, detect_injection
from app.application.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


def build_rag_graph(
    retrieval: RetrievalService,
    max_chars_per_chunk: int = 4000,
    with_generation: bool = True,
) -> CompiledStateGraph:
    async def guard(state: RAGState) -> RAGState:
        result = detect_injection(state["question"])
        if result.flagged:
            logger.warning(
                "Possible prompt injection (%s): %.200r", ", ".join(result.reasons), state["question"]
            )
        return {"guard": result}

    async def retrieve(state: RAGState) -> RAGState:
        return {"candidates": await retrieval.hybrid_search(state["collection"], state["question"])}

    async def rerank(state: RAGState) -> RAGState:
        return {"reranked": await retrieval.rerank(state["question"], state["candidates"])}

    async def build_prompt(state: RAGState) -> RAGState:
        return {
            "system_prompt": build_system_prompt(state["guard"]),
            "messages": build_messages(
                state["question"], state.get("history", []), state["reranked"], max_chars_per_chunk
            ),
        }

    async def generate(state: RAGState) -> RAGState:
        return {"answer": await state["llm"].generate(state["system_prompt"], state["messages"])}

    graph = StateGraph(RAGState)
    graph.add_node("guard", guard)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("build_prompt", build_prompt)
    graph.add_edge(START, "guard")
    graph.add_edge("guard", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "build_prompt")
    if with_generation:
        graph.add_node("generate", generate)
        graph.add_edge("build_prompt", "generate")
        graph.add_edge("generate", END)
    else:
        graph.add_edge("build_prompt", END)
    return graph.compile()


class RAGPipeline:
    def __init__(self, retrieval: RetrievalService, max_chars_per_chunk: int = 4000) -> None:
        # guard -> retrieve -> rerank -> build_prompt -> generate
        self.full: CompiledStateGraph = build_rag_graph(retrieval, max_chars_per_chunk, with_generation=True)
        # guard -> retrieve -> rerank -> build_prompt (LLM is streamed outside the graph)
        self.prepare: CompiledStateGraph = build_rag_graph(
            retrieval, max_chars_per_chunk, with_generation=False
        )
