"""`LexicalIndex` adapter: BM25 search for Ukrainian text, built on LangChain's `BM25Retriever`.

Preprocessing (passed to the retriever as `preprocess_func`):
  lowercase -> Unicode word regex -> NLTK stopwords (ukrainian + english) -> pymorphy3 (uk) lemmatization.

NLTK has no built-in Ukrainian stopword list, so the project ships one as an NLTK corpus file
(`app/resources/nltk_data/corpora/stopwords/ukrainian`) and loads it through the standard
`nltk.corpus.stopwords` API.

One retriever is built lazily per collection from the chunks stored in the vector DB and
invalidated whenever the collection changes.
"""

import asyncio
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from uuid import UUID

import nltk
import pymorphy3
from langchain_core.documents import Document as LCDocument
from nltk.corpus import stopwords

from app.domain.entities import Chunk
from app.domain.ports import VectorStore

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.retrievers import BM25Retriever

_NLTK_DATA = Path(__file__).resolve().parent.parent / "resources" / "nltk_data"
if str(_NLTK_DATA) not in nltk.data.path:
    nltk.data.path.insert(0, str(_NLTK_DATA))

# Words of letters/digits, keeping inner apostrophes and hyphens (п'ять, будь-який).
_TOKEN_RE = re.compile(r"\w+(?:['’ʼ-]\w+)*", re.UNICODE)


@lru_cache(maxsize=1)
def stopword_set() -> frozenset[str]:
    return frozenset(stopwords.words("ukrainian")) | frozenset(stopwords.words("english"))


@lru_cache(maxsize=1)
def _morph() -> pymorphy3.MorphAnalyzer:
    return pymorphy3.MorphAnalyzer(lang="uk")


@lru_cache(maxsize=200_000)
def _lemma(word: str) -> str:
    return _morph().parse(word)[0].normal_form


def tokenize(text: str) -> list[str]:
    """Preprocessing function handed to LangChain's BM25Retriever."""
    stop = stopword_set()
    tokens = (t.replace("’", "'").replace("ʼ", "'") for t in _TOKEN_RE.findall(text.lower()))
    return [_lemma(t) for t in tokens if t not in stop]


def _to_lc(chunk: Chunk) -> LCDocument:
    return LCDocument(page_content=chunk.text, metadata={"chunk_id": str(chunk.id)}, id=str(chunk.id))


@dataclass
class _Entry:
    chunk_count: int
    retriever: BM25Retriever | None
    by_id: dict[UUID, Chunk]


class BM25Index:
    """Per-process cache of BM25 retrievers.

    Each entry remembers the collection's chunk count; if the count in the vector store differs
    (another worker/replica added or deleted documents) the index is rebuilt. Local writes also
    call `invalidate` explicitly.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store
        self._entries: dict[str, _Entry] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def invalidate(self, collection: str) -> None:
        self._entries.pop(collection, None)

    async def _get(self, collection: str) -> _Entry:
        count = await self.vector_store.count(collection)
        entry = self._entries.get(collection)
        if entry is not None and entry.chunk_count == count:
            return entry
        async with self._locks.setdefault(collection, asyncio.Lock()):
            entry = self._entries.get(collection)
            if entry is None or entry.chunk_count != count:
                chunks = await self.vector_store.list_chunks(collection)
                retriever = (
                    await asyncio.to_thread(
                        BM25Retriever.from_documents, [_to_lc(c) for c in chunks], preprocess_func=tokenize
                    )
                    if chunks
                    else None
                )
                entry = _Entry(len(chunks), retriever, {c.id: c for c in chunks})
                self._entries[collection] = entry
        return entry

    async def search(self, collection: str, query: str, top_k: int) -> list[Chunk]:
        """Chunks ranked by LangChain's BM25Retriever (best first). Only the rank is used downstream (RRF)."""
        entry = await self._get(collection)
        query_terms = set(tokenize(query))
        if entry.retriever is None or not query_terms:
            return []
        # Ask for a few extra documents: in tiny collections BM25Okapi's IDF can be zero, and the
        # retriever then returns `k` documents even if they share no words with the query.
        retriever = entry.retriever.model_copy(update={"k": top_k * 4})
        documents = await retriever.ainvoke(query)
        matching = [d for d in documents if query_terms & set(tokenize(d.page_content))]
        return [entry.by_id[UUID(d.metadata["chunk_id"])] for d in matching[:top_k]]
