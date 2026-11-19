from uuid import uuid4

from app.application.services.retrieval_service import reciprocal_rank_fusion
from app.domain.entities import Chunk
from app.infra.bm25 import BM25Index, stopword_set, tokenize
from tests.stubs import FakeEmbedder, InMemoryVectorStore


def _chunk(text: str) -> Chunk:
    return Chunk(id=uuid4(), document_id=uuid4(), collection_id="c", index=0, text=text)


def test_rrf_prefers_items_ranked_high_in_both_lists():
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    fused = reciprocal_rank_fusion([(a, 0.9), (b, 0.8)], [b, c], k=60)
    assert [s.chunk for s in fused] == [b, a, c]
    assert fused[0].dense_score == 0.8 and fused[0].bm25_rank == 1
    assert fused[2].dense_score is None


def test_rrf_respects_weights():
    a, b = _chunk("a"), _chunk("b")
    fused = reciprocal_rank_fusion([(a, 1.0)], [b], dense_weight=0.1, bm25_weight=1.0)
    assert fused[0].chunk == b


def test_nltk_ukrainian_stopwords_loaded():
    stop = stopword_set()
    assert {"і", "що", "це", "та"} <= stop  # ukrainian (bundled NLTK corpus)
    assert {"the", "and"} <= stop  # english (NLTK)


def test_tokenizer_lemmatizes_ukrainian_and_drops_stopwords():
    tokens = tokenize("Захист від фішингових атак та кібератаки")
    assert "та" not in tokens and "від" not in tokens
    assert "захист" in tokens
    assert "атака" in tokens  # атак -> атака


async def test_bm25_finds_inflected_forms():
    store = InMemoryVectorStore()
    await store.create_collection("c", 64)
    docs = [
        _chunk("Фішингові атаки спрямовані на викрадення паролів."),
        _chunk("Резервні копії захищають від програм-вимагачів."),
    ]
    await store.upsert("c", docs, await FakeEmbedder().embed_documents([d.text for d in docs]))
    bm25 = BM25Index(store)
    results = await bm25.search("c", "як викрадають пароль", top_k=5)
    assert results and results[0].id == docs[0].id


async def test_bm25_rebuilds_when_collection_changes():
    store = InMemoryVectorStore()
    await store.create_collection("c", 64)
    bm25 = BM25Index(store)
    assert await bm25.search("c", "пароль", 5) == []
    doc = _chunk("Надійний пароль має бути довгим.")
    await store.upsert("c", [doc], await FakeEmbedder().embed_documents([doc.text]))
    # no explicit invalidate: the changed chunk count (e.g. a write by another worker) triggers a rebuild
    assert (await bm25.search("c", "пароль", 5))[0].id == doc.id
    await store.delete_document("c", doc.document_id)
    assert await bm25.search("c", "пароль", 5) == []
