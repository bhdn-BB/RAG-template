"""End-to-end API flow with dishka-injected fakes (no GPU, no databases)."""

import json

from tests.conftest import UA_DOCS


async def _setup(client) -> None:
    assert (await client.post("/collections", json={"name": "cyber"})).status_code == 201
    files = [("files", (name, text.encode(), "text/plain")) for name, text in UA_DOCS.items()]
    response = await client.post("/collections/cyber/documents", files=files)
    assert response.status_code == 201, response.text
    assert len(response.json()["documents"]) == 3


async def test_health(client):
    assert (await client.get("/health")).json() == {"status": "ok"}


async def test_providers(client):
    body = (await client.get("/providers")).json()
    assert body["vector_store"]["active"] == "qdrant"
    assert ".pdf" in body["supported_file_types"]


async def test_collection_crud(client):
    assert (await client.post("/collections", json={"name": "Bad-Name"})).status_code == 422
    assert (await client.post("/collections", json={"name": "kb"})).status_code == 201
    assert (await client.post("/collections", json={"name": "kb"})).status_code == 409
    assert [c["name"] for c in (await client.get("/collections")).json()] == ["kb"]
    assert (await client.delete("/collections/kb")).status_code == 204
    assert (await client.get("/collections/kb")).status_code == 404


async def test_document_management(client):
    await _setup(client)
    docs = (await client.get("/collections/cyber/documents")).json()
    assert {d["filename"] for d in docs} == set(UA_DOCS)

    doc_id = docs[0]["id"]
    chunks = (await client.get(f"/collections/cyber/documents/{doc_id}/chunks")).json()
    assert chunks and chunks[0]["index"] == 0
    assert (await client.post(f"/collections/cyber/documents/{doc_id}/reindex")).status_code == 200

    text_doc = await client.post(
        "/collections/cyber/documents/text", json={"title": "note", "text": "Нотатка"}
    )
    assert text_doc.status_code == 201

    assert (await client.delete(f"/collections/cyber/documents/{doc_id}")).status_code == 204
    assert (await client.get(f"/collections/cyber/documents/{doc_id}")).status_code == 404
    assert (await client.get("/collections/cyber")).json()["document_count"] == 3


async def test_unsupported_file_is_reported(client):
    await client.post("/collections", json={"name": "kb"})
    response = await client.post("/collections/kb/documents", files=[("files", ("x.exe", b"MZ", "x"))])
    assert response.status_code == 201
    assert "x.exe" in response.json()["errors"]


async def test_hybrid_search_exposes_scores(client):
    await _setup(client)
    response = await client.post("/collections/cyber/search", json={"query": "як захиститися від фішингу"})
    hits = response.json()["hits"]
    assert hits[0]["filename"] == "phishing.txt"
    assert hits[0]["rrf_score"] is not None and hits[0]["rerank_score"] is not None
    assert hits[0]["bm25_rank"] == 1  # lexical rank from LangChain's BM25Retriever


async def test_chat_and_regenerate(client, fake_llm):
    await _setup(client)
    response = await client.post(
        "/chat", json={"collection": "cyber", "question": "Що таке фішинг?", "llm_provider": "fake"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == fake_llm.answer
    assert body["sources"][0]["n"] == 1 and body["sources"][0]["filename"] == "phishing.txt"

    history = [
        {"role": "user", "content": "Що таке фішинг?"},
        {"role": "assistant", "content": body["answer"]},
    ]
    regen = await client.post(
        "/chat/regenerate",
        json={"question": "Що таке фішинг?", "history": history, "sources": body["sources"]},
    )
    assert regen.status_code == 200
    assert [s["chunk_id"] for s in regen.json()["sources"]] == [s["chunk_id"] for s in body["sources"]]


async def test_chat_stream(client, fake_llm):
    await _setup(client)
    async with client.stream(
        "POST", "/chat/stream", json={"collection": "cyber", "question": "Що таке фішинг?"}
    ) as response:
        raw = "".join([chunk async for chunk in response.aiter_text()])
    events = [line.removeprefix("event: ").strip() for line in raw.splitlines() if line.startswith("event:")]
    assert events[0] == "sources" and events[-1] == "done" and "token" in events
    data = [json.loads(line[5:]) for line in raw.splitlines() if line.startswith("data:")]
    text = "".join(d.get("text", "") for d in data[1:])
    assert text.strip() == fake_llm.answer


async def test_chat_unknown_collection_404(client):
    response = await client.post("/chat", json={"collection": "missing", "question": "?"})
    assert response.status_code == 404


PAGE_URL = "https://pro.kyiv.ua/vartist-proizdu-kyiv/"


async def test_url_ingestion_and_source_attribution(client):
    await client.post("/collections", json={"name": "web"})
    response = await client.post("/collections/web/documents/url", json={"url": PAGE_URL})
    assert response.status_code == 201, response.text
    doc = response.json()
    assert doc["source_url"] == PAGE_URL and doc["filename"] == "Вартість проїзду в Києві"

    chunks = (await client.get(f"/collections/web/documents/{doc['id']}/chunks")).json()
    text = " ".join(c["text"] for c in chunks)
    assert "30 гривень" in text and "pro.kyiv.ua" not in text  # <title> line not duplicated into content
    assert chunks[0]["stored_payload"]["chunk_id"] == chunks[0]["id"]

    # re-adding the same URL replaces the old version instead of duplicating it
    await client.post("/collections/web/documents/url", json={"url": PAGE_URL})
    assert len((await client.get("/collections/web/documents")).json()) == 1

    body = (
        await client.post(
            "/chat",
            json={"collection": "web", "question": "Скільки коштує проїзд?", "include_llm_input": True},
        )
    ).json()
    assert body["sources"][0]["source_url"] == PAGE_URL
    assert body["sources"][0]["source"] == PAGE_URL
    llm_input = body["llm_input"]
    assert "untrusted DATA" in llm_input["system_prompt"]
    assert f'source="{PAGE_URL}"' in llm_input["messages"][-1]["content"]


async def test_llm_input_hidden_by_default(client):
    await _setup(client)
    body = (await client.post("/chat", json={"collection": "cyber", "question": "Що таке фішинг?"})).json()
    assert body["llm_input"] is None


async def test_ssrf_urls_are_rejected():
    from app.core.exceptions import ValidationError
    from app.infra.url_resolver import SafeUrlResolver

    for url in ["http://localhost:6333/collections", "http://169.254.169.254/", "ftp://example.com/x"]:
        try:
            await SafeUrlResolver().resolve(url)
        except ValidationError:
            continue
        raise AssertionError(f"not blocked: {url}")


async def test_edit_and_delete_chunk(client):
    await _setup(client)
    doc = next(
        d for d in (await client.get("/collections/cyber/documents")).json() if d["filename"] == "mfa.txt"
    )
    base = f"/collections/cyber/documents/{doc['id']}/chunks"
    chunk = (await client.get(base)).json()[0]

    edited = await client.patch(
        f"{base}/{chunk['id']}", json={"text": "Квантовий ключ шифрування захищає вхід."}
    )
    assert edited.status_code == 200 and edited.json()["metadata"].get("edited_at")
    assert (await client.get(base)).json()[0]["text"].startswith("Квантовий")

    # the edited text is searchable right away (dense + BM25 re-indexed)
    hits = (await client.post("/collections/cyber/search", json={"query": "квантовий ключ"})).json()["hits"]
    assert hits[0]["chunk_id"] == chunk["id"]

    assert (await client.delete(f"{base}/{chunk['id']}")).status_code == 204
    assert (await client.get(f"/collections/cyber/documents/{doc['id']}")).status_code == 404
