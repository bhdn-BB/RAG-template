"""Streamlit demo UI for the Ukrainian RAG API.

Tabs:
- 💬 Chat            : answers with clickable sources, the exact prompt sent to the LLM, regenerate.
- 📚 Knowledge base  : add sources (URL / files / text), inspect how chunks are stored in the
                       vector DB, edit / delete chunks and documents.
- 🔍 Search debug    : hybrid search scores (dense, BM25, RRF, reranker).

The API is stateless, so the conversation (history + last sources) lives in `st.session_state`.
"""

import html
import json
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1").rstrip("/")
TIMEOUT = 900

PROVIDER_LABELS = {
    "anthropic": "Anthropic · Claude",
    "openai": "OpenAI",
    "gemini": "Google Gemini",
    "openrouter": "OpenRouter",
    "groq": "GroqCloud",
    "vllm": "vLLM · self-hosted Qwen",
    "ollama": "Ollama · self-hosted",
}

st.set_page_config(page_title="Ukrainian RAG", page_icon="🇺🇦", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.hero {
  background: linear-gradient(120deg, #0057B7 0%, #1f6fd1 55%, #FFD500 140%);
  color: #fff; border-radius: 18px; padding: 22px 28px; margin-bottom: 18px;
  box-shadow: 0 8px 24px rgba(0, 87, 183, .18);
}
.hero h1 {color: #fff; margin: 0 0 4px 0; font-size: 1.9rem;}
.hero p {margin: 0; opacity: .92;}
.pipeline {display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; align-items: center;}
.chip {background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.35);
  border-radius: 999px; padding: 3px 12px; font-size: .82rem; white-space: nowrap;}
.chip b {color: #FFD500;}
.arrow {opacity: .7;}
.src-card {background: #fff; border: 1px solid #dfe7f3; border-left: 4px solid #0057B7;
  border-radius: 12px; padding: 10px 14px; margin: 6px 0;}
.src-card .meta {font-size: .8rem; color: #5b6b80; margin-bottom: 4px;}
.src-card .meta a {color: #0057B7; font-weight: 600; text-decoration: none;}
.score {display: inline-block; background: #FFF4C2; color: #6b5400; border-radius: 6px;
  padding: 0 6px; font-size: .75rem; margin-left: 6px;}
.src-card .text {font-size: .86rem; color: #2a3646; white-space: pre-wrap;}
.side-info {font-size: .8rem; color: #4a5a70; line-height: 1.5;}
.badge-ok {color: #137333; font-weight: 600;} .badge-lock {color: #b3261e; font-weight: 600;}
div[data-testid="stChatMessage"] {background: #fff; border-radius: 14px; border: 1px solid #e6edf6;}
</style>
""",
    unsafe_allow_html=True,
)


# --- API helpers -------------------------------------------------------------


def api(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        st.error(f"API is unreachable ({API_URL}): {exc}")
        return None
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text
        st.error(f"{response.status_code}: {detail}")
        return None
    return response.json() if response.content else {}


def stream_chat(path: str, payload: dict):
    """Consume the SSE stream; yields text tokens and stores metadata in session_state."""
    with requests.post(f"{API_URL}{path}", json=payload, stream=True, timeout=TIMEOUT) as response:
        if response.status_code >= 400:
            st.error(f"{response.status_code}: {response.text}")
            return
        event = None
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip() or "{}")
                if event == "sources":
                    st.session_state.pending_meta = data
                elif event == "token":
                    yield data["text"]
                elif event == "error":
                    st.error(data.get("detail", "Streaming error"))
                    return


state = st.session_state
state.setdefault("messages", [])
state.setdefault("pending_meta", None)


def history_payload(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]


providers = api("GET", "/providers") or {}
retrieval = providers.get("retrieval", {"top_k": "?", "rerank_top_n": "?"})
emb, rer = providers.get("embedding", {}), providers.get("reranker", {})
vector_db = providers.get("vector_store", {}).get("active", "?")


# --- header --------------------------------------------------------------------

st.markdown(
    f"""
<div class="hero">
  <h1>🇺🇦 Ukrainian RAG</h1>
  <p>Hybrid retrieval over your documents and web pages: answers grounded in sources.</p>
  <div class="pipeline">
    <span class="chip">❓ Question</span><span class="arrow">→</span>
    <span class="chip">🧬 Embedder + BM25 · <b>top {retrieval["top_k"]}</b></span><span class="arrow">→</span>
    <span class="chip">⚖️ RRF fusion</span><span class="arrow">→</span>
    <span class="chip">🎯 Reranker · <b>top {retrieval["rerank_top_n"]}</b></span><span class="arrow">→</span>
    <span class="chip">🤖 LLM</span><span class="arrow">→</span>
    <span class="chip">💬 Answer + [n] sources</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# --- sidebar -------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📁 Collection")
    collections = api("GET", "/collections") or []
    names = [c["name"] for c in collections]
    collection = st.selectbox(
        "Collection",
        names,
        index=0 if names else None,
        placeholder="Create one below",
        label_visibility="collapsed",
    )
    with st.expander("➕ New collection", expanded=not names):
        new_name = st.text_input("Name (a-z, 0-9, _)", key="new_collection")
        if (
            st.button("Create", use_container_width=True)
            and new_name
            and api("POST", "/collections", json={"name": new_name})
        ):
            st.rerun()

    st.markdown("### 🤖 LLM")
    llm_info = providers.get("llm", {})
    all_providers: dict = llm_info.get("providers", {})
    options = list(all_providers)
    default = llm_info.get("default")

    def provider_label(name: str) -> str:
        info = all_providers.get(name, {})
        self_hosted = info.get("key_env") is None
        icon = "🖥️" if self_hosted else ("✅" if info.get("configured") else "🔒")
        return f"{icon}  {PROVIDER_LABELS.get(name, name)}"

    llm_provider = st.selectbox(
        "Provider",
        options,
        index=options.index(default) if default in options else 0,
        format_func=provider_label,
    )
    provider_info = all_providers.get(llm_provider, {})
    provider_ready = provider_info.get("configured", False)
    if not provider_ready:
        st.warning(f"No API key. Add `{provider_info.get('key_env')}=...` to `.env` and restart the API.")
    elif provider_info.get("key_env") is None:
        st.caption("🖥️ Self-hosted model server: make sure it is running.")

    model = st.text_input("Model", placeholder=provider_info.get("default_model", ""), help="Empty = default")
    use_stream = st.toggle("Stream tokens", value=True)
    show_context = st.toggle("Show context sent to LLM", value=True)
    if st.button("🧹 New chat", use_container_width=True):
        state.messages = []
        st.rerun()

    st.markdown("### ⚙️ Pipeline")
    st.markdown(
        f"""<div class="side-info">
🗄️ Vector DB: <b>{vector_db}</b><br>
🧬 Embedder: <b>{emb.get("model", "?")}</b> · {emb.get("dim", "?")}d<br>
🎯 Reranker: <b>{rer.get("model", "?")}</b><br>
🔢 Retrieve <b>{retrieval["top_k"]}</b> → rerank to <b>{retrieval["rerank_top_n"]}</b>
</div>""",
        unsafe_allow_html=True,
    )

selection = {"llm_provider": llm_provider, "model": model or None, "include_llm_input": show_context}


# --- rendering helpers -----------------------------------------------------------


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})", expanded=True):
        for s in sources:
            title = html.escape(s.get("metadata", {}).get("title") or s.get("filename") or "")
            if s.get("source_url"):
                url = html.escape(s["source_url"])
                origin = f'<a href="{url}" target="_blank">{title or url}</a> · {url}'
            else:
                origin = f"<b>{html.escape(s.get('source') or title)}</b>"
            score = s.get("rerank_score")
            badge = f'<span class="score">rerank {score:.3f}</span>' if score is not None else ""
            text = html.escape(s["text"][:700] + ("…" if len(s["text"]) > 700 else ""))
            st.markdown(
                f'<div class="src-card"><div class="meta">[{s["n"]}] {origin} · '
                f"chunk {s['index']}{badge}</div>"
                f'<div class="text">{text}</div></div>',
                unsafe_allow_html=True,
            )


def render_llm_input(llm_input: dict | None) -> None:
    if not llm_input:
        return
    with st.expander("🧠 Context sent to the LLM (exact prompt)"):
        st.markdown("**System prompt**")
        st.code(llm_input["system_prompt"], language="markdown")
        for i, m in enumerate(llm_input["messages"], start=1):
            st.markdown(f"**Message {i} · {m['role']}**")
            st.code(m["content"], language="xml")


def run_llm(path: str, payload: dict) -> None:
    with st.chat_message("assistant", avatar="🤖"):
        if use_stream:
            state.pending_meta = None
            with st.spinner("Searching: embedder + BM25 → reranker…"):
                answer = st.write_stream(stream_chat(f"{path}/stream", payload))
            meta = state.pending_meta or {}
        else:
            with st.spinner("Searching and generating…"):
                meta = api("POST", path, json=payload) or {}
            answer = meta.get("answer", "")
            st.markdown(answer)
        if meta.get("flagged"):
            st.warning(f"⚠️ Prompt-injection heuristics matched: {', '.join(meta.get('flag_reasons', []))}")
        if meta.get("provider"):
            st.caption(f"{meta.get('provider')} · {meta.get('model')}")
        render_sources(meta.get("sources", []))
        render_llm_input(meta.get("llm_input"))
    if answer:
        state.messages.append(
            {
                "role": "assistant",
                "content": answer if isinstance(answer, str) else "".join(answer),
                "sources": meta.get("sources", []),
                "provider": meta.get("provider"),
                "model": meta.get("model"),
                "llm_input": meta.get("llm_input"),
            }
        )


# --- tabs ----------------------------------------------------------------------------

chat_tab, kb_tab, search_tab = st.tabs(["💬 Chat", "📚 Knowledge base", "🔍 Search debug"])

with chat_tab:
    if not collection:
        st.info("Create a collection in the sidebar and add sources in **📚 Knowledge base**.")
    elif not state.messages:
        st.markdown(
            "**Try:** «Скільки коштує безлімітний місячний проїзний?» · «How much is a 2-day guest ticket?»"
        )

    for message in state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "🧑"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                st.caption(f"{message.get('provider')} · {message.get('model')}")
                render_sources(message.get("sources", []))
                render_llm_input(message.get("llm_input"))

    can_regenerate = bool(state.messages) and state.messages[-1]["role"] == "assistant"
    if can_regenerate and st.button(
        "🔁 Regenerate (same context, no new search)", disabled=not provider_ready
    ):
        last_answer = state.messages.pop()
        question = state.messages[-1]["content"]
        run_llm(
            "/chat/regenerate",
            {
                **selection,
                "question": question,
                "history": history_payload(state.messages[:-1]),
                "sources": last_answer["sources"],
            },
        )
        st.rerun()

    placeholder = "Запитайте щось українською…" if provider_ready else "Select a provider with an API key"
    if question := st.chat_input(placeholder, disabled=not (collection and provider_ready)):
        history = history_payload(state.messages)
        state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)
        run_llm("/chat", {**selection, "collection": collection, "question": question, "history": history})


with kb_tab:
    if not collection:
        st.info("Create or select a collection first.")
    else:
        documents = api("GET", f"/collections/{collection}/documents") or []
        m1, m2, m3 = st.columns(3)
        m1.metric("Documents", len(documents))
        m2.metric("Chunks (vectors)", sum(d["chunk_count"] for d in documents))
        m3.metric("Vector DB", vector_db)

        st.markdown(f"#### ➕ Add a source to `{collection}`")
        url_col, file_col, text_col = st.tabs(["🌐 Web page", "📄 Files", "✍️ Text"])
        with url_col:
            url = st.text_input("Page URL", placeholder="https://pro.kyiv.ua/vartist-proizdu-kyiv/")
            st.caption(
                "Parsed with LangChain WebBaseLoader. Adding the same URL again replaces the old version."
            )
            if st.button("Parse & index page", type="primary", disabled=not url):
                with st.spinner("Downloading, parsing, embedding…"):
                    doc = api("POST", f"/collections/{collection}/documents/url", json={"url": url})
                if doc:
                    st.success(f"Indexed **{doc['filename']}** · {doc['chunk_count']} chunks")
                    st.rerun()
        with file_col:
            files = st.file_uploader(
                "PDF, DOCX, TXT, MD, HTML",
                accept_multiple_files=True,
                type=[ext.lstrip(".") for ext in providers.get("supported_file_types", [])] or None,
            )
            if st.button("Upload & index", type="primary", disabled=not files):
                with st.spinner("Parsing, embedding, indexing…"):
                    result = api(
                        "POST",
                        f"/collections/{collection}/documents",
                        files=[("files", (f.name, f.getvalue(), f.type)) for f in files],
                    )
                if result:
                    st.success(f"Indexed {len(result['documents'])} document(s)")
                    for name, error in result.get("errors", {}).items():
                        st.warning(f"{name}: {error}")
        with text_col:
            title = st.text_input("Title")
            body = st.text_area("Text", height=150)
            if st.button("Index text", type="primary", disabled=not (title and body)):
                doc = api(
                    "POST", f"/collections/{collection}/documents/text", json={"title": title, "text": body}
                )
                if doc:
                    st.success(f"Indexed **{doc['filename']}** · {doc['chunk_count']} chunks")
                    st.rerun()

        st.markdown("#### 🗄️ What is stored in the vector DB")
        if not documents:
            st.caption("No documents yet.")
        for doc in documents:
            origin = (
                f"[{doc['source_url']}]({doc['source_url']})" if doc.get("source_url") else doc["filename"]
            )
            icon = "🌐" if doc.get("source_url") else "📄"
            with st.expander(f"{icon} {doc['filename']} · {doc['chunk_count']} chunks"):
                st.markdown(
                    f"**Source:** {origin}  \n**Document ID:** `{doc['id']}`  \n"
                    f"**Added:** {doc['uploaded_at']}"
                )
                if st.button("🗑 Delete whole document", key=f"deldoc-{doc['id']}"):
                    api("DELETE", f"/collections/{collection}/documents/{doc['id']}")
                    st.rerun()
                if not st.toggle("Show chunks", key=f"show-{doc['id']}"):
                    continue
                chunks = api("GET", f"/collections/{collection}/documents/{doc['id']}/chunks") or []
                for chunk in chunks:
                    edited = " · ✏️ edited" if chunk.get("metadata", {}).get("edited_at") else ""
                    st.markdown(f"---\n**Chunk #{chunk['index']}** · {len(chunk['text'])} chars{edited}")
                    new_text = st.text_area(
                        "Text",
                        chunk["text"],
                        height=180,
                        key=f"txt-{chunk['id']}",
                        label_visibility="collapsed",
                    )
                    c1, c2, c3 = st.columns([1, 1, 4])
                    save_path = f"/collections/{collection}/documents/{doc['id']}/chunks/{chunk['id']}"
                    if c1.button("💾 Save", key=f"save-{chunk['id']}", disabled=new_text == chunk["text"]):
                        with st.spinner("Re-embedding…"):
                            ok = api("PATCH", save_path, json={"text": new_text})
                        if ok:
                            st.rerun()
                    if c2.button("🗑 Delete", key=f"delc-{chunk['id']}"):
                        api("DELETE", save_path)
                        st.rerun()
                    with c3.popover("🔬 Raw record in DB"):
                        st.caption(
                            f"Point ID `{chunk['id']}` · vector: {emb.get('dim', '?')} floats (cosine)"
                        )
                        st.json(chunk.get("stored_payload") or {})

        st.divider()
        if st.button("⚠️ Delete collection"):
            api("DELETE", f"/collections/{collection}")
            state.messages = []
            st.rerun()


with search_tab:
    if collection:
        query = st.text_input("Query", placeholder="Скільки коштує пересадковий квиток?")
        rerank = st.checkbox("Apply cross-encoder reranker", value=True)
        if st.button("Search", type="primary", disabled=not query):
            with st.spinner("Searching…"):
                result = api(
                    "POST", f"/collections/{collection}/search", json={"query": query, "rerank": rerank}
                )
            for i, hit in enumerate((result or {}).get("hits", []), start=1):
                parts = [
                    f"{k.replace('_score', '')}: {hit[k]:.4f}"
                    for k in ("dense_score", "rrf_score", "rerank_score")
                    if hit.get(k) is not None
                ]
                if hit.get("bm25_rank") is not None:
                    parts.insert(1, f"bm25 rank: {hit['bm25_rank']}")
                scores = " · ".join(parts)
                st.markdown(f"**{i}. {hit['filename']}** · chunk {hit['index']}  \n`{scores}`")
                st.caption(hit["text"][:500])
