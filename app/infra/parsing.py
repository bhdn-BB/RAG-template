"""Document parsing with LangChain loaders + recursive character splitting.

- Files: loaders are selected through a registry keyed by file extension. Add a new format by
  registering a loader class that accepts a file path.
- Web pages: LangChain's `WebBaseLoader` downloads and parses the page. Only the main content
  (`<main>` / `<article>`) is parsed, so menus, footers and scripts don't pollute the chunks;
  pages without those tags fall back to the whole document.
  URLs must be validated by the SSRF guard (`app/infra/url_resolver.py`) BEFORE calling `parse_url`.
"""

import os
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path

from bs4 import SoupStrainer
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.exceptions import UnsupportedFileTypeError, ValidationError
from app.infra.url_resolver import USER_AGENT

os.environ.setdefault("USER_AGENT", USER_AGENT)  # silences WebBaseLoader's warning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.document_loaders import (
        BSHTMLLoader,
        Docx2txtLoader,
        PyPDFLoader,
        TextLoader,
        WebBaseLoader,
    )

LoaderFactory = Callable[[str], BaseLoader]

LOADERS: dict[str, LoaderFactory] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
    ".md": lambda path: TextLoader(path, encoding="utf-8"),
    ".html": lambda path: BSHTMLLoader(path, open_encoding="utf-8"),
    ".htm": lambda path: BSHTMLLoader(path, open_encoding="utf-8"),
}

# Tags parsed from web pages. `title` is kept so WebBaseLoader can fill metadata["title"].
_MAIN_CONTENT = SoupStrainer(["title", "main", "article"])

WebLoaderFactory = Callable[[str, bool], BaseLoader]


def web_loader(url: str, main_only: bool = True) -> BaseLoader:
    """LangChain WebBaseLoader configured for RAG ingestion.

    Redirects are disabled: the SSRF guard has already resolved and validated the final URL.
    """
    return WebBaseLoader(
        web_path=url,
        header_template={"User-Agent": USER_AGENT},
        requests_kwargs={"allow_redirects": False, "timeout": 20},
        raise_for_status=True,
        bs_kwargs={"parse_only": _MAIN_CONTENT} if main_only else None,
        bs_get_text_kwargs={"separator": "\n", "strip": True},
        show_progress=False,
    )


# Separators tuned for prose: paragraphs -> lines -> sentences -> words.
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


def _clean_title(title: str, url: str) -> str:
    title = (title or "").strip() or url
    # "Article title - site.name" -> "Article title"
    return title.rsplit(" - ", 1)[0].strip() if " - " in title else title


class LangChainParser:
    """`DocumentParser` adapter built on LangChain loaders and splitters."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        web_loader_factory: WebLoaderFactory = web_loader,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=_SEPARATORS
        )
        self.web_loader_factory = web_loader_factory

    @staticmethod
    def supported_extensions() -> list[str]:
        return sorted(LOADERS)

    def parse(self, filename: str, data: bytes) -> list[tuple[str, dict]]:
        suffix = Path(filename).suffix.lower()
        loader_factory = LOADERS.get(suffix)
        if loader_factory is None:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{suffix}'. Supported: {', '.join(self.supported_extensions())}"
            )
        # LangChain loaders work with paths, so spill the upload into a temp file.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(data)
            documents = loader_factory(str(path)).load()

        chunks: list[tuple[str, dict]] = []
        for doc in documents:
            meta = {k: v for k, v in doc.metadata.items() if k in {"page", "page_label", "title"}}
            chunks.extend(self.split_text(doc.page_content, meta))
        if not chunks:
            raise ValidationError(f"No text could be extracted from '{filename}'.")
        return chunks

    def parse_url(self, url: str) -> tuple[str, list[tuple[str, dict]]]:
        """Load a web page with LangChain's WebBaseLoader. Returns (title, chunks). Blocking (requests)."""
        try:
            documents = self.web_loader_factory(url, True).load()
            raw_title = documents[0].metadata.get("title", "") if documents else ""
            # Main content is the page text minus the <title> line.
            text = documents[0].page_content.replace(raw_title, "", 1).strip() if documents else ""
            if len(text) < 200:  # no <main>/<article> on this page -> parse the whole document
                documents = self.web_loader_factory(url, False).load()
                text = documents[0].page_content.strip() if documents else ""
        except Exception as exc:  # requests/bs4 errors
            raise ValidationError(f"Could not load {url}: {exc}") from exc

        title = _clean_title(raw_title, url)
        meta = {"title": title}
        if description := documents[0].metadata.get("description"):
            meta["description"] = description
        chunks = self.split_text(text, meta)
        if not chunks:
            raise ValidationError(f"No readable text found at {url}.")
        return title, chunks

    def split_text(self, text: str, metadata: dict) -> list[tuple[str, dict]]:
        return [(piece, dict(metadata)) for piece in self.splitter.split_text(text) if piece.strip()]
