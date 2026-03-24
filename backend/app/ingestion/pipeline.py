"""
Document ingestion pipeline.

Supported formats: PDF, HTML, Markdown, DOCX, TXT.
Each document is parsed into text elements, chunked, embedded, and stored in Qdrant.
"""

import io
import re
from pathlib import Path

from app.retrieval.bm25_retrieval import index_chunks
from app.retrieval.dense import upsert_chunks

CHUNK_SIZE = 512  # characters per chunk
CHUNK_OVERLAP = 64


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _parse_pdf(content: bytes) -> str:
    import fitz  # pymupdf

    doc = fitz.open(stream=content, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def _parse_docx(content: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(content))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _parse_html(content: bytes) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(content, "html.parser")
    # Remove scripts and styles
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _parse_markdown(content: bytes) -> str:
    import markdown
    from bs4 import BeautifulSoup

    html = markdown.markdown(content.decode("utf-8", errors="ignore"))
    return BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)


def _parse_txt(content: bytes) -> str:
    return content.decode("utf-8", errors="ignore")


def parse_document(filename: str, content: bytes) -> str:
    """Parse a document and return plain text."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(content)
    elif ext == ".docx":
        return _parse_docx(content)
    elif ext in (".html", ".htm"):
        return _parse_html(content)
    elif ext in (".md", ".markdown"):
        return _parse_markdown(content)
    else:
        return _parse_txt(content)


def ingest_document(filename: str, content: bytes, document_id: int | None = None) -> int:
    """
    Parse, chunk, embed, and index a document.
    Returns the number of chunks indexed.
    """
    text = parse_document(filename, content)
    # Clean up excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    raw_chunks = _chunk_text(text)
    if not raw_chunks:
        return 0

    chunk_dicts = [
        {"text": chunk, "source": filename, "chunk_index": i}
        for i, chunk in enumerate(raw_chunks)
    ]

    upsert_chunks(chunk_dicts, document_id=document_id)
    index_chunks(chunk_dicts, document_id=document_id)
    return len(chunk_dicts)

