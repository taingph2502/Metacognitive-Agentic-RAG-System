import contextlib
import hashlib
import io
import logging
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from huggingface_hub import login as _hf_login

from app.config import settings

_client: QdrantClient | None = None
_embedder: SentenceTransformer | None = None
_hf_runtime_configured = False

EMBEDDING_DIM = 384  # bge-small-en-v1.5


@contextlib.contextmanager
def _quiet_huggingface_output():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield


def _configure_huggingface_runtime() -> None:
    """Keep Hugging Face downloads quiet and authenticated when HF_TOKEN exists."""
    global _hf_runtime_configured
    if _hf_runtime_configured:
        return

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        with _quiet_huggingface_output():
            _hf_login(token=token, add_to_git_credential=False)

    _hf_runtime_configured = True


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return _client


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _configure_huggingface_runtime()
        with _quiet_huggingface_output():
            _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def embed(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def ensure_collection() -> None:
    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection not in existing:
        client.create_collection(
            settings.qdrant_collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def text_to_id(text: str, source: str) -> int:
    """Deterministic integer ID from text content."""
    h = hashlib.md5(f"{source}::{text[:200]}".encode()).hexdigest()
    return int(h[:15], 16)


def upsert_chunks(chunks: list[dict], document_id: int | None = None) -> None:
    """
    chunks: list of {"text": str, "source": str, "chunk_index": int}
    """
    ensure_collection()
    client = get_qdrant_client()
    vectors = embed([c["text"] for c in chunks])
    points = [
        PointStruct(
            id=text_to_id(c["text"], c["source"]),
            vector=v,
            payload={
                "text": c["text"],
                "source": c["source"],
                "chunk_index": c.get("chunk_index", 0),
                "document_id": document_id,
            },
        )
        for c, v in zip(chunks, vectors)
    ]
    # Upsert in batches of 100
    for i in range(0, len(points), 100):
        client.upsert(
            collection_name=settings.qdrant_collection, points=points[i : i + 100]
        )


def dense_search(query: str, top_k: int = 5, document_ids: list[int] | None = None) -> list[dict]:
    ensure_collection()
    
    # If a filter is requested but no IDs provided, return empty
    if document_ids is not None and len(document_ids) == 0:
        return []

    client = get_qdrant_client()
    q_vec = embed([query])[0]
    
    # Filtering to ensure we only retrieve from existing documents
    search_filter = None
    if document_ids is not None:
        from qdrant_client.models import FieldCondition, Filter, MatchAny
        search_filter = Filter(
            must=[
                FieldCondition(key="document_id", match=MatchAny(any=document_ids)),
            ]
        )

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=q_vec,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
    )
    return [
        {
            "id": r.id,
            "text": r.payload["text"],
            "source": r.payload["source"],
            "score": r.score,
        }
        for r in response.points
    ]


def wipe_all_embeddings() -> None:
    """Completely clear the research_papers collection."""
    client = get_qdrant_client()
    client.delete_collection(collection_name=settings.qdrant_collection)
    ensure_collection()
