"""
End-to-end integration tests.
These require a running Qdrant instance; they are skipped otherwise.
"""

import pytest
import pytest_asyncio


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qdrant_available():
    """Check if Qdrant is reachable; skip tests if not."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333, timeout=2)
        client.get_collections()
        return True
    except Exception:
        pytest.skip("Qdrant not available — skipping e2e tests")


@pytest.fixture(scope="session")
def test_collection(qdrant_available):
    """Create a temporary test collection and tear it down after."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(host="localhost", port=6333)
    col = "test_ara_e2e"
    # Ensure clean state
    try:
        client.delete_collection(col)
    except Exception:
        pass
    client.create_collection(col, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    yield col
    client.delete_collection(col)


# ──────────────────────────────────────────────────────────────────────────────
# Ingestion + retrieval round-trip
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_and_dense_search(test_collection, monkeypatch):
    import app.config as cfg_mod
    import app.retrieval.dense as dense_mod

    monkeypatch.setattr(cfg_mod.settings, "qdrant_collection", test_collection)
    # Reset cached client/embedder to pick up monkeypatched collection name
    monkeypatch.setattr(dense_mod, "_client", None)

    chunks = [
        {"text": "Transformers use self-attention mechanisms.", "source": "paper1.pdf", "chunk_index": 0},
        {"text": "BERT is a bidirectional encoder representation.", "source": "paper2.pdf", "chunk_index": 0},
    ]
    dense_mod.upsert_chunks(chunks)

    results = dense_mod.dense_search("self-attention transformer", top_k=2)
    assert len(results) >= 1
    assert any("Transformer" in r["text"] or "attention" in r["text"].lower() for r in results)


@pytest.mark.asyncio
async def test_hybrid_search_returns_results(test_collection, monkeypatch):
    import app.config as cfg_mod
    import app.retrieval.dense as dense_mod
    import app.retrieval.bm25_retrieval as bm25_mod

    monkeypatch.setattr(cfg_mod.settings, "qdrant_collection", test_collection)
    monkeypatch.setattr(dense_mod, "_client", None)
    # Mock ES client so hybrid search doesn't require a running ElasticSearch
    from unittest.mock import MagicMock
    mock_es = MagicMock()
    mock_es.indices.exists.return_value = True
    mock_es.search.return_value = {"hits": {"hits": []}}
    monkeypatch.setattr(bm25_mod, "_es_client", mock_es)

    from app.retrieval.hybrid import hybrid_search
    results = hybrid_search("BERT encoder", top_k=5)
    assert isinstance(results, list)


# ──────────────────────────────────────────────────────────────────────────────
# Planner
# ──────────────────────────────────────────────────────────────────────────────

def test_rule_based_classifier_simple():
    from app.agent.planner import classify_rule_based
    assert classify_rule_based("What is the learning rate used?") == "simple"


def test_rule_based_classifier_complex():
    from app.agent.planner import classify_rule_based
    assert classify_rule_based("Compare BERT vs GPT") == "complex"


def test_rule_based_classifier_multi_hop():
    from app.agent.planner import classify_rule_based
    assert classify_rule_based("Why does attention help in NLP?") == "multi-hop"


# ──────────────────────────────────────────────────────────────────────────────
# Ingestion pipeline
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_txt():
    from app.ingestion.pipeline import parse_document
    content = b"Hello world. This is a test document."
    text = parse_document("test.txt", content)
    assert "Hello world" in text


def test_parse_markdown():
    from app.ingestion.pipeline import parse_document
    content = b"# Title\n\nSome **bold** content here."
    text = parse_document("test.md", content)
    assert "Title" in text
    assert "bold" in text


def test_chunk_text_overlap():
    from app.ingestion.pipeline import _chunk_text
    text = "a" * 1200
    chunks = _chunk_text(text, chunk_size=512, overlap=64)
    assert len(chunks) >= 2
    # Check overlap: end of chunk 0 should appear at start of chunk 1
    assert chunks[0][-64:] == chunks[1][:64]


def test_chunk_text_empty():
    from app.ingestion.pipeline import _chunk_text
    assert _chunk_text("") == []
    assert _chunk_text("   ") == []
