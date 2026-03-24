"""Unit tests for the retrieval layer."""

from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# ElasticSearch-backed BM25 search
# ──────────────────────────────────────────────────────────────────────────────

def _make_es_response(hits: list[dict]) -> dict:
    """Build a minimal ES search response dict."""
    return {
        "hits": {
            "hits": [
                {"_id": str(h["id"]), "_source": h["source_doc"], "_score": h["score"]}
                for h in hits
            ]
        }
    }


def test_bm25_search_returns_empty_on_no_hits(monkeypatch):
    import app.retrieval.bm25_retrieval as bm25_mod

    mock_client = MagicMock()
    mock_client.indices.exists.return_value = True
    mock_client.search.return_value = _make_es_response([])
    monkeypatch.setattr(bm25_mod, "_es_client", mock_client)

    results = bm25_mod.bm25_search("anything", top_k=5)
    assert results == []


def test_bm25_search_returns_ranked_results(monkeypatch):
    import app.retrieval.bm25_retrieval as bm25_mod

    hits = [
        {
            "id": 1,
            "source_doc": {"text": "transformer attention mechanism in NLP", "source": "a.pdf", "document_id": None, "chunk_index": 0},
            "score": 8.5,
        },
        {
            "id": 3,
            "source_doc": {"text": "self-attention and transformer architecture", "source": "c.pdf", "document_id": None, "chunk_index": 0},
            "score": 7.2,
        },
    ]

    mock_client = MagicMock()
    mock_client.indices.exists.return_value = True
    mock_client.search.return_value = _make_es_response(hits)
    monkeypatch.setattr(bm25_mod, "_es_client", mock_client)

    results = bm25_mod.bm25_search("transformer architecture", top_k=3)
    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[0]["score"] > results[1]["score"]


def test_bm25_search_filters_by_document_ids(monkeypatch):
    import app.retrieval.bm25_retrieval as bm25_mod

    mock_client = MagicMock()
    mock_client.indices.exists.return_value = True
    mock_client.search.return_value = _make_es_response([])
    monkeypatch.setattr(bm25_mod, "_es_client", mock_client)

    bm25_mod.bm25_search("test query", top_k=5, document_ids=[1, 2])

    # Verify the ES query includes a terms filter for document_id
    call_kwargs = mock_client.search.call_args
    query = call_kwargs.kwargs.get("query") or call_kwargs[1].get("query")
    assert "bool" in query
    assert any(
        "terms" in f and "document_id" in f["terms"]
        for f in query["bool"]["filter"]
    )


def test_bm25_search_returns_empty_for_empty_document_ids(monkeypatch):
    import app.retrieval.bm25_retrieval as bm25_mod

    mock_client = MagicMock()
    mock_client.indices.exists.return_value = True
    monkeypatch.setattr(bm25_mod, "_es_client", mock_client)

    results = bm25_mod.bm25_search("test", top_k=5, document_ids=[])
    assert results == []
    mock_client.search.assert_not_called()


def test_index_chunks_calls_bulk(monkeypatch):
    import app.retrieval.bm25_retrieval as bm25_mod

    mock_client = MagicMock()
    mock_client.indices.exists.return_value = True
    monkeypatch.setattr(bm25_mod, "_es_client", mock_client)

    chunks = [
        {"text": "hello world", "source": "doc.pdf", "chunk_index": 0},
        {"text": "foo bar", "source": "doc.pdf", "chunk_index": 1},
    ]
    bm25_mod.index_chunks(chunks, document_id=42)

    mock_client.bulk.assert_called_once()
    call_kwargs = mock_client.bulk.call_args
    operations = call_kwargs.kwargs.get("operations") or call_kwargs[1].get("operations")
    # Each chunk produces 2 entries (action + body)
    assert len(operations) == 4
    assert operations[1]["document_id"] == 42


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid / RRF
# ──────────────────────────────────────────────────────────────────────────────

def test_rrf_score_decreases_with_rank():
    from app.retrieval.hybrid import _rrf_score

    assert _rrf_score(0) > _rrf_score(1) > _rrf_score(10)


def test_hybrid_deduplicates(monkeypatch):
    from app.retrieval import hybrid

    doc = {"id": 42, "text": "shared doc", "source": "x.pdf", "score": 0.9}

    monkeypatch.setattr(hybrid, "dense_search", lambda q, top_k, document_ids=None: [doc])
    monkeypatch.setattr(hybrid, "bm25_search", lambda q, top_k, document_ids=None: [doc])

    results = hybrid.hybrid_search("test", top_k=5)
    # Same doc should appear only once
    ids = [r["id"] for r in results]
    assert ids.count(42) == 1


def test_hybrid_fuses_scores(monkeypatch):
    from app.retrieval import hybrid

    dense_docs = [
        {"id": 1, "text": "doc one", "source": "a", "score": 0.9},
        {"id": 2, "text": "doc two", "source": "b", "score": 0.8},
    ]
    bm25_docs = [
        {"id": 2, "text": "doc two", "source": "b", "score": 5.0},
        {"id": 3, "text": "doc three", "source": "c", "score": 4.0},
    ]
    monkeypatch.setattr(hybrid, "dense_search", lambda q, top_k, document_ids=None: dense_docs)
    monkeypatch.setattr(hybrid, "bm25_search", lambda q, top_k, document_ids=None: bm25_docs)

    results = hybrid.hybrid_search("test", top_k=5)
    # doc 2 appears in both sources so should rank highly
    ids = [r["id"] for r in results]
    assert 2 in ids
    assert ids.index(2) == 0  # highest fused score


# ──────────────────────────────────────────────────────────────────────────────
# Dense text_to_id
# ──────────────────────────────────────────────────────────────────────────────

def test_text_to_id_deterministic():
    from app.retrieval.dense import text_to_id

    id1 = text_to_id("hello world", "doc.pdf")
    id2 = text_to_id("hello world", "doc.pdf")
    assert id1 == id2


def test_text_to_id_different_inputs_differ():
    from app.retrieval.dense import text_to_id

    assert text_to_id("hello", "a.pdf") != text_to_id("world", "a.pdf")
