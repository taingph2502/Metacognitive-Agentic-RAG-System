from app.retrieval.bm25_retrieval import bm25_search
from app.retrieval.dense import dense_search


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank + 1)


def hybrid_search(query: str, top_k: int = 10, document_ids: list[int] | None = None) -> list[dict]:
    """
    Fuse dense + BM25 results using Reciprocal Rank Fusion.
    Returns top_k deduplicated documents sorted by fused score.
    """
    dense_results = dense_search(query, top_k=top_k, document_ids=document_ids)
    bm25_results = bm25_search(query, top_k=top_k, document_ids=document_ids)

    fused_scores: dict[int, float] = {}
    seen: dict[int, dict] = {}

    for rank, doc in enumerate(dense_results):
        doc_id = doc["id"]
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + _rrf_score(rank)
        seen[doc_id] = doc

    for rank, doc in enumerate(bm25_results):
        doc_id = doc["id"]
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + _rrf_score(rank)
        seen[doc_id] = doc

    sorted_ids = sorted(fused_scores, key=lambda x: fused_scores[x], reverse=True)[
        :top_k
    ]
    return [
        {**{k: v for k, v in seen[i].items() if k != "score"}, "score": fused_scores[i]}
        for i in sorted_ids
    ]



