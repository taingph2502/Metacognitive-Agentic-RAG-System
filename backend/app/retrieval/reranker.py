from sentence_transformers import CrossEncoder

from app.config import settings

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def rerank(query: str, docs: list[dict], top_k: int | None = None) -> list[dict]:
    """Re-score docs using cross-encoder and return sorted by descending score."""
    if not docs:
        return docs

    reranker = get_reranker()
    pairs = [(query, d["text"]) for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    limit = top_k if top_k is not None else len(docs)
    return [{**doc, "score": float(score)} for doc, score in ranked[:limit]]
