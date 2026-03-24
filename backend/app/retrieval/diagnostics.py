from collections import Counter


def _tokenize(text: str) -> set[str]:
    return {t for t in text.lower().split() if t}


def compute_retrieval_diagnostics(
    query: str,
    query_variants: list[str],
    docs: list[dict],
) -> dict[str, float]:
    """Compute cheap retrieval observability metrics for analysis/debugging."""
    if not docs:
        return {
            "query_coverage": 0.0,
            "document_diversity": 0.0,
            "retrieval_redundancy": 1.0,
            "estimated_recall_proxy": 0.0,
        }

    query_terms = _tokenize(query)
    query_hits = 0
    all_doc_terms: set[str] = set()
    sources = [d.get("source", "") for d in docs]

    for d in docs:
        terms = _tokenize(d.get("text", ""))
        all_doc_terms.update(terms)
        if query_terms and len(query_terms & terms) > 0:
            query_hits += 1

    query_coverage = query_hits / max(1, len(docs))

    source_counts = Counter(sources)
    unique_sources = len(source_counts)
    document_diversity = unique_sources / max(1, len(docs))

    duplicate_docs = sum(c - 1 for c in source_counts.values() if c > 1)
    retrieval_redundancy = duplicate_docs / max(1, len(docs))

    variant_terms = set()
    for q in query_variants:
        variant_terms.update(_tokenize(q))
    coverage_terms = variant_terms & all_doc_terms
    estimated_recall_proxy = len(coverage_terms) / max(1, len(variant_terms))

    return {
        "query_coverage": round(query_coverage, 4),
        "document_diversity": round(document_diversity, 4),
        "retrieval_redundancy": round(retrieval_redundancy, 4),
        "estimated_recall_proxy": round(estimated_recall_proxy, 4),
    }
