from app.text_utils import tokenize_for_overlap as _tokenize


def estimate_evidence_coverage(query: str, docs: list[dict]) -> float:
    """Estimate whether retrieved evidence sufficiently covers query terms."""
    if not docs:
        return 0.0

    q_terms = _tokenize(query)
    if not q_terms:
        return 0.0

    covered: set[str] = set()
    for d in docs:
        covered.update(q_terms & _tokenize(d.get("text", "")))

    return round(len(covered) / max(1, len(q_terms)), 4)
