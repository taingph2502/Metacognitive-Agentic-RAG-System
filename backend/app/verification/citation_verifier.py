"""
Citation verification — checks whether generated claims are supported by
retrieved documents.

Two modes:
  - Token overlap (fast, no model loading) — default for streaming/real-time
  - Cross-encoder (model-based, more accurate) — for benchmarks and evaluation

The cross-encoder approach is inspired by Paper §4.3 which uses NLI to
double-check each statement against references.  We reuse the existing
reranker cross-encoder as a lightweight entailment proxy.
"""

from app.text_utils import tokenize_for_overlap as _tokenize
from app.verification.claim_extractor import extract_claims


def verify_citations(
    answer: str,
    docs: list[dict],
    use_cross_encoder: bool = False,
) -> dict:
    """
    Claim-level support verification.

    Args:
        answer: Generated answer text.
        docs: Retrieved documents with 'text' field.
        use_cross_encoder: If True, use the reranker cross-encoder for
            more accurate entailment scoring (Paper §4.3).

    Returns keys:
    - citation_precision
    - unsupported_claim_rate
    - evidence_alignment_score
    - supported_claims
    - total_claims
    """
    claims = extract_claims(answer)
    if not claims:
        return {
            "citation_precision": 0.0,
            "unsupported_claim_rate": 1.0,
            "evidence_alignment_score": 0.0,
            "supported_claims": 0,
            "total_claims": 0,
        }

    if use_cross_encoder:
        return _verify_cross_encoder(claims, docs)
    return _verify_token_overlap(claims, docs)


def _verify_token_overlap(claims: list[str], docs: list[dict]) -> dict:
    """Original token-overlap heuristic (fast, no model)."""
    doc_terms = [_tokenize(d.get("text", "")) for d in docs]
    supported = 0

    for claim in claims:
        c_terms = _tokenize(claim)
        if not c_terms:
            continue
        matched = False
        for d_terms in doc_terms:
            overlap = len(c_terms & d_terms) / max(1, len(c_terms))
            if overlap >= 0.3:
                matched = True
                break
        if matched:
            supported += 1

    total = len(claims)
    precision = supported / max(1, total)
    unsupported_rate = 1.0 - precision
    return {
        "citation_precision": round(precision, 4),
        "unsupported_claim_rate": round(unsupported_rate, 4),
        "evidence_alignment_score": round(precision, 4),
        "supported_claims": supported,
        "total_claims": total,
    }


def _verify_cross_encoder(
    claims: list[str],
    docs: list[dict],
    support_threshold: float = 0.5,
) -> dict:
    """
    Cross-encoder based verification — Paper §4.3 NLI double-check.

    Uses the reranker cross-encoder to score (claim, document) pairs.
    A claim is "supported" if any document scores above the threshold.
    """
    from app.retrieval.reranker import get_reranker

    reranker = get_reranker()
    doc_texts = [d.get("text", "") for d in docs]
    if not doc_texts:
        return {
            "citation_precision": 0.0,
            "unsupported_claim_rate": 1.0,
            "evidence_alignment_score": 0.0,
            "supported_claims": 0,
            "total_claims": len(claims),
        }

    supported = 0
    total_alignment = 0.0

    for claim in claims:
        if len(claim.strip()) < 10:
            continue
        pairs = [(claim, dt) for dt in doc_texts]
        scores = reranker.predict(pairs)
        max_score = float(max(scores)) if len(scores) > 0 else 0.0
        # Normalize cross-encoder score to [0, 1] via sigmoid-like mapping
        norm_score = 1.0 / (1.0 + 2.718 ** (-max_score))
        total_alignment += norm_score
        if norm_score >= support_threshold:
            supported += 1

    total = len(claims)
    precision = supported / max(1, total)
    avg_alignment = total_alignment / max(1, total)
    return {
        "citation_precision": round(precision, 4),
        "unsupported_claim_rate": round(1.0 - precision, 4),
        "evidence_alignment_score": round(avg_alignment, 4),
        "supported_claims": supported,
        "total_claims": total,
    }
