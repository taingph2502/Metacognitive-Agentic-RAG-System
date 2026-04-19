"""Evidence graph builder — tracks claim → source provenance chains."""

from dataclasses import dataclass, field

from app.text_utils import tokenize_for_overlap as _tokenize


@dataclass
class EvidenceLink:
    """A single claim with its supporting document indices."""
    claim: str
    supporting_doc_indices: list[int] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class EvidenceGraph:
    """Lightweight provenance graph: claims → supporting documents."""
    links: list[EvidenceLink] = field(default_factory=list)

    @property
    def total_claims(self) -> int:
        return len(self.links)

    @property
    def supported_claims(self) -> int:
        return sum(1 for link in self.links if link.supporting_doc_indices)

    @property
    def coverage_ratio(self) -> float:
        if not self.links:
            return 0.0
        return self.supported_claims / self.total_claims

    def to_dict(self) -> dict:
        return {
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "coverage_ratio": round(self.coverage_ratio, 4),
            "links": [
                {
                    "claim": link.claim,
                    "supporting_docs": link.supporting_doc_indices,
                    "confidence": round(link.confidence, 4),
                }
                for link in self.links
            ],
        }

def build_evidence_graph(claims: list[str], docs: list[dict], overlap_threshold: float = 0.3) -> EvidenceGraph:
    """
    Build a provenance graph linking each claim to supporting documents.
    Uses token-overlap heuristic consistent with the citation verifier.
    """
    doc_terms = [_tokenize(d.get("text", "")) for d in docs]
    links: list[EvidenceLink] = []

    for claim in claims:
        c_terms = _tokenize(claim)
        if not c_terms:
            links.append(EvidenceLink(claim=claim))
            continue

        supporting = []
        best_overlap = 0.0
        for idx, d_terms in enumerate(doc_terms):
            overlap = len(c_terms & d_terms) / max(1, len(c_terms))
            if overlap >= overlap_threshold:
                supporting.append(idx)
            best_overlap = max(best_overlap, overlap)

        links.append(
            EvidenceLink(
                claim=claim,
                supporting_doc_indices=supporting,
                confidence=best_overlap,
            )
        )

    return EvidenceGraph(links=links)
