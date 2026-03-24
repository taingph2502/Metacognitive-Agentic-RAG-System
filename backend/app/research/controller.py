from dataclasses import dataclass


@dataclass
class ResearchSignals:
    evidence_coverage: float
    retrieval_diversity: float
    evaluator_confidence: float
    estimated_recall_proxy: float
    hop: int
    max_hops: int
    has_followup_query: bool


def decide_next_action(signals: ResearchSignals) -> str:
    """
    Returns one of: stop | extra_hop | reformulate | abstain
    """
    if signals.hop >= signals.max_hops:
        # All hops exhausted — abstain if evidence is very weak
        if signals.evidence_coverage < 0.15 and signals.estimated_recall_proxy < 0.2:
            return "abstain"
        return "stop"

    if signals.evidence_coverage < 0.45:
        # If reader already proposed followup, continue directly.
        if signals.has_followup_query:
            return "extra_hop"
        return "reformulate"

    if signals.retrieval_diversity < 0.25 and signals.hop < signals.max_hops:
        return "reformulate"

    if signals.evaluator_confidence < 0.4 and signals.hop < signals.max_hops:
        return "extra_hop"

    # Low recall proxy suggests sparse index is thin — more retrieval may help
    if signals.estimated_recall_proxy < 0.2 and signals.hop < signals.max_hops:
        return "reformulate"

    return "stop"
