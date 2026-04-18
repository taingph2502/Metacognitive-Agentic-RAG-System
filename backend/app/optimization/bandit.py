def compute_utility(
    faithfulness: float,
    cost: float,
    latency: float,
    lambda1: float = 0.3,
    lambda2: float = 0.2,
) -> float:
    """
    Utility = Faithfulness - λ₁ × Cost_norm - λ₂ × Latency_norm
    Cost_norm  = cost / 0.005
    Latency_norm = latency / 15.0
    """
    cost_norm = cost / 0.005
    latency_norm = latency / 15.0
    utility = faithfulness - lambda1 * cost_norm - lambda2 * latency_norm
    return round(max(0.0, min(1.0, utility)), 4)


def compute_reward(
    faithfulness: float,
    citation_precision: float,
    answer_completeness: float,
    latency: float,
    w1: float = 0.45,
    w2: float = 0.3,
    w3: float = 0.2,
    w4: float = 0.15,
) -> float:
    latency_penalty = min(1.0, latency / 15.0)
    score = (
        w1 * faithfulness
        + w2 * citation_precision
        + w3 * answer_completeness
        - w4 * latency_penalty
    )
    return round(max(0.0, min(1.0, score)), 4)


def estimate_cost(input_chars: int, output_chars: int) -> float:
    """
    Rough cost estimate using Gemini 2.5 Flash pricing.
    ~4 chars per token. Input: $0.075/1M, Output: $0.30/1M tokens.
    """
    input_tokens = input_chars / 4
    output_tokens = output_chars / 4
    return input_tokens * 0.075 / 1_000_000 + output_tokens * 0.30 / 1_000_000
