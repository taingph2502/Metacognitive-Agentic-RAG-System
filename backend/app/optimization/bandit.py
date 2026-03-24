import numpy as np
from dataclasses import dataclass, field
from random import random

# Four retrieval configurations
CONFIGS: dict[str, dict] = {
    "A": {
        "top_k": 5,
        "chunk_size": 256,
        "rerank": False,
        "query_rewrite": False,
        "rewrite_count": 1,
        "hop_limit": 1,
    },
    "B": {
        "top_k": 10,
        "chunk_size": 512,
        "rerank": False,
        "query_rewrite": True,
        "rewrite_count": 3,
        "hop_limit": 1,
    },
    "C": {
        "top_k": 10,
        "chunk_size": 512,
        "rerank": True,
        "query_rewrite": True,
        "rewrite_count": 4,
        "hop_limit": 2,
    },
    "D": {
        "top_k": 8,
        "chunk_size": 512,
        "rerank": False,
        "query_rewrite": True,
        "rewrite_count": 5,
        "hop_limit": 2,
        "llm_rewrite": True,
    },
}

CONFIG_NAMES = list(CONFIGS.keys())


@dataclass
class ThompsonSamplingBandit:
    """Thompson Sampling bandit with Beta(1,1) priors over retrieval configs."""

    alpha: dict[str, float] = field(default_factory=lambda: {c: 1.0 for c in CONFIG_NAMES})
    beta: dict[str, float] = field(default_factory=lambda: {c: 1.0 for c in CONFIG_NAMES})

    epsilon: float = 0.1

    def _ensure_keys(self) -> None:
        for c in CONFIG_NAMES:
            self.alpha.setdefault(c, 1.0)
            self.beta.setdefault(c, 1.0)

    def select_config(self, exclude: str | None = None) -> str:
        self._ensure_keys()
        candidates = [c for c in CONFIG_NAMES if c != exclude]
        if candidates and random() < self.epsilon:
            return str(np.random.choice(candidates))
        samples = {
            c: float(np.random.beta(self.alpha[c], self.beta[c])) for c in candidates
        }
        return max(samples, key=samples.__getitem__)

    def update(self, config: str, utility: float, threshold: float = 0.7) -> None:
        self._ensure_keys()
        if utility >= threshold:
            self.alpha[config] += 1.0
        else:
            self.beta[config] += 1.0

    def stats(self) -> dict[str, dict]:
        result = {}
        for c in CONFIG_NAMES:
            a, b = self.alpha[c], self.beta[c]
            result[c] = {
                "alpha": a,
                "beta": b,
                "win_rate": round(a / (a + b), 4),
                "trials": int(a + b - 2),  # subtract Beta(1,1) prior
            }
        return result


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
