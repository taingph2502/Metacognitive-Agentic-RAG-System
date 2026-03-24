"""
Evaluation metrics following the paper's experimental setup (§5.1).

Metrics:
  - Exact Match (EM): binary, prediction matches reference after normalization
  - Token-level F1: harmonic mean of token precision and recall
  - Token Precision: fraction of prediction tokens found in reference
  - Token Recall: fraction of reference tokens found in prediction

These match the standard QA evaluation used in HotpotQA and 2WikiMultiHopQA.
"""

import re
import string
from collections import Counter


def _normalize_answer(text: str) -> str:
    """Normalize answer text for fair comparison (standard QA normalization)."""
    # Lowercase
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def _get_tokens(text: str) -> list[str]:
    return _normalize_answer(text).split()


def exact_match(prediction: str, reference: str) -> float:
    """Binary exact match after normalization."""
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(reference) else 0.0


def token_f1(prediction: str, reference: str) -> dict[str, float]:
    """
    Compute token-level F1, precision, and recall.

    Returns:
        {"f1": float, "precision": float, "recall": float}
    """
    pred_tokens = Counter(_get_tokens(prediction))
    ref_tokens = Counter(_get_tokens(reference))

    # Count common tokens
    common = pred_tokens & ref_tokens
    num_common = sum(common.values())

    if num_common == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    precision = num_common / max(1, sum(pred_tokens.values()))
    recall = num_common / max(1, sum(ref_tokens.values()))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    return {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def compute_metrics(prediction: str, reference: str) -> dict[str, float]:
    """Compute all metrics for a single prediction-reference pair."""
    em = exact_match(prediction, reference)
    f1_scores = token_f1(prediction, reference)
    return {
        "exact_match": em,
        "f1": f1_scores["f1"],
        "precision": f1_scores["precision"],
        "recall": f1_scores["recall"],
    }


def aggregate_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    """Average metrics across all examples."""
    if not results:
        return {"exact_match": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    n = len(results)
    return {
        "exact_match": round(sum(r["exact_match"] for r in results) / n, 4),
        "f1": round(sum(r["f1"] for r in results) / n, 4),
        "precision": round(sum(r["precision"] for r in results) / n, 4),
        "recall": round(sum(r["recall"] for r in results) / n, 4),
        "num_examples": n,
    }
