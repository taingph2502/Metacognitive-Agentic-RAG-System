"""Unit tests for the optimization layer (bandit + utility)."""

import pytest
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Utility function
# ──────────────────────────────────────────────────────────────────────────────

def test_compute_utility_perfect():
    from app.optimization.bandit import compute_utility

    # High faithfulness, zero cost/latency → close to 1.0
    u = compute_utility(1.0, 0.0, 0.0)
    assert u == 1.0


def test_compute_utility_penalizes_cost():
    from app.optimization.bandit import compute_utility

    u_cheap = compute_utility(0.8, 0.001, 5.0)
    u_expensive = compute_utility(0.8, 0.005, 5.0)
    assert u_cheap > u_expensive


def test_compute_utility_penalizes_latency():
    from app.optimization.bandit import compute_utility

    u_fast = compute_utility(0.8, 0.001, 3.0)
    u_slow = compute_utility(0.8, 0.001, 15.0)
    assert u_fast > u_slow


def test_compute_utility_clamped():
    from app.optimization.bandit import compute_utility

    # Should not go below 0
    u = compute_utility(0.0, 0.1, 100.0)
    assert u == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Thompson Sampling Bandit
# ──────────────────────────────────────────────────────────────────────────────

def test_bandit_initial_priors():
    from app.optimization.bandit import ThompsonSamplingBandit

    bandit = ThompsonSamplingBandit()
    for c in ["A", "B", "C"]:
        assert bandit.alpha[c] == 1.0
        assert bandit.beta[c] == 1.0


def test_bandit_update_success():
    from app.optimization.bandit import ThompsonSamplingBandit

    bandit = ThompsonSamplingBandit()
    bandit.update("A", utility=0.8)  # above threshold
    assert bandit.alpha["A"] == 2.0
    assert bandit.beta["A"] == 1.0


def test_bandit_update_failure():
    from app.optimization.bandit import ThompsonSamplingBandit

    bandit = ThompsonSamplingBandit()
    bandit.update("B", utility=0.3)  # below threshold
    assert bandit.alpha["B"] == 1.0
    assert bandit.beta["B"] == 2.0


def test_bandit_select_excludes():
    from app.optimization.bandit import ThompsonSamplingBandit

    np.random.seed(42)
    bandit = ThompsonSamplingBandit()
    for _ in range(20):
        selected = bandit.select_config(exclude="A")
        assert selected != "A"


def test_bandit_stats_win_rate():
    from app.optimization.bandit import ThompsonSamplingBandit

    bandit = ThompsonSamplingBandit()
    # 3 wins, 1 loss for config A
    for _ in range(3):
        bandit.update("A", 0.9)
    bandit.update("A", 0.1)

    stats = bandit.stats()
    # alpha=4, beta=2 → win_rate = 4/6 ≈ 0.667
    assert abs(stats["A"]["win_rate"] - 4 / 6) < 0.01
    assert stats["A"]["trials"] == 4


def test_bandit_converges_to_better_config():
    """After many updates, bandit should prefer config with higher win rate."""
    from app.optimization.bandit import ThompsonSamplingBandit

    np.random.seed(0)
    bandit = ThompsonSamplingBandit()

    # Config A always succeeds, C always fails
    for _ in range(20):
        bandit.update("A", 0.9)
        bandit.update("C", 0.2)

    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for _ in range(100):
        counts[bandit.select_config()] += 1

    # A should be selected most often
    assert counts["A"] > counts["C"]


# ──────────────────────────────────────────────────────────────────────────────
# Cost estimation
# ──────────────────────────────────────────────────────────────────────────────

def test_estimate_cost_positive():
    from app.optimization.bandit import estimate_cost

    cost = estimate_cost(1000, 200)
    assert cost > 0


def test_estimate_cost_scales_with_tokens():
    from app.optimization.bandit import estimate_cost

    c1 = estimate_cost(1000, 100)
    c2 = estimate_cost(10000, 1000)
    assert c2 > c1
