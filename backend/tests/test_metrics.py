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


# (Thompson Sampling Bandit tests removed as the component was refactored out)


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
