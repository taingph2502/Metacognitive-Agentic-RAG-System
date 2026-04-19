"""Unit tests for the optimization layer (bandit + utility)."""

import pytest
import numpy as np





# ──────────────────────────────────────────────────────────────────────────────
# Cost estimation
# ──────────────────────────────────────────────────────────────────────────────

def test_estimate_cost_positive():
    from app.cost import estimate_cost

    cost = estimate_cost(1000, 200)
    assert cost > 0


def test_estimate_cost_scales_with_tokens():
    from app.cost import estimate_cost

    c1 = estimate_cost(1000, 100)
    c2 = estimate_cost(10000, 1000)
    assert c2 > c1
