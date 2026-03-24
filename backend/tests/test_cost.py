"""Tests for the cost tracking module."""

import json
import tempfile
from pathlib import Path


def test_compute_call_cost():
    from app.cost import compute_call_cost

    # 1M tokens at each tier
    cost = compute_call_cost(1_000_000, 1_000_000, 1_000_000)
    expected = 0.028 + 0.28 + 0.42
    assert abs(cost - expected) < 1e-6


def test_compute_call_cost_zero():
    from app.cost import compute_call_cost

    assert compute_call_cost(0, 0, 0) == 0.0


def test_tracker_record_and_summary():
    from app.cost import CostTracker

    tracker = CostTracker()
    tracker.record("writer", cache_hit_tokens=500, cache_miss_tokens=100, completion_tokens=200)
    tracker.record("evaluator", cache_hit_tokens=300, cache_miss_tokens=50, completion_tokens=100)

    s = tracker.summary()
    assert s["total_calls"] == 2
    assert s["cache_hit_tokens"] == 800
    assert s["cache_miss_tokens"] == 150
    assert s["completion_tokens"] == 300
    assert s["total_cost_usd"] > 0


def test_tracker_flush_to_jsonl():
    from app.cost import CostTracker

    tracker = CostTracker()
    tracker.record("planner", cache_miss_tokens=100, completion_tokens=50)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cost_log.jsonl"
        tracker.flush_to_jsonl(path)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["call_site"] == "planner"
        assert entry["prompt_cache_miss_tokens"] == 100
        assert entry["completion_tokens"] == 50
        assert entry["cost_usd"] > 0


def test_tracker_reset():
    from app.cost import CostTracker

    tracker = CostTracker()
    tracker.record("test", completion_tokens=100)
    assert tracker.summary()["total_calls"] == 1
    tracker.reset()
    assert tracker.summary()["total_calls"] == 0


def test_llm_provider_config():
    from app.config import settings

    # Default provider should be deepseek
    assert settings.llm_provider == "deepseek"
    assert settings.deepseek_model == "deepseek-chat"
    assert settings.deepseek_base_url == "https://api.deepseek.com"
