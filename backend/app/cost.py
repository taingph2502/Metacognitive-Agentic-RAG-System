"""
Per-call cost logging for LLM API usage (DeepSeek V3 + Gemini 2.5 Flash).

Tracks token usage and calculates cost per call using each provider's pricing:

DeepSeek V3:
  - Cache hit:  $0.028  / 1M tokens
  - Cache miss: $0.28   / 1M tokens
  - Completion: $0.42   / 1M tokens

Gemini 2.5 Flash:
  - Prompt:     $0.30   / 1M tokens
  - Completion: $2.50   / 1M tokens

Cost data is accumulated in-memory per session and can be flushed to
a JSONL file for benchmark cost reporting.
"""

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# DeepSeek V3 pricing (USD per 1M tokens)
PRICE_CACHE_HIT = 0.028
PRICE_CACHE_MISS = 0.28
PRICE_COMPLETION = 0.42

# Gemini 2.5 Flash pricing (USD per 1M tokens)
PRICE_GEMINI_PROMPT = 0.30
PRICE_GEMINI_COMPLETION = 2.50


@dataclass
class CallCostEntry:
    timestamp: str
    call_site: str
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    provider: str = ""

    def to_dict(self) -> dict:
        d = {
            "timestamp": self.timestamp,
            "call_site": self.call_site,
            "prompt_cache_hit_tokens": self.prompt_cache_hit_tokens,
            "prompt_cache_miss_tokens": self.prompt_cache_miss_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
        }
        if self.provider:
            d["provider"] = self.provider
        return d


def compute_call_cost(
    cache_hit_tokens: int,
    cache_miss_tokens: int,
    completion_tokens: int,
) -> float:
    """Calculate USD cost for a single DeepSeek API call."""
    return (
        cache_hit_tokens * PRICE_CACHE_HIT
        + cache_miss_tokens * PRICE_CACHE_MISS
        + completion_tokens * PRICE_COMPLETION
    ) / 1_000_000


def compute_gemini_cost(
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Calculate USD cost for a single Gemini API call."""
    return (
        prompt_tokens * PRICE_GEMINI_PROMPT
        + completion_tokens * PRICE_GEMINI_COMPLETION
    ) / 1_000_000


def estimate_cost(input_chars: int, output_chars: int) -> float:
    """
    Rough cost estimate using Gemini 2.5 Flash pricing.
    ~4 chars per token. Input: $0.075/1M, Output: $0.30/1M tokens.
    """
    input_tokens = input_chars / 4
    output_tokens = output_chars / 4
    return input_tokens * 0.075 / 1_000_000 + output_tokens * 0.30 / 1_000_000


class CostTracker:
    """Thread-safe accumulator for per-call cost entries within a session."""

    def __init__(self) -> None:
        self._entries: list[CallCostEntry] = []
        self._lock = threading.Lock()

    def record(
        self,
        call_site: str,
        cache_hit_tokens: int = 0,
        cache_miss_tokens: int = 0,
        completion_tokens: int = 0,
        provider: str = "",
    ) -> CallCostEntry:
        if provider == "gemini":
            # For Gemini: cache_miss_tokens holds prompt tokens (no cache split)
            cost = compute_gemini_cost(cache_miss_tokens, completion_tokens)
        else:
            cost = compute_call_cost(cache_hit_tokens, cache_miss_tokens, completion_tokens)
        entry = CallCostEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            call_site=call_site,
            prompt_cache_hit_tokens=cache_hit_tokens,
            prompt_cache_miss_tokens=cache_miss_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            provider=provider,
        )
        with self._lock:
            self._entries.append(entry)
        return entry

    @property
    def entries(self) -> list[CallCostEntry]:
        with self._lock:
            return list(self._entries)

    def summary(self) -> dict:
        """Aggregate cost summary for the current session."""
        with self._lock:
            entries = list(self._entries)

        total_cache_hit = sum(e.prompt_cache_hit_tokens for e in entries)
        total_cache_miss = sum(e.prompt_cache_miss_tokens for e in entries)
        total_completion = sum(e.completion_tokens for e in entries)
        total_cost = sum(e.cost_usd for e in entries)

        cost_cache_hit = total_cache_hit * PRICE_CACHE_HIT / 1_000_000
        cost_cache_miss = total_cache_miss * PRICE_CACHE_MISS / 1_000_000
        cost_completion = total_completion * PRICE_COMPLETION / 1_000_000

        return {
            "total_calls": len(entries),
            "total_tokens": total_cache_hit + total_cache_miss + total_completion,
            "cache_hit_tokens": total_cache_hit,
            "cache_miss_tokens": total_cache_miss,
            "completion_tokens": total_completion,
            "cost_cache_hit_usd": round(cost_cache_hit, 6),
            "cost_cache_miss_usd": round(cost_cache_miss, 6),
            "cost_completion_usd": round(cost_completion, 6),
            "total_cost_usd": round(total_cost, 6),
        }

    def flush_to_jsonl(self, path: str | Path) -> None:
        """Append all entries to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            entries = list(self._entries)
        with open(path, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry.to_dict()) + "\n")

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()

    def print_summary(self) -> None:
        s = self.summary()
        lines = [
            f"\n{'=' * 40}",
            "Cost Summary",
            f"{'=' * 40}",
            f"  Total calls      : {s['total_calls']}",
            f"  Total tokens     : {s['total_tokens']}",
            f"    - cache hits   : {s['cache_hit_tokens']}  (${s['cost_cache_hit_usd']:.4f})",
            f"    - cache misses : {s['cache_miss_tokens']}  (${s['cost_cache_miss_usd']:.4f})",
            f"    - completion   : {s['completion_tokens']}  (${s['cost_completion_usd']:.4f})",
            f"  Total cost (calc): ${s['total_cost_usd']:.4f}",
            f"{'=' * 40}\n",
        ]
        logger.info("\n".join(lines))


async def fetch_deepseek_balance(api_key: str) -> dict | None:
    """Fetch account balance from DeepSeek API.

    Returns dict with currency, total_balance, topped_up_balance,
    or None on failure. Wrapped in try/except to never crash callers.
    """
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.deepseek.com/user/balance",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            infos = data.get("balance_infos", [])
            if infos:
                info = infos[0]
                return {
                    "currency": info.get("currency", "USD"),
                    "total_balance": float(info.get("total_balance", 0)),
                    "topped_up_balance": float(info.get("topped_up_balance", 0)),
                }
    except Exception as e:
        logger.warning(f"Failed to fetch DeepSeek balance: {e}", exc_info=True)
    return None


# Global tracker instance for the current session
cost_tracker = CostTracker()
