"""
Centralized LLM provider factory with per-call cost tracking.

Supports two model tiers via the DeepSeek OpenAI-compatible API:
  - "flash" (default): DeepSeek V4 Flash — fast and cheap, for simple tasks.
  - "strong": DeepSeek V4 Pro — more capable, for complex reasoning tasks.

All agent modules call ainvoke(messages, call_site=..., tier=...) for LLM calls,
which handles invocation and cost logging.
"""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.config import settings
from app.cost import cost_tracker

ModelTier = Literal["flash", "strong"]

# Cache one instance per (model, temperature) tuple to avoid
# re-creating clients on every call while still supporting different temps.
_cache: dict[tuple[str, float], BaseChatModel] = {}


def _resolve_model_name(tier: ModelTier) -> str:
    """Map a tier label to the concrete model name from settings."""
    if tier == "strong":
        return settings.deepseek_model_strong
    return settings.deepseek_model


def get_llm(temperature: float = 0.0, tier: ModelTier = "flash") -> BaseChatModel:
    """Return a chat LLM for the specified tier."""
    return _get_deepseek(temperature, tier)


async def ainvoke(
    messages: list,
    *,
    call_site: str,
    temperature: float = 0.0,
    tier: ModelTier = "flash",
) -> BaseMessage:
    """Invoke a DeepSeek model and log cost.

    Args:
        messages: List of LangChain message objects.
        call_site: Identifier for which node/function triggered the call
                   (e.g. "writer", "diagnose", "monitor_reference").
        temperature: Sampling temperature.
        tier: Which model tier to use — "flash" (cheap/fast) or
              "strong" (expensive/capable).

    Returns:
        The LLM response message.
    """
    llm = get_llm(temperature=temperature, tier=tier)
    response = await llm.ainvoke(messages)
    _log_response_cost(response, call_site, tier)
    return response


def _log_response_cost(response: BaseMessage, call_site: str, tier: ModelTier) -> None:
    """Extract DeepSeek token usage and record cost."""
    _log_deepseek_cost(response, call_site, tier)


def _log_deepseek_cost(response: BaseMessage, call_site: str, tier: ModelTier) -> None:
    """Extract DeepSeek token usage (with cache fields) and record cost."""
    meta = getattr(response, "response_metadata", {}) or {}
    usage = meta.get("token_usage", {}) or {}

    cache_hit = int(usage.get("prompt_cache_hit_tokens", 0))
    cache_miss = int(usage.get("prompt_cache_miss_tokens", 0))
    completion = int(usage.get("completion_tokens", 0))

    # Fallback: if DeepSeek-specific fields missing, use prompt_tokens as cache_miss
    if cache_hit == 0 and cache_miss == 0:
        cache_miss = int(usage.get("prompt_tokens", 0))

    provider = f"deepseek-{tier}"
    if cache_hit or cache_miss or completion:
        cost_tracker.record(
            call_site=call_site,
            cache_hit_tokens=cache_hit,
            cache_miss_tokens=cache_miss,
            completion_tokens=completion,
            provider=provider,
        )


def _get_deepseek(temperature: float, tier: ModelTier) -> BaseChatModel:
    model_name = _resolve_model_name(tier)
    key = (model_name, temperature)
    if key not in _cache:
        from langchain_openai import ChatOpenAI

        _cache[key] = ChatOpenAI(
            model=model_name,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=temperature,
        )
    return _cache[key]
