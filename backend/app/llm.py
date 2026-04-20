"""
Centralized LLM provider factory with per-call cost tracking.

Uses DeepSeek V3.2 exclusively via the OpenAI-compatible API at api.deepseek.com.
All agent modules call ainvoke(messages, call_site=...) for LLM calls,
which handles invocation and cost logging.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.config import settings
from app.cost import cost_tracker

# Cache one instance per (model, temperature) tuple to avoid
# re-creating clients on every call while still supporting different temps.
_cache: dict[tuple[str, float], BaseChatModel] = {}


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Return a chat LLM for DeepSeek."""
    return _get_deepseek(temperature)


async def ainvoke(
    messages: list,
    *,
    call_site: str,
    temperature: float = 0.0,
) -> BaseMessage:
    """Invoke DeepSeek and log cost.

    Args:
        messages: List of LangChain message objects.
        call_site: Identifier for which node/function triggered the call
                   (e.g. "planner", "writer", "evaluator").
        temperature: Sampling temperature.

    Returns:
        The LLM response message.
    """
    llm = get_llm(temperature=temperature)
    response = await llm.ainvoke(messages)
    _log_response_cost(response, call_site)
    return response


def _log_response_cost(response: BaseMessage, call_site: str) -> None:
    """Extract DeepSeek token usage and record cost."""
    _log_deepseek_cost(response, call_site)


def _log_deepseek_cost(response: BaseMessage, call_site: str) -> None:
    """Extract DeepSeek token usage (with cache fields) and record cost."""
    meta = getattr(response, "response_metadata", {}) or {}
    usage = meta.get("token_usage", {}) or {}

    cache_hit = int(usage.get("prompt_cache_hit_tokens", 0))
    cache_miss = int(usage.get("prompt_cache_miss_tokens", 0))
    completion = int(usage.get("completion_tokens", 0))

    # Fallback: if DeepSeek-specific fields missing, use prompt_tokens as cache_miss
    if cache_hit == 0 and cache_miss == 0:
        cache_miss = int(usage.get("prompt_tokens", 0))

    if cache_hit or cache_miss or completion:
        cost_tracker.record(
            call_site=call_site,
            cache_hit_tokens=cache_hit,
            cache_miss_tokens=cache_miss,
            completion_tokens=completion,
            provider="deepseek",
        )


def _get_deepseek(temperature: float) -> BaseChatModel:
    key = (settings.deepseek_model, temperature)
    if key not in _cache:
        from langchain_openai import ChatOpenAI

        _cache[key] = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=temperature,
        )
    return _cache[key]


def clear_cache() -> None:
    """Clear cached LLM instances (useful for tests)."""
    _cache.clear()
