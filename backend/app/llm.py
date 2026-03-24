"""
Centralized LLM provider factory with per-call cost tracking.

Supports multiple providers selectable via config:
  - "deepseek" (default) — OpenAI-compatible API at api.deepseek.com
  - "gemini" — Google Generative AI (Gemini 2.5 Flash)

All agent modules call ainvoke(messages, call_site=...) for LLM calls,
which handles provider selection, invocation, and cost logging.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.config import settings
from app.cost import cost_tracker

# Cache one instance per (provider, model, temperature) tuple to avoid
# re-creating clients on every call while still supporting different temps.
_cache: dict[tuple[str, str, float], BaseChatModel] = {}


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Return a chat LLM for the configured provider."""
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        return _get_gemini(temperature)
    if provider == "deepseek":
        return _get_deepseek(temperature)
    raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'deepseek' or 'gemini'.")


async def ainvoke(
    messages: list,
    *,
    call_site: str,
    temperature: float = 0.0,
) -> BaseMessage:
    """Invoke the LLM and log cost for DeepSeek and Gemini calls.

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
    """Extract token usage from the response and record cost."""
    provider = settings.llm_provider.lower()

    if provider == "deepseek":
        _log_deepseek_cost(response, call_site)
    elif provider == "gemini":
        _log_gemini_cost(response, call_site)


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


def _log_gemini_cost(response: BaseMessage, call_site: str) -> None:
    """Extract Gemini token usage (promptTokenCount, candidatesTokenCount) and record cost."""
    meta = getattr(response, "response_metadata", {}) or {}
    usage = meta.get("usage_metadata", {}) or {}

    prompt_tokens = int(usage.get("prompt_token_count", 0))
    completion = int(usage.get("candidates_token_count", 0))

    # Fallback: check alternative field names from LangChain's normalization
    if prompt_tokens == 0 and completion == 0:
        usage_alt = meta.get("token_usage", {}) or {}
        prompt_tokens = int(usage_alt.get("prompt_tokens", 0))
        completion = int(usage_alt.get("completion_tokens", 0))

    if prompt_tokens or completion:
        cost_tracker.record(
            call_site=call_site,
            cache_hit_tokens=0,
            cache_miss_tokens=prompt_tokens,  # stored in cache_miss slot (no cache split for Gemini)
            completion_tokens=completion,
            provider="gemini",
        )


def _get_gemini(temperature: float) -> BaseChatModel:
    key = ("gemini", settings.gemini_model, temperature)
    if key not in _cache:
        from langchain_google_genai import ChatGoogleGenerativeAI

        _cache[key] = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=temperature,
        )
    return _cache[key]


def _get_deepseek(temperature: float) -> BaseChatModel:
    key = ("deepseek", settings.deepseek_model, temperature)
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
