"""Retrieval guardrails — filter retrieved documents before they reach the LLM."""

import re

# Patterns that suggest prompt injection or adversarial content in retrieved docs
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|above)\s+(?:instructions?|prompts?)", re.IGNORECASE),
    re.compile(r"(system|user)\s*:\s*", re.IGNORECASE),
    re.compile(r"<\s*/?\s*(system|instruction|prompt)\s*>", re.IGNORECASE),
    re.compile(r"you\s+are\s+(now|a)\s+(evil|hacker|malicious)", re.IGNORECASE),
    re.compile(r"disregard\s+(everything|all|the)\s+(above|previous)", re.IGNORECASE),
]

# Minimum useful text length — very short chunks rarely contain real evidence
_MIN_CHUNK_LENGTH = 20


def filter_retrieved_docs(docs: list[dict]) -> list[dict]:
    """
    Remove documents that fail safety or quality checks.

    Filters:
    1. Prompt injection patterns
    2. Extremely short chunks with no useful content
    """
    safe = []
    for doc in docs:
        text = doc.get("text", "")
        if len(text.strip()) < _MIN_CHUNK_LENGTH:
            continue
        if _contains_injection(text):
            continue
        safe.append(doc)
    return safe


def _contains_injection(text: str) -> bool:
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False
