"""Shared text utilities used across retrieval, verification, and research modules."""

import json
import re


def tokenize_for_overlap(text: str) -> set[str]:
    """Tokenize text into a set of lowercase words (length > 2) for overlap comparison."""
    return {t for t in text.lower().split() if len(t) > 2}


def extract_json_object(text: str) -> dict | None:
    """Extract the first JSON object from LLM response text.

    Returns the parsed dict, or None if no valid JSON object is found.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            return None
    return None
