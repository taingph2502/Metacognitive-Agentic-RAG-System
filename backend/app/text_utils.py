"""Shared text utilities used by the research runtime."""

import json
import re


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
