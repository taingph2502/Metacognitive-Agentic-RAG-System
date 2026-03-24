import re


def extract_claims(answer: str) -> list[str]:
    """Split generated answer into coarse atomic claims."""
    text = re.sub(r"\[[\d\s,]+\]", "", answer)
    # Split by sentence punctuation and markdown list boundaries.
    chunks = re.split(r"[\n\r]+|(?<=[.!?])\s+", text)
    claims = []
    for c in chunks:
        claim = c.strip(" -*\t")
        if len(claim) < 20:
            continue
        claims.append(claim)
    return claims[:40]
