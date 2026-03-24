import json
import re

from langchain_core.messages import HumanMessage

from app.llm import ainvoke


def _normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


REWRITE_PROMPT = """\
Generate {num} diverse search queries for the following research question.
Each query should focus on a different aspect or use different keywords to maximize retrieval recall.

Original query: {query}

Respond with a JSON array of strings only, e.g. ["query1", "query2", "query3"]"""


class QueryRewriter:
    """Generate alternative search queries for better recall."""

    def rewrite(
        self,
        query: str,
        query_type: str | None = None,
        num_rewrites: int = 4,
    ) -> list[str]:
        """Template-based rewriting (synchronous, no LLM cost)."""
        base = _normalize_query(query)
        if not base:
            return []

        candidates = [
            base,
            f"{base} key findings",
            f"{base} evidence",
            f"{base} limitations",
            f"{base} benchmark results",
        ]

        if query_type == "comparative":
            candidates.extend(
                [
                    f"{base} comparison",
                    f"{base} tradeoffs",
                ]
            )
        elif query_type == "multi_hop":
            candidates.extend(
                [
                    f"{base} causal factors",
                    f"{base} historical development",
                ]
            )

        uniq = _unique_keep_order(candidates)
        min_count = 3
        target = max(min_count, min(5, num_rewrites))
        return uniq[:target]

    async def rewrite_with_llm(
        self,
        query: str,
        num_rewrites: int = 4,
    ) -> list[str]:
        """LLM-based rewriting for semantically diverse queries."""
        base = _normalize_query(query)
        if not base:
            return []

        prompt = REWRITE_PROMPT.format(query=base, num=num_rewrites)
        try:
            response = await ainvoke(
                [HumanMessage(content=prompt)], call_site="query_rewriter", temperature=0.7,
            )
            text = response.content.strip()
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                variants = json.loads(match.group())
                if isinstance(variants, list) and variants:
                    # Always include original query first
                    result = _unique_keep_order([base] + [str(v) for v in variants])
                    return result[: num_rewrites + 1]
        except Exception:
            pass
        # Fallback to template-based
        return self.rewrite(query, num_rewrites=num_rewrites)
