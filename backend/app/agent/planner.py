from langchain_core.messages import HumanMessage

from app.llm import ainvoke
from app.text_utils import extract_json_object


def classify_rule_based(query: str) -> str:
    """Lightweight rule-based fallback (as per spec: no KMeans at runtime)."""
    q = query.lower()
    if any(w in q for w in ["compare", "difference", "vs", "versus", "contrast", "better than", "which is"]):
        return "comparative"
    if any(w in q for w in ["why", "how does", "explain", "what caused", "what led", "relationship between", "connection between"]):
        return "multi_hop"
    return "factual"


PLAN_PROMPT = """\
Classify the following research query into one of three types:
- "factual": simple fact lookup, single-document answer
- "comparative": comparing multiple things, methods, or papers
- "multi_hop": requires reasoning across multiple documents or steps

Query: {query}

Respond with JSON only:
{{"query_type": "factual|comparative|multi_hop", "max_hops": 1}}

Use max_hops=2 only for multi_hop queries. For all others use max_hops=1."""


async def plan_query(query: str) -> dict:
    """Returns {"query_type": str, "max_hops": int}"""
    prompt = PLAN_PROMPT.format(query=query)
    try:
        response = await ainvoke([HumanMessage(content=prompt)], call_site="planner")
        result = extract_json_object(response.content.strip())
        if result:
            if result.get("query_type") not in ("factual", "comparative", "multi_hop"):
                result["query_type"] = classify_rule_based(query)
            result["max_hops"] = 2 if result["query_type"] == "multi_hop" else 1
            return result
    except Exception:
        pass

    query_type = classify_rule_based(query)
    return {"query_type": query_type, "max_hops": 2 if query_type == "multi_hop" else 1}
