from langchain_core.messages import HumanMessage

from app.llm import ainvoke
from app.text_utils import extract_json_object


READ_PROMPT = """\
You are a research reader. Given a query and retrieved documents, extract relevant evidence spans.

Query: {query}

Documents:
{context}

Respond with JSON only:
{{
  "evidence": ["<direct quote or paraphrase with citation [N]>", ...],
  "missing": "<key information still missing to answer the query, or null>",
  "followup_query": "<a targeted follow-up search query to retrieve missing info, or null>"
}}

Set "followup_query" only if important information is truly missing. Do not set it for minor details."""


async def read_documents(query: str, docs: list[dict]) -> dict:
    """
    Extract evidence spans and identify information gaps.
    Returns {"evidence": list[str], "missing": str|None, "followup_query": str|None}
    """
    context = "\n\n".join(
        [f"[{i + 1}] (Source: {d['source']})\n{d['text']}" for i, d in enumerate(docs)]
    )
    prompt = READ_PROMPT.format(query=query, context=context[:4000])
    try:
        response = await ainvoke([HumanMessage(content=prompt)], call_site="reader")
        result = extract_json_object(response.content.strip())
        if result:
            result.setdefault("evidence", [])
            result.setdefault("missing", None)
            result.setdefault("followup_query", None)
            return result
    except Exception:
        pass

    return {
        "evidence": [d["text"][:300] for d in docs[:3]],
        "missing": None,
        "followup_query": None,
    }
