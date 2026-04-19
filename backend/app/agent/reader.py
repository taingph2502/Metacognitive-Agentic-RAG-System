import logging

from langchain_core.messages import HumanMessage

from app.llm import ainvoke
from app.text_utils import extract_json_object

logger = logging.getLogger(__name__)


READ_PROMPT = """\
You are a research reader. Given a query and retrieved documents, extract relevant evidence spans.

Query: {query}

Documents:
{context}

Respond with JSON only:
{{
  "evidence": ["<direct quote or paraphrase with citation [N]>", ...],
  "missing": "<key information still missing to answer the query, or null>"
}}"""


async def read_documents(query: str, docs: list[dict]) -> dict:
    """
    Extract evidence spans and identify information gaps.
    Returns {"evidence": list[str], "missing": str|None}
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
            return result
    except Exception as e:
        logger.warning(f"Failed to parse reader LLM output: {e}", exc_info=True)

    return {
        "evidence": [d["text"][:300] for d in docs[:3]],
        "missing": None,
    }
