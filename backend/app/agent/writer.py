from langchain_core.messages import HumanMessage

from app.llm import ainvoke


WRITE_PROMPT = """\
You are a research writer. Write a comprehensive, grounded answer to the query using only the provided source documents.

Query: {query}

Evidence:
{evidence}

Source documents:
{context}
{directive_block}
Instructions:
- Write a clear, structured Markdown response
- ALWAYS cite sources inline using [1], [2], etc. notation whenever you use information from a document.
- Inline citations are MANDATORY for every factual claim.
- Be concise but thorough
- Do not add information not present in the documents
- IMPORTANT: DO NOT include a "References" or "Sources" section at the end. I will handle the list separately.

Answer:"""


async def write_answer(
    query: str,
    evidence: list[str],
    docs: list[dict],
    writing_directive: str | None = None,
) -> str:
    """Generate a grounded Markdown answer with inline citations.

    If a writing_directive is provided (from metacognitive remediation),
    it is injected into the prompt to shape the answer strategy.
    """
    context = "\n\n".join(
        [f"[{i + 1}] **{d['source']}**\n{d['text']}" for i, d in enumerate(docs)]
    )
    evidence_block = "\n".join([f"- {e}" for e in evidence]) if evidence else "(no evidence extracted)"
    directive_block = f"\n{writing_directive}\n" if writing_directive else ""
    prompt = WRITE_PROMPT.format(
        query=query,
        evidence=evidence_block,
        context=context[:5000],
        directive_block=directive_block,
    )
    response = await ainvoke(
        [HumanMessage(content=prompt)], call_site="writer", temperature=0.3,
    )
    return response.content.strip()
