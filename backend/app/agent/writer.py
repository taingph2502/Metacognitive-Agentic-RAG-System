from langchain_core.messages import HumanMessage

from app.llm import ainvoke


EXTERNAL_ONLY_INSTRUCTIONS = """\
Answer using only the provided source documents.
- Cite source documents inline using [1], [2], etc. whenever you use document information.
- Do not add information not present in the documents.
- If the documents are insufficient, say what is missing."""

INTERNAL_ONLY_INSTRUCTIONS = """\
The retrieved documents may be incomplete or misleading for this question.
- Answer from your own knowledge.
- Do not invent citations.
- Cite a provided document only if it directly supports a claim."""

REASONING_ERROR_INSTRUCTIONS = """\
Answer from the provided documents, but repair the reasoning.
- Connect the relevant evidence step by step.
- Avoid repeating the same point.
- Address exactly what the question asks.
- Cite source documents inline using [1], [2], etc. when document evidence is used."""


def _format_docs(docs: list[dict]) -> str:
    if not docs:
        return "(no source documents provided)"
    return "\n\n".join(
        f"[{i + 1}] **{d.get('source', 'unknown')}**\n{d.get('text', '')}"
        for i, d in enumerate(docs)
    )


def _mode_instructions(answer_mode: str) -> str:
    if answer_mode == "internal_only":
        return INTERNAL_ONLY_INSTRUCTIONS
    if answer_mode == "reasoning_error":
        return REASONING_ERROR_INSTRUCTIONS
    return EXTERNAL_ONLY_INSTRUCTIONS


def build_writer_prompt(
    query: str,
    docs: list[dict],
    writing_directive: str | None = None,
    answer_mode: str = "external_only",
) -> str:
    directive_block = f"\nAdditional planning directive:\n{writing_directive}\n" if writing_directive else ""
    return f"""\
You are the cognition component of a metacognitive RAG system.

Query:
{query}

Source documents:
{_format_docs(docs)[:5000]}

Answer mode:
{answer_mode}

Instructions:
{_mode_instructions(answer_mode)}
{directive_block}
Write a concise, direct Markdown answer.

Answer:"""


async def write_answer(
    query: str,
    docs: list[dict],
    *,
    writing_directive: str | None = None,
    answer_mode: str = "external_only",
    metacognitive_round: int = 0,
) -> str:
    # First draft: use the fast/cheap Flash model.
    # Remediation rounds (round > 0): upgrade to the strong Pro model
    # because the writer must follow complex corrective directives.
    tier = "strong" if metacognitive_round > 0 else "flash"
    prompt = build_writer_prompt(
        query=query,
        docs=docs,
        writing_directive=writing_directive,
        answer_mode=answer_mode,
    )
    response = await ainvoke(
        [HumanMessage(content=prompt)],
        call_site="writer",
        temperature=0.0,
        tier=tier,
    )
    return response.content.strip()

