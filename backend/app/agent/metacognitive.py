"""
Paper-like metacognitive regulation for Meta-Agent-RAG.

The runtime loop follows the paper's core shape:
  1. Monitoring  — compare the cognition answer with a reference answer.
  2. Evaluating  — classify knowledge condition and reasoning errors.
  3. Planning    — generate a follow-up query or writer directive.

This implementation deliberately avoids heavyweight NLI models. External
knowledge sufficiency is judged by the evaluator-critic LLM, while monitoring
uses the existing embedding model for answer/reference similarity.
"""

import logging
import math
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

from app.config import settings
from app.llm import ainvoke
from app.text_utils import extract_json_object

logger = logging.getLogger(__name__)


SATISFACTORY = "satisfactory"
INSUFFICIENT = "insufficient_knowledge"
INTERNAL_ONLY = "internal_knowledge_only"
EXTERNAL_ONLY = "external_knowledge_only"
REASONING_ERROR = "reasoning_error"

KNOWLEDGE_CATEGORIES = (
    SATISFACTORY,
    INSUFFICIENT,
    INTERNAL_ONLY,
    EXTERNAL_ONLY,
    REASONING_ERROR,
)

ERROR_INCOMPLETE = "incomplete_reasoning"
ERROR_REDUNDANCE = "answer_redundance"
ERROR_AMBIGUITY = "ambiguity_understanding"

ERROR_TYPES = (ERROR_INCOMPLETE, ERROR_REDUNDANCE, ERROR_AMBIGUITY)


@dataclass
class MonitorResult:
    reference_answer: str = ""
    score: float = 0.0


@dataclass
class Diagnosis:
    category: str = SATISFACTORY
    error_types: list[str] = field(default_factory=list)
    internal_sufficient: bool = True
    external_sufficient: bool = True
    reasoning: str = ""
    suggested_query: str | None = None
    suggestion: str | None = None


REFERENCE_ANSWER_PROMPT = """\
You are the expert monitoring model in a metacognitive RAG system.

Generate a concise reference answer to the question from the provided documents.
If the documents are insufficient, answer from your best judgement and state uncertainty briefly.

Question:
{query}

Documents:
{context}

Reference answer:"""


DIAGNOSIS_PROMPT = """\
You are an evaluator-critic system performing metacognitive analysis.

The answer failed the monitoring gate because its similarity to the reference
answer is below the threshold.

Classify the situation using the paper's metacognitive RAG categories:
- "satisfactory": the answer is acceptable despite the monitor score
- "insufficient_knowledge": neither internal nor external knowledge is sufficient
- "internal_knowledge_only": the model can answer from its own knowledge but documents are noisy or insufficient
- "external_knowledge_only": documents contain the answer but model knowledge is unreliable
- "reasoning_error": both knowledge sources are adequate but the answer has logical or structural problems

Check the paper's declarative error types:
- "incomplete_reasoning"
- "answer_redundance"
- "ambiguity_understanding"

If knowledge is insufficient, produce a targeted follow-up search query.

Question:
{query}

Retrieved documents:
{context}

Generated answer:
{answer}

Reference answer:
{reference_answer}

Monitor similarity score: {monitor_score}
Monitor threshold: {monitor_threshold}

Return JSON only:
{{
  "internal_sufficient": true/false,
  "external_sufficient": true/false,
  "category": "<one of the five categories>",
  "error_types": ["<error_type>", ...],
  "reasoning": "<1-2 sentence explanation>",
  "suggested_query": "<follow-up search query or null>",
  "suggestion": "<corrective suggestion or null>"
}}"""


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def is_answer_satisfactory(monitor_score: float) -> bool:
    return monitor_score >= settings.monitor_similarity_threshold


def _format_context(docs: list[dict], limit: int = 10) -> str:
    return "\n\n".join(
        f"[{i + 1}] ({d.get('source', 'unknown')})\n{d.get('text', '')}"
        for i, d in enumerate(docs[:limit])
    )


async def generate_reference_answer(query: str, docs: list[dict]) -> str:
    prompt = REFERENCE_ANSWER_PROMPT.format(
        query=query[:800],
        context=_format_context(docs)[:4000],
    )
    response = await ainvoke(
        [HumanMessage(content=prompt)],
        call_site="monitor_reference",
        temperature=0.0,
        tier="flash",
    )
    return response.content.strip()


def embed_texts(texts: list[str]) -> list[list[float]]:
    from app.retrieval.dense import embed

    return embed(texts)


async def monitor_answer(query: str, answer: str, docs: list[dict]) -> MonitorResult:
    reference_answer = await generate_reference_answer(query, docs)
    vectors = embed_texts([answer, reference_answer])
    score = cosine_similarity(vectors[0], vectors[1]) if len(vectors) == 2 else 0.0
    return MonitorResult(reference_answer=reference_answer, score=round(score, 4))


async def diagnose_answer(
    query: str,
    answer: str,
    docs: list[dict],
    monitor_score: float,
    reference_answer: str = "",
) -> Diagnosis:
    if is_answer_satisfactory(monitor_score):
        return Diagnosis(
            category=SATISFACTORY,
            reasoning="Answer/reference similarity passed the monitoring threshold.",
        )

    prompt = DIAGNOSIS_PROMPT.format(
        query=query[:800],
        context=_format_context(docs)[:3000],
        answer=answer[:2200],
        reference_answer=reference_answer[:1200],
        monitor_score=round(monitor_score, 4),
        monitor_threshold=settings.monitor_similarity_threshold,
    )

    try:
        response = await ainvoke([HumanMessage(content=prompt)], call_site="diagnose", tier="strong")
        result = extract_json_object(response.content.strip())
        if result:
            category = result.get("category", REASONING_ERROR)
            if category not in KNOWLEDGE_CATEGORIES:
                category = REASONING_ERROR
            error_types = [e for e in result.get("error_types", []) if e in ERROR_TYPES]
            return Diagnosis(
                category=category,
                error_types=error_types,
                internal_sufficient=bool(result.get("internal_sufficient", False)),
                external_sufficient=bool(result.get("external_sufficient", False)),
                reasoning=str(result.get("reasoning", "")),
                suggested_query=result.get("suggested_query"),
                suggestion=result.get("suggestion"),
            )
    except Exception as exc:
        logger.warning("Failed to parse metacognitive diagnosis: %s", exc, exc_info=True)

    return _heuristic_diagnosis(query, docs, monitor_score)


def _heuristic_diagnosis(query: str, docs: list[dict], monitor_score: float) -> Diagnosis:
    if not docs:
        return Diagnosis(
            category=INSUFFICIENT,
            internal_sufficient=False,
            external_sufficient=False,
            reasoning="No retrieved documents are available.",
            suggested_query=f"{query} supporting evidence",
        )
    if monitor_score < 0.2:
        return Diagnosis(
            category=INSUFFICIENT,
            internal_sufficient=False,
            external_sufficient=False,
            reasoning="Very low monitor similarity suggests missing or conflicting knowledge.",
            suggested_query=f"{query} detailed evidence",
        )
    return Diagnosis(
        category=REASONING_ERROR,
        error_types=[ERROR_INCOMPLETE],
        reasoning="Monitor similarity is below threshold; defaulting to incomplete reasoning.",
        suggestion="Think step by step and connect all relevant evidence before answering.",
    )


def answer_mode_for_diagnosis(diagnosis: Diagnosis) -> str:
    if diagnosis.category == INTERNAL_ONLY:
        return "internal_only"
    if diagnosis.category == REASONING_ERROR:
        return "reasoning_error"
    return "external_only"


def build_writing_directive(diagnosis: Diagnosis) -> str | None:
    if diagnosis.category == SATISFACTORY:
        return None
    if diagnosis.category == INSUFFICIENT:
        return (
            "Additional evidence may be needed. Use newly retrieved documents if available, "
            "and explicitly state remaining uncertainty if the evidence is still incomplete."
        )
    if diagnosis.category == INTERNAL_ONLY:
        return (
            "Retrieved documents appear noisy or insufficient. Use intrinsic knowledge, "
            "and cite provided documents only when they directly support the answer."
        )
    if diagnosis.category == EXTERNAL_ONLY:
        return (
            "Rely only on the provided source documents. Do not use unsupported intrinsic knowledge."
        )

    parts: list[str] = []
    if ERROR_INCOMPLETE in diagnosis.error_types:
        parts.append("Complete the reasoning chain using all relevant evidence.")
    if ERROR_REDUNDANCE in diagnosis.error_types:
        parts.append("Remove redundant statements and consolidate repeated points.")
    if ERROR_AMBIGUITY in diagnosis.error_types:
        parts.append("Re-read the question and answer exactly the requested relation.")
    if diagnosis.suggestion:
        parts.append(diagnosis.suggestion)
    if not parts:
        parts.append("Think step by step.")
    return " ".join(parts)


def check_convergence(previous_answer: str, current_answer: str, threshold: float | None = None) -> bool:
    if not previous_answer:
        return False
    threshold = settings.metacognitive_convergence_threshold if threshold is None else threshold
    prev_tokens = set(previous_answer.lower().split())
    curr_tokens = set(current_answer.lower().split())
    if not prev_tokens or not curr_tokens:
        return False
    return len(prev_tokens & curr_tokens) / max(1, len(prev_tokens | curr_tokens)) >= threshold
