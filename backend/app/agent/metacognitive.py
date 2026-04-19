"""
Metacognitive evaluation module — inspired by MetaRAG paper §4.

Implements the three-phase metacognitive regulation pipeline:
  1. Monitoring  — fast-path check: is the answer good enough?
  2. Evaluating  — diagnose WHY the answer is weak (knowledge categories + error types)
  3. Planning    — prescribe targeted remediation (writing directives, re-retrieval)

Integrated into the existing LangGraph pipeline as diagnose_node / remediate_node.
"""

import logging
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

from app.config import settings
from app.llm import ainvoke
from app.text_utils import extract_json_object

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Knowledge categories (Paper §4.2 — Procedural Knowledge)
# ──────────────────────────────────────────────────────────────────────────────

SATISFACTORY = "satisfactory"
INSUFFICIENT = "insufficient_knowledge"       # neither source can answer
INTERNAL_ONLY = "internal_knowledge_only"     # LLM knows, docs are noisy
EXTERNAL_ONLY = "external_knowledge_only"     # docs have it, LLM doesn't
REASONING_ERROR = "reasoning_error"           # both sources adequate, logic flawed

KNOWLEDGE_CATEGORIES = (SATISFACTORY, INSUFFICIENT, INTERNAL_ONLY, EXTERNAL_ONLY, REASONING_ERROR)

# ──────────────────────────────────────────────────────────────────────────────
# Error types (Paper §4.2 — Declarative Knowledge)
# ──────────────────────────────────────────────────────────────────────────────

ERROR_INCOMPLETE = "incomplete_reasoning"
ERROR_REDUNDANCE = "answer_redundance"
ERROR_AMBIGUITY = "ambiguity_understanding"

ERROR_TYPES = (ERROR_INCOMPLETE, ERROR_REDUNDANCE, ERROR_AMBIGUITY)


@dataclass
class Diagnosis:
    """Result of metacognitive evaluation."""
    category: str = SATISFACTORY
    error_types: list[str] = field(default_factory=list)
    internal_sufficient: bool = True
    external_sufficient: bool = True
    reasoning: str = ""
    suggested_query: str | None = None   # for INSUFFICIENT — targeted sub-query
    suggestion: str | None = None        # corrective guidance for the writer


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — Monitoring: fast-path satisfaction check (Paper §4.1)
# ──────────────────────────────────────────────────────────────────────────────

def is_answer_satisfactory(
    faithfulness: float,
    completeness: float,
    citation_precision: float,
) -> bool:
    """
    Quick threshold gate — if all metrics are acceptable, skip the full
    metacognitive evaluation (mirrors the paper's monitoring phase where
    only uncertain answers trigger the evaluator-critic).
    """
    return (
        faithfulness >= settings.faithfulness_threshold
        and completeness >= settings.completeness_threshold
        and citation_precision >= settings.citation_precision_threshold
    )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — Evaluating: diagnose answer limitations (Paper §4.2)
# ──────────────────────────────────────────────────────────────────────────────

# Single prompt covering both procedural and declarative knowledge assessment.
# Paper §4.2: procedural = internal/external knowledge sufficiency,
#              declarative = common error patterns.
DIAGNOSIS_PROMPT = """\
You are an evaluator-critic system performing metacognitive analysis on a \
question-answering system's output.

## Your task
Analyze the generated answer and diagnose what went wrong (if anything).

### Step 1 — Procedural knowledge assessment (Paper §4.2)
Evaluate two knowledge sources independently:
- **Internal knowledge**: Could an LLM reliably answer this question from \
training knowledge alone, without any references? (yes/no)
- **External knowledge**: Do the retrieved documents contain sufficient, \
relevant information to correctly answer this question? (yes/no)

### Step 2 — Classify the situation into exactly one category:
- "satisfactory" — answer is adequate (faith >= {faith_thresh}, completeness >= {comp_thresh})
- "insufficient_knowledge" — neither internal nor external knowledge is sufficient
- "internal_knowledge_only" — LLM could answer from own knowledge but retrieved \
docs are insufficient or misleading
- "external_knowledge_only" — retrieved docs contain the answer but the model's \
own knowledge is wrong or absent on this topic
- "reasoning_error" — both knowledge sources are adequate but the answer has \
logical or structural problems

### Step 3 — Declarative knowledge: check for common error patterns
- "incomplete_reasoning" — failed to follow a complete chain of thought, \
missed relevant evidence fragments, or skipped reasoning steps
- "answer_redundance" — overly verbose or repetitious; similar points repeated \
instead of consolidated
- "ambiguity_understanding" — misunderstood the query's intent or nuances, \
answered a related but different question

### Step 4 — If category is "insufficient_knowledge", generate a targeted \
follow-up search query that would retrieve the missing information.

### Step 5 — Provide a brief corrective suggestion for the answer generator.

## Input

Question: {query}

Retrieved documents:
{context}

Generated answer:
{answer}

Evaluation scores:
- Faithfulness: {faithfulness}
- Answer completeness: {completeness}
- Citation precision: {citation_precision}

## Response format — JSON only:
{{
  "internal_sufficient": true/false,
  "external_sufficient": true/false,
  "category": "<one of the five categories>",
  "error_types": ["<error_type>", ...],
  "reasoning": "<1-2 sentence explanation>",
  "suggested_query": "<follow-up search query or null>",
  "suggestion": "<corrective suggestion for rewriting, or null>"
}}"""




async def diagnose_answer(
    query: str,
    answer: str,
    docs: list[dict],
    faithfulness: float,
    completeness: float,
    citation_precision: float,
    evaluator_confidence: float = 0.0,
) -> Diagnosis:
    """
    Full metacognitive diagnosis — Paper §4.1 + §4.2.

    Runs the monitoring gate first; if the answer passes, returns SATISFACTORY
    without an LLM call.  Otherwise invokes the evaluator-critic LLM for
    procedural + declarative knowledge assessment.
    """
    # Phase 1 — Monitoring fast path (original threshold check)
    if is_answer_satisfactory(faithfulness, completeness, citation_precision):
        return Diagnosis(
            category=SATISFACTORY,
            reasoning="Answer meets quality thresholds — no metacognitive intervention needed.",
        )

    # Phase 1b — Confidence-based fast path (Phase 3 optimization)
    # If the evaluator is highly confident AND metrics are reasonably close to
    # thresholds, skip the expensive LLM diagnosis call.
    fast_thresh = settings.fast_path_threshold
    if fast_thresh is not None and evaluator_confidence >= fast_thresh:
        # Only fast-path if metrics are borderline (within 0.15 of thresholds)
        if faithfulness >= (settings.faithfulness_threshold - 0.15) and completeness >= (settings.completeness_threshold - 0.15):
            return Diagnosis(
                category=SATISFACTORY,
                reasoning=f"Fast-path: evaluator confidence {evaluator_confidence:.2f} >= {fast_thresh} with near-threshold metrics.",
            )

    # Phase 2 — Full evaluator-critic diagnosis
    context = "\n\n".join(
        f"[{i + 1}] ({d.get('source', 'unknown')})\n{d.get('text', '')}"
        for i, d in enumerate(docs[:10])
    )

    prompt = DIAGNOSIS_PROMPT.format(
        query=query[:800],
        context=context[:3000],
        answer=answer[:2200],
        faithfulness=round(faithfulness, 3),
        completeness=round(completeness, 3),
        citation_precision=round(citation_precision, 3),
        faith_thresh=settings.faithfulness_threshold,
        comp_thresh=settings.completeness_threshold,
    )

    try:
        response = await ainvoke([HumanMessage(content=prompt)], call_site="diagnose")
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
    except Exception as e:
        logger.warning(f"Failed to parse metacognitive diagnosis LLM output: {e}", exc_info=True)

    # Heuristic fallback when LLM diagnosis fails
    return _heuristic_diagnosis(query, faithfulness, completeness, citation_precision)


def _heuristic_diagnosis(
    query: str,
    faithfulness: float,
    completeness: float,
    citation_precision: float,
) -> Diagnosis:
    """Rule-based fallback when the LLM evaluator-critic is unavailable."""
    if faithfulness < 0.35 and completeness < 0.35:
        return Diagnosis(
            category=INSUFFICIENT,
            internal_sufficient=False,
            external_sufficient=False,
            reasoning="Very low faithfulness and completeness suggest knowledge gap.",
            suggested_query=f"{query} detailed explanation evidence",
        )
    if faithfulness < 0.4 and citation_precision < 0.3:
        return Diagnosis(
            category=EXTERNAL_ONLY,
            internal_sufficient=False,
            external_sufficient=True,
            reasoning="Low faithfulness with low citation precision suggests model hallucination.",
            suggestion="Rely strictly on the provided references for every claim.",
        )
    if completeness < 0.4 and faithfulness >= 0.5:
        return Diagnosis(
            category=REASONING_ERROR,
            error_types=[ERROR_INCOMPLETE],
            reasoning="Decent faithfulness but low completeness — incomplete reasoning.",
            suggestion="Follow a complete chain of reasoning using all relevant evidence.",
        )
    return Diagnosis(
        category=REASONING_ERROR,
        error_types=[ERROR_INCOMPLETE],
        reasoning="Metrics below threshold; defaulting to reasoning error.",
        suggestion="Think step by step. Ensure each claim is supported by evidence.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — Planning: build writing directives (Paper §4.3)
# ──────────────────────────────────────────────────────────────────────────────

def build_writing_directive(diagnosis: Diagnosis) -> str | None:
    """
    Convert a diagnosis into a concrete instruction that shapes how the
    writer re-generates the answer.  Returns None if no remediation needed.

    Maps to the four planning strategies in Paper §4.3:
      - Insufficient  → handled via re-retrieval, not writer directive
      - Internal only  → discard external references
      - External only  → rely strictly on references
      - Reasoning error → error-specific corrective suggestions
    """
    if diagnosis.category == SATISFACTORY:
        return None

    if diagnosis.category == INSUFFICIENT:
        # Re-retrieval handles this; give the writer a general nudge
        return (
            "Additional evidence has been retrieved to fill knowledge gaps. "
            "Use ALL available evidence to provide a comprehensive answer. "
            "Cite every source you use."
        )

    if diagnosis.category == INTERNAL_ONLY:
        # Paper §4.3: discard external references, rely on intrinsic knowledge
        return (
            "IMPORTANT DIRECTIVE: The retrieved documents appear to contain "
            "misleading or insufficient information for this question. "
            "Rely primarily on your own knowledge to answer accurately. "
            "Only cite a reference if it clearly and directly supports a claim."
        )

    if diagnosis.category == EXTERNAL_ONLY:
        # Paper §4.3: force reliance on provided references
        return (
            "IMPORTANT DIRECTIVE: For this question, answer ONLY using "
            "information explicitly stated in the provided source documents. "
            "Do NOT rely on your own knowledge — it may be inaccurate here. "
            "Every factual claim MUST have an inline citation [N]."
        )

    # REASONING_ERROR — Paper §4.3: error-specific suggestions
    parts: list[str] = []

    if ERROR_INCOMPLETE in diagnosis.error_types:
        parts.append(
            "Your previous answer had incomplete reasoning. "
            "Follow a COMPLETE chain of thought: identify all relevant evidence "
            "fragments, connect them logically, and ensure no reasoning step is skipped."
        )
    if ERROR_REDUNDANCE in diagnosis.error_types:
        parts.append(
            "Your previous answer was redundant. "
            "Consolidate similar information into single, distinct points. "
            "Each sentence should contribute new information."
        )
    if ERROR_AMBIGUITY in diagnosis.error_types:
        parts.append(
            "Your previous answer may have misunderstood the question. "
            "Re-read the query carefully and address exactly what is asked. "
            "If the query is ambiguous, state your interpretation explicitly."
        )

    # Append any custom suggestion from the evaluator-critic
    if diagnosis.suggestion:
        parts.append(diagnosis.suggestion)

    # Default fallback (Paper §4.3: "Please think step by step")
    if not parts:
        parts.append(
            "Think step by step. Ensure each statement is grounded in evidence "
            "and follows logically from the previous one."
        )

    return "IMPROVEMENT GUIDANCE: " + " ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Convergence detection (Paper §6.5 — avoid over-thinking)
# ──────────────────────────────────────────────────────────────────────────────

def check_convergence(previous_answer: str, current_answer: str, threshold: float | None = None) -> bool:
    """
    Detect when consecutive answers are too similar to justify another
    metacognitive round.  Paper §6.5 shows performance degrades when the
    model can no longer extract useful improvements.
    """
    if not previous_answer:
        return False

    if threshold is None:
        threshold = settings.metacognitive_convergence_threshold

    prev_tokens = set(previous_answer.lower().split())
    curr_tokens = set(current_answer.lower().split())
    if not prev_tokens or not curr_tokens:
        return False

    intersection = prev_tokens & curr_tokens
    union = prev_tokens | curr_tokens
    jaccard = len(intersection) / max(1, len(union))
    return jaccard >= threshold
