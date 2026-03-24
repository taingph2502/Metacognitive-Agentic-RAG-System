from langchain_core.messages import HumanMessage

from app.llm import ainvoke
from app.text_utils import extract_json_object


MULTI_EVAL_PROMPT = """\
Given the answer and retrieved context below, score the following metrics from 0.0 to 1.0:
- faithfulness: fraction of statements supported by context
- answer_completeness: how fully the answer addresses the query
- confidence: confidence in the evaluation itself

Query:
{query}

Context:
{context}

Answer:
{answer}

Respond with JSON only:
{{"faithfulness": <float>, "answer_completeness": <float>, "confidence": <float>, "reasoning": "<brief explanation>"}}"""


async def evaluate_answer(query: str, answer: str, context: str) -> dict:
    """Multi-metric runtime evaluation with a single LLM call."""
    prompt = MULTI_EVAL_PROMPT.format(
        query=query[:800],
        context=context[:3000],
        answer=answer[:2200],
    )
    try:
        response = await ainvoke([HumanMessage(content=prompt)], call_site="evaluator")
        result = extract_json_object(response.content.strip())
        if result:
            return {
                "faithfulness": max(0.0, min(1.0, float(result.get("faithfulness", 0.5)))),
                "answer_completeness": max(0.0, min(1.0, float(result.get("answer_completeness", 0.5)))),
                "confidence": max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
                "reasoning": str(result.get("reasoning", "")),
            }
    except Exception:
        pass
    return {
        "faithfulness": 0.5,
        "answer_completeness": 0.5,
        "confidence": 0.4,
        "reasoning": "evaluation unavailable",
    }
