"""
Simplified Meta-Agent-RAG graph.

Flow:
    retrieve -> write -> diagnose -> [remediate -> write]* -> END
"""

import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.agent.metacognitive import (
    INSUFFICIENT,
    SATISFACTORY,
    Diagnosis,
    answer_mode_for_diagnosis,
    build_writing_directive,
    check_convergence,
    diagnose_answer,
    monitor_answer,
)
from app.agent.writer import write_answer
from app.config import settings
from app.cost import estimate_cost
from app.retrieval.guardrails import filter_retrieved_docs
from app.retrieval.hybrid import hybrid_search


class AgentState(TypedDict):
    query: str
    document_ids: list[int] | None
    current_query: str
    all_docs: list[dict]
    hop: int
    answer: str
    reference_answer: str
    monitor_score: float
    cost: float
    latency: float
    start_time: float
    total_input_chars: int
    total_output_chars: int
    diagnosis: str
    diagnosis_reasoning: str
    error_types: list[str]
    internal_sufficient: bool
    external_sufficient: bool
    writing_directive: str | None
    answer_mode: str
    metacognitive_round: int
    previous_answer: str
    diagnosis_suggested_query: str | None
    diagnosis_suggestion: str | None


async def retrieve_node(state: AgentState) -> dict[str, Any]:
    new_docs = hybrid_search(
        state["current_query"],
        top_k=settings.retrieval_top_k,
        document_ids=state.get("document_ids"),
    )
    new_docs = filter_retrieved_docs(new_docs)
    existing_ids = {d["id"] for d in state.get("all_docs", [])}
    fresh = [d for d in new_docs if d["id"] not in existing_ids]
    return {
        "all_docs": state.get("all_docs", []) + fresh,
        "hop": state.get("hop", 0) + 1,
    }


async def write_node(state: AgentState) -> dict[str, Any]:
    answer = await write_answer(
        state["query"],
        state["all_docs"],
        writing_directive=state.get("writing_directive"),
        answer_mode=state.get("answer_mode", "external_only"),
        metacognitive_round=state.get("metacognitive_round", 0),
    )
    prompt_len = len(state["query"]) + sum(len(d.get("text", "")) for d in state["all_docs"])
    return {
        "answer": answer,
        "total_input_chars": state["total_input_chars"] + prompt_len,
        "total_output_chars": state["total_output_chars"] + len(answer),
    }


async def diagnose_node(state: AgentState) -> dict[str, Any]:
    monitor = await monitor_answer(state["query"], state["answer"], state["all_docs"])
    diag = await diagnose_answer(
        query=state["query"],
        answer=state["answer"],
        docs=state["all_docs"],
        monitor_score=monitor.score,
        reference_answer=monitor.reference_answer,
    )
    latency = time.monotonic() - state["start_time"]
    cost = estimate_cost(state["total_input_chars"], state["total_output_chars"])
    return {
        "reference_answer": monitor.reference_answer,
        "monitor_score": monitor.score,
        "diagnosis": diag.category,
        "diagnosis_reasoning": diag.reasoning,
        "error_types": diag.error_types,
        "internal_sufficient": diag.internal_sufficient,
        "external_sufficient": diag.external_sufficient,
        "diagnosis_suggested_query": diag.suggested_query,
        "diagnosis_suggestion": diag.suggestion,
        "cost": cost,
        "latency": round(latency, 2),
        "total_input_chars": state["total_input_chars"] + 500,
        "total_output_chars": state["total_output_chars"] + 200,
    }


async def remediate_node(state: AgentState) -> dict[str, Any]:
    diag = Diagnosis(
        category=state["diagnosis"],
        error_types=state.get("error_types", []),
        internal_sufficient=state.get("internal_sufficient", True),
        external_sufficient=state.get("external_sufficient", True),
        reasoning=state.get("diagnosis_reasoning", ""),
        suggested_query=state.get("diagnosis_suggested_query"),
        suggestion=state.get("diagnosis_suggestion"),
    )
    updates: dict[str, Any] = {
        "metacognitive_round": state["metacognitive_round"] + 1,
        "previous_answer": state["answer"],
        "writing_directive": build_writing_directive(diag),
        "answer_mode": answer_mode_for_diagnosis(diag),
    }

    if diag.category == INSUFFICIENT and diag.suggested_query:
        new_docs = hybrid_search(
            diag.suggested_query,
            top_k=settings.retrieval_top_k,
            document_ids=state.get("document_ids"),
        )
        new_docs = filter_retrieved_docs(new_docs)
        existing_ids = {d["id"] for d in state["all_docs"]}
        fresh = [d for d in new_docs if d["id"] not in existing_ids]
        if fresh:
            updates["all_docs"] = state["all_docs"] + fresh
            updates["hop"] = state.get("hop", 0) + 1
            updates["total_input_chars"] = state["total_input_chars"] + sum(
                len(d.get("text", "")) for d in fresh
            )

    return updates


def should_remediate(state: AgentState) -> str:
    if state.get("diagnosis") == SATISFACTORY:
        return END
    if state["metacognitive_round"] >= settings.max_metacognitive_rounds:
        return END
    if check_convergence(state.get("previous_answer", ""), state.get("answer", "")):
        return END
    return "remediate"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("write", write_node)
    g.add_node("diagnose", diagnose_node)
    g.add_node("remediate", remediate_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "write")
    g.add_edge("write", "diagnose")
    g.add_conditional_edges("diagnose", should_remediate, {"remediate": "remediate", END: END})
    g.add_edge("remediate", "write")
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def build_initial_state(query: str, document_ids: list[int] | None = None) -> AgentState:
    return {
        "query": query,
        "document_ids": document_ids,
        "current_query": query,
        "all_docs": [],
        "hop": 0,
        "answer": "",
        "reference_answer": "",
        "monitor_score": 0.0,
        "cost": 0.0,
        "latency": 0.0,
        "start_time": time.monotonic(),
        "total_input_chars": 0,
        "total_output_chars": 0,
        "diagnosis": SATISFACTORY,
        "diagnosis_reasoning": "",
        "error_types": [],
        "internal_sufficient": True,
        "external_sufficient": True,
        "writing_directive": None,
        "answer_mode": "external_only",
        "metacognitive_round": 0,
        "previous_answer": "",
        "diagnosis_suggested_query": None,
        "diagnosis_suggestion": None,
    }
