"""
LangGraph agent orchestration for the Autonomous Research Agent.

Flow:
    plan → retrieve → read → controller → [hop?] → ...
                                                     ↓
    write → claim_extract → citation_verify → evidence_graph → evaluate → diagnose → [remediate?] → END
      ↑                                                                                   ↓
      └──────────────────────────────────────── remediate ←───────────────────────────────┘

The metacognitive loop (diagnose → remediate → write → ... → diagnose) implements
the paper's three-phase regulation pipeline: monitoring, evaluating, planning (§4).
"""

import time
import asyncio
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.agent.metacognitive import (
    INSUFFICIENT,
    SATISFACTORY,
    Diagnosis,
    build_writing_directive,
    check_convergence,
    diagnose_answer,
)
from app.agent.planner import plan_query
from app.agent.reader import read_documents
from app.agent.writer import write_answer
from app.config import settings
from app.cost import estimate_cost
from app.optimization.evaluator import evaluate_answer
from app.research.evidence_graph import build_evidence_graph
from app.retrieval.hybrid import hybrid_search
from app.retrieval.diagnostics import compute_retrieval_diagnostics
from app.retrieval.reranker import rerank
from app.retrieval.guardrails import filter_retrieved_docs
from app.verification.claim_extractor import extract_claims
from app.verification.citation_verifier import verify_citations

ROUTING_CONFIGS = {
    "simple": {"top_k": 5, "rerank": False, "max_hops": 1},
    "complex": {"top_k": 10, "rerank": True, "max_hops": 1},
    "multi-hop": {"top_k": 10, "rerank": True, "max_hops": 3},
}

class AgentState(TypedDict):
    # Input
    query: str
    document_ids: list[int] | None

    # Planning output
    complexity: str
    max_hops: int

    # Retrieval tracking
    current_query: str         # original query
    hop: int                   # number of retrieve calls made

    # Accumulated results
    all_docs: list[dict]
    evidence: list[str]
    claims: list[str]

    # Control / observability
    retrieval_diagnostics: dict[str, float]
    evaluator_confidence: float
    evidence_graph: dict

    # Output
    answer: str
    faithfulness: float
    citation_precision: float
    unsupported_claim_rate: float
    evidence_alignment_score: float
    answer_completeness: float
    cost: float
    latency: float
    latency: float

    # Internal cost tracking
    start_time: float
    total_input_chars: int
    total_output_chars: int

    # Metacognitive state (Paper §4)
    diagnosis: str                        # knowledge category from evaluator-critic
    diagnosis_reasoning: str              # explanation from diagnosis
    error_types: list[str]                # detected error patterns (declarative knowledge)
    internal_sufficient: bool             # procedural: LLM knows the answer?
    external_sufficient: bool             # procedural: docs contain the answer?
    writing_directive: str | None         # prompt modifier for targeted remediation
    metacognitive_round: int              # current iteration (Paper §6.5)
    previous_answer: str                  # for convergence detection
    diagnosis_suggested_query: str | None # sub-query for re-retrieval
    diagnosis_suggestion: str | None      # custom corrective suggestion


# ──────────────────────────────────────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────────────────────────────────────

async def plan_node(state: AgentState) -> dict[str, Any]:
    result = await plan_query(state["query"])
    comp = result.get("complexity", "simple")
    
    # We strictly enforce raw query usage for academic benchmark integrity
    cfg = ROUTING_CONFIGS.get(comp, ROUTING_CONFIGS["simple"])

    return {
        "complexity": comp,
        "max_hops": cfg["max_hops"],
        "current_query": state["query"],
        "hop": 0,
        "all_docs": [],
        "evidence": [],
        "claims": [],
        "retrieval_diagnostics": {},
        "evaluator_confidence": 0.5,
        "start_time": time.monotonic(),
        "total_input_chars": len(state["query"]) * 2,  # plan prompt approx
        "total_output_chars": 50,
        # Metacognitive init
        "diagnosis": SATISFACTORY,
        "diagnosis_reasoning": "",
        "error_types": [],
        "internal_sufficient": True,
        "external_sufficient": True,
        "writing_directive": None,
        "metacognitive_round": 0,
        "previous_answer": "",
        "diagnosis_suggested_query": None,
        "diagnosis_suggestion": None,
    }


async def retrieve_node(state: AgentState) -> dict[str, Any]:
    query_to_use = state["current_query"]
    
    comp = state.get("complexity", "simple")
    cfg = ROUTING_CONFIGS.get(comp, ROUTING_CONFIGS["simple"])
    
    # If rerank is enabled, fetch a larger pool (top_k=30)
    fetch_k = 30 if cfg["rerank"] else cfg["top_k"]

    new_docs = hybrid_search(
        query_to_use,
        top_k=fetch_k,
        document_ids=state.get("document_ids"),
    )
    diagnostics = compute_retrieval_diagnostics(query_to_use, new_docs)

    if cfg["rerank"] and new_docs:
        new_docs = rerank(query_to_use, new_docs, top_k=cfg["top_k"])

    # Apply retrieval guardrails to filter unsafe/low-quality chunks
    new_docs = filter_retrieved_docs(new_docs)

    # Deduplicate by doc id
    existing_ids = {d["id"] for d in state.get("all_docs", [])}
    fresh = [d for d in new_docs if d["id"] not in existing_ids]
    all_docs = state.get("all_docs", []) + fresh

    return {
        "all_docs": all_docs,
        "retrieval_diagnostics": diagnostics,
        "hop": state.get("hop", 0) + 1,
    }


async def read_node(state: AgentState) -> dict[str, Any]:
    query = state["current_query"]
    result = await read_documents(query, state["all_docs"])
    prompt_len = len(query) + sum(len(d["text"]) for d in state["all_docs"][:10])
    answer_len = sum(len(e) for e in result.get("evidence", []))

    return {
        "evidence": state["evidence"] + result.get("evidence", []),
        "total_input_chars": state["total_input_chars"] + prompt_len,
        "total_output_chars": state["total_output_chars"] + answer_len,
    }


async def write_node(state: AgentState) -> dict[str, Any]:
    # Paper §4.3: pass writing_directive from metacognitive planning
    answer = await write_answer(
        state["current_query"],
        state["evidence"],
        state["all_docs"],
        writing_directive=state.get("writing_directive"),
    )
    prompt_len = (
        len(state["current_query"])
        + sum(len(e) for e in state["evidence"])
        + sum(len(d["text"]) for d in state["all_docs"][:10])
    )
    return {
        "answer": answer,
        "abstained": False,
        "total_input_chars": state["total_input_chars"] + prompt_len,
        "total_output_chars": state["total_output_chars"] + len(answer),
    }


async def claim_extract_node(state: AgentState) -> dict[str, Any]:
    return {"claims": extract_claims(state["answer"]) }


async def citation_verify_node(state: AgentState) -> dict[str, Any]:
    metrics = verify_citations(state["answer"], state["all_docs"])
    return {
        "citation_precision": metrics["citation_precision"],
        "unsupported_claim_rate": metrics["unsupported_claim_rate"],
        "evidence_alignment_score": metrics["evidence_alignment_score"],
    }


async def evidence_graph_node(state: AgentState) -> dict[str, Any]:
    graph = build_evidence_graph(state["claims"], state["all_docs"])
    return {"evidence_graph": graph.to_dict()}


async def evaluate_node(state: AgentState) -> dict[str, Any]:
    context = "\n\n".join(d["text"] for d in state["all_docs"][:10])
    eval_result = await evaluate_answer(state["current_query"], state["answer"], context)
    faithfulness = eval_result["faithfulness"]
    answer_completeness = eval_result["answer_completeness"]
    evaluator_confidence = eval_result["confidence"]

    latency = time.monotonic() - state["start_time"]
    cost = estimate_cost(state["total_input_chars"], state["total_output_chars"])

    return {
        "faithfulness": faithfulness,
        "answer_completeness": answer_completeness,
        "evaluator_confidence": evaluator_confidence,
        "cost": cost,
        "latency": round(latency, 2),
    }


async def parallel_verify_and_evaluate_node(state: AgentState) -> dict[str, Any]:
    """
    Phase 3 optimization: run claim extraction, citation verification,
    evidence graph building, and LLM evaluation in parallel.
    """
    async def _claims():
        return extract_claims(state["answer"])

    async def _citations():
        return verify_citations(state["answer"], state["all_docs"])

    async def _evidence_graph(claims: list[str]):
        return build_evidence_graph(claims, state["all_docs"])

    async def _evaluate():
        context = "\n\n".join(d["text"] for d in state["all_docs"][:10])
        return await evaluate_answer(state["current_query"], state["answer"], context)

    claims_result, citation_metrics, eval_result = await asyncio.gather(
        _claims(), _citations(), _evaluate()
    )

    graph = await _evidence_graph(claims_result)

    faithfulness = eval_result["faithfulness"]
    answer_completeness = eval_result["answer_completeness"]
    evaluator_confidence = eval_result["confidence"]

    latency = time.monotonic() - state["start_time"]
    cost = estimate_cost(state["total_input_chars"], state["total_output_chars"])

    return {
        "claims": claims_result,
        "citation_precision": citation_metrics["citation_precision"],
        "unsupported_claim_rate": citation_metrics["unsupported_claim_rate"],
        "evidence_alignment_score": citation_metrics["evidence_alignment_score"],
        "evidence_graph": graph.to_dict(),
        "faithfulness": faithfulness,
        "answer_completeness": answer_completeness,
        "evaluator_confidence": evaluator_confidence,
        "cost": cost,
        "latency": round(latency, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Metacognitive nodes (Paper §4)
# ──────────────────────────────────────────────────────────────────────────────

async def diagnose_node(state: AgentState) -> dict[str, Any]:
    diag = await diagnose_answer(
        query=state["current_query"],
        answer=state["answer"],
        docs=state["all_docs"],
        faithfulness=state["faithfulness"],
        completeness=state["answer_completeness"],
        citation_precision=state.get("citation_precision", 0.0),
        evaluator_confidence=state.get("evaluator_confidence", 0.0),
    )

    return {
        "diagnosis": diag.category,
        "diagnosis_reasoning": diag.reasoning,
        "error_types": diag.error_types,
        "internal_sufficient": diag.internal_sufficient,
        "external_sufficient": diag.external_sufficient,
        "diagnosis_suggested_query": diag.suggested_query,
        "diagnosis_suggestion": diag.suggestion,
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
    }

    if diag.category == INSUFFICIENT and diag.suggested_query:
        comp = state.get("complexity", "simple")
        cfg = ROUTING_CONFIGS.get(comp, ROUTING_CONFIGS["simple"])
        fetch_k = 30 if cfg["rerank"] else cfg["top_k"]

        new_docs = hybrid_search(
            diag.suggested_query,
            top_k=fetch_k,
            document_ids=state.get("document_ids"),
        )

        if cfg["rerank"] and new_docs:
            new_docs = rerank(diag.suggested_query, new_docs, top_k=cfg["top_k"])
        new_docs = filter_retrieved_docs(new_docs)

        existing_ids = {d["id"] for d in state["all_docs"]}
        fresh = [d for d in new_docs if d["id"] not in existing_ids]

        if fresh:
            all_docs = state["all_docs"] + fresh
            read_result = await read_documents(state["current_query"], fresh)
            new_evidence = read_result.get("evidence", [])
            updates["all_docs"] = all_docs
            updates["evidence"] = state["evidence"] + new_evidence
            updates["total_input_chars"] = (
                state["total_input_chars"]
                + sum(len(d["text"]) for d in fresh)
            )
            updates["total_output_chars"] = (
                state["total_output_chars"]
                + sum(len(e) for e in new_evidence)
            )

    updates["writing_directive"] = build_writing_directive(diag)

    return updates


# ──────────────────────────────────────────────────────────────────────────────
# Conditional edge functions
# ──────────────────────────────────────────────────────────────────────────────

def should_remediate(state: AgentState) -> str:
    if state.get("diagnosis") == SATISFACTORY:
        return END
    if state["metacognitive_round"] >= settings.max_metacognitive_rounds:
        return END
    if check_convergence(state.get("previous_answer", ""), state.get("answer", "")):
        return END

    return "remediate"


# ──────────────────────────────────────────────────────────────────────────────
# Build the graph
# ──────────────────────────────────────────────────────────────────────────────

def build_graph(parallel: bool = True):
    g = StateGraph(AgentState)

    g.add_node("plan", plan_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("read", read_node)
    g.add_node("write", write_node)
    g.add_node("diagnose", diagnose_node)
    g.add_node("remediate", remediate_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "read")
    g.add_edge("read", "write")

    if parallel:
        g.add_node("verify_and_evaluate", parallel_verify_and_evaluate_node)
        g.add_edge("write", "verify_and_evaluate")
        g.add_edge("verify_and_evaluate", "diagnose")
    else:
        g.add_node("claim_extract", claim_extract_node)
        g.add_node("citation_verify", citation_verify_node)
        g.add_node("evidence_graph", evidence_graph_node)
        g.add_node("evaluate", evaluate_node)
        g.add_edge("write", "claim_extract")
        g.add_edge("claim_extract", "citation_verify")
        g.add_edge("citation_verify", "evidence_graph")
        g.add_edge("evidence_graph", "evaluate")
        g.add_edge("evaluate", "diagnose")

    g.add_conditional_edges("diagnose", should_remediate, {"remediate": "remediate", END: END})
    g.add_edge("remediate", "write")

    return g.compile()


_graph = None
_graph_parallel: bool | None = None


def get_graph():
    global _graph, _graph_parallel
    parallel = not settings.disable_parallel
    if _graph is None or _graph_parallel != parallel:
        _graph = build_graph(parallel=parallel)
        _graph_parallel = parallel
    return _graph


def build_initial_state(
    query: str,
    document_ids: list[int] | None = None,
) -> AgentState:
    return {
        "query": query,
        "document_ids": document_ids,
        "complexity": "simple",
        "max_hops": 1,
        "current_query": query,
        "hop": 0,
        "all_docs": [],
        "evidence": [],
        "claims": [],
        "retrieval_diagnostics": {},
        "evaluator_confidence": 0.5,
        "evidence_graph": {},
        "answer": "",
        "faithfulness": 0.0,
        "citation_precision": 0.0,
        "unsupported_claim_rate": 1.0,
        "evidence_alignment_score": 0.0,
        "answer_completeness": 0.0,
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
        "metacognitive_round": 0,
        "previous_answer": "",
        "diagnosis_suggested_query": None,
        "diagnosis_suggestion": None,
    }


async def stream_agent_updates(
    query: str,
    document_ids: list[int] | None = None,
):
    graph = get_graph()
    initial_state = build_initial_state(query, document_ids)
    async for update in graph.astream(initial_state, stream_mode="updates"):
        yield update


async def run_agent(
    query: str,
    document_ids: list[int] | None = None
) -> AgentState:
    graph = get_graph()
    initial_state = build_initial_state(query, document_ids)
    final_state = await graph.ainvoke(initial_state)
    return final_state
