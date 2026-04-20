import json
import re
from collections import Counter
from pathlib import Path

import logging
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import build_initial_state, run_agent, stream_agent_updates
from app.config import settings
from app.database import get_db
from app.ingestion.pipeline import ingest_document
from app.memory.strategy_memory import (
    log_provenance_snapshot,
    log_retrieval_diagnostics,
    log_run,
)
from app.agent.graph import ROUTING_CONFIGS
from app.models.db_models import Document, ProvenanceSnapshot, RetrievalDiagnostic, RunLog
from app.models.schemas import (
    CitedChunk,
    DocumentRead,
    DocumentUploadResponse,
    EvidenceGraphLinkRead,
    EvidenceGraphRead,
    HealthResponse,
    QueryRequest,
    QueryProvenance,
    QueryResponse,
    RetrievedDocumentRead,
    RunMetrics,
)
from app.retrieval.bm25_retrieval import (
    delete_es_by_document_id,
    delete_es_by_source,
    wipe_es_index,
)
from app.retrieval.dense import delete_by_document_id, delete_by_source, wipe_all_embeddings

logger = logging.getLogger(__name__)
router = APIRouter()
BACKEND_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_RESULTS_DIR = BACKEND_ROOT / "benchmarks" / "results" / "evaluations"

ALLOWED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".markdown", ".docx", ".txt"}


# ──────────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health(db: AsyncSession = Depends(get_db)):
    qdrant_ok = False
    db_ok = False

    try:
        from app.retrieval.dense import get_qdrant_client
        get_qdrant_client().get_collections()
        qdrant_ok = True
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")

    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")

    return HealthResponse(
        status="ok" if qdrant_ok and db_ok else "degraded",
        qdrant=qdrant_ok,
        database=db_ok,
    )


@router.get("/system/overview")
async def system_overview():
    benchmark_summaries = _load_benchmark_summaries(limit=6)
    return {
        "project": {
            "name": "Meta-RAG Research Workbench",
            "description": "Autonomous research agent with metacognitive remediation, adaptive retrieval, and benchmark tooling.",
        },
        "infrastructure": {
            "llm_provider": settings.llm_provider,
            "deepseek_model": settings.deepseek_model,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "qdrant_collection": settings.qdrant_collection,
        },
        "pipeline": [
            {
                "id": "planner",
                "title": "Planner",
                "summary": "Classifies the query into factual, comparative, or multi-hop and initializes hop limits.",
            },
            {
                "id": "query_rewrite",
                "title": "Query Rewriter",
                "summary": "Expands or reformulates the query with rule-based or LLM-guided variants.",
            },
            {
                "id": "retrieval",
                "title": "Hybrid Retrieval",
                "summary": "Combines dense search, ElasticSearch BM25, reranking, and guardrails to assemble candidate evidence.",
            },
            {
                "id": "reader",
                "title": "Reader",
                "summary": "Extracts evidence spans, estimates coverage, and proposes follow-up queries for extra hops.",
            },
            {
                "id": "controller",
                "title": "Research Controller",
                "summary": "Decides whether to stop, reformulate, hop again, or abstain based on retrieval and confidence signals.",
            },
            {
                "id": "writer",
                "title": "Writer",
                "summary": "Produces grounded answers with inline citations or an abstention when evidence is too weak.",
            },
            {
                "id": "verification",
                "title": "Verification Stack",
                "summary": "Runs claim extraction, citation verification, evidence graph construction, and answer evaluation.",
            },
            {
                "id": "metacognition",
                "title": "Metacognitive Loop",
                "summary": "Diagnoses failure modes and applies targeted remediation through re-retrieval or writing directives.",
            },
        ],
        "retrieval_configs": [
            {
                "name": name,
                "top_k": cfg["top_k"],
                "rerank": cfg["rerank"],
                "max_hops": cfg["max_hops"],
            }
            for name, cfg in ROUTING_CONFIGS.items()
        ],
        "metacognition": {
            "max_rounds": settings.max_metacognitive_rounds,
            "faithfulness_threshold": settings.faithfulness_threshold,
            "convergence_threshold": settings.metacognitive_convergence_threshold,
            "fast_path_threshold": settings.fast_path_threshold,
            "parallel_verification_enabled": not settings.disable_parallel,
        },
        "api_surface": {
            "query": ["/api/query", "/api/query/stream"],
            "documents": ["/api/documents", "/api/documents/{id}", "/api/documents/wipe"],
            "ingestion": ["/api/ingest"],
            "analytics": ["/api/bandit", "/api/analytics/overview", "/api/analytics/provenance", "/api/benchmarks/summaries"],
        },
        "benchmark_summaries": benchmark_summaries,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Query
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest, db: AsyncSession = Depends(get_db)):
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query_context = await _prepare_query_context(query, db)

    final_state = await run_agent(
        query,
        document_ids=query_context["active_doc_ids"],
    )

    return await _finalize_query_response(
        db=db,
        query=query,
        final_state=final_state,
    )


@router.post("/query/stream")
async def query_stream_endpoint(body: QueryRequest, db: AsyncSession = Depends(get_db)):
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query_context = await _prepare_query_context(query, db)

    async def event_generator():
        current_state = build_initial_state(
            query,
            document_ids=query_context["active_doc_ids"],
        )

        try:
            async for update in stream_agent_updates(
                query,
                document_ids=query_context["active_doc_ids"],
            ):
                for node_name, payload in update.items():
                    previous_state = dict(current_state)
                    current_state.update(payload)

                    if node_name == "remediate":
                        yield _sse_event(
                            {
                                "type": "retrying",
                                "details": {
                                    "retry_count": current_state.get("metacognitive_round", 0),
                                    "diagnosis": previous_state.get("diagnosis", ""),
                                    "diagnosis_reasoning": previous_state.get("diagnosis_reasoning", ""),
                                    "error_types": previous_state.get("error_types", []),
                                    "reason": f"metacognitive remediation: {previous_state.get('diagnosis', '')}",
                                    "failed_faithfulness": previous_state.get("faithfulness", 0.0),
                                },
                            }
                        )
                        continue

                    event = _build_stream_event(node_name, current_state)
                    if event is not None:
                        yield _sse_event(event)

            response = await _finalize_query_response(
                db=db,
                query=query,
                final_state=current_state,
            )
            yield _sse_event({"type": "query_complete", "response": response.model_dump()})
        except Exception as exc:
            yield _sse_event({"type": "query_error", "error": str(exc)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _prepare_query_context(query: str, db: AsyncSession) -> dict:
    doc_result = await db.execute(select(Document.id))
    active_doc_ids = [row[0] for row in doc_result.all()]

    return {
        "active_doc_ids": active_doc_ids,
    }


async def _finalize_query_response(
    db: AsyncSession,
    query: str,
    final_state: dict,
) -> QueryResponse:
    query_type = final_state.get("complexity", "simple")
    final_config = query_type

    faithfulness = final_state["faithfulness"]
    citation_precision = final_state.get("citation_precision", 0.0)
    unsupported_claim_rate = final_state.get("unsupported_claim_rate", 1.0)
    answer_completeness = final_state.get("answer_completeness", 0.0)
    cost = final_state["cost"]
    latency = final_state["latency"]
    retrieval_diagnostics = final_state.get("retrieval_diagnostics", {})

    metacognitive_round = final_state.get("metacognitive_round", 0)

    await log_run(db, query, query_type, final_config, faithfulness, cost, latency, is_retry=metacognitive_round > 0)
    await log_retrieval_diagnostics(db, query, query_type, final_config, retrieval_diagnostics)

    docs = final_state["all_docs"]
    citations = _extract_citations(final_state["answer"], docs)
    provenance = _build_provenance(final_state, citations)
    await log_provenance_snapshot(
        db,
        query=query,
        query_type=query_type,
        config_name=final_config,
        diagnosis=final_state.get("diagnosis", "satisfactory"),
        faithfulness=faithfulness,
        citation_precision=citation_precision,
        hops=int(final_state.get("hop", 0)),
        metacognitive_rounds=int(final_state.get("metacognitive_round", 0)),
        answer_preview=str(final_state.get("answer", ""))[:500],
        provenance_payload=provenance.model_dump(),
        retrieval_diagnostics=retrieval_diagnostics,
    )

    return QueryResponse(
        answer=final_state["answer"],
        citations=citations,
        metrics=RunMetrics(
            faithfulness=round(faithfulness, 4),
            citation_precision=round(citation_precision, 4),
            unsupported_claim_rate=round(unsupported_claim_rate, 4),
            answer_completeness=round(answer_completeness, 4),
            cost=round(cost, 6),
            latency=round(latency, 2),
            config=final_config,
            query_type=query_type,
            hops=final_state["hop"],
            retrieval_diagnostics=retrieval_diagnostics,
            diagnosis=final_state.get("diagnosis", "satisfactory"),
            diagnosis_reasoning=final_state.get("diagnosis_reasoning", ""),
            error_types=final_state.get("error_types", []),
            metacognitive_rounds=final_state.get("metacognitive_round", 0),
        ),
        provenance=provenance,
        abstained=final_state.get("abstained", False),
    )


def _build_stream_event(node_name: str, state: dict) -> dict | None:
    if node_name == "plan":
        return {
            "type": "step_completed",
            "step": "planner",
            "details": {
                "complexity": state.get("complexity", "simple"),
                "primary_query": (state.get("query_variants") or [state.get("current_query", "")])[0],
            },
        }

    if node_name == "retrieve":
        diagnostics = state.get("retrieval_diagnostics", {})
        return {
            "type": "step_completed",
            "step": "retrieval",
            "details": {
                "documents_retrieved": len(state.get("all_docs", [])),
                "query_coverage": diagnostics.get("query_coverage", 0.0),
                "document_diversity": diagnostics.get("document_diversity", 0.0),
                "retrieval_redundancy": diagnostics.get("retrieval_redundancy", 0.0),
                "recall_proxy": diagnostics.get("estimated_recall_proxy", 0.0),
                "top_sources": [doc.get("source", "unknown") for doc in state.get("all_docs", [])[:3]],
            },
        }

    if node_name == "read":
        return {
            "type": "step_completed",
            "step": "reader",
            "details": {
                "evidence_spans": len(state.get("evidence", [])),
                "followup_query": state.get("followup_query") or "none",
                "evidence_preview": (state.get("evidence") or ["none"])[0][:140],
            },
        }

    if node_name == "controller":
        diagnostics = state.get("retrieval_diagnostics", {})
        return {
            "type": "step_completed",
            "step": "research_controller",
            "details": {
                "evidence_coverage": state.get("evidence_coverage", 0.0),
                "recall_proxy": diagnostics.get("estimated_recall_proxy", 0.0),
                "decision": state.get("controller_action", "stop"),
            },
        }

    if node_name == "write":
        return {
            "type": "step_completed",
            "step": "writer",
            "details": {
                "answer_length": len(state.get("answer", "")),
                "abstained": state.get("abstained", False),
            },
        }

    if node_name == "claim_extract":
        return {
            "type": "step_completed",
            "step": "claim_extraction",
            "details": {
                "claims_extracted": len(state.get("claims", [])),
            },
        }

    if node_name == "citation_verify":
        return {
            "type": "step_completed",
            "step": "citation_verification",
            "details": {
                "citation_precision": state.get("citation_precision", 0.0),
                "unsupported_claim_rate": state.get("unsupported_claim_rate", 0.0),
            },
        }

    if node_name == "evaluate":
        return {
            "type": "step_completed",
            "step": "evaluation",
            "details": {
                "faithfulness": state.get("faithfulness", 0.0),
                "answer_completeness": state.get("answer_completeness", 0.0),
                "confidence": state.get("evaluator_confidence", 0.0),
            },
        }

    if node_name == "verify_and_evaluate":
        return {
            "type": "step_completed",
            "step": "verification_and_evaluation",
            "details": {
                "claims_extracted": len(state.get("claims", [])),
                "citation_precision": state.get("citation_precision", 0.0),
                "unsupported_claim_rate": state.get("unsupported_claim_rate", 0.0),
                "faithfulness": state.get("faithfulness", 0.0),
                "answer_completeness": state.get("answer_completeness", 0.0),
                "confidence": state.get("evaluator_confidence", 0.0),
            },
        }

    if node_name == "diagnose":
        return {
            "type": "step_completed",
            "step": "diagnosis",
            "details": {
                "category": state.get("diagnosis", ""),
                "reasoning": state.get("diagnosis_reasoning", ""),
                "error_types": state.get("error_types", []),
                "internal_sufficient": state.get("internal_sufficient", True),
                "external_sufficient": state.get("external_sufficient", True),
                "metacognitive_round": state.get("metacognitive_round", 0),
            },
        }

    return None


def _extract_citations(answer: str, docs: list[dict]) -> list[CitedChunk]:
    """Extract [N] citation references from the answer and map to source docs."""
    cited_indices = set()
    # Find all brackets containing numbers, potentially separated by commas/spaces
    # e.g., [1], [1, 2], [1, 2, 3]
    for m in re.finditer(r"\[([\d\s,]+)\]", answer):
        content = m.group(1)
        # Split by comma and clean up
        parts = content.split(",")
        for p in parts:
            try:
                idx = int(p.strip())
                if 1 <= idx <= len(docs):
                    cited_indices.add(idx)
            except ValueError:
                continue

    citations = []
    for idx in sorted(cited_indices):
        doc = docs[idx - 1]
        citations.append(
            CitedChunk(
                index=idx,
                text=doc["text"][:300],
                source=doc["source"],
            )
        )
    return citations


def _build_provenance(final_state: dict, citations: list[CitedChunk]) -> QueryProvenance:
    docs = final_state.get("all_docs", [])
    evidence_graph = final_state.get("evidence_graph", {}) or {}
    cited_doc_indices = {citation.index for citation in citations}

    links = [
        EvidenceGraphLinkRead(
            claim_index=index + 1,
            claim=str(link.get("claim", "")),
            supporting_docs=[int(doc_idx) + 1 for doc_idx in link.get("supporting_docs", [])],
            confidence=round(float(link.get("confidence", 0.0)), 4),
        )
        for index, link in enumerate(evidence_graph.get("links", []))
    ]

    support_map: dict[int, list[int]] = {}
    for link in links:
        for doc_index in link.supporting_docs:
            support_map.setdefault(doc_index, []).append(link.claim_index)

    retrieved_documents = [
        RetrievedDocumentRead(
            rank=index + 1,
            source=str(doc.get("source", "unknown")),
            text=str(doc.get("text", ""))[:420],
            score=round(float(doc.get("score", 0.0)), 4),
            cited=(index + 1) in cited_doc_indices,
            supporting_claims=support_map.get(index + 1, []),
        )
        for index, doc in enumerate(docs)
    ]

    return QueryProvenance(
        evidence_spans=[str(item) for item in final_state.get("evidence", [])[:12]],
        followup_query=final_state.get("followup_query"),
        retrieved_documents=retrieved_documents,
        evidence_graph=EvidenceGraphRead(
            total_claims=int(evidence_graph.get("total_claims", 0)),
            supported_claims=int(evidence_graph.get("supported_claims", 0)),
            coverage_ratio=round(float(evidence_graph.get("coverage_ratio", 0.0)), 4),
            links=links,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Documents
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/documents", response_model=list[DocumentRead])
async def get_documents(db: AsyncSession = Depends(get_db)):
    """List all documents in the system."""
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    docs = result.scalars().all()
    return docs


@router.get("/analytics/overview")
async def analytics_overview(db: AsyncSession = Depends(get_db)):
    docs_total = await db.scalar(select(func.count()).select_from(Document)) or 0
    docs_indexed = await db.scalar(select(func.count()).select_from(Document).where(Document.status == "indexed")) or 0
    docs_processing = await db.scalar(select(func.count()).select_from(Document).where(Document.status == "processing")) or 0
    docs_failed = await db.scalar(select(func.count()).select_from(Document).where(Document.status == "failed")) or 0

    runs_total = await db.scalar(select(func.count()).select_from(RunLog)) or 0

    run_averages = await db.execute(
        select(
            func.avg(RunLog.faithfulness),
            func.avg(RunLog.latency),
            func.avg(RunLog.cost),
        )
    )
    avg_faithfulness, avg_latency, avg_cost = run_averages.one()

    recent_runs_result = await db.execute(
        select(RunLog).order_by(RunLog.created_at.desc()).limit(12)
    )
    recent_runs = recent_runs_result.scalars().all()

    recent_diagnostics_result = await db.execute(
        select(RetrievalDiagnostic)
        .order_by(RetrievalDiagnostic.created_at.desc())
        .limit(12)
    )
    recent_diagnostics = recent_diagnostics_result.scalars().all()

    recent_provenance_result = await db.execute(
        select(ProvenanceSnapshot)
        .order_by(ProvenanceSnapshot.created_at.desc())
        .limit(8)
    )
    recent_provenance = recent_provenance_result.scalars().all()

    query_type_rows = await db.execute(
        select(RunLog.query_type, func.count())
        .group_by(RunLog.query_type)
        .order_by(func.count().desc())
    )
    config_rows = await db.execute(
        select(RunLog.config_name, func.count())
        .group_by(RunLog.config_name)
        .order_by(func.count().desc())
    )

    recent_query_counter = Counter(run.query for run in recent_runs)

    return {
        "documents": {
            "total": docs_total,
            "indexed": docs_indexed,
            "processing": docs_processing,
            "failed": docs_failed,
        },
        "runs": {
            "total": runs_total,
            "avg_faithfulness": round(float(avg_faithfulness or 0.0), 4),
            "avg_latency": round(float(avg_latency or 0.0), 2),
            "avg_cost": round(float(avg_cost or 0.0), 6),
        },
        "query_type_distribution": [
            {"label": label, "count": count}
            for label, count in query_type_rows.all()
        ],
        "config_distribution": [
            {"label": label, "count": count}
            for label, count in config_rows.all()
        ],
        "recent_runs": [
            {
                "query": run.query,
                "query_type": run.query_type,
                "config_name": run.config_name,
                "faithfulness": round(run.faithfulness, 4),
                "cost": round(run.cost, 6),
                "latency": round(run.latency, 2),
                "is_retry": run.is_retry,
                "created_at": run.created_at.isoformat(),
            }
            for run in recent_runs
        ],
        "recent_diagnostics": [
            {
                "query": item.query,
                "query_type": item.query_type,
                "config_name": item.config_name,
                "query_coverage": round(item.query_coverage, 4),
                "document_diversity": round(item.document_diversity, 4),
                "retrieval_redundancy": round(item.retrieval_redundancy, 4),
                "estimated_recall_proxy": round(item.estimated_recall_proxy, 4),
                "created_at": item.created_at.isoformat(),
            }
            for item in recent_diagnostics
        ],
        "recent_provenance": [
            {
                "query": item.query,
                "query_type": item.query_type,
                "config_name": item.config_name,
                "diagnosis": item.diagnosis,
                "faithfulness": round(item.faithfulness, 4),
                "citation_precision": round(item.citation_precision, 4),
                "hops": item.hops,
                "metacognitive_rounds": item.metacognitive_rounds,
                "answer_preview": item.answer_preview[:220],
                "retrieval_diagnostics": item.retrieval_diagnostics,
                "provenance": item.provenance_payload,
                "created_at": item.created_at.isoformat(),
            }
            for item in recent_provenance
        ],
        "recent_query_frequency": [
            {"query": query, "count": count}
            for query, count in recent_query_counter.most_common(6)
        ],
    }


@router.get("/analytics/provenance")
async def analytics_provenance(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ProvenanceSnapshot)
        .order_by(ProvenanceSnapshot.created_at.desc())
        .limit(30)
    )
    items = result.scalars().all()
    return [
        {
            "query": item.query,
            "query_type": item.query_type,
            "config_name": item.config_name,
            "diagnosis": item.diagnosis,
            "faithfulness": round(item.faithfulness, 4),
            "citation_precision": round(item.citation_precision, 4),
            "hops": item.hops,
            "metacognitive_rounds": item.metacognitive_rounds,
            "answer_preview": item.answer_preview,
            "retrieval_diagnostics": item.retrieval_diagnostics,
            "provenance": item.provenance_payload,
            "created_at": item.created_at.isoformat(),
        }
        for item in items
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Document upload
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=DocumentUploadResponse)
async def ingest_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    import os

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:  # 50 MB cap
        raise HTTPException(status_code=413, detail="File too large (max 50 MB).")

    # Create document record
    doc = Document(
        filename=file.filename,
        source=file.filename,
        status="processing"
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    async def run_ingestion(doc_id: int, filename: str, file_content: bytes):
        from app.database import async_session_factory
        async with async_session_factory() as session:
            try:
                count = ingest_document(filename, file_content, document_id=doc_id)
                db_doc = await session.get(Document, doc_id)
                if db_doc:
                    db_doc.status = "indexed"
                    db_doc.chunks_count = count
                    await session.commit()
            except Exception as e:
                db_doc = await session.get(Document, doc_id)
                if db_doc:
                    db_doc.status = "failed"
                    db_doc.error_message = str(e)
                    await session.commit()

    background_tasks.add_task(run_ingestion, doc.id, file.filename, content)

    return DocumentUploadResponse(
        message=f"Upload started for '{file.filename}'. It will be processed in the background.",
        chunks_indexed=0,
    )


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a document and its associated vector embeddings."""
    doc = await db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete from Qdrant using the unified document_id
    delete_by_document_id(doc.id)
    # Also delete by source as a backup (important for legacy data)
    delete_by_source(doc.source)
    # Delete from ElasticSearch
    delete_es_by_document_id(doc.id)
    delete_es_by_source(doc.source)
    
    await db.delete(doc)
    await db.commit()
    return {"message": "Document deleted successfully"}


@router.post("/documents/wipe")
async def wipe_all_documents(db: AsyncSession = Depends(get_db)):
    """Wipe all documents from the database and all embeddings from Qdrant."""
    from sqlalchemy import delete
    
    # 1. Clear database
    await db.execute(delete(Document))
    await db.commit()
    
    # 2. Clear Qdrant collection
    wipe_all_embeddings()
    # 3. Clear ElasticSearch index
    wipe_es_index()

    return {"message": "All database records and vector embeddings have been wiped."}


# (Bandit stats endpoints removed)


@router.get("/benchmarks/summaries")
async def benchmark_summaries():
    return _load_benchmark_summaries(limit=20)


def _load_benchmark_summaries(limit: int = 20) -> list[dict]:
    if not BENCHMARK_RESULTS_DIR.exists():
        return []

    summaries: list[dict] = []
    files = sorted(
        BENCHMARK_RESULTS_DIR.glob("*_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for path in files[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        metrics = payload.get("metrics", {})
        cost = payload.get("cost", {})
        summaries.append(
            {
                "file": path.name,
                "dataset": payload.get("dataset", "unknown"),
                "mode": payload.get("mode", "unknown"),
                "provider": payload.get("provider") or payload.get("llm_provider", settings.llm_provider),
                "timestamp": payload.get("timestamp"),
                "metrics": {
                    "exact_match": metrics.get("exact_match", 0.0),
                    "f1": metrics.get("f1", 0.0),
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "num_examples": metrics.get("num_examples", payload.get("n", 0)),
                },
                "cost": {
                    "total_cost_usd": cost.get("total_cost_usd", 0.0),
                    "total_tokens": cost.get("total_tokens", 0),
                    "total_calls": cost.get("total_calls", 0),
                },
                "total_time_seconds": payload.get("total_time_seconds", 0.0),
                "latency_percentiles": payload.get("latency_percentiles", {}),
                "convergence_threshold": payload.get("convergence_threshold"),
                "fast_path_threshold": payload.get("fast_path_threshold"),
                "disable_parallel": payload.get("disable_parallel"),
            }
        )

    return summaries
