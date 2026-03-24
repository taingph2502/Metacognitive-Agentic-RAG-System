"""
Benchmark evaluation runner — Paper §5.

Runs the Meta-RAG agent pipeline against HotpotQA and 2WikiMultiHopQA,
computing EM, F1, Precision, and Recall as specified in the paper.

Two evaluation modes:
  - "gold": Use provided supporting documents (tests reasoning quality)
  - "open_domain": Use the full retrieval pipeline (tests end-to-end)

Usage:
    python -m benchmarks.runner --dataset hotpotqa --n 500 --mode gold
    python -m benchmarks.runner --dataset hotpotqa --n 500 --mode open_domain
    python -m benchmarks.runner --dataset 2wikimultihopqa --n 100 --mode gold
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent to path for module resolution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.datasets import (
    load_2wikimultihopqa,
    load_hotpotqa,
    save_results,
)
from benchmarks.metrics import aggregate_metrics, compute_metrics


def _ingest_benchmark_corpus(examples: list[dict]) -> int:
    """
    Ingest all context documents from benchmark examples into Qdrant.

    For open-domain evaluation, the agent must retrieve from a document store
    rather than receiving gold context. This function builds that store by
    ingesting all unique context paragraphs from the dataset examples.

    Returns the total number of chunks indexed.
    """
    from app.ingestion.pipeline import _chunk_text
    from app.retrieval.bm25_retrieval import index_chunks, wipe_es_index
    from app.retrieval.dense import upsert_chunks, wipe_all_embeddings

    # Collect all unique documents across all examples
    seen: set[str] = set()
    all_chunks: list[dict] = []

    for example in examples:
        for doc in example.get("context", []):
            source = doc.get("source", "unknown")
            text = doc.get("text", "").strip()
            if not text:
                continue
            # Deduplicate by source+text prefix
            key = f"{source}::{text[:200]}"
            if key in seen:
                continue
            seen.add(key)

            chunks = _chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "source": source,
                    "chunk_index": i,
                })

    if not all_chunks:
        return 0

    # Wipe existing stores and re-ingest
    print("  Wiping existing Qdrant collection and ElasticSearch index...")
    wipe_all_embeddings()
    wipe_es_index()

    # Batch upsert (100 chunks at a time for progress)
    total = len(all_chunks)
    batch_size = 100
    for i in range(0, total, batch_size):
        batch = all_chunks[i : i + batch_size]
        upsert_chunks(batch)
        index_chunks(batch)
        if (i + batch_size) % 500 == 0 or i + batch_size >= total:
            print(f"  Indexed {min(i + batch_size, total)}/{total} chunks...")

    return total


def _compute_latency_percentiles(results: list[dict]) -> dict[str, float]:
    """Compute p50, p75, p90, p99 latency in ms from result dicts."""
    latencies = sorted(r.get("latency_ms", 0) for r in results if "latency_ms" in r)
    if not latencies:
        return {}

    def _pct(p: float) -> float:
        idx = int(len(latencies) * p / 100)
        idx = min(idx, len(latencies) - 1)
        return round(latencies[idx], 1)

    return {"p50": _pct(50), "p75": _pct(75), "p90": _pct(90), "p99": _pct(99)}


async def evaluate_single_gold(
    example: dict,
    agent_fn,
) -> dict:
    """
    Evaluate a single example using gold context (provided documents).

    Runs the simplified pipeline: plan → write → evaluate → diagnose → [remediate → write]*
    bypassing retrieval to isolate reasoning quality while preserving the metacognitive loop.
    """
    import time as _time

    from app.agent.metacognitive import (
        SATISFACTORY,
        Diagnosis,
        build_writing_directive,
        check_convergence,
        diagnose_answer,
    )
    from app.agent.planner import plan_query
    from app.agent.writer import write_answer
    from app.config import settings
    from app.optimization.evaluator import evaluate_answer

    query_start = _time.monotonic()

    question = example["question"]
    reference = example["answer"]
    context_docs = example.get("context", [])

    # Build synthetic doc list matching the agent's expected format
    docs_for_agent = [
        {
            "id": i,
            "text": doc["text"],
            "source": doc["source"],
            "score": 1.0,
        }
        for i, doc in enumerate(context_docs)
    ]

    plan = await plan_query(question)
    query_type = plan["query_type"]

    # Collect evidence from gold docs
    evidence = [doc["text"][:500] for doc in context_docs if doc["text"].strip()]

    # Generate initial answer
    answer = await write_answer(question, evidence, docs_for_agent)

    # Metacognitive loop: evaluate → diagnose → remediate → write
    metacognitive_round = 0
    previous_answer = ""
    context_str = "\n\n".join(d["text"] for d in docs_for_agent[:10])

    for _ in range(settings.max_metacognitive_rounds):
        eval_result = await evaluate_answer(question, answer, context_str)
        faithfulness = eval_result["faithfulness"]
        completeness = eval_result["answer_completeness"]
        confidence = eval_result.get("confidence", 0.0)

        diag = await diagnose_answer(
            query=question,
            answer=answer,
            docs=docs_for_agent,
            faithfulness=faithfulness,
            completeness=completeness,
            citation_precision=0.5,  # no citation verify in gold mode
            evaluator_confidence=confidence,
        )

        if diag.category == SATISFACTORY:
            break

        if check_convergence(previous_answer, answer):
            break

        # Remediate: apply writing directive and re-generate
        directive = build_writing_directive(diag)
        previous_answer = answer
        answer = await write_answer(question, evidence, docs_for_agent, writing_directive=directive)
        metacognitive_round += 1

    latency_ms = (_time.monotonic() - query_start) * 1000

    # Extract short answer from the generated text
    short_answer = await _extract_short_answer(answer, question)

    # Compute metrics
    metrics = compute_metrics(short_answer, reference)
    metrics["question"] = question
    metrics["reference"] = reference
    metrics["prediction"] = short_answer
    metrics["full_answer"] = answer
    metrics["query_type"] = query_type
    metrics["example_id"] = example.get("id", "")
    metrics["metacognitive_rounds"] = metacognitive_round
    metrics["latency_ms"] = round(latency_ms, 1)

    return metrics


async def evaluate_single_open_domain(
    example: dict,
    agent_fn,
) -> dict:
    """
    Evaluate a single example using open-domain retrieval.

    Runs the full agent pipeline (plan → rewrite → retrieve → read → ... → diagnose)
    without any gold context — the system must find its own supporting documents.
    """
    import time as _time

    from app.agent.graph import build_initial_state, get_graph
    from app.optimization.bandit import CONFIG_NAMES

    query_start = _time.monotonic()

    question = example["question"]
    reference = example["answer"]

    # Default uniform bandit priors (no pre-trained state for benchmarks)
    alpha = {c: 1.0 for c in CONFIG_NAMES}
    beta = {c: 1.0 for c in CONFIG_NAMES}

    graph = get_graph()
    initial_state = build_initial_state(question, alpha, beta, document_ids=None)
    final_state = await graph.ainvoke(initial_state)

    answer = final_state.get("answer", "")
    query_type = final_state.get("query_type", "unknown")
    metacognitive_round = final_state.get("metacognitive_round", 0)

    latency_ms = (_time.monotonic() - query_start) * 1000

    # Extract short answer from the generated text
    short_answer = await _extract_short_answer(answer, question)

    # Compute metrics
    metrics = compute_metrics(short_answer, reference)
    metrics["question"] = question
    metrics["reference"] = reference
    metrics["prediction"] = short_answer
    metrics["full_answer"] = answer
    metrics["query_type"] = query_type
    metrics["example_id"] = example.get("id", "")
    metrics["metacognitive_rounds"] = metacognitive_round
    metrics["latency_ms"] = round(latency_ms, 1)
    metrics["num_docs_retrieved"] = len(final_state.get("all_docs", []))
    metrics["abstained"] = final_state.get("abstained", False)

    return metrics


async def _extract_short_answer(full_answer: str, question: str) -> str:
    """
    Extract a concise short answer from the generated markdown response
    using an LLM call for accurate extraction.

    For EM comparison, we need the minimal answer span (e.g. "Paris",
    not "The capital of France is Paris according to [1]").
    """
    import re

    from langchain_core.messages import HumanMessage

    from app.llm import ainvoke

    # If the full answer is already very short, use it directly
    cleaned = re.sub(r"\[\d+\]", "", full_answer).strip()
    cleaned = re.sub(r"[#*\[\]()]", "", cleaned).strip()
    if len(cleaned.split()) <= 5:
        return cleaned

    # LLM-based extraction for precise short answers
    prompt = (
        "Extract ONLY the shortest possible answer to the question from the text below. "
        "Output just the answer — no explanation, no citations, no formatting.\n\n"
        f"Question: {question}\n\n"
        f"Text: {full_answer[:1500]}\n\n"
        "Short answer:"
    )
    try:
        response = await ainvoke(
            [HumanMessage(content=prompt)], call_site="answer_extractor",
        )
        extracted = response.content.strip()
        # Clean up any remaining formatting
        extracted = re.sub(r"\[\d+\]", "", extracted).strip()
        extracted = re.sub(r"^['\"]|['\"]$", "", extracted).strip()
        if extracted and len(extracted.split()) <= 20:
            return extracted
    except Exception:
        pass

    # Heuristic fallback
    bold_match = re.search(r"\*\*(.+?)\*\*", full_answer)
    if bold_match:
        candidate = bold_match.group(1).strip()
        if len(candidate.split()) <= 15:
            return candidate

    sentences = re.split(r"[.!?]\s+", full_answer)
    if sentences:
        first = sentences[0].strip()
        first = re.sub(r"[#*\[\]()]", "", first).strip()
        first = re.sub(r"\[\d+\]", "", first).strip()
        return first

    return cleaned[:200].strip()


async def _eval_one(index: int, example: dict, mode: str, semaphore: asyncio.Semaphore) -> tuple[int, dict]:
    """Evaluate a single example under a concurrency semaphore."""
    async with semaphore:
        try:
            if mode == "gold":
                result = await evaluate_single_gold(example, None)
            elif mode == "open_domain":
                result = await evaluate_single_open_domain(example, None)
            else:
                raise ValueError(f"Unknown evaluation mode: {mode}")
            return index, result
        except Exception as e:
            return index, {
                "exact_match": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "question": example["question"],
                "reference": example["answer"],
                "prediction": "",
                "error": str(e),
                "example_id": example.get("id", ""),
            }


async def run_benchmark(
    dataset_name: str,
    n: int = 500,
    mode: str = "gold",
    seed: int = 42,
    output_dir: str = "benchmarks/results",
    concurrency: int = 1,
    provider: str | None = None,
    convergence_threshold: float | None = None,
    fast_path_threshold: float | None = None,
    disable_parallel: bool = False,
) -> dict:
    """
    Run full benchmark evaluation.

    Args:
        dataset_name: "hotpotqa" or "2wikimultihopqa"
        n: Number of examples to evaluate
        mode: "gold" (use provided context) or "open_domain" (full retrieval pipeline)
        seed: Random seed for subsampling
        output_dir: Directory for results output
        concurrency: Number of parallel queries (1 = sequential)
        provider: Override LLM provider ("deepseek" or "gemini"); None uses config default
        convergence_threshold: Override metacognitive convergence threshold
        fast_path_threshold: Evaluator confidence for fast-path skip (None = disabled)
        disable_parallel: Disable parallel post-write verification calls
    """
    from app.config import settings
    from app.cost import cost_tracker, fetch_deepseek_balance

    # Override provider if requested
    if provider:
        from app.llm import clear_cache
        settings.llm_provider = provider
        clear_cache()  # ensure new provider instances are created

    # Override convergence threshold if requested
    if convergence_threshold is not None:
        settings.metacognitive_convergence_threshold = convergence_threshold

    # Store fast-path and parallel settings
    if fast_path_threshold is not None:
        settings.fast_path_threshold = fast_path_threshold
    settings.disable_parallel = disable_parallel

    print(f"\n{'='*60}")
    print(f"Meta-RAG Benchmark: {dataset_name} ({mode} mode, n={n})")
    print(f"  LLM provider: {settings.llm_provider}")
    print(f"  Concurrency : {concurrency}")
    print(f"{'='*60}\n")

    # Reset cost tracker for this benchmark run
    cost_tracker.reset()

    # Fetch balance before (DeepSeek only, non-blocking)
    balance_before = None
    if settings.llm_provider.lower() == "deepseek":
        try:
            balance_before = await fetch_deepseek_balance(settings.deepseek_api_key)
            if balance_before:
                print(f"  Account balance (before): {balance_before['topped_up_balance']:.2f} {balance_before['currency']}")
        except Exception:
            pass

    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    if dataset_name == "hotpotqa":
        examples = load_hotpotqa(n=n, seed=seed)
    elif dataset_name == "2wikimultihopqa":
        examples = load_2wikimultihopqa(n=n, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Loaded {len(examples)} examples.\n")

    # For open_domain mode: ingest benchmark contexts into Qdrant
    if mode == "open_domain":
        print("Ingesting benchmark corpus into Qdrant for open-domain retrieval...")
        num_chunks = _ingest_benchmark_corpus(examples)
        print(f"  Ingested {num_chunks} chunks from {len(examples)} examples.\n")

    start_time = time.time()
    semaphore = asyncio.Semaphore(concurrency)

    if concurrency <= 1:
        # Sequential execution with live progress
        all_results = []
        for i, example in enumerate(examples):
            _, result = await _eval_one(i, example, mode, semaphore)
            all_results.append(result)

            if "error" in result:
                print(f"  [{i+1}] ERROR: {result['error']}")
            elif (i + 1) % 10 == 0 or i == 0:
                running_agg = aggregate_metrics(all_results)
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(examples) - i - 1) / max(0.01, rate)
                cs = cost_tracker.summary()
                print(
                    f"  [{i+1}/{len(examples)}] "
                    f"EM={running_agg['exact_match']:.3f} "
                    f"F1={running_agg['f1']:.3f} "
                    f"Prec={running_agg['precision']:.3f} "
                    f"Rec={running_agg['recall']:.3f} "
                    f"(${cs['total_cost_usd']:.4f}, {rate:.1f} q/s, ETA {eta:.0f}s)"
                )
    else:
        # Concurrent execution in batches for progress reporting
        all_results = [None] * len(examples)
        completed = 0
        batch_size = concurrency * 2  # process in batches for progress updates

        for batch_start in range(0, len(examples), batch_size):
            batch_end = min(batch_start + batch_size, len(examples))
            batch = examples[batch_start:batch_end]
            tasks = [
                _eval_one(batch_start + j, ex, mode, semaphore)
                for j, ex in enumerate(batch)
            ]
            for coro in asyncio.as_completed(tasks):
                idx, result = await coro
                all_results[idx] = result
                completed += 1

                if "error" in result:
                    print(f"  [{completed}/{len(examples)}] ERROR: {result['error']}")

            # Progress after each batch
            done = [r for r in all_results if r is not None]
            running_agg = aggregate_metrics(done)
            elapsed = time.time() - start_time
            rate = len(done) / max(0.01, elapsed)
            eta = (len(examples) - len(done)) / max(0.01, rate)
            cs = cost_tracker.summary()
            print(
                f"  [{len(done)}/{len(examples)}] "
                f"EM={running_agg['exact_match']:.3f} "
                f"F1={running_agg['f1']:.3f} "
                f"Prec={running_agg['precision']:.3f} "
                f"Rec={running_agg['recall']:.3f} "
                f"(${cs['total_cost_usd']:.4f}, {rate:.1f} q/s, ETA {eta:.0f}s)"
            )

        all_results = [r for r in all_results if r is not None]

    # Final aggregation
    final_metrics = aggregate_metrics(all_results)
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"RESULTS: {dataset_name} ({mode} mode)")
    print(f"{'='*60}")
    print(f"  Exact Match : {final_metrics['exact_match']:.4f}")
    print(f"  F1          : {final_metrics['f1']:.4f}")
    print(f"  Precision   : {final_metrics['precision']:.4f}")
    print(f"  Recall      : {final_metrics['recall']:.4f}")
    print(f"  Examples    : {final_metrics['num_examples']}")
    print(f"  Total time  : {total_time:.1f}s")

    # Latency percentiles
    pcts = _compute_latency_percentiles(all_results)
    if pcts:
        print(f"  Latency p50 : {pcts['p50']:.0f} ms")
        print(f"  Latency p75 : {pcts['p75']:.0f} ms")
        print(f"  Latency p90 : {pcts['p90']:.0f} ms")
        print(f"  Latency p99 : {pcts['p99']:.0f} ms")

    # Optimization settings
    print(f"  Convergence : {settings.metacognitive_convergence_threshold}")
    if settings.fast_path_threshold is not None:
        print(f"  Fast-path   : {settings.fast_path_threshold}")
    print(f"  Parallel    : {'OFF' if settings.disable_parallel else 'ON'}")
    print(f"{'='*60}\n")

    # Cost summary and balance (DeepSeek only)
    cost_tracker.print_summary()

    balance_after = None
    if settings.llm_provider.lower() == "deepseek":
        try:
            balance_after = await fetch_deepseek_balance(settings.deepseek_api_key)
        except Exception:
            pass

    if balance_before and balance_after:
        session_spend = balance_before["topped_up_balance"] - balance_after["topped_up_balance"]
        print(f"  Account balance  : {balance_after['topped_up_balance']:.2f} {balance_after['currency']}")
        print(f"  Session spend    : ${max(0, session_spend):.4f}")
        print()

    # Flush cost log to JSONL
    cost_log_path = Path(output_dir) / "cost_logs" / "cost_log.jsonl"
    cost_tracker.flush_to_jsonl(cost_log_path)

    # Save results
    output_path = Path(output_dir) / "evaluations"
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    provider_tag = settings.llm_provider.lower()
    results_file = output_path / f"{dataset_name}_{mode}_{provider_tag}_{timestamp}.json"
    summary_file = output_path / f"{dataset_name}_{mode}_{provider_tag}_{timestamp}_summary.json"

    cost_summary = cost_tracker.summary()
    save_results(all_results, results_file)
    save_results(
        {
            "dataset": dataset_name,
            "mode": mode,
            "provider": provider_tag,
            "n": n,
            "seed": seed,
            "llm_provider": settings.llm_provider,
            "metrics": final_metrics,
            "cost": cost_summary,
            "balance_before": balance_before,
            "balance_after": balance_after,
            "total_time_seconds": round(total_time, 2),
            "timestamp": timestamp,
            "convergence_threshold": settings.metacognitive_convergence_threshold,
            "fast_path_threshold": settings.fast_path_threshold,
            "disable_parallel": settings.disable_parallel,
            "latency_percentiles": _compute_latency_percentiles(all_results),
        },
        summary_file,
    )
    print(f"Results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"Cost log saved to: {cost_log_path}")

    return final_metrics


# Paper Table 1 baselines for reference
PAPER_BASELINES = {
    "hotpotqa": {
        "Standard RAG":  {"EM": 24.6, "F1": 33.0, "Prec": 34.1, "Rec": 34.5},
        "ReAct":         {"EM": 24.8, "F1": 41.7, "Prec": 42.6, "Rec": 44.7},
        "Flare":         {"EM": 29.2, "F1": 42.4, "Prec": 42.8, "Rec": 43.0},
        "IR-CoT":        {"EM": 31.4, "F1": 40.3, "Prec": 41.6, "Rec": 41.2},
        "Self-Ask":      {"EM": 28.2, "F1": 43.1, "Prec": 43.4, "Rec": 44.8},
        "Reflexion":     {"EM": 30.0, "F1": 43.4, "Prec": 43.2, "Rec": 44.3},
        "MetaRAG":       {"EM": 37.8, "F1": 49.9, "Prec": 52.1, "Rec": 50.9},
    },
    "2wikimultihopqa": {
        "Standard RAG":  {"EM": 18.8, "F1": 25.2, "Prec": 25.6, "Rec": 26.2},
        "ReAct":         {"EM": 21.0, "F1": 28.0, "Prec": 27.6, "Rec": 30.0},
        "Flare":         {"EM": 28.2, "F1": 39.8, "Prec": 40.0, "Rec": 40.8},
        "IR-CoT":        {"EM": 30.8, "F1": 42.6, "Prec": 42.3, "Rec": 40.9},
        "Self-Ask":      {"EM": 28.6, "F1": 37.5, "Prec": 36.5, "Rec": 42.8},
        "Reflexion":     {"EM": 31.8, "F1": 41.7, "Prec": 40.6, "Rec": 44.2},
        "MetaRAG":       {"EM": 42.8, "F1": 50.8, "Prec": 50.7, "Rec": 52.2},
    },
}


def print_comparison(dataset_name: str, our_metrics: dict) -> None:
    """Print our results alongside paper baselines for comparison."""
    baselines = PAPER_BASELINES.get(dataset_name, {})
    if not baselines:
        return

    print(f"\n{'='*70}")
    print(f"COMPARISON WITH PAPER BASELINES — {dataset_name}")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'EM':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}")
    print("-" * 70)
    for method, scores in baselines.items():
        print(f"{method:<20} {scores['EM']:>8.1f} {scores['F1']:>8.1f} {scores['Prec']:>8.1f} {scores['Rec']:>8.1f}")
    print("-" * 70)
    print(
        f"{'Ours (Meta-RAG)':<20} "
        f"{our_metrics['exact_match']*100:>8.1f} "
        f"{our_metrics['f1']*100:>8.1f} "
        f"{our_metrics['precision']*100:>8.1f} "
        f"{our_metrics['recall']*100:>8.1f}"
    )
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Meta-RAG Benchmark Runner")
    parser.add_argument(
        "--dataset",
        choices=["hotpotqa", "2wikimultihopqa", "both"],
        default="hotpotqa",
        help="Dataset to evaluate on",
    )
    parser.add_argument("--n", type=int, default=500, help="Number of examples (paper uses 500)")
    parser.add_argument("--mode", choices=["gold", "open_domain"], default="gold", help="Evaluation mode: gold (provided docs) or open_domain (full retrieval)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    parser.add_argument("--output-dir", default="benchmarks/results", help="Output directory")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel queries (default 1, try 5-10 for speed)")
    parser.add_argument("--provider", choices=["deepseek", "gemini"], default=None, help="Override LLM provider (default: use config)")
    parser.add_argument("--convergence-threshold", type=float, default=None, help="Override metacognitive convergence threshold (default: 0.85)")
    parser.add_argument("--fast-path-threshold", type=float, default=None, help="Evaluator confidence threshold for fast-path skip (default: None = disabled)")
    parser.add_argument("--disable-parallel", action="store_true", help="Disable parallel post-write verification calls")
    args = parser.parse_args()

    datasets = (
        ["hotpotqa", "2wikimultihopqa"] if args.dataset == "both"
        else [args.dataset]
    )

    for ds in datasets:
        metrics = asyncio.run(
            run_benchmark(ds, n=args.n, mode=args.mode, seed=args.seed,
                          output_dir=args.output_dir, concurrency=args.concurrency,
                          provider=args.provider,
                          convergence_threshold=args.convergence_threshold,
                          fast_path_threshold=args.fast_path_threshold,
                          disable_parallel=args.disable_parallel)
        )
        print_comparison(ds, metrics)


if __name__ == "__main__":
    main()
