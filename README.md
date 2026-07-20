# Meta-Agentic RAG | Adaptive Multi-Hop QA Research System

## Overview

Multi-hop question answering over retrieved documents requires more than a single retrieval-generation pass: models must recognize when their own answer is inadequate and understand why.

This work implements and extends the MetaRAG framework (Yao et al., 2024), which introduces a three-phase metacognitive regulation loop — monitoring, evaluating, and planning — directly into the RAG pipeline. The system classifies answer failures into four knowledge-deficit categories (insufficient knowledge, internal-only, external-only, reasoning error) and applies category-specific remediation strategies including targeted re-retrieval and constrained writing directives.

## Architecture

Runtime flow (LangGraph):

```text
retrieve -> write -> diagnose -> [remediate -> write]* -> END
```

The `diagnose` node combines **monitoring** (reference-answer similarity) and
**evaluating** (evaluator-critic LLM classification) into a single graph step.
The conditional edge `should_remediate` decides whether to loop or stop based on
three criteria: satisfactory diagnosis, round budget, or convergence detection.

### Component Mapping

| Paper stage | Implementation                                                                                                                                                                   | File                         |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| Cognition   | `write_answer()` generates the answer from query + documents. First draft uses Flash, remediation rounds upgrade to Pro.                                                         | `app/agent/writer.py`        |
| Monitoring  | `monitor_answer()` generates a reference answer (Flash) and computes embedding cosine similarity against the cognition answer.                                                   | `app/agent/metacognitive.py` |
| Evaluating  | `diagnose_answer()` classifies knowledge conditions and reasoning errors. Uses the Pro model as evaluator-critic. Falls back to heuristic rules on LLM failure.                  | `app/agent/metacognitive.py` |
| Planning    | `build_writing_directive()` maps diagnosis categories to writer instructions. `answer_mode_for_diagnosis()` selects `external_only`, `internal_only`, or `reasoning_error` mode. | `app/agent/metacognitive.py` |
| Remediation | `remediate_node()` applies the directive and optionally triggers follow-up retrieval when diagnosis is `insufficient_knowledge` with a `suggested_query`.                        | `app/agent/graph.py`         |

The LangGraph state machine is defined in `backend/app/agent/graph.py`.

### Dual-Model Routing

The system uses **two DeepSeek model tiers** to balance cost and capability:

| Tier     | Model               | Used by                                                                          |
| -------- | ------------------- | -------------------------------------------------------------------------------- |
| `flash`  | `deepseek-v4-flash` | Writer (first draft), monitor reference answer generation                        |
| `strong` | `deepseek-v4-pro`   | Writer (remediation rounds), evaluator-critic diagnosis, short-answer extraction |

All LLM calls go through the centralized `ainvoke()` function in `app/llm.py`,
which handles model selection and per-call cost logging.

### Convergence Detection

The metacognitive loop has three stop conditions:

1. **Satisfactory**: monitoring similarity ≥ threshold → skip evaluation.
2. **Round budget**: `max_metacognitive_rounds` reached.
3. **Convergence**: Jaccard overlap between consecutive answers exceeds
   `metacognitive_convergence_threshold` (default `0.85`).

## Module Map

```text
backend/
├── app/
│   ├── agent/
│   │   ├── graph.py            # LangGraph state machine
│   │   ├── metacognitive.py    # Monitor, diagnose, plan, convergence
│   │   └── writer.py           # Cognition: prompt construction + answer generation
│   ├── retrieval/
│   │   ├── hybrid.py           # Reciprocal Rank Fusion (dense + BM25)
│   │   ├── dense.py            # Qdrant + BGE embedding
│   │   ├── bm25_retrieval.py   # ElasticSearch BM25 lexical search
│   │   └── guardrails.py       # Prompt-injection filter + min-length check
│   ├── ingestion/
│   │   └── pipeline.py         # Document parser (PDF/DOCX/HTML/MD/TXT) + chunker
│   ├── config.py               # Pydantic settings (.env loader)
│   ├── llm.py                  # Centralized LLM factory with dual-tier routing
│   ├── cost.py                 # Per-call cost tracking (DeepSeek pricing)
│   └── text_utils.py           # JSON extraction from LLM responses
├── benchmarks/
│   ├── runner.py               # Benchmark evaluation (gold + open_domain modes)
│   ├── datasets.py             # HotpotQA / 2WikiMultiHopQA loaders
│   ├── metrics.py              # EM, F1, Precision, Recall
│   └── results/                # Evaluation outputs + cost logs
├── demo.py                     # Gradio interactive demo
└── pyproject.toml              # Project metadata (Python ≥ 3.13, uv)
```

## Default Research Settings

| Setting                      | Default                  | Source                  |
| ---------------------------- | ------------------------ | ----------------------- |
| Flash model                  | `deepseek-v4-flash`      | `config.py`             |
| Strong model                 | `deepseek-v4-pro`        | `config.py`             |
| Retrieval top-k              | `5`                      | `config.py`             |
| Monitor similarity threshold | `0.4`                    | `config.py`             |
| Max metacognitive rounds     | `5`                      | `config.py`             |
| Convergence threshold        | `0.85`                   | `config.py`             |
| Writer temperature           | `0.0`                    | `llm.py`                |
| Embedding model              | `BAAI/bge-small-en-v1.5` | `config.py`             |
| Chunk size / overlap         | `512 / 64` chars         | `ingestion/pipeline.py` |

Configuration lives in:

- `.env.example` / `.env`
- `backend/app/config.py`
- `docker-compose.yml`

## Retrieval Stack

Retrieval is hybrid:

- **Dense**: Qdrant + `BAAI/bge-small-en-v1.5` (384-dim, cosine) in
  `backend/app/retrieval/dense.py`
- **Lexical**: ElasticSearch BM25 in `backend/app/retrieval/bm25_retrieval.py`
- **Fusion**: Reciprocal Rank Fusion (k=60) in `backend/app/retrieval/hybrid.py`
- **Guardrails**: Prompt-injection pattern filter + minimum chunk length (20
  chars) in `backend/app/retrieval/guardrails.py`

## Ingestion Pipeline

`backend/app/ingestion/pipeline.py` parses uploaded documents (PDF, DOCX, HTML,
Markdown, TXT), splits them into overlapping character-level chunks
(512 chars / 64 overlap), and dual-indexes into both Qdrant and ElasticSearch.

## Cost Tracking

`backend/app/cost.py` records every LLM call with DeepSeek-specific token
usage (cache hit / miss / completion) and computes per-call USD cost using
published pricing. A session-level `CostTracker` accumulates entries and can
flush to JSONL for benchmark cost reporting. The module also supports querying
the DeepSeek account balance API.

## Local Setup

1. Copy environment variables:

```powershell
Copy-Item .env.example .env
```

2. Set `DEEPSEEK_API_KEY` in `.env`.

3. Start retrieval infrastructure:

```powershell
docker compose up -d qdrant elasticsearch
```

4. Optional: set `HF_TOKEN` in `.env` to authenticate Hugging Face downloads.
   When present, the embedding runtime logs in automatically and suppresses
   Hugging Face progress/load-report noise during model loading.

5. Install and test the research package:

```powershell
cd backend
uv sync
```

## Gradio Demo

An interactive Gradio demo walks through each metacognitive stage
(Cognition → Monitoring → Evaluating → Planning → Remediation) with real-time
visibility. It loads HotpotQA / 2WikiMultiHopQA examples and streams per-round
status, monitoring scores, diagnosis details, planning directives, and final
metrics (EM, F1).

```powershell
cd backend
uv run python demo.py
```

The demo launches at `http://localhost:7860`.

## Benchmarks

Benchmark scripts live under `backend/benchmarks/`. Two evaluation modes:

| Mode          | Description                                                     |
| ------------- | --------------------------------------------------------------- |
| `gold`        | Uses provided supporting documents — isolates reasoning quality |
| `open_domain` | Uses the full retrieval pipeline — tests end-to-end             |

The runner records cost tracking, latency percentiles (p50/p75/p90/p99), and
compares results against paper Table 1 baselines (Standard RAG, ReAct, Flare,
IR-CoT, Self-Ask, Reflexion, MetaRAG).

### Results

| Method                    | HotpotQA EM | HotpotQA F1 | HotpotQA Prec. | HotpotQA Rec. | 2WikiMultiHopQA EM | 2WikiMultiHopQA F1 | 2WikiMultiHopQA Prec. | 2WikiMultiHopQA Rec. |
| ------------------------- | ----------: | ----------: | -------------: | ------------: | -----------------: | -----------------: | --------------------: | -------------------: |
| Standard RAG (paper)      |        24.6 |        33.0 |           34.1 |          34.5 |               18.8 |               25.2 |                  25.6 |                 26.2 |
| ReAct (paper)             |        24.8 |        41.7 |           42.6 |          44.7 |               21.0 |               28.0 |                  27.6 |                 30.0 |
| Flare (paper)             |        29.2 |        42.4 |           42.8 |          43.0 |               28.2 |               39.8 |                  40.0 |                 40.8 |
| IR-CoT (paper)            |        31.4 |        40.3 |           41.6 |          41.2 |               30.8 |               42.6 |                  42.3 |                 40.9 |
| Self-Ask (paper)          |        28.2 |        43.1 |           43.4 |          44.8 |               28.6 |               37.5 |                  36.5 |                 42.8 |
| Reflexion (paper)         |        30.0 |        43.4 |           43.2 |          44.3 |               31.8 |               41.7 |                  40.6 |                 44.2 |
| MetaRAG (paper)           |        37.8 |        49.9 |           52.1 |          50.9 |               42.8 |               50.8 |                  50.7 |                 52.2 |
| **Meta-Agent-RAG (ours)** |    **49.8** |    **62.7** |       **66.8** |      **62.4** |           **65.0** |           **70.3** |              **70.2** |             **72.0** |

Examples:

```powershell
cd backend
uv run python -m benchmarks.runner --dataset hotpotqa --mode gold --n 50
uv run python -m benchmarks.runner --dataset 2wikimultihopqa --mode gold --n 50
uv run python -m benchmarks.runner --dataset hotpotqa --mode open_domain --n 50
```

Full options:

```powershell
uv run python -m benchmarks.runner --help
```

Key flags: `--concurrency` (parallel queries), `--convergence-threshold`
(override default 0.85), `--output-dir`.
