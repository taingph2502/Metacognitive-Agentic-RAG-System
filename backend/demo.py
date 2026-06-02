"""
Gradio demo for the MetaRAG metacognitive pipeline.

Loads HotpotQA examples and walks through each metacognitive stage
(Cognition → Monitoring → Evaluating → Planning → Remediation)
with real-time visibility into every step.

Usage:
    cd backend
    uv run python demo.py
"""

import asyncio
import sys
import time
from pathlib import Path

import gradio as gr

# Ensure backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.agent.metacognitive import (
    SATISFACTORY,
    answer_mode_for_diagnosis,
    build_writing_directive,
    check_convergence,
    diagnose_answer,
    monitor_answer,
)
from app.agent.writer import write_answer
from app.config import settings
from benchmarks.datasets import load_hotpotqa, load_2wikimultihopqa
from benchmarks.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Short answer extraction (same as benchmarks.runner)
# ---------------------------------------------------------------------------
async def _extract_short_answer(full_answer: str, question: str) -> str:
    import re
    from langchain_core.messages import HumanMessage
    from app.llm import ainvoke

    cleaned = re.sub(r"\[\d+\]", "", full_answer).strip()
    cleaned = re.sub(r"[#*\[\]()]", "", cleaned).strip()
    if len(cleaned.split()) <= 5:
        return cleaned

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
        extracted = re.sub(r"\[\d+\]", "", extracted).strip()
        extracted = re.sub(r"^['\"]|['\"]$", "", extracted).strip()
        if extracted and len(extracted.split()) <= 20:
            return extracted
    except Exception:
        pass

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

# ---------------------------------------------------------------------------
# Dataset cache — sorted by real monitor scores from hard_questions.json
# ---------------------------------------------------------------------------
import json as _json
from pathlib import Path as _Path

_dataset_cache: dict[str, list[dict]] = {}
DATASET_CHOICES = ["HotpotQA", "2WikiMultiHopQA"]

# Pre-computed monitor scores (question → score)
_SCORE_FILE = _Path(__file__).resolve().parent / "hard_questions.json"
_precomputed_scores: dict[str, float] = {}
if _SCORE_FILE.exists():
    for item in _json.loads(_SCORE_FILE.read_text()):
        if "monitor_score" in item:
            _precomputed_scores[item["question"]] = item["monitor_score"]

# Override threshold so hard questions actually trigger multi-round
settings.monitor_similarity_threshold = 0.80


def get_examples(dataset_name: str = "HotpotQA", n: int = 50) -> list[dict]:
    if dataset_name not in _dataset_cache:
        if dataset_name == "2WikiMultiHopQA":
            raw = load_2wikimultihopqa(n=n, seed=42)
        else:
            raw = load_hotpotqa(n=n, seed=42)
        # Sort by real monitor score (lowest = hardest = first)
        raw.sort(key=lambda ex: _precomputed_scores.get(ex["question"], 0.80))
        _dataset_cache[dataset_name] = raw
    return _dataset_cache[dataset_name]


def _build_choices(dataset_name: str) -> list[str]:
    examples = get_examples(dataset_name)
    choices = []
    for i, ex in enumerate(examples):
        choices.append(f"{i}: {ex['question'][:110]}")
    return choices


# ---------------------------------------------------------------------------
# Helpers for panel rendering
# ---------------------------------------------------------------------------
STAGE_PENDING = "⬜"
STAGE_RUNNING = "🔄"
STAGE_DONE = "✅"
STAGE_SKIP = "⏭️"

_EMPTY_MONITORING = "*Waiting…*"
_EMPTY_EVALUATING = "*Waiting…*"
_EMPTY_PLANNING = "*Waiting…*"


def _render_monitoring(status: str, mon=None) -> str:
    header = f"### {status} Monitoring\n"
    if mon is None:
        return header + _EMPTY_MONITORING
    passed = mon.score >= settings.monitor_similarity_threshold
    icon = "✅ PASS" if passed else "❌ FAIL"
    bar_pct = min(int(mon.score * 100), 100)
    bar_color = "#2ecc71" if passed else "#e74c3c"
    bar_html = (
        f'<div style="background:#eee;border-radius:6px;height:18px;width:100%;margin:4px 0">'
        f'<div style="background:{bar_color};height:100%;border-radius:6px;width:{bar_pct}%;'
        f'text-align:center;color:#fff;font-size:12px;line-height:18px">{mon.score:.4f}</div></div>'
    )
    return (
        f"{header}"
        f"**Reference answer:** {mon.reference_answer}\n\n"
        f"**Similarity:** {bar_html}\n\n"
        f"Score `{mon.score:.4f}` / threshold `{settings.monitor_similarity_threshold}` → **{icon}**"
    )


def _render_evaluating(status: str, diag=None) -> str:
    header = f"### {status} Evaluating\n"
    if diag is None:
        return header + _EMPTY_EVALUATING
    lines = [
        header,
        f"| Field | Value |",
        f"|---|---|",
        f"| Category | `{diag.category}` |",
        f"| Internal knowledge | {'✅' if diag.internal_sufficient else '❌'} `{diag.internal_sufficient}` |",
        f"| External knowledge | {'✅' if diag.external_sufficient else '❌'} `{diag.external_sufficient}` |",
    ]
    if diag.error_types:
        lines.append(f"| Error types | {', '.join(f'`{e}`' for e in diag.error_types)} |")
    lines.append(f"\n**Reasoning:** {diag.reasoning}")
    if diag.suggested_query:
        lines.append(f"\n**Suggested query:** {diag.suggested_query}")
    if diag.suggestion:
        lines.append(f"\n**Suggestion:** {diag.suggestion}")
    return "\n".join(lines)


def _render_planning(status: str, directive=None, answer_mode=None) -> str:
    header = f"### {status} Planning\n"
    if directive is None:
        return header + _EMPTY_PLANNING
    return (
        f"{header}"
        f"**Answer mode:** `{answer_mode}`\n\n"
        f"**Writing directive:** {directive}"
    )


def _make_round_header(rnd: int, total_rounds: int) -> str:
    dots = " → ".join(
        f"**R{i}**" if i == rnd else f"R{i}"
        for i in range(total_rounds + 1)
    )
    return f"### Metacognitive Round {rnd}\n{dots}"


# ---------------------------------------------------------------------------
# State tuple order (must match outputs list):
#   0: status_md, 1: docs_md, 2: ground_truth_md, 3: cognition_md,
#   4: round_header_md, 5: monitoring_md, 6: evaluating_md, 7: planning_md,
#   8: answer_md, 9: metrics_md, 10: full_log_md
# ---------------------------------------------------------------------------
N_OUTPUTS = 11


def _pack(
    status="", docs="", ground_truth="", cognition="", round_header="",
    monitoring="", evaluating="", planning="", answer="", metrics="",
    full_log="",
):
    return (status, docs, ground_truth, cognition, round_header, monitoring, evaluating, planning, answer, metrics, full_log)


def _render_docs(doc_list: list[dict]) -> str:
    """Render the retrieved documents panel."""
    lines = [f"**Documents retrieved: {len(doc_list)}**\n"]
    for d in doc_list:
        src = d.get('source', 'unknown')
        score = d.get('score', 0)
        text = d.get('text', '')
        text_preview = text[:400] + ('…' if len(text) > 400 else '')
        lines.append(
            f"<details><summary><b>[{d.get('id', '?')+1}] {src}</b> "
            f"(score: {score:.2f}, {len(text)} chars)</summary>\n\n"
            f"{text_preview}\n\n</details>\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core pipeline — yields tuples for each UI panel
# ---------------------------------------------------------------------------
async def run_metarag_pipeline(example_index: int, dataset_name: str = "HotpotQA"):
    examples = get_examples(dataset_name)
    if example_index is None or int(example_index) < 0 or int(example_index) >= len(examples):
        yield _pack(status="⚠️ Invalid example index.")
        return

    example = examples[int(example_index)]
    question = example["question"]
    reference = example["answer"]
    context_docs = example.get("context", [])

    docs = [
        {"id": i, "text": doc["text"], "source": doc["source"], "score": 1.0}
        for i, doc in enumerate(context_docs)
    ]

    full_log: list[str] = []
    t0 = time.monotonic()

    def _elapsed():
        return f"{time.monotonic() - t0:.1f}s"

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------
    gt_lines = [f"**Gold Answer:** `{reference}`\n"]
    gt_lines.append(f"**Question type:** `{example.get('type', 'unknown')}` · **Level:** `{example.get('level', 'unknown')}`\n")
    gt_lines.append(f"**Supporting documents ({len(context_docs)}):**\n")
    for i, doc in enumerate(context_docs):
        text_preview = doc['text'][:300] + ('…' if len(doc['text']) > 300 else '')
        gt_lines.append(f"<details><summary><b>[{i+1}] {doc['source']}</b> ({len(doc['text'])} chars)</summary>\n\n{text_preview}\n\n</details>\n")
    gt_md = "\n".join(gt_lines)

    # ------------------------------------------------------------------
    # Documents panel
    # ------------------------------------------------------------------
    docs_md = _render_docs(docs)

    # ------------------------------------------------------------------
    # Cognition
    # ------------------------------------------------------------------
    cog_md = (
        f"**Question:** {question}\n\n"
        f"**Documents:** {len(docs)}\n\n"
        f"*Generating initial answer…*"
    )
    yield _pack(
        status=f"🔄 **Cognition** — generating initial answer… ({_elapsed()})",
        docs=docs_md,
        ground_truth=gt_md,
        cognition=cog_md,
    )

    answer = await write_answer(question, docs)

    cog_md = (
        f"**Question:** {question}\n\n"
        f"**Documents:** {len(docs)}\n\n"
        f"### Initial Answer\n{answer}"
    )
    full_log.append(f"## Cognition\n{cog_md}")

    yield _pack(
        status=f"✅ **Cognition** done ({_elapsed()})",
        docs=docs_md,
        ground_truth=gt_md,
        cognition=cog_md,
    )

    # ------------------------------------------------------------------
    # Metacognitive loop
    # ------------------------------------------------------------------
    metacognitive_round = 0
    previous_answer = ""

    for rnd in range(settings.max_metacognitive_rounds):
        round_hdr = _make_round_header(rnd + 1, settings.max_metacognitive_rounds)

        # ---- Monitoring: running ----
        yield _pack(
            status=f"🔄 **Round {rnd + 1} · Monitoring** — computing similarity… ({_elapsed()})",
            docs=docs_md,
            ground_truth=gt_md,
            cognition=cog_md,
            round_header=round_hdr,
            monitoring=_render_monitoring(STAGE_RUNNING),
            evaluating=_render_evaluating(STAGE_PENDING),
            planning=_render_planning(STAGE_PENDING),
        )

        mon = await monitor_answer(question, answer, docs)
        mon_md = _render_monitoring(STAGE_DONE, mon)
        full_log.append(f"### Round {rnd+1} — Monitoring\n{mon_md}")

        # If monitoring passes, skip evaluating & planning entirely
        if mon.score >= settings.monitor_similarity_threshold:
            eval_md = f"### {STAGE_SKIP} Evaluating\n*Skipped — monitoring passed (score ≥ threshold).*"
            plan_md = f"### {STAGE_SKIP} Planning\n*Skipped — answer is satisfactory.*"
            yield _pack(
                status=f"✅ **Satisfactory** — monitoring passed in round {rnd + 1} ({_elapsed()})",
                docs=docs_md,
                ground_truth=gt_md,
                cognition=cog_md,
                round_header=round_hdr,
                monitoring=mon_md,
                evaluating=eval_md,
                planning=plan_md,
            )
            full_log.append(f"### Round {rnd+1} — Evaluating\nSkipped — monitoring passed.")
            full_log.append(f"### Round {rnd+1} — Planning\nSkipped — answer is satisfactory.")
            break

        # ---- Monitoring failed → Evaluating: running ----
        yield _pack(
            status=f"🔄 **Round {rnd + 1} · Evaluating** — running diagnosis… ({_elapsed()})",
            docs=docs_md,
            ground_truth=gt_md,
            cognition=cog_md,
            round_header=round_hdr,
            monitoring=mon_md,
            evaluating=_render_evaluating(STAGE_RUNNING),
            planning=_render_planning(STAGE_PENDING),
        )

        diag = await diagnose_answer(
            query=question,
            answer=answer,
            docs=docs,
            monitor_score=mon.score,
            reference_answer=mon.reference_answer,
        )
        eval_md = _render_evaluating(STAGE_DONE, diag)
        full_log.append(f"### Round {rnd+1} — Evaluating\n{eval_md}")

        # ---- Check convergence ----
        if check_convergence(previous_answer, answer):
            plan_md = _render_planning(STAGE_SKIP, "N/A — convergence detected", "—")
            yield _pack(
                status=f"🔄 **Converged** after round {rnd + 1} ({_elapsed()})",
                docs=docs_md,
                ground_truth=gt_md,
                cognition=cog_md,
                round_header=round_hdr,
                monitoring=mon_md,
                evaluating=eval_md,
                planning=plan_md,
            )
            full_log.append(f"### Round {rnd+1} — Planning\nConvergence — loop stopped.")
            break

        # ---- Planning ----
        directive = build_writing_directive(diag)
        answer_mode = answer_mode_for_diagnosis(diag)
        plan_md = _render_planning(STAGE_DONE, directive, answer_mode)
        full_log.append(f"### Round {rnd+1} — Planning\n{plan_md}")

        yield _pack(
            status=f"🔄 **Round {rnd + 1} · Remediation** — re-generating answer… ({_elapsed()})",
            docs=docs_md,
            ground_truth=gt_md,
            cognition=cog_md,
            round_header=round_hdr,
            monitoring=mon_md,
            evaluating=eval_md,
            planning=plan_md,
        )

        # ---- Remediation ----
        previous_answer = answer
        answer = await write_answer(
            question, docs,
            writing_directive=directive,
            answer_mode=answer_mode,
        )
        metacognitive_round += 1

        ans_md = f"### Answer after round {metacognitive_round}\n{answer}"
        full_log.append(f"### Round {rnd+1} — Remediation\n{ans_md}")

        yield _pack(
            status=f"✅ **Round {rnd + 1}** complete ({_elapsed()})",
            docs=docs_md,
            ground_truth=gt_md,
            cognition=cog_md,
            round_header=round_hdr,
            monitoring=mon_md,
            evaluating=eval_md,
            planning=plan_md,
            answer=ans_md,
        )

    # ------------------------------------------------------------------
    # Final metrics
    # ------------------------------------------------------------------
    short = await _extract_short_answer(answer, question)
    m = compute_metrics(short, reference)
    metrics_md = (
        f"### 🤖 Prediction\n\n"
        f"**Short answer:** {short}\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Exact Match | `{m['exact_match']:.2f}` |\n"
        f"| F1 | `{m['f1']:.4f}` |\n"
        f"| Precision | `{m['precision']:.4f}` |\n"
        f"| Recall | `{m['recall']:.4f}` |\n"
        f"| Rounds | `{metacognitive_round}` |\n"
        f"| Time | `{_elapsed()}` |"
    )
    # Ground truth panel now shows alongside prediction for comparison
    gt_compare_md = (
        f"### 🎯 Ground Truth\n\n"
        f"**Gold answer:** `{reference}`\n\n"
        f"**Question type:** `{example.get('type', 'unknown')}` · "
        f"**Level:** `{example.get('level', 'unknown')}`\n\n"
        f"**Supporting documents ({len(context_docs)}):**\n\n"
    )
    for i, doc in enumerate(context_docs):
        text_preview = doc['text'][:300] + ('…' if len(doc['text']) > 300 else '')
        gt_compare_md += (
            f"<details><summary><b>[{i+1}] {doc['source']}</b> ({len(doc['text'])} chars)</summary>\n\n"
            f"{text_preview}\n\n</details>\n\n"
        )
    full_log.append(f"## Final Metrics\n{metrics_md}\n\n{gt_compare_md}")

    yield _pack(
        status=f"🏁 **Done** — {metacognitive_round} metacognitive round(s) in {_elapsed()}",
        docs=docs_md,
        ground_truth=gt_compare_md,
        cognition=cog_md,
        round_header=_make_round_header(metacognitive_round, metacognitive_round),
        monitoring=mon_md,
        evaluating=eval_md,
        planning=plan_md,
        answer=f"### Final Answer\n{answer}",
        metrics=metrics_md,
        full_log="\n\n---\n\n".join(full_log),
    )


# ---------------------------------------------------------------------------
# Synchronous wrapper — drives the async generator for Gradio streaming
# ---------------------------------------------------------------------------
def run_pipeline_sync(dataset_name, example_index):
    loop = asyncio.new_event_loop()
    gen = run_metarag_pipeline(example_index, dataset_name=dataset_name or "HotpotQA")
    try:
        while True:
            result = loop.run_until_complete(gen.__anext__())
            yield result
    except StopAsyncIteration:
        pass
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
CSS = """
.stage-panel { border: 1px solid #ddd; border-radius: 8px; padding: 12px; min-height: 120px; }
.status-bar { font-size: 1.1em; padding: 8px 14px; border-radius: 6px; background: #f0f4ff; }
"""


def _on_dataset_change(dataset_name):
    choices = _build_choices(dataset_name)
    return gr.update(choices=choices, value=choices[0] if choices else None)


def build_demo():
    # Pre-load default dataset
    default_dataset = "HotpotQA"
    choices = _build_choices(default_dataset)

    with gr.Blocks(title="MetaRAG Demo", css=CSS) as demo:
        gr.Markdown(
            "# 🧠 MetaRAG — Metacognitive RAG Demo\n"
            "Select a dataset and question, then watch the metacognitive loop run **in real time**.\n\n"
            "Pipeline: **Cognition → Monitoring → Evaluating → Planning → Remediation** (repeat until satisfactory)"
        )

        with gr.Row():
            dataset_dropdown = gr.Dropdown(
                choices=DATASET_CHOICES,
                label="Dataset",
                value=default_dataset,
                scale=1,
            )
            question_dropdown = gr.Dropdown(
                choices=choices,
                label="Select question",
                type="index",
                value=choices[0] if choices else None,
                scale=4,
            )
            run_btn = gr.Button("▶  Run MetaRAG", variant="primary", scale=1)

        dataset_dropdown.change(
            fn=_on_dataset_change,
            inputs=[dataset_dropdown],
            outputs=[question_dropdown],
        )

        # Live status bar
        status_md = gr.Markdown("*Ready — pick a question and press Run.*", elem_classes=["status-bar"])

        # Retrieved documents panel
        with gr.Accordion("📄 Retrieved Documents", open=False):
            docs_panel_md = gr.Markdown("")

        # Cognition panel
        with gr.Accordion("📝 Cognition (Initial Answer)", open=True):
            cognition_md = gr.Markdown("")

        # Metacognitive round panels
        round_header_md = gr.Markdown("")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=300):
                with gr.Group(elem_classes=["stage-panel"]):
                    monitoring_md = gr.Markdown(f"### {STAGE_PENDING} Monitoring\n*Waiting…*")
            with gr.Column(scale=1, min_width=300):
                with gr.Group(elem_classes=["stage-panel"]):
                    evaluating_md = gr.Markdown(f"### {STAGE_PENDING} Evaluating\n*Waiting…*")
            with gr.Column(scale=1, min_width=300):
                with gr.Group(elem_classes=["stage-panel"]):
                    planning_md = gr.Markdown(f"### {STAGE_PENDING} Planning\n*Waiting…*")

        # Revised answer
        with gr.Accordion("✍️ Revised Answer", open=True):
            answer_md = gr.Markdown("")

        # Prediction vs Ground Truth comparison
        with gr.Accordion("📊 Final Metrics & Comparison", open=True):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    metrics_md = gr.Markdown("")
                with gr.Column(scale=1):
                    ground_truth_md = gr.Markdown("")

        # Full log (collapsed by default)
        with gr.Accordion("📜 Full Log", open=False):
            full_log_md = gr.Markdown("")

        gr.Markdown(
            "---\n"
            f"**Config:** model=`{settings.deepseek_model}` · "
            f"monitor_threshold=`{settings.monitor_similarity_threshold}` · "
            f"max_rounds=`{settings.max_metacognitive_rounds}` · "
            f"embedding=`{settings.embedding_model}`"
        )

        run_btn.click(
            fn=run_pipeline_sync,
            inputs=[dataset_dropdown, question_dropdown],
            outputs=[
                status_md, docs_panel_md, ground_truth_md, cognition_md,
                round_header_md, monitoring_md, evaluating_md, planning_md,
                answer_md, metrics_md, full_log_md,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
