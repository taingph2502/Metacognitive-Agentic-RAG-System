"""
Comparison report generator for dual-provider benchmark results.

Accepts two result summary JSON files and produces a side-by-side Markdown
report with per-metric comparison and architecture contribution analysis.

Usage:
    python -m benchmarks.compare \
        benchmarks/results/evaluations/hotpotqa_gold_deepseek_..._summary.json \
        benchmarks/results/evaluations/hotpotqa_gold_gemini_..._summary.json \
        -o benchmarks/results/reports/comparison.md
"""

import argparse
import json
import sys
from pathlib import Path


# Paper Table 1 Standard RAG baselines (no metacognitive architecture)
STANDARD_RAG_BASELINES = {
    "hotpotqa": {"EM": 24.6, "F1": 33.0, "Prec": 34.1, "Rec": 34.5},
    "2wikimultihopqa": {"EM": 18.8, "F1": 25.2, "Prec": 25.6, "Rec": 26.2},
}


def load_summary(path: str | Path) -> dict:
    """Load a benchmark summary JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(val: float, pct: bool = True) -> str:
    """Format a metric value as percentage string."""
    if pct:
        return f"{val * 100:.1f}" if val <= 1.0 else f"{val:.1f}"
    return f"{val:.4f}"


def generate_comparison(summary_a: dict, summary_b: dict) -> str:
    """Generate Markdown comparison report from two summary dicts."""
    lines: list[str] = []

    provider_a = summary_a.get("provider", summary_a.get("llm_provider", "A"))
    provider_b = summary_b.get("provider", summary_b.get("llm_provider", "B"))
    dataset = summary_a.get("dataset", summary_b.get("dataset", "unknown"))
    mode = summary_a.get("mode", summary_b.get("mode", "unknown"))

    ma = summary_a["metrics"]
    mb = summary_b["metrics"]

    lines.append(f"# Benchmark Comparison: {provider_a} vs {provider_b}")
    lines.append(f"")
    lines.append(f"**Dataset:** {dataset} | **Mode:** {mode}")
    lines.append(f"**Samples:** {ma.get('num_examples', 'N/A')} ({provider_a}), {mb.get('num_examples', 'N/A')} ({provider_b})")
    lines.append(f"")

    # Side-by-side metrics table
    lines.append(f"## Performance Comparison")
    lines.append(f"")
    lines.append(f"| Metric | {provider_a} | {provider_b} | Delta |")
    lines.append(f"|--------|{'---:|' * 3}")

    for metric_key, label in [
        ("exact_match", "Exact Match"),
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]:
        va = ma.get(metric_key, 0)
        vb = mb.get(metric_key, 0)
        delta = (va - vb) * 100 if va <= 1.0 else va - vb
        va_s = _fmt(va)
        vb_s = _fmt(vb)
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {label} | {va_s} | {vb_s} | {sign}{delta:.1f} |")

    lines.append(f"")

    # Cost comparison
    cost_a = summary_a.get("cost", {})
    cost_b = summary_b.get("cost", {})
    if cost_a or cost_b:
        lines.append(f"## Cost Comparison")
        lines.append(f"")
        lines.append(f"| Metric | {provider_a} | {provider_b} |")
        lines.append(f"|--------|---:|---:|")
        lines.append(f"| Total calls | {cost_a.get('total_calls', 'N/A')} | {cost_b.get('total_calls', 'N/A')} |")
        lines.append(f"| Total tokens | {cost_a.get('total_tokens', 'N/A')} | {cost_b.get('total_tokens', 'N/A')} |")
        lines.append(f"| Total cost (USD) | ${cost_a.get('total_cost_usd', 0):.4f} | ${cost_b.get('total_cost_usd', 0):.4f} |")
        lines.append(f"")

    # Timing comparison
    time_a = summary_a.get("total_time_seconds", 0)
    time_b = summary_b.get("total_time_seconds", 0)
    if time_a or time_b:
        lines.append(f"## Timing")
        lines.append(f"")
        lines.append(f"| | {provider_a} | {provider_b} |")
        lines.append(f"|--|---:|---:|")
        lines.append(f"| Total time (s) | {time_a:.1f} | {time_b:.1f} |")
        lines.append(f"")

    # Architecture contribution analysis
    baseline = STANDARD_RAG_BASELINES.get(dataset)
    if baseline:
        lines.append(f"## Architecture Contribution Analysis")
        lines.append(f"")
        lines.append(f"Decomposing improvement into **model contribution** (swapping LLM) vs **architecture contribution** (metacognitive pipeline).")
        lines.append(f"")
        lines.append(f"| Configuration | EM | F1 | Prec | Rec |")
        lines.append(f"|--------------|---:|---:|---:|---:|")

        # Row 1: Standard RAG baseline from paper (no metacognitive, Gemini-class)
        lines.append(
            f"| Standard RAG (paper baseline) | {baseline['EM']:.1f} | {baseline['F1']:.1f} | {baseline['Prec']:.1f} | {baseline['Rec']:.1f} |"
        )

        # Determine which summary is gemini/deepseek
        gemini_m = ma if provider_a.lower() == "gemini" else mb if provider_b.lower() == "gemini" else None
        deepseek_m = ma if provider_a.lower() == "deepseek" else mb if provider_b.lower() == "deepseek" else None

        if gemini_m:
            lines.append(
                f"| Gemini + Meta-RAG (ours) | {_fmt(gemini_m['exact_match'])} | {_fmt(gemini_m['f1'])} | {_fmt(gemini_m['precision'])} | {_fmt(gemini_m['recall'])} |"
            )
        if deepseek_m:
            lines.append(
                f"| DeepSeek + Meta-RAG (ours) | {_fmt(deepseek_m['exact_match'])} | {_fmt(deepseek_m['f1'])} | {_fmt(deepseek_m['precision'])} | {_fmt(deepseek_m['recall'])} |"
            )

        lines.append(f"")

        # Delta rows
        if gemini_m:
            gem_em = gemini_m["exact_match"] * 100 if gemini_m["exact_match"] <= 1.0 else gemini_m["exact_match"]
            gem_f1 = gemini_m["f1"] * 100 if gemini_m["f1"] <= 1.0 else gemini_m["f1"]
            arch_em = gem_em - baseline["EM"]
            arch_f1 = gem_f1 - baseline["F1"]
            lines.append(f"**Architecture contribution** (Gemini + metacognitive vs Standard RAG): EM +{arch_em:.1f}, F1 +{arch_f1:.1f}")

        if gemini_m and deepseek_m:
            gem_em = gemini_m["exact_match"] * 100 if gemini_m["exact_match"] <= 1.0 else gemini_m["exact_match"]
            ds_em = deepseek_m["exact_match"] * 100 if deepseek_m["exact_match"] <= 1.0 else deepseek_m["exact_match"]
            gem_f1 = gemini_m["f1"] * 100 if gemini_m["f1"] <= 1.0 else gemini_m["f1"]
            ds_f1 = deepseek_m["f1"] * 100 if deepseek_m["f1"] <= 1.0 else deepseek_m["f1"]
            model_em = ds_em - gem_em
            model_f1 = ds_f1 - gem_f1
            sign_em = "+" if model_em >= 0 else ""
            sign_f1 = "+" if model_f1 >= 0 else ""
            lines.append(f"")
            lines.append(f"**Model contribution** (DeepSeek vs Gemini, same architecture): EM {sign_em}{model_em:.1f}, F1 {sign_f1}{model_f1:.1f}")

    lines.append(f"")
    lines.append(f"---")
    lines.append(f"*Generated by `benchmarks/compare.py`*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare two benchmark result summaries")
    parser.add_argument("file_a", help="First summary JSON file")
    parser.add_argument("file_b", help="Second summary JSON file")
    parser.add_argument("-o", "--output", default=None, help="Output Markdown file (default: print to stdout)")
    args = parser.parse_args()

    summary_a = load_summary(args.file_a)
    summary_b = load_summary(args.file_b)
    report = generate_comparison(summary_a, summary_b)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Comparison report saved to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
