"""
Dataset loaders for the paper's benchmark evaluation (§5.1).

Datasets:
  - HotpotQA (validation set, 500 examples)
  - 2WikiMultiHopQA (validation set, 500 examples)

Both are multi-hop QA datasets built on Wikipedia.

Usage:
    from benchmarks.datasets import load_hotpotqa, load_2wikimultihopqa

    examples = load_hotpotqa(n=500, seed=42)
    # Each example: {"id": str, "question": str, "answer": str,
    #                "context": list[dict], "type": str}
"""

import json
import random
from pathlib import Path
from typing import Any


def _download_dataset(name: str, split: str = "validation") -> list[dict]:
    """Download dataset via HuggingFace datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install the `datasets` package to use benchmarks:\n"
            "  uv add datasets --group dev"
        )

    if name == "hotpot_qa":
        ds = load_dataset("hotpot_qa", "distractor", split=split)
        return list(ds)
    elif name == "2wikimultihopqa":
        # Load from parquet files directly (dataset script is deprecated)
        parquet_map = {"validation": "dev.parquet", "train": "train.parquet", "test": "test.parquet"}
        fname = parquet_map.get(split, "dev.parquet")
        ds = load_dataset("parquet", data_files=f"hf://datasets/xanhho/2WikiMultihopQA/{fname}", split="train")
        return list(ds)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def _format_hotpotqa(raw: dict) -> dict[str, Any]:
    """Convert a HotpotQA example to our standard format."""
    # Build context from the provided supporting documents
    context = []
    titles = raw.get("context", {}).get("title", [])
    sentences_list = raw.get("context", {}).get("sentences", [])
    for title, sentences in zip(titles, sentences_list):
        text = " ".join(sentences)
        context.append({"source": title, "text": text})

    return {
        "id": raw.get("id", ""),
        "question": raw["question"],
        "answer": raw["answer"],
        "context": context,
        "type": raw.get("type", "unknown"),
        "level": raw.get("level", "unknown"),
    }


def _format_2wikimultihopqa(raw: dict) -> dict[str, Any]:
    """Convert a 2WikiMultiHopQA example to our standard format."""
    context = []
    raw_context = raw.get("context", [])

    # xanhho/2WikiMultihopQA stores context as a JSON string
    if isinstance(raw_context, str):
        try:
            raw_context = json.loads(raw_context)
        except (json.JSONDecodeError, TypeError):
            raw_context = []

    if isinstance(raw_context, dict):
        titles = raw_context.get("title", [])
        sentences_list = raw_context.get("content", [])
        for title, sentences in zip(titles, sentences_list):
            text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
            context.append({"source": title, "text": text})
    elif isinstance(raw_context, list):
        for item in raw_context:
            if isinstance(item, list) and len(item) >= 2:
                title = item[0]
                sentences = item[1] if isinstance(item[1], list) else [item[1]]
                text = " ".join(str(s) for s in sentences)
                context.append({"source": title, "text": text})
            elif isinstance(item, dict):
                context.append({
                    "source": item.get("title", item.get("source", "unknown")),
                    "text": item.get("text", item.get("content", "")),
                })

    return {
        "id": raw.get("_id", raw.get("id", "")),
        "question": raw["question"],
        "answer": raw["answer"],
        "context": context,
        "type": raw.get("type", "unknown"),
    }


def load_hotpotqa(
    n: int = 500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Load HotpotQA validation examples.

    Paper §5.1: "we sub-sample 500 questions from the validation set"
    """
    raw_data = _download_dataset("hotpot_qa", split="validation")
    formatted = [_format_hotpotqa(r) for r in raw_data]

    # Sub-sample as per paper
    if n and n < len(formatted):
        rng = random.Random(seed)
        formatted = rng.sample(formatted, n)

    return formatted


def load_2wikimultihopqa(
    n: int = 500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Load 2WikiMultiHopQA validation examples.

    Paper §5.1: "we sub-sample 500 questions from the validation set"
    """
    raw_data = _download_dataset("2wikimultihopqa", split="validation")
    formatted = [_format_2wikimultihopqa(r) for r in raw_data]

    if n and n < len(formatted):
        rng = random.Random(seed)
        formatted = rng.sample(formatted, n)

    return formatted


def save_results(
    results: list[dict],
    output_path: str | Path,
) -> None:
    """Save evaluation results to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
