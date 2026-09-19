"""Benchmark loading. Files are JSON lists of ``{"question", "answer", ...}``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

BENCHMARKS = {
    "gsm8k": "gsm8k.json",
    "gsm_hard": "gsm_hard.json",
    "multiarith": "multiarith.json",
    "svamp": "svamp.json",
    "asdiv_a": "asdiv_a.json",
    "gsm_plus": "gsm_plus.json",
}


@dataclass
class Example:
    idx: int
    question: str
    gold: float


def parse_gold(answer) -> float | None:
    try:
        return float(str(answer).replace(",", ""))
    except ValueError:
        return None


def load_examples(path: str | Path, max_examples: int | None = None) -> list[Example]:
    """Load a JSON benchmark file; rows whose gold does not parse are skipped."""
    with open(path) as f:
        raw = json.load(f)
    examples = []
    for i, item in enumerate(raw):
        gold = parse_gold(item["answer"])
        if gold is not None:
            examples.append(Example(idx=i, question=item["question"], gold=gold))
    return examples[:max_examples] if max_examples is not None else examples


def benchmark_path(name: str) -> Path:
    """A benchmark name from :data:`BENCHMARKS`, or a path to a JSON file."""
    if name in BENCHMARKS:
        return DATA_DIR / BENCHMARKS[name]
    path = Path(name)
    if not path.exists():
        raise FileNotFoundError(f"{name!r} is neither a benchmark name {list(BENCHMARKS)} nor a file")
    return path
