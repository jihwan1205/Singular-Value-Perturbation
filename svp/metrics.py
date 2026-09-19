"""Correctness and pass@k metrics."""

from __future__ import annotations

from collections import Counter
from math import comb

_TOL = 1e-6


def _decimals(gold: float) -> int:
    text = f"{gold:.10g}"
    if "e" in text.lower() or "." not in text:
        return 0
    return len(text.split(".")[1])


def is_correct(pred: float | None, gold: float) -> bool:
    """``|pred - gold| < 1e-6``; non-integer golds also match after rounding
    ``pred`` to the gold's own decimal count (GSM-Plus stores rounded golds)."""
    if pred is None:
        return False
    if abs(pred - gold) < _TOL:
        return True
    d = _decimals(gold)
    return d > 0 and abs(round(pred, d) - gold) < _TOL


def pass_at_k(corrects: list[list[bool]], k: int) -> float:
    """Unbiased pass@k estimator averaged over problems."""
    values = []
    for flags in corrects:
        n, c = len(flags), sum(flags)
        if k > n:
            values.append(1.0 if c > 0 else 0.0)
        elif c == 0:
            values.append(0.0)
        else:
            values.append(1.0 - comb(n - c, k) / comb(n, k))
    return sum(values) / len(values) if values else 0.0


def pass_at_k_curve(corrects: list[list[bool]], n: int) -> dict[str, float]:
    ks, k = [], 1
    while k <= n:
        ks.append(k)
        k *= 2
    return {f"pass@{k}": pass_at_k(corrects, k) for k in ks}


def coverage(corrects: list[list[bool]]) -> float:
    return sum(any(f) for f in corrects) / len(corrects) if corrects else 0.0


def sample_accuracy(corrects: list[list[bool]]) -> float:
    total = sum(len(f) for f in corrects)
    return sum(sum(f) for f in corrects) / total if total else 0.0


def majority_vote_accuracy(answers: list[list[float | None]], golds: list[float]) -> float:
    if not answers:
        return 0.0
    hits = sum(is_correct(Counter(row).most_common(1)[0][0], gold) for row, gold in zip(answers, golds, strict=True))
    return hits / len(answers)
