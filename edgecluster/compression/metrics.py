"""Classification metrics without a scikit-learn dependency."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    """Fraction of exact matches."""
    if len(gold) != len(pred):
        raise ValueError("length mismatch")
    return sum(g == p for g, p in zip(gold, pred, strict=True)) / max(len(gold), 1)


def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    """Unweighted mean F1 over the labels present in ``gold``."""
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    for g, p in zip(gold, pred, strict=True):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    scores = []
    for label in set(gold):
        prec = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        rec = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(scores) / max(len(scores), 1)
