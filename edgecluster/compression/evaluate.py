"""Evaluate an exported ONNX classifier on a JSONL split."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from edgecluster.compression.metrics import accuracy, macro_f1
from edgecluster.data.schema import Example, load_many
from edgecluster.serving.classifier import OnnxClassifier


def evaluate_onnx(clf: OnnxClassifier, examples: Sequence[Example], batch_size: int = 32) -> dict[str, Any]:
    """Accuracy / macro-F1 overall and per language."""
    preds: list[str] = []
    for i in range(0, len(examples), batch_size):
        chunk = [ex.text for ex in examples[i : i + batch_size]]
        preds += [p.label for p in clf.predict(chunk)]
    gold = [ex.label for ex in examples]
    per_lang: dict[str, list[int]] = defaultdict(list)
    for idx, ex in enumerate(examples):
        per_lang[ex.lang].append(idx)
    return {
        "accuracy": accuracy(gold, preds),
        "macro_f1": macro_f1(gold, preds),
        "n": len(gold),
        "per_lang": {
            lang: {"accuracy": accuracy([gold[i] for i in ids], [preds[i] for i in ids]), "n": len(ids)}
            for lang, ids in sorted(per_lang.items())
        },
    }


def local_latency(clf: OnnxClassifier, text: str, runs: int = 30, warmup: int = 5) -> dict[str, float]:
    """Quick batch-1 latency on *this* machine (not a Pi benchmark; see bench/)."""
    feeds = clf.encode([text])
    for _ in range(warmup):
        clf.run(feeds)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        clf.run(feeds)
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-file", default="model.onnx")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--max-seq-len", type=int, default=None)
    args = ap.parse_args()
    clf = OnnxClassifier(args.model_dir, args.model_file, args.max_seq_len)
    print(json.dumps(evaluate_onnx(clf, load_many(args.data)), indent=2))


if __name__ == "__main__":
    main()
