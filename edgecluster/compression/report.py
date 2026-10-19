"""Size / accuracy report for fp32 vs pruned vs pruned+INT8 ONNX variants.

    python -m edgecluster.compression.report --run-dir runs/xlmr-bbca

Reads ``<run-dir>/onnx/*.onnx`` and the test split recorded in
``<run-dir>/run.json``, evaluates every variant with ONNX Runtime, and writes
``report.json`` and ``report.md``. Every number in the output is measured
by this script. Nothing is copied from elsewhere.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import onnxruntime as ort

from edgecluster.compression.evaluate import evaluate_onnx, local_latency
from edgecluster.data.schema import load_many
from edgecluster.serving.classifier import OnnxClassifier

VARIANTS = (
    ("fp32", "model.fp32.onnx"),
    ("fp32+int8", "model.fp32.int8.onnx"),
    ("pruned", "model.pruned.onnx"),
    ("pruned+int8", "model.pruned.int8.onnx"),
)


def file_mb(path: Path) -> float:
    """Size on disk in MB (10^6 bytes), including external-data files if any."""
    total = path.stat().st_size
    for extra in path.parent.glob(path.name + ".data"):
        total += extra.stat().st_size
    return total / 1e6


def build_report(run_dir: str | Path, latency_runs: int = 30) -> dict[str, Any]:
    """Evaluate all variants present in ``run_dir/onnx``."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    onnx_dir = run_dir / "onnx"
    test = load_many(run["test_files"])
    max_len = int(run["max_seq_len"])
    probe = test[0].text
    rows = []
    for name, fname in VARIANTS:
        path = onnx_dir / fname
        if not path.exists():
            continue
        clf = OnnxClassifier(onnx_dir, fname, max_len, pad_to_max=True)
        ev = evaluate_onnx(clf, test)
        lat = local_latency(clf, probe, runs=latency_runs)
        rows.append(
            {
                "variant": name,
                "file": fname,
                "size_mb": round(file_mb(path), 1),
                "accuracy": ev["accuracy"],
                "macro_f1": ev["macro_f1"],
                "n_test": ev["n"],
                "per_lang": ev["per_lang"],
                "local_p50_ms": round(lat["p50_ms"], 1),
                "local_p95_ms": round(lat["p95_ms"], 1),
            }
        )
    base = next((r for r in rows if r["variant"] == "fp32"), None)
    for r in rows:
        if base:
            r["size_ratio_vs_fp32"] = round(base["size_mb"] / r["size_mb"], 2)
            r["accuracy_retained_pct"] = round(100 * r["accuracy"] / base["accuracy"], 1) if base["accuracy"] else None
    return {
        "model": run["base_model"],
        "dataset": run["dataset"],
        "max_seq_len": max_len,
        "pruning": run.get("pruning", {}),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor() or platform.machine(),
            "system": f"{platform.system()} {platform.release()}",
            "onnxruntime": ort.__version__,
        },
        "rows": rows,
    }


def to_markdown(rep: dict[str, Any]) -> str:
    """Render a report as a markdown table."""
    p = rep.get("pruning", {})
    lines = [
        f"### {rep['model']} on {rep['dataset']} (max_seq_len={rep['max_seq_len']})",
        "",
        f"Pruning: {p.get('heads_pruned', 0)} of {p.get('heads_total', '?')} attention heads removed "
        f"(ratio {p.get('head_ratio', 0)}, via {p.get('head_mechanism', 'n/a')}), FFN ratio {p.get('ffn_ratio', 0)}, "
        f"recovery fine-tune {p.get('recovery_epochs', 0)} epoch(s). "
        f"Measured on {rep['hardware']['system']} / {rep['hardware']['machine']}, onnxruntime {rep['hardware']['onnxruntime']}.",
        "",
        "| variant | size on disk (MB) | compression vs fp32 | test accuracy | accuracy retained | macro-F1 | local p50 / p95 (ms, bs=1, padded) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rep["rows"]:
        lines.append(
            f"| {r['variant']} | {r['size_mb']:.1f} | {r.get('size_ratio_vs_fp32', 1):.2f}x | {100 * r['accuracy']:.2f}% | "
            f"{r.get('accuracy_retained_pct', 100):.1f}% | {r['macro_f1']:.4f} | {r['local_p50_ms']:.1f} / {r['local_p95_ms']:.1f} |"
        )
    lines.append("")
    lines.append(
        "Local latency is measured on the machine that ran the report, *not* on a Raspberry Pi. "
        "Use `make bench` on the Pi for edge numbers."
    )
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--latency-runs", type=int, default=30)
    args = ap.parse_args()
    rep = build_report(args.run_dir, args.latency_runs)
    out = Path(args.run_dir)
    (out / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    md = to_markdown(rep)
    (out / "report.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
