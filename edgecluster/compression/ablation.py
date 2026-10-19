"""Quantization ablation on an existing pipeline run (no retraining).

    python -m edgecluster.compression.ablation --run-dir runs/xlmr-bbca

Re-quantizes ``onnx/model.pruned.onnx`` with different settings, then
evaluates size and test accuracy of each:

* per-tensor vs per-channel weight scales
* quantizing ``MatMul`` only (fp32 embeddings) vs ``MatMul`` + ``Gather``

Writes ``ablation.json`` / ``ablation.md`` into the run directory. Variant
files are written to ``onnx/ablation/`` and can be served directly.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from edgecluster.compression.evaluate import evaluate_onnx, local_latency
from edgecluster.compression.quantize import quantize_int8
from edgecluster.compression.report import file_mb
from edgecluster.data.schema import load_many
from edgecluster.serving.classifier import OnnxClassifier

SETTINGS = (
    ("per-tensor, MatMul+Gather", {"per_channel": False, "op_types": ("MatMul", "Gather")}),
    ("per-channel, MatMul+Gather", {"per_channel": True, "op_types": ("MatMul", "Gather")}),
    ("per-tensor, MatMul only", {"per_channel": False, "op_types": ("MatMul",)}),
    ("per-channel, MatMul only", {"per_channel": True, "op_types": ("MatMul",)}),
)


def run_ablation(run_dir: str | Path, source: str = "model.pruned.onnx") -> dict[str, Any]:
    """Quantize ``source`` with every setting and evaluate it."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    onnx_dir = run_dir / "onnx"
    out_dir = onnx_dir / "ablation"
    out_dir.mkdir(exist_ok=True)
    for f in ("tokenizer.json", "meta.json"):
        shutil.copy(onnx_dir / f, out_dir / f)
    test = load_many(run["test_files"])
    max_len = int(run["max_seq_len"])
    rows = []
    for i, (name, kw) in enumerate(SETTINGS):
        fname = f"{Path(source).stem}.q{i}.onnx"
        quantize_int8(onnx_dir / source, out_dir / fname, **kw)
        clf = OnnxClassifier(out_dir, fname, max_len, pad_to_max=True)
        ev = evaluate_onnx(clf, test)
        lat = local_latency(clf, test[0].text)
        rows.append(
            {
                "setting": name,
                "file": f"ablation/{fname}",
                "size_mb": round(file_mb(out_dir / fname), 1),
                "accuracy": ev["accuracy"],
                "macro_f1": ev["macro_f1"],
                "local_p50_ms": round(lat["p50_ms"], 1),
            }
        )
    return {"run": run["name"], "source": source, "n_test": len(test), "rows": rows}


def to_markdown(res: dict[str, Any]) -> str:
    """Render the ablation table."""
    lines = [
        f"### INT8 ablation on {res['run']} ({res['source']}, test n={res['n_test']})",
        "",
        "| setting | size (MB) | test accuracy | macro-F1 | local p50 (ms) |",
        "|---|---|---|---|---|",
    ]
    for r in res["rows"]:
        lines.append(
            f"| {r['setting']} | {r['size_mb']:.1f} | {100 * r['accuracy']:.2f}% | {r['macro_f1']:.4f} | {r['local_p50_ms']:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--source", default="model.pruned.onnx")
    args = ap.parse_args()
    res = run_ablation(args.run_dir, args.source)
    (Path(args.run_dir) / "ablation.json").write_text(json.dumps(res, indent=2))
    md = to_markdown(res)
    (Path(args.run_dir) / "ablation.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
