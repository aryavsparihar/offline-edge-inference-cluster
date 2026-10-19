"""End-to-end compression pipeline driven by a YAML config.

    python -m edgecluster.compression.pipeline --config configs/compress/xlmr-bbca.yaml

Stages (each writes into ``out_dir``):

1. data: load JSONL splits (or fetch the public IndicGLUE config first)
2. fine-tune the base model (reused if ``finetuned/`` already exists)
3. export fp32 ONNX, then INT8 of it (reference point)
4. head + FFN importance on the validation split, structured pruning
5. short recovery fine-tune of the pruned model
6. export the pruned ONNX model, then INT8 of it
7. report: size on disk and test accuracy of every variant
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from edgecluster.compression.data import pick_device
from edgecluster.compression.export import export_onnx, write_runtime_files
from edgecluster.compression.finetune import TrainConfig, evaluate_torch, load_model, train
from edgecluster.compression.pruning import (
    PruneSpec,
    compute_importance,
    count_parameters,
    head_counts,
    prune_ffn,
    prune_heads,
    save_pruned,
    select_heads,
)
from edgecluster.compression.quantize import quantize_int8
from edgecluster.compression.report import build_report, to_markdown
from edgecluster.data.schema import label_list, load_many, write_jsonl

log = logging.getLogger("pipeline")


def resolve_data(cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Return JSONL paths per split, downloading the public dataset if needed."""
    ds = cfg["dataset"]
    if ds.get("kind") == "indic_glue":
        root = Path(ds.get("dir", "data/public")) / ds["config"]
        files = {s: [str(root / f"{s}.jsonl")] for s in ("train", "validation", "test")}
        if not all(Path(f[0]).exists() for f in files.values()):
            from edgecluster.data.public import load_indic_glue

            for split, examples in load_indic_glue(ds["config"], ds.get("revision")).items():
                write_jsonl(root / f"{split}.jsonl", examples)
        return files
    return {s: list(ds[s]) for s in ("train", "validation", "test")}


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute the pipeline described by ``cfg``."""
    t_start = time.time()
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    max_len = int(cfg.get("max_seq_len", 128))
    files = resolve_data(cfg)
    train_set, val_set, test_set = (
        load_many(files[s], include_needs_review=False) for s in ("train", "validation", "test")
    )
    labels = label_list(train_set)
    label2id = {lab: i for i, lab in enumerate(labels)}
    log.info("data: %d train / %d val / %d test, %d labels", len(train_set), len(val_set), len(test_set), len(labels))
    ft_cfg = TrainConfig(max_seq_len=max_len, **cfg.get("finetune", {}))
    device = pick_device(ft_cfg.device)
    timings: dict[str, float] = {}

    # 2. fine-tune ----------------------------------------------------------------
    # ``finetuned_dir`` lets pruning-ratio variants share one fine-tuned checkpoint.
    ft_dir = Path(cfg.get("finetuned_dir") or out / "finetuned")
    t0 = time.time()
    if (ft_dir / "config.json").exists() and cfg.get("reuse_finetuned", True):
        log.info("reusing fine-tuned model in %s", ft_dir)
        model, tok = load_model(str(ft_dir), labels)
        history = json.loads((ft_dir / "train_history.json").read_text())["history"]
    else:
        model, tok = load_model(cfg["base_model"], labels)
        model, history = train(model, tok, train_set, val_set, label2id, ft_cfg)
        model.save_pretrained(ft_dir)
        tok.save_pretrained(ft_dir)
        (ft_dir / "train_history.json").write_text(json.dumps({"config": asdict(ft_cfg), "history": history}, indent=2))
    timings["finetune_s"] = time.time() - t0
    params_fp32 = count_parameters(model)

    # 3. fp32 export + int8 reference -------------------------------------------
    onnx_dir = out / "onnx"
    meta = {
        "name": cfg.get("name", out.name),
        "base_model": cfg["base_model"],
        "labels": labels,
        "max_seq_len": max_len,
        "pad_token_id": int(tok.pad_token_id if tok.pad_token_id is not None else 0),
        "dataset": cfg["dataset"].get("config", "jsonl"),
    }
    write_runtime_files(tok, onnx_dir, meta)
    probe = [ex.text for ex in val_set[:16]]
    t0 = time.time()
    qcfg = cfg.get("quantization", {})
    qkw = {
        "per_channel": bool(qcfg.get("per_channel", False)),
        "op_types": tuple(qcfg.get("op_types", ("MatMul", "Gather"))),
    }
    baseline = cfg.get("baseline_onnx_dir")
    if baseline:
        # Same fine-tuned checkpoint as another run: link its fp32 files instead of re-exporting 1+ GB.
        for name in ("model.fp32.onnx", "model.fp32.int8.onnx"):
            link = onnx_dir / name
            link.unlink(missing_ok=True)
            link.symlink_to(Path(baseline).resolve() / name)
        verify_fp32 = json.loads((Path(baseline).parent / "run.json").read_text())["verify"]["fp32"]
    else:
        verify_fp32 = export_onnx(model, tok, onnx_dir / "model.fp32.onnx", max_len, probe)
        quantize_int8(onnx_dir / "model.fp32.onnx", onnx_dir / "model.fp32.int8.onnx", **qkw)
    timings["export_fp32_s"] = time.time() - t0
    log.info("fp32 export verified: %s", verify_fp32)

    # 4. importance + pruning ----------------------------------------------------
    pcfg = cfg.get("pruning", {})
    head_ratio = float(pcfg.get("head_ratio", 0.0))
    ffn_ratio = float(pcfg.get("ffn_ratio", 0.0))
    n_imp = int(pcfg.get("importance_examples", 512))
    t0 = time.time()
    val_before = evaluate_torch(model, tok, val_set, label2id, max_len, device)
    head_imp, neuron_imp = compute_importance(model, tok, val_set[:n_imp], label2id, max_len, device)
    heads_before = head_counts(model)
    spec = PruneSpec(heads=select_heads(head_imp, head_ratio))
    mechanism = prune_heads(model, spec.heads)
    spec.ffn_keep = prune_ffn(model, neuron_imp, ffn_ratio)
    model.cpu()
    val_pruned = evaluate_torch(model, tok, val_set, label2id, max_len, device)
    model.cpu()
    timings["prune_s"] = time.time() - t0
    log.info(
        "pruned %s heads via %s; val acc %.4f -> %.4f",
        sum(map(len, spec.heads.values())),
        mechanism,
        val_before["accuracy"],
        val_pruned["accuracy"],
    )

    # 5. recovery fine-tune ------------------------------------------------------
    rec_epochs = float(pcfg.get("recovery_epochs", 0))
    t0 = time.time()
    rec_history: list[dict[str, float]] = []
    if rec_epochs > 0 and (spec.heads or spec.ffn_keep):
        rec_cfg = TrainConfig(
            **{**asdict(ft_cfg), "epochs": rec_epochs, "lr": float(pcfg.get("recovery_lr", ft_cfg.lr))}
        )
        model, rec_history = train(model, tok, train_set, val_set, label2id, rec_cfg)
    timings["recovery_s"] = time.time() - t0
    if cfg.get("save_pruned_checkpoint", True):
        save_pruned(model, tok, spec, out / "pruned")

    # 6. pruned export + int8 ----------------------------------------------------
    t0 = time.time()
    verify_pruned = export_onnx(model, tok, onnx_dir / "model.pruned.onnx", max_len, probe)
    quantize_int8(onnx_dir / "model.pruned.onnx", onnx_dir / "model.pruned.int8.onnx", **qkw)
    timings["export_pruned_s"] = time.time() - t0

    run_info = {
        "name": cfg.get("name", out.name),
        "base_model": cfg["base_model"],
        "dataset": cfg["dataset"].get("config", "jsonl"),
        "test_files": files["test"],
        "max_seq_len": max_len,
        "labels": labels,
        "params_fp32": params_fp32,
        "params_pruned": count_parameters(model),
        "finetune_history": history,
        "recovery_history": rec_history,
        "val_before_pruning": val_before,
        "val_after_pruning_no_recovery": val_pruned,
        "verify": {"fp32": verify_fp32, "pruned": verify_pruned},
        "pruning": {
            "head_ratio": head_ratio,
            "ffn_ratio": ffn_ratio,
            "recovery_epochs": rec_epochs,
            "head_mechanism": mechanism,
            "heads_total": sum(heads_before.values()),
            "heads_pruned": sum(len(v) for v in spec.heads.values()),
            "heads_remaining": head_counts(model),
            "ffn_keep": spec.ffn_keep,
        },
        "quantization": qkw | {"op_types": list(qkw["op_types"])},
        "timings": timings,
    }
    (out / "run.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False, default=str))

    # 7. report -------------------------------------------------------------------
    rep = build_report(out)
    rep["timings"] = timings | {"total_s": time.time() - t_start}
    (out / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    md = to_markdown(rep)
    (out / "report.md").write_text(md + "\n")
    print(md)
    return rep


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", help="override out_dir")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    run(cfg)


if __name__ == "__main__":
    main()
