"""Full compression pipeline on a tiny Hub model (hf-internal-testing), as run by `make compress`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_pipeline_end_to_end(tmp_path: Path) -> None:
    from edgecluster.compression.pipeline import run

    cfg = yaml.safe_load((ROOT / "configs/compress/tiny-sample.yaml").read_text())
    cfg["out_dir"] = str(tmp_path / "run")
    for split in ("train", "validation", "test"):
        cfg["dataset"][split] = [str(ROOT / p) for p in cfg["dataset"][split]]
    rep = run(cfg)
    variants = {r["variant"]: r for r in rep["rows"]}
    assert set(variants) == {"fp32", "fp32+int8", "pruned", "pruned+int8"}
    assert variants["pruned"]["size_mb"] < variants["fp32"]["size_mb"]
    assert variants["pruned+int8"]["size_mb"] < variants["pruned"]["size_mb"]
    for r in variants.values():
        assert 0.0 <= r["accuracy"] <= 1.0 and r["n_test"] == 84
    run_info = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run_info["pruning"]["heads_pruned"] > 0
    assert run_info["verify"]["fp32"]["max_abs_diff"] < 1e-3
    assert run_info["verify"]["pruned"]["max_abs_diff"] < 1e-3
    assert (tmp_path / "run" / "report.md").exists()
    assert (tmp_path / "run" / "onnx" / "tokenizer.json").exists()
