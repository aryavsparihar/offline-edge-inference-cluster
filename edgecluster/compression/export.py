"""Export a sequence classifier to ONNX and verify it against PyTorch."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from edgecluster.serving.runtime import make_session

OPSET = 17


class _LogitsOnly(nn.Module):
    """Wrapper with a positional (input_ids, attention_mask) -> logits signature."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def write_runtime_files(tokenizer: Any, out_dir: str | Path, meta: dict[str, Any]) -> None:
    """Write ``tokenizer.json`` (Rust fast tokenizer) and ``meta.json`` next to the model."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("a fast tokenizer is required to produce tokenizer.json")
    tokenizer.backend_tokenizer.save(str(out_dir / "tokenizer.json"))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def export_onnx(
    model: nn.Module,
    tokenizer: Any,
    out_path: str | Path,
    max_seq_len: int,
    sample_texts: Sequence[str],
) -> dict[str, float]:
    """Export ``model`` to ``out_path`` and check ORT logits match PyTorch.

    Uses the TorchScript-based exporter (``dynamo=False``), which handles
    pruned layers with per-layer head counts reliably. Batch and sequence axes
    are dynamic. Returns the max absolute logit difference on ``sample_texts``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    enc = tokenizer(
        list(sample_texts[:2]) or ["test"], truncation=True, max_length=max_seq_len, padding=True, return_tensors="pt"
    )
    args = (enc["input_ids"], enc["attention_mask"])
    # The wrapper must be in eval mode too: the exporter restores the wrapper's
    # original training flag afterwards, and a fresh Module defaults to train(),
    # which would silently re-enable dropout in ``model``.
    wrapper = _LogitsOnly(model).eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            str(out_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    return verify_onnx(model, tokenizer, out_path, max_seq_len, sample_texts)


def verify_onnx(
    model: nn.Module, tokenizer: Any, onnx_path: str | Path, max_seq_len: int, texts: Sequence[str]
) -> dict[str, float]:
    """Max |logit difference| and argmax agreement between PyTorch and ORT."""
    model.eval()
    sess = make_session(onnx_path, intra_op_threads=0)
    enc = tokenizer(list(texts), truncation=True, max_length=max_seq_len, padding=True, return_tensors="pt")
    with torch.no_grad():
        ref = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.numpy()
    (got,) = sess.run(
        ["logits"], {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()}
    )
    return {
        "max_abs_diff": float(np.abs(ref - got).max()),
        "argmax_agreement": float((ref.argmax(-1) == got.argmax(-1)).mean()),
    }
