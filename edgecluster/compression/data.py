"""Torch-side dataset helpers shared by fine-tuning, importance scoring and evaluation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import torch

from edgecluster.data.schema import Example


def pick_device(preference: str = "auto") -> torch.device:
    """CUDA, then Apple MPS, then CPU (or whatever ``preference`` names)."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batches(
    examples: Sequence[Example],
    tokenizer: Any,
    label2id: dict[str, int],
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Tokenize and yield model-ready batches (dynamic padding per batch)."""
    order = (
        torch.randperm(len(examples), generator=generator).tolist() if shuffle else list(range(len(examples)))
    )
    for start in range(0, len(order), batch_size):
        chunk = [examples[i] for i in order[start : start + batch_size]]
        enc = tokenizer(
            [ex.text for ex in chunk],
            truncation=True,
            max_length=max_seq_len,
            padding=True,
            return_tensors="pt",
        )
        enc.pop("token_type_ids", None)  # single-segment task; keeps ONNX inputs uniform
        enc["labels"] = torch.tensor([label2id[ex.label] for ex in chunk], dtype=torch.long)
        yield {k: v.to(device) for k, v in enc.items()}
