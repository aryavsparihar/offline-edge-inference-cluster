"""Intent classifier backed by ONNX Runtime and a Rust ``tokenizers`` tokenizer.

A model directory (written by :mod:`edgecluster.compression.export`) contains:

* ``model.onnx`` (or another ``.onnx`` file, e.g. ``model.pruned.int8.onnx``),
* ``tokenizer.json``, a fast tokenizer, so neither ``transformers`` nor
  ``torch`` is needed on the Pi,
* ``meta.json``: ``{"labels": [...], "max_seq_len": 128, "pad_token_id": 1, ...}``.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from edgecluster.serving.runtime import make_session


@dataclass(frozen=True)
class Prediction:
    """Classifier output for one text."""

    label: str
    confidence: float
    scores: dict[str, float]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def load_meta(model_dir: str | Path) -> dict[str, Any]:
    """Read ``meta.json`` from a model directory."""
    with open(Path(model_dir) / "meta.json", encoding="utf-8") as fh:
        meta: dict[str, Any] = json.load(fh)
    if not meta.get("labels"):
        raise ValueError(f"{model_dir}/meta.json has no labels")
    return meta


class OnnxClassifier:
    """Sequence classifier.

    Args:
        model_dir: Directory with the ONNX model, ``tokenizer.json`` and ``meta.json``.
        model_file: ONNX file inside ``model_dir``.
        max_seq_len: Truncation length (defaults to ``meta.json``).
        pad_to_max: Pad every input to ``max_seq_len``. This gives fixed-shape,
            worst-case latency (what the benchmark reports). When False, inputs
            are padded only to the longest text in the batch.
        intra_op_threads / inter_op_threads / graph_optimization: Session tuning.
    """

    def __init__(
        self,
        model_dir: str | Path,
        model_file: str = "model.onnx",
        max_seq_len: int | None = None,
        pad_to_max: bool = False,
        intra_op_threads: int = 4,
        inter_op_threads: int = 1,
        graph_optimization: str = "all",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / model_file
        self.meta = load_meta(model_dir)
        self.labels: list[str] = list(self.meta["labels"])
        self.max_seq_len = int(max_seq_len or self.meta.get("max_seq_len", 128))
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=self.max_seq_len)
        pad_id = int(self.meta.get("pad_token_id", 0))
        pad_token = self.tokenizer.id_to_token(pad_id) or "[PAD]"
        self.tokenizer.enable_padding(
            pad_id=pad_id, pad_token=pad_token, length=self.max_seq_len if pad_to_max else None
        )
        self.session = make_session(self.model_path, intra_op_threads, inter_op_threads, graph_optimization)
        self.input_names = {i.name for i in self.session.get_inputs()}
        # Concurrent session.run calls would oversubscribe the 4 cores; serialize them.
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        return str(self.meta.get("name", self.model_path.name))

    def encode(self, texts: list[str]) -> dict[str, np.ndarray]:
        """Tokenize a batch into ONNX inputs."""
        encs = self.tokenizer.encode_batch(texts)
        feeds = {
            "input_ids": np.asarray([e.ids for e in encs], dtype=np.int64),
            "attention_mask": np.asarray([e.attention_mask for e in encs], dtype=np.int64),
        }
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.asarray([e.type_ids for e in encs], dtype=np.int64)
        return {k: v for k, v in feeds.items() if k in self.input_names}

    def run(self, feeds: dict[str, np.ndarray]) -> np.ndarray:
        """Run the session on pre-tokenized inputs; returns logits."""
        with self._lock:
            (out,) = self.session.run(["logits"], feeds)
        return np.asarray(out, dtype=np.float32)

    def logits(self, texts: list[str]) -> np.ndarray:
        """Raw logits, shape ``(len(texts), num_labels)``."""
        return self.run(self.encode(texts))

    def decode(self, logits: np.ndarray) -> list[Prediction]:
        """Turn logits into labeled predictions."""
        probs = softmax(logits)
        preds = []
        for row in probs:
            idx = int(row.argmax())
            preds.append(
                Prediction(
                    self.labels[idx],
                    float(row[idx]),
                    {lab: float(p) for lab, p in zip(self.labels, row, strict=True)},
                )
            )
        return preds

    def predict(self, texts: list[str]) -> list[Prediction]:
        """Classify a batch of texts."""
        return self.decode(self.logits(texts))
