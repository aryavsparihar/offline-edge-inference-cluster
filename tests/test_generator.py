"""Generative model decode loop (runs only if the ~520 MB model was fetched with `make generator`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from edgecluster.serving.generator import chatml_prompt

MODEL = Path(__file__).resolve().parents[1] / "models" / "qwen2.5-0.5b-instruct"


def test_chatml_prompt_format() -> None:
    p = chatml_prompt("sys", "hello")
    assert p.startswith("<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n")
    assert p.endswith("<|im_start|>assistant\n")


@pytest.mark.skipif(not (MODEL / "onnx" / "model_quantized.onnx").exists(), reason="generator model not downloaded")
def test_kv_cache_decoding_matches_full_recompute() -> None:
    """Greedy decoding with the KV cache must equal decoding without it (full prefix every step)."""
    import numpy as np

    from edgecluster.serving.generator import OnnxGenerator

    g = OnnxGenerator(MODEL)
    ids = g.tokenizer.encode(chatml_prompt("Answer briefly.", "Name a primary colour."), add_special_tokens=False).ids
    cached, _ = g.generate_ids(ids, 6, min_new_tokens=6)
    assert len(cached) == 6

    seq = list(ids)
    for _ in range(6):
        feeds = {
            "input_ids": np.asarray([seq], dtype=np.int64),
            "attention_mask": np.ones((1, len(seq)), dtype=np.int64),
            "position_ids": np.arange(len(seq), dtype=np.int64)[None, :],
            **g._empty_past(),
        }
        logits = g.session.run(["logits"], feeds)[0][0, -1].copy()
        logits[list(g.eos_ids)] = -np.inf
        seq.append(int(np.argmax(logits)))
    assert seq[len(ids) :] == cached

    out = g.generate("Name a primary colour.", "Answer in one word.", max_new_tokens=8)
    assert out.text.strip() and out.new_tokens <= 8 and out.total_ms >= out.ttft_ms > 0
