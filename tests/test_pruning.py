"""Structured pruning, export and quantization on tiny randomly initialised encoders (offline)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from edgecluster.compression.export import export_onnx, write_runtime_files  # noqa: E402
from edgecluster.compression.pruning import (  # noqa: E402
    PruneSpec,
    compute_importance,
    count_parameters,
    find_units,
    head_counts,
    load_pruned,
    prune_ffn,
    prune_heads,
    save_pruned,
    select_heads,
)
from edgecluster.compression.quantize import quantize_int8  # noqa: E402
from edgecluster.data.schema import load_many  # noqa: E402
from edgecluster.serving.classifier import OnnxClassifier  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"
EXAMPLES = load_many([SAMPLE / "hi.jsonl", SAMPLE / "ur.jsonl"])
LABELS = sorted({e.label for e in EXAMPLES})
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory: pytest.TempPathFactory):
    """A character-level WordPiece vocab built from the sample, as a fast BERT tokenizer."""
    from transformers import BertTokenizerFast

    chars = sorted({ch for e in EXAMPLES for ch in e.text if not ch.isspace()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *chars, *[f"##{c}" for c in chars]]
    path = tmp_path_factory.mktemp("tok") / "vocab.txt"
    path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    return BertTokenizerFast(vocab_file=str(path), do_lower_case=False, tokenize_chinese_chars=False)


def tiny_model(kind: str, vocab_size: int):
    common = dict(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=len(LABELS),
    )
    if kind == "bert":
        cfg = transformers.BertConfig(**common)
    elif kind == "xlm-roberta":
        cfg = transformers.XLMRobertaConfig(**common, pad_token_id=0, bos_token_id=2, eos_token_id=3)
    else:
        cfg = transformers.AlbertConfig(**common, embedding_size=16, num_hidden_groups=1, inner_group_num=1)
    torch.manual_seed(0)
    return transformers.AutoModelForSequenceClassification.from_config(cfg, attn_implementation="eager").eval()


@pytest.mark.parametrize("kind", ["bert", "xlm-roberta", "albert"])
def test_importance_prune_export_quantize(kind: str, tokenizer, tmp_path: Path) -> None:
    model = tiny_model(kind, tokenizer.vocab_size)
    device = torch.device("cpu")
    head_imp, neuron_imp = compute_importance(model, tokenizer, EXAMPLES, LABEL2ID, 32, device, batch_size=16)
    attn_units, _ = find_units(model)
    expected_units = 1 if kind == "albert" else 2  # ALBERT shares one block across layers
    assert len(attn_units) == len(head_imp) == expected_units
    assert all(v.shape == (4,) and torch.isfinite(v).all() and v.sum() > 0 for v in head_imp.values())
    assert all(v.shape == (64,) and v.sum() > 0 for v in neuron_imp.values())

    params_before = count_parameters(model)
    spec = PruneSpec(heads=select_heads(head_imp, ratio=0.5))
    assert sum(len(v) for v in spec.heads.values()) == (2 if kind == "albert" else 4)
    mechanism = prune_heads(model, spec.heads)
    assert mechanism in {"model.prune_heads", "manual-slicing"}
    assert sum(head_counts(model).values()) == (2 if kind == "albert" else 4)
    spec.ffn_keep = prune_ffn(model, neuron_imp, ratio=0.25)
    assert set(spec.ffn_keep.values()) == {48}
    assert count_parameters(model) < params_before

    # The pruned model still runs and keeps its output shape.
    enc = tokenizer(["पेंशन कब आएगी"], return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
    assert logits.shape == (1, len(LABELS))

    # Save -> load reproduces the exact same function.
    save_pruned(model, tokenizer, spec, tmp_path / "pruned")
    reloaded, _, spec2 = load_pruned(tmp_path / "pruned")
    assert spec2.heads == spec.heads and spec2.ffn_keep == spec.ffn_keep
    with torch.no_grad():
        again = reloaded.eval()(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
    torch.testing.assert_close(logits, again)

    # ONNX export matches PyTorch; INT8 is smaller and still runs through the serving classifier.
    onnx_dir = tmp_path / "onnx"
    write_runtime_files(tokenizer, onnx_dir, {"labels": LABELS, "max_seq_len": 32, "pad_token_id": 0})
    texts = [e.text for e in EXAMPLES[:8]]
    check = export_onnx(model, tokenizer, onnx_dir / "model.onnx", 32, texts)
    assert check["max_abs_diff"] < 1e-4 and check["argmax_agreement"] == 1.0
    quantize_int8(onnx_dir / "model.onnx", onnx_dir / "model.int8.onnx")
    assert (onnx_dir / "model.int8.onnx").stat().st_size < (onnx_dir / "model.onnx").stat().st_size
    clf = OnnxClassifier(onnx_dir, "model.onnx")
    # The Rust tokenizer used at serving time must produce the same ids as the HF tokenizer.
    hf_ids = tokenizer(texts, truncation=True, max_length=32, padding=True)["input_ids"]
    assert clf.encode(texts)["input_ids"].tolist() == hf_ids
    with torch.no_grad():
        enc = tokenizer(texts, truncation=True, max_length=32, padding=True, return_tensors="pt")
        ref = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.numpy()
    np.testing.assert_allclose(clf.logits(texts), ref, atol=1e-4)
    q = OnnxClassifier(onnx_dir, "model.int8.onnx")
    assert len(q.predict(texts)) == len(texts)


def test_select_heads_keeps_one_head_per_layer() -> None:
    imp = {0: torch.tensor([0.1, 0.2]), 1: torch.tensor([0.9, 0.8])}
    assert select_heads(imp, ratio=1.0) == {0: [0], 1: [1]}
    assert select_heads(imp, ratio=0.0) == {}
