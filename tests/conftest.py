"""Shared fixtures: a tiny hand-built ONNX classifier that needs no download or torch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from edgecluster.data.schema import WELFARE_LABELS

# word -> label it votes for; every other token embeds to zero
KEYWORDS = {
    "पेंशन": "pension",
    "پنشن": "pension",
    "राशन": "ration_card",
    "راشن": "ration_card",
    "आवास": "housing",
    "مکان": "housing",
    "आयुष्मान": "health_scheme",
    "ہسپتال": "health_scheme",
    "छात्रवृत्ति": "education_scholarship",
    "اسکالرشپ": "education_scholarship",
    "आधार": "documents_id",
    "سرٹیفکیٹ": "documents_id",
    "बारिश": "other",
}


def build_tiny_classifier(out: Path) -> Path:
    """Write model.onnx / tokenizer.json / meta.json for a bag-of-keywords classifier."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from tokenizers import Tokenizer, models, pre_tokenizers

    out.mkdir(parents=True, exist_ok=True)
    labels = list(WELFARE_LABELS)
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for w in KEYWORDS:
        vocab[w] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(out / "tokenizer.json"))

    emb = np.zeros((len(vocab), len(labels)), dtype=np.float32)
    for w, lab in KEYWORDS.items():
        emb[vocab[w], labels.index(lab)] = 5.0
    w_out = np.eye(len(labels), dtype=np.float32)
    bias = np.zeros(len(labels), dtype=np.float32)
    bias[labels.index("other")] = 0.1  # no keyword -> "other"
    nodes = [
        helper.make_node("Gather", ["emb", "input_ids"], ["e"]),
        helper.make_node("Cast", ["attention_mask"], ["m"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["m", "axis2"], ["m3"]),
        helper.make_node("Mul", ["e", "m3"], ["em"]),
        helper.make_node("ReduceSum", ["em", "axis1"], ["s"], keepdims=0),
        helper.make_node("ReduceSum", ["m3", "axis1"], ["cnt"], keepdims=0),
        helper.make_node("Max", ["cnt", "one"], ["cnt1"]),
        helper.make_node("Div", ["s", "cnt1"], ["pooled"]),
        helper.make_node("MatMul", ["pooled", "w"], ["mm"]),
        helper.make_node("Add", ["mm", "b"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"]),
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", len(labels)])],
        initializer=[
            numpy_helper.from_array(emb, "emb"),
            numpy_helper.from_array(w_out, "w"),
            numpy_helper.from_array(bias, "b"),
            numpy_helper.from_array(np.array([2], dtype=np.int64), "axis2"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "axis1"),
            numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(out / "model.onnx"))
    (out / "meta.json").write_text(
        json.dumps({"name": "tiny-keyword", "labels": labels, "max_seq_len": 32, "pad_token_id": 0})
    )
    return out


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_classifier(tmp_path_factory.mktemp("tiny-model"))
