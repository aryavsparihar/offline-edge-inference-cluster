"""Fine-tune a multilingual encoder for intent / topic classification.

A plain PyTorch loop (AdamW + linear warmup/decay). It keeps the checkpoint
with the best validation accuracy, and the test split is never consulted.

    python -m edgecluster.compression.finetune --model xlm-roberta-base \\
        --train data/public/bbca.hi/train.jsonl --val data/public/bbca.hi/validation.jsonl \\
        --out models/xlmr-bbca/finetuned
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from edgecluster.compression.data import batches, pick_device
from edgecluster.compression.metrics import accuracy, macro_f1
from edgecluster.data.schema import Example, label_list, load_many

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Hyper-parameters for one fine-tuning (or recovery) run."""

    epochs: float = 3.0
    batch_size: int = 16
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_len: int = 128
    grad_clip: float = 1.0
    seed: int = 13
    device: str = "auto"
    eval_every: int = 0  # steps between validation runs (0 = once per epoch)


def load_model(name_or_path: str, labels: Sequence[str]) -> tuple[nn.Module, Any]:
    """Load a sequence-classification head on ``name_or_path`` with our label map."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        name_or_path,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={lab: i for i, lab in enumerate(labels)},
        attn_implementation="eager",
        ignore_mismatched_sizes=True,
    )
    return model, tok


@torch.no_grad()
def predict_torch(
    model: nn.Module,
    tokenizer: Any,
    examples: Sequence[Example],
    label2id: dict[str, int],
    max_seq_len: int,
    device: torch.device,
    batch_size: int = 64,
) -> list[str]:
    """Predicted label names for ``examples``."""
    model.eval()
    id2label = {i: lab for lab, i in label2id.items()}
    preds: list[str] = []
    for batch in batches(examples, tokenizer, label2id, batch_size, max_seq_len, device):
        batch.pop("labels")
        logits = model(**batch).logits
        preds += [id2label[int(i)] for i in logits.argmax(-1).tolist()]
    return preds


def evaluate_torch(
    model: nn.Module,
    tokenizer: Any,
    examples: Sequence[Example],
    label2id: dict[str, int],
    max_seq_len: int,
    device: torch.device,
) -> dict[str, float]:
    """Accuracy and macro-F1 of a PyTorch model."""
    preds = predict_torch(model, tokenizer, examples, label2id, max_seq_len, device)
    gold = [ex.label for ex in examples]
    return {"accuracy": accuracy(gold, preds), "macro_f1": macro_f1(gold, preds), "n": len(gold)}


def train(
    model: nn.Module,
    tokenizer: Any,
    train_set: Sequence[Example],
    val_set: Sequence[Example],
    label2id: dict[str, int],
    cfg: TrainConfig,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train and return the best-on-validation model plus the eval history."""
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    model.to(device)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "LayerNorm.bias")
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(params, lr=cfg.lr)
    steps_per_epoch = math.ceil(len(train_set) / cfg.batch_size)
    total = max(1, int(steps_per_epoch * cfg.epochs))
    sched = get_linear_schedule_with_warmup(optim, int(total * cfg.warmup_ratio), total)
    eval_every = cfg.eval_every or steps_per_epoch
    gen = torch.Generator().manual_seed(cfg.seed)
    best_acc, best_state, history, step = -1.0, None, [], 0
    t0 = time.time()
    while step < total:
        model.train()
        for batch in batches(train_set, tokenizer, label2id, cfg.batch_size, cfg.max_seq_len, device, True, gen):
            loss = model(**batch).loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                log.info("step %d/%d loss %.4f (%.0fs)", step, total, loss.item(), time.time() - t0)
            if step % eval_every == 0 or step == total:
                metrics = evaluate_torch(model, tokenizer, val_set, label2id, cfg.max_seq_len, device)
                metrics["step"] = step
                history.append(metrics)
                log.info("eval step %d: val acc %.4f macro-F1 %.4f", step, metrics["accuracy"], metrics["macro_f1"])
                if metrics["accuracy"] > best_acc:
                    best_acc = metrics["accuracy"]
                    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                model.train()
            if step >= total:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), history


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--train", nargs="+", required=True, help="training JSONL file(s)")
    ap.add_argument("--val", nargs="+", required=True, help="validation JSONL file(s)")
    ap.add_argument("--out", required=True)
    for name, value in asdict(TrainConfig()).items():
        ap.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = TrainConfig(**{k: getattr(args, k) for k in asdict(TrainConfig())})
    train_set, val_set = load_many(args.train), load_many(args.val)
    labels = label_list(train_set)
    model, tok = load_model(args.model, labels)
    model, history = train(model, tok, train_set, val_set, {lab: i for i, lab in enumerate(labels)}, cfg)
    out = Path(args.out)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "train_history.json").write_text(json.dumps({"config": asdict(cfg), "history": history}, indent=2))
    print(json.dumps(history[-1] if history else {}, indent=2))


if __name__ == "__main__":
    main()
