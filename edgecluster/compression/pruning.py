"""Structured pruning: attention heads and FFN neurons, driven by importance scores.

Head importance follows Michel et al. (2019), "Are Sixteen Heads Really Better
than One?": a gate ``ξ_h = 1`` multiplies each head's context vector, and the
importance of head ``h`` is ``Σ |∂L/∂ξ_h|`` over a validation set. Scores are
L2-normalised per layer and ranked globally. FFN-neuron importance is the
first-order Taylor score ``Σ |a · ∂L/∂a|`` of each intermediate activation.

Gates are applied with forward pre-hooks on the attention output projection,
so no model code needs patching. Supported architectures:

* **BERT / mBERT, RoBERTa / XLM-R**: one attention block per layer. Heads are
  removed with ``model.prune_heads`` when the installed ``transformers``
  provides it, otherwise by slicing the Q/K/V/O projections directly (same
  result).
* **ALBERT / IndicBERT**: all 12 layers *share one* attention block, so
  removing a head removes it from every layer at once. Importance is
  accumulated over all 12 applications of the shared block. Per-layer head
  pruning is impossible for ALBERT by construction.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from edgecluster.compression.data import batches
from edgecluster.data.schema import Example


@dataclass
class AttentionUnit:
    """One prunable (possibly shared) multi-head attention block."""

    key: int  # index understood by ``model.prune_heads``
    module: nn.Module  # holds query/key/value and num_attention_heads
    out_proj: nn.Linear  # consumes the concatenated head contexts
    uses: int = 1  # how many layers apply this block (12 for ALBERT)

    @property
    def num_heads(self) -> int:
        """Current number of heads."""
        return int(self.module.num_attention_heads)

    @property
    def head_dim(self) -> int:
        """Per-head dimension."""
        return int(self.module.attention_head_size)


@dataclass
class FfnUnit:
    """One prunable feed-forward block (``up`` -> activation -> ``down``)."""

    key: int
    activation_module: nn.Module  # its output is the intermediate activation
    up: nn.Linear
    down: nn.Linear


@dataclass
class PruneSpec:
    """What was pruned, enough to rebuild the shapes before loading weights."""

    heads: dict[int, list[int]] = field(default_factory=dict)  # unit key -> pruned head ids (original numbering)
    ffn_keep: dict[int, int] = field(default_factory=dict)  # unit key -> remaining neurons

    def to_json(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "heads": {str(k): v for k, v in self.heads.items()},
            "ffn_keep": {str(k): v for k, v in self.ffn_keep.items()},
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> PruneSpec:
        """Inverse of :meth:`to_json`."""
        return cls(
            {int(k): list(v) for k, v in raw.get("heads", {}).items()},
            {int(k): int(v) for k, v in raw.get("ffn_keep", {}).items()},
        )


def architecture(model: nn.Module) -> str:
    """``bert``-like (one block per layer) or ``albert`` (shared blocks)."""
    return "albert" if getattr(model.config, "model_type", "") == "albert" else "bert"


def find_units(model: nn.Module) -> tuple[list[AttentionUnit], list[FfnUnit]]:
    """Locate attention and FFN blocks for supported encoders."""
    base = model.base_model
    if architecture(model) == "albert":
        cfg = model.config
        attn, ffn = [], []
        per_group = cfg.num_hidden_layers // cfg.num_hidden_groups
        for g, group in enumerate(base.encoder.albert_layer_groups):
            for i, layer in enumerate(group.albert_layers):
                key = g * cfg.inner_group_num + i
                attn.append(AttentionUnit(key, layer.attention, layer.attention.dense, uses=per_group))
                ffn.append(FfnUnit(key, layer.ffn, layer.ffn, layer.ffn_output))
        return attn, ffn
    layers = base.encoder.layer
    attn = [AttentionUnit(i, lay.attention.self, lay.attention.output.dense) for i, lay in enumerate(layers)]
    ffn = [FfnUnit(i, lay.intermediate, lay.intermediate.dense, lay.output.dense) for i, lay in enumerate(layers)]
    return attn, ffn


def compute_importance(
    model: nn.Module,
    tokenizer: Any,
    examples: Sequence[Example],
    label2id: dict[str, int],
    max_seq_len: int,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Head and FFN-neuron importance on ``examples`` (use the validation split)."""
    model.to(device).eval()
    attn_units, ffn_units = find_units(model)
    gates = {u.key: torch.ones(u.num_heads, device=device, requires_grad=True) for u in attn_units}
    head_imp = {u.key: torch.zeros(u.num_heads) for u in attn_units}
    neuron_imp = {u.key: torch.zeros(u.up.out_features) for u in ffn_units}
    acts: dict[int, list[torch.Tensor]] = {u.key: [] for u in ffn_units}
    handles = []

    def gate_hook(unit: AttentionUnit):
        def hook(_mod: nn.Module, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            ctx = args[0]
            b, t, _ = ctx.shape
            gated = ctx.view(b, t, unit.num_heads, unit.head_dim) * gates[unit.key].view(1, 1, -1, 1)
            return (gated.view(b, t, -1), *args[1:])

        return hook

    def act_hook(unit: FfnUnit):
        def hook(_mod: nn.Module, _args: Any, out: torch.Tensor) -> torch.Tensor:
            out.retain_grad()
            acts[unit.key].append(out)
            return out

        return hook

    for u in attn_units:
        handles.append(u.out_proj.register_forward_pre_hook(gate_hook(u)))
    is_albert = architecture(model) == "albert"
    for u in ffn_units:
        if is_albert:
            # ALBERT applies the activation inside AlbertLayer.ff_chunk, after ``ffn``.
            # Score the input of ``ffn_output`` (post-activation) instead.
            def pre(unit: FfnUnit):
                def hook(_mod: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
                    a = args[0]
                    a.retain_grad()
                    acts[unit.key].append(a)

                return hook

            handles.append(u.down.register_forward_pre_hook(pre(u)))
        else:
            handles.append(u.activation_module.register_forward_hook(act_hook(u)))
    try:
        for batch in batches(examples, tokenizer, label2id, batch_size, max_seq_len, device):
            for v in acts.values():
                v.clear()
            model.zero_grad(set_to_none=True)
            for g in gates.values():
                g.grad = None
            loss = model(**batch).loss
            loss.backward()
            for u in attn_units:
                head_imp[u.key] += gates[u.key].grad.detach().abs().cpu()
            mask = batch["attention_mask"].unsqueeze(-1).float()
            for key, tensors in acts.items():
                for a in tensors:
                    if a.grad is not None:
                        neuron_imp[key] += (a * a.grad * mask).abs().sum(dim=(0, 1)).detach().cpu()
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)
    # Michel et al.: normalise head scores per layer (L2) before global ranking.
    for k, v in head_imp.items():
        head_imp[k] = v / (v.norm(p=2) + 1e-12)
    return head_imp, neuron_imp


def select_heads(head_imp: dict[int, torch.Tensor], ratio: float, min_heads: int = 1) -> dict[int, list[int]]:
    """Globally remove the ``ratio`` least important heads, keeping ``min_heads`` per block."""
    flat = sorted((float(s), k, h) for k, v in head_imp.items() for h, s in enumerate(v))
    budget = round(ratio * len(flat))
    remaining = {k: len(v) for k, v in head_imp.items()}
    pruned: dict[int, list[int]] = {}
    for _, k, h in flat:
        if budget == 0:
            break
        if remaining[k] <= min_heads:
            continue
        pruned.setdefault(k, []).append(h)
        remaining[k] -= 1
        budget -= 1
    return {k: sorted(v) for k, v in pruned.items()}


def _slice_linear(layer: nn.Linear, index: torch.Tensor, dim: int) -> nn.Linear:
    """New Linear keeping only ``index`` along output (dim=0) or input (dim=1)."""
    index = index.to(layer.weight.device)
    w = layer.weight.index_select(dim, index).detach().clone()
    b = None
    if layer.bias is not None:
        b = (layer.bias if dim == 1 else layer.bias.index_select(0, index)).detach().clone()
    new = nn.Linear(w.shape[1], w.shape[0], bias=b is not None).to(layer.weight.device, layer.weight.dtype)
    new.weight.data.copy_(w)
    if b is not None:
        new.bias.data.copy_(b)
    return new


def _prune_heads_manually(unit: AttentionUnit, heads: list[int], owner: nn.Module, out_attr: str) -> None:
    """Slice Q/K/V outputs and the output projection's inputs to drop ``heads``."""
    mod, d = unit.module, unit.head_dim
    already = set(getattr(mod, "pruned_heads", set()))
    # Map original head ids to current positions (earlier prunes shift indices).
    current = [h for h in range(unit.num_heads + len(already)) if h not in already]
    drop_pos = {current.index(h) for h in heads if h in current}
    keep = torch.tensor(
        [p * d + j for p in range(unit.num_heads) if p not in drop_pos for j in range(d)], dtype=torch.long
    )
    mod.query = _slice_linear(mod.query, keep, 0)
    mod.key = _slice_linear(mod.key, keep, 0)
    mod.value = _slice_linear(mod.value, keep, 0)
    new_out = _slice_linear(unit.out_proj, keep, 1)
    setattr(owner, out_attr, new_out)
    unit.out_proj = new_out
    mod.num_attention_heads = unit.num_heads - len(drop_pos)
    mod.all_head_size = mod.num_attention_heads * d
    mod.pruned_heads = already | set(heads)


def prune_heads(model: nn.Module, spec: dict[int, list[int]]) -> str:
    """Remove heads in place. Returns which mechanism was used."""
    if not spec:
        return "none"
    if hasattr(model, "prune_heads"):
        try:
            model.prune_heads(spec)
            return "model.prune_heads"
        except (NotImplementedError, AttributeError):
            pass
    attn_units, _ = find_units(model)
    base = model.base_model
    for unit in attn_units:
        if unit.key not in spec:
            continue
        if architecture(model) == "albert":
            owner, attr = unit.module, "dense"
        else:
            owner, attr = base.encoder.layer[unit.key].attention.output, "dense"
        _prune_heads_manually(unit, spec[unit.key], owner, attr)
    return "manual-slicing"


def _replace_ffn(model: nn.Module, key: int, up: nn.Linear, down: nn.Linear) -> None:
    base = model.base_model
    if architecture(model) == "albert":
        inner = model.config.inner_group_num
        layer = base.encoder.albert_layer_groups[key // inner].albert_layers[key % inner]
        layer.ffn, layer.ffn_output = up, down
    else:
        layer = base.encoder.layer[key]
        layer.intermediate.dense, layer.output.dense = up, down


def prune_ffn(model: nn.Module, neuron_imp: dict[int, torch.Tensor], ratio: float) -> dict[int, int]:
    """Drop the ``ratio`` least important intermediate neurons in every FFN block."""
    if ratio <= 0:
        return {}
    _, ffn_units = find_units(model)
    kept: dict[int, int] = {}
    for unit in ffn_units:
        scores = neuron_imp[unit.key]
        n_keep = max(1, round(scores.numel() * (1 - ratio)))
        keep = torch.sort(torch.topk(scores, n_keep).indices).values
        _replace_ffn(model, unit.key, _slice_linear(unit.up, keep, 0), _slice_linear(unit.down, keep, 1))
        kept[unit.key] = n_keep
    return kept


def apply_spec_shapes(model: nn.Module, spec: PruneSpec) -> None:
    """Recreate pruned shapes on a fresh model so a pruned state_dict can be loaded."""
    prune_heads(model, spec.heads)
    _, ffn_units = find_units(model)
    for unit in ffn_units:
        if unit.key in spec.ffn_keep:
            keep = torch.arange(spec.ffn_keep[unit.key])  # values are overwritten by load_state_dict
            _replace_ffn(model, unit.key, _slice_linear(unit.up, keep, 0), _slice_linear(unit.down, keep, 1))


def count_parameters(model: nn.Module) -> int:
    """Total parameter count."""
    return sum(p.numel() for p in model.parameters())


def head_counts(model: nn.Module) -> dict[int, int]:
    """Remaining heads per attention unit."""
    attn_units, _ = find_units(model)
    return {u.key: u.num_heads for u in attn_units}


def save_pruned(model: nn.Module, tokenizer: Any, spec: PruneSpec, out: str | Path) -> None:
    """Save a pruned model as (config, structure spec, safetensors weights, tokenizer)."""
    from safetensors.torch import save_file

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = model.config.to_dict()
    cfg["pruned_heads"] = {}  # shapes come from prune_spec.json, not from config
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "prune_spec.json").write_text(json.dumps(spec.to_json(), indent=2))
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    # Tied weights would be saved twice; safetensors refuses shared storage.
    seen: dict[int, str] = {}
    unique = {}
    for k, v in state.items():
        ptr = v.data_ptr()
        if ptr in seen:
            continue
        seen[ptr] = k
        unique[k] = v
    save_file(unique, str(out / "model.safetensors"))
    tokenizer.save_pretrained(out)


def load_pruned(path: str | Path) -> tuple[nn.Module, Any, PruneSpec]:
    """Inverse of :func:`save_pruned`."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    path = Path(path)
    config = AutoConfig.from_pretrained(path)
    config.pruned_heads = {}
    model = AutoModelForSequenceClassification.from_config(config, attn_implementation="eager")
    spec = PruneSpec.from_json(json.loads((path / "prune_spec.json").read_text()))
    apply_spec_shapes(model, spec)
    missing, unexpected = model.load_state_dict(load_file(str(path / "model.safetensors")), strict=False)
    if unexpected or any("embeddings" not in m and "decoder" not in m for m in missing):
        raise RuntimeError(f"state dict mismatch: missing={missing} unexpected={unexpected}")
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model, AutoTokenizer.from_pretrained(path), spec
