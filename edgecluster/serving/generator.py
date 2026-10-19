"""Optional small generative model on ONNX Runtime with a KV-cache decode loop.

Model choice: **Qwen2.5-0.5B-Instruct** (494M params, Apache-2.0), using the
pre-exported ONNX files published at ``onnx-community/Qwen2.5-0.5B-Instruct``.
The INT8 file ``onnx/model_quantized.onnx`` is 512 MB on disk, which leaves
plenty of headroom on a 4 GB Pi next to the classifier. Its training data
includes Hindi and Urdu, but answers in those languages are weak at this size.
Treat it as a drafting aid behind a flag, not an authority on scheme rules.

Larger models (e.g. 1.5B+) technically load on a 4 GB Pi but leave no room
for the classifier and page cache, and decode at well under 2 tokens/s, so
they are not supported here.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from edgecluster.serving.runtime import make_session

DEFAULT_REPO = "onnx-community/Qwen2.5-0.5B-Instruct"
DEFAULT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "onnx/model_quantized.onnx",
)


@dataclass(frozen=True)
class Generation:
    """Result of one generate call."""

    text: str
    new_tokens: int
    prompt_tokens: int
    ttft_ms: float
    total_ms: float

    @property
    def ms_per_token(self) -> float:
        """Average decode time per generated token."""
        return self.total_ms / max(self.new_tokens, 1)


def chatml_prompt(system: str, user: str) -> str:
    """Qwen2.5 chat template (ChatML) for a single-turn conversation."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


class OnnxGenerator:
    """Greedy decoder for decoder-only ONNX models with ``past_key_values`` I/O.

    Args:
        model_dir: Directory with ``config.json``, ``tokenizer.json`` and the ONNX file.
        model_file: Path of the ONNX file relative to ``model_dir``.
        intra_op_threads: Threads for the session (4 on a Pi 4).
    """

    def __init__(
        self,
        model_dir: str | Path,
        model_file: str = "onnx/model_quantized.onnx",
        intra_op_threads: int = 4,
    ) -> None:
        self.model_dir = Path(model_dir)
        cfg = json.loads((self.model_dir / "config.json").read_text())
        gen_cfg_path = self.model_dir / "generation_config.json"
        gen_cfg = json.loads(gen_cfg_path.read_text()) if gen_cfg_path.exists() else {}
        self.num_layers = int(cfg["num_hidden_layers"])
        self.num_kv_heads = int(cfg.get("num_key_value_heads", cfg["num_attention_heads"]))
        self.head_dim = int(cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]))
        eos = gen_cfg.get("eos_token_id", cfg.get("eos_token_id"))
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.session = make_session(self.model_dir / model_file, intra_op_threads=intra_op_threads)
        inputs = {i.name: i for i in self.session.get_inputs()}
        self.input_names = set(inputs)
        self.past_names = sorted(n for n in inputs if n.startswith("past_key_values"))
        self.present_names = [o.name for o in self.session.get_outputs() if o.name.startswith("present")]
        # present.{i}.key -> past_key_values.{i}.key (match by name, never by position)
        self.present_to_past = {n: n.replace("present", "past_key_values", 1) for n in self.present_names}
        if set(self.present_to_past.values()) != set(self.past_names):
            raise ValueError("model's present.* outputs do not match its past_key_values.* inputs")
        past_type = inputs[self.past_names[0]].type if self.past_names else "tensor(float)"
        self.kv_dtype = np.float16 if "float16" in past_type else np.float32
        self._lock = threading.Lock()

    def _empty_past(self) -> dict[str, np.ndarray]:
        shape = (1, self.num_kv_heads, 0, self.head_dim)
        return {n: np.zeros(shape, dtype=self.kv_dtype) for n in self.past_names}

    def generate_ids(
        self, prompt_ids: list[int], max_new_tokens: int, min_new_tokens: int = 0
    ) -> tuple[list[int], float]:
        """Greedy-decode up to ``max_new_tokens``. Returns (new ids, time-to-first-token ms).

        EOS is suppressed for the first ``min_new_tokens`` steps, which the
        benchmark uses to time responses of a fixed length.
        """
        past = self._empty_past()
        ids = np.asarray([prompt_ids], dtype=np.int64)
        past_len, out_ids, ttft = 0, [], 0.0
        start = time.perf_counter()
        for step in range(max_new_tokens):
            n = ids.shape[1]
            feeds = {
                "input_ids": ids,
                "attention_mask": np.ones((1, past_len + n), dtype=np.int64),
                **past,
            }
            if "position_ids" in self.input_names:
                feeds["position_ids"] = np.arange(past_len, past_len + n, dtype=np.int64)[None, :]
            outputs = self.session.run(["logits", *self.present_names], feeds)
            logits = outputs[0][0, -1]
            if step < min_new_tokens:
                logits = logits.copy()
                logits[list(self.eos_ids)] = -np.inf
            next_id = int(np.argmax(logits))
            if step == 0:
                ttft = (time.perf_counter() - start) * 1000
            if next_id in self.eos_ids:
                break
            out_ids.append(next_id)
            past = {self.present_to_past[n]: v for n, v in zip(self.present_names, outputs[1:], strict=True)}
            past_len += n
            ids = np.asarray([[next_id]], dtype=np.int64)
        return out_ids, ttft

    def generate(self, user: str, system: str, max_new_tokens: int = 32) -> Generation:
        """Answer a single-turn chat message."""
        prompt_ids = self.tokenizer.encode(chatml_prompt(system, user), add_special_tokens=False).ids
        with self._lock:
            start = time.perf_counter()
            new_ids, ttft = self.generate_ids(prompt_ids, max_new_tokens)
            total = (time.perf_counter() - start) * 1000
        return Generation(
            self.tokenizer.decode(new_ids, skip_special_tokens=True), len(new_ids), len(prompt_ids), ttft, total
        )


def download(dest: str | Path, repo: str = DEFAULT_REPO, files: tuple[str, ...] = DEFAULT_FILES) -> Path:
    """Fetch the generator files from the Hub into ``dest``."""
    from huggingface_hub import hf_hub_download

    dest = Path(dest)
    for f in files:
        hf_hub_download(repo, f, local_dir=dest)
    return dest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download and smoke-test the generative model.")
    ap.add_argument("--dest", default="models/qwen2.5-0.5b-instruct")
    ap.add_argument("--prompt", default="What documents are needed for a ration card in India?")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()
    d = download(args.dest)
    g = OnnxGenerator(d)
    res = g.generate(args.prompt, "You are a helpful assistant. Answer briefly.", args.max_new_tokens)
    print(json.dumps(res.__dict__ | {"ms_per_token": res.ms_per_token}, ensure_ascii=False, indent=2))
