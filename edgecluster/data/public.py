"""Loader for a public Indic classification dataset from the Hugging Face Hub.

The original Hindi/Urdu/Dogri welfare-query corpus is not part of this
repository. To train and evaluate the compression pipeline on real data we
use **IndicGLUE** (``ai4bharat/indic_glue``, Kakwani et al., 2020), which
the Hub serves as parquet files (no loading script needed).

Supported configs are those with a ``text`` and a ``label`` column. The
default is ``bbca.hi``: BBC Hindi news articles labeled by topic (train 3,467 /
test 866). It is a *topic* classification task, not welfare-intent
classification, so it serves as a realistic multilingual proxy for measuring
how much accuracy compression costs.

Usage::

    python -m edgecluster.data.public --config bbca.hi --out data/public
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from edgecluster.data.schema import Example, stratified_split, write_jsonl

DATASET = "ai4bharat/indic_glue"
TEXT_CONFIGS = {"bbca.hi": "hi", "iitp-mr.hi": "hi", "iitp-pr.hi": "hi", "actsa-sc.te": "te"}


def load_indic_glue(config: str = "bbca.hi", revision: str | None = None) -> dict[str, list[Example]]:
    """Download an IndicGLUE text-classification config and convert it to :class:`Example`.

    Returns ``train``/``validation``/``test`` splits. If the config has no
    validation split (``bbca.hi``), 10% of train is held out, stratified with a
    fixed seed, so the test split is only ever used for final evaluation.
    """
    if config not in TEXT_CONFIGS:
        raise ValueError(f"unsupported config {config!r}; choose from {sorted(TEXT_CONFIGS)}")
    from datasets import load_dataset  # heavy import, only needed here

    ds = load_dataset(DATASET, config, revision=revision)
    lang = TEXT_CONFIGS[config]
    feature = ds["train"].features["label"]
    to_name = feature.int2str if hasattr(feature, "int2str") else str

    def convert(split: str) -> list[Example]:
        return [
            Example(str(row["text"]).strip(), str(to_name(row["label"])), lang)
            for row in ds[split]
            if str(row["text"]).strip()
        ]

    splits = {"train": convert("train"), "test": convert("test")}
    if "validation" in ds:
        splits["validation"] = convert("validation")
    else:
        splits["train"], splits["validation"] = stratified_split(splits["train"], [0.9, 0.1], seed=13)
    return splits


def main() -> None:
    """CLI: download, convert and write JSONL splits."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="bbca.hi", choices=sorted(TEXT_CONFIGS))
    ap.add_argument("--out", default="data/public")
    ap.add_argument("--revision", default=None, help="pin a Hub commit for reproducibility")
    args = ap.parse_args()
    splits = load_indic_glue(args.config, args.revision)
    out = Path(args.out) / args.config
    for name, examples in splits.items():
        write_jsonl(out / f"{name}.jsonl", examples)
        dist = Counter(ex.label for ex in examples)
        print(f"{name:10s} {len(examples):6d} examples -> {out / f'{name}.jsonl'}  labels={dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
