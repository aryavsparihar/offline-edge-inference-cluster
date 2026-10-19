"""The JSONL example format used throughout the project (see ``data/README.md``)."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

WELFARE_LABELS = (
    "pension",
    "ration_card",
    "housing",
    "health_scheme",
    "education_scholarship",
    "documents_id",
    "other",
)
LANGS = ("hi", "ur", "doi")


@dataclass(frozen=True)
class Example:
    """One labeled query."""

    text: str
    label: str
    lang: str


def load_jsonl(path: str | Path, include_needs_review: bool = True) -> list[Example]:
    """Read examples from a JSONL file.

    Blank lines and lines starting with ``#`` are comments. Records with
    ``"needs_review": true`` (unreviewed Dogri drafts) can be excluded.
    """
    out: list[Example] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            missing = {"text", "label", "lang"} - rec.keys()
            if missing:
                raise ValueError(f"{path}:{lineno}: missing fields {sorted(missing)}")
            if not str(rec["text"]).strip():
                raise ValueError(f"{path}:{lineno}: empty text")
            if rec.get("needs_review") and not include_needs_review:
                continue
            out.append(Example(str(rec["text"]), str(rec["label"]), str(rec["lang"])))
    return out


def load_many(paths: Iterable[str | Path], include_needs_review: bool = True) -> list[Example]:
    """Concatenate several JSONL files."""
    return [ex for p in paths for ex in load_jsonl(p, include_needs_review)]


def write_jsonl(path: str | Path, examples: Iterable[Example]) -> None:
    """Write examples as JSONL (UTF-8, no ASCII escaping)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def stratified_split(examples: Sequence[Example], fractions: Sequence[float], seed: int = 13) -> list[list[Example]]:
    """Split per label so every part keeps the label distribution.

    ``fractions`` must sum to 1. Each label contributes at least one example to
    every part when it has enough examples.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    by_label: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_label[ex.label].append(ex)
    rng = random.Random(seed)
    parts: list[list[Example]] = [[] for _ in fractions]
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        n = len(items)
        counts = [round(f * n) for f in fractions[:-1]]
        if n >= len(fractions):
            counts = [max(1, c) for c in counts]
        start = 0
        for i, c in enumerate(counts):
            parts[i].extend(items[start : start + c])
            start += c
        parts[-1].extend(items[start:])
    for p in parts:
        rng.shuffle(p)
    return parts


def label_list(examples: Iterable[Example]) -> list[str]:
    """Sorted unique labels."""
    return sorted({ex.label for ex in examples})
