"""Dataset format and the shipped demo sample."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from edgecluster.compression.metrics import accuracy, macro_f1
from edgecluster.data.schema import WELFARE_LABELS, Example, load_jsonl, stratified_split, write_jsonl

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


def test_sample_files_follow_schema() -> None:
    for lang in ("hi", "ur", "doi"):
        rows = load_jsonl(SAMPLE / f"{lang}.jsonl")
        assert rows, lang
        assert {r.lang for r in rows} == {lang}
        assert {r.label for r in rows} <= set(WELFARE_LABELS)
    assert Counter(r.label for r in load_jsonl(SAMPLE / "hi.jsonl")) == {lab: 6 for lab in WELFARE_LABELS}


def test_dogri_drafts_are_flagged_for_review() -> None:
    text = (SAMPLE / "doi.jsonl").read_text(encoding="utf-8")
    assert text.count("# TODO: native-speaker review") == len(load_jsonl(SAMPLE / "doi.jsonl"))
    assert load_jsonl(SAMPLE / "doi.jsonl", include_needs_review=False) == []


def test_jsonl_roundtrip_and_errors(tmp_path: Path) -> None:
    rows = [Example("پنشن کب آئے گی", "pension", "ur")]
    write_jsonl(tmp_path / "x.jsonl", rows)
    assert load_jsonl(tmp_path / "x.jsonl") == rows
    (tmp_path / "bad.jsonl").write_text('{"text": "x"}\n')
    with pytest.raises(ValueError, match="missing fields"):
        load_jsonl(tmp_path / "bad.jsonl")


def test_stratified_split_keeps_every_label() -> None:
    rows = load_jsonl(SAMPLE / "hi.jsonl") + load_jsonl(SAMPLE / "ur.jsonl")
    train, val, test = stratified_split(rows, [0.6, 0.2, 0.2], seed=1)
    assert len(train) + len(val) + len(test) == len(rows)
    for part in (train, val, test):
        assert {r.label for r in part} == set(WELFARE_LABELS)
    assert stratified_split(rows, [0.6, 0.2, 0.2], seed=1)[0] == train


def test_metrics() -> None:
    assert accuracy(["a", "b", "b"], ["a", "b", "a"]) == pytest.approx(2 / 3)
    assert macro_f1(["a", "b"], ["a", "b"]) == 1.0
    assert macro_f1(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx((2 / 3 + 2 / 3) / 2)
