"""Behavioural tests for the CRDTs beyond the lattice laws."""

from __future__ import annotations

import random

import pytest

from edgecluster.sync.crdt import GCounter, LWWMap, LWWRegister, ORSet, PNCounter
from edgecluster.sync.hlc import Timestamp


def ts(wall: int, node: str = "a", logical: int = 0) -> Timestamp:
    return Timestamp(wall, logical, node)


def test_gcounter_counts_across_replicas() -> None:
    a, b = GCounter(), GCounter()
    a.increment("a", 3)
    b.increment("b", 2)
    b.increment("b")
    assert a.merge(b).value == 6
    with pytest.raises(ValueError):
        a.increment("a", -1)


def test_pncounter_decrements() -> None:
    a, b = PNCounter(), PNCounter()
    a.increment("a", 5)
    b.decrement("b", 2)
    a.increment("a", -1)
    assert a.merge(b).value == 2


def test_lww_register_latest_wins_and_ties_deterministic() -> None:
    r = LWWRegister()
    assert r.assign("old", ts(1))
    assert r.assign("new", ts(2))
    assert not r.assign("stale", ts(1, "z"))
    assert r.value == "new"
    x, y = LWWRegister("x", ts(5, "a")), LWWRegister("y", ts(5, "b"))
    assert x.merge(y).value == y.merge(x).value == "y"  # node id breaks the tie


def test_orset_add_wins_over_concurrent_remove() -> None:
    a, b = ORSet(), ORSet()
    a.add("pension", "a")
    b = b.merge(a)
    b.remove("pension")  # b removes the tag it observed
    a.add("pension", "a")  # concurrently, a re-adds with a fresh tag
    merged = a.merge(b)
    assert "pension" in merged
    assert merged == b.merge(a)


def test_orset_observed_remove() -> None:
    s = ORSet()
    s.add("x", "a")
    s.add("y", "b")
    s.remove("x")
    assert s.value == frozenset({"y"})
    s.add("x", "a")
    assert "x" in s


def test_lwwmap_delete_beats_older_write_but_not_newer() -> None:
    a, b = LWWMap(), LWWMap()
    a.set("case-1", {"status": "open"}, ts(1, "a"))
    b = b.merge(a)
    b.delete("case-1", ts(2, "b"))
    assert "case-1" not in a.merge(b)
    a.set("case-1", {"status": "reopened"}, ts(3, "a"))
    assert a.merge(b).get("case-1") == {"status": "reopened"}


def test_random_ops_converge_under_any_delivery_order() -> None:
    """Three replicas apply random ops, then exchange states in random orders."""
    rng = random.Random(7)
    for _ in range(50):
        reps = [LWWMap() for _ in range(3)]
        clocks = [0, 0, 0]
        for _ in range(30):
            i = rng.randrange(3)
            clocks[i] += rng.randint(0, 2)
            t = Timestamp(clocks[i], 0, f"n{i}")
            key = rng.choice(["a", "b", "c"])
            if rng.random() < 0.2:
                reps[i].delete(key, t)
            else:
                reps[i].set(key, rng.randint(0, 9), t)
            if rng.random() < 0.3:  # partial gossip
                j = rng.randrange(3)
                reps[j] = reps[j].merge(reps[i])
        for _ in range(4):
            order = [(i, j) for i in range(3) for j in range(3) if i != j]
            rng.shuffle(order)
            for i, j in order:
                reps[j] = reps[j].merge(reps[i])
        assert reps[0] == reps[1] == reps[2]
