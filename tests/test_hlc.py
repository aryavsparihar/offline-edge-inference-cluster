"""Hybrid logical clock properties."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from edgecluster.sync.hlc import ClockDriftError, HybridLogicalClock, Timestamp


@settings(max_examples=200, deadline=None)
@given(
    physical=st.lists(st.integers(0, 1000), min_size=1, max_size=50),
    remotes=st.lists(st.tuples(st.booleans(), st.integers(0, 2000), st.integers(0, 10)), max_size=50),
)
def test_hlc_is_strictly_monotonic_even_if_physical_clock_jumps_back(
    physical: list[int], remotes: list[tuple[bool, int, int]]
) -> None:
    it = iter(physical * 3)
    clock = HybridLogicalClock("a", physical_ms=lambda: next(it, physical[-1]))
    last = clock.now()
    for is_remote, wall, logical in remotes[: len(physical) * 2]:
        if is_remote:
            remote = Timestamp(wall, logical, "b")
            t = clock.update(remote)
            assert t > remote
        else:
            t = clock.now()
        assert t > last
        last = t


def test_hlc_survives_restart_with_clock_behind() -> None:
    c1 = HybridLogicalClock("a", physical_ms=lambda: 10_000)
    t1 = c1.now()
    c2 = HybridLogicalClock("a", physical_ms=lambda: 5_000, last=t1)  # RTC-less reboot
    assert c2.now() > t1


def test_hlc_encode_roundtrip_and_order() -> None:
    a, b = Timestamp(5, 1, "x"), Timestamp(5, 2, "a")
    assert Timestamp.decode(a.encode()) == a
    assert (a < b) == (a.encode() < b.encode())


def test_hlc_drift_guard() -> None:
    clock = HybridLogicalClock("a", physical_ms=lambda: 1000, max_drift_ms=500)
    clock.update(Timestamp(1400, 0, "b"))
    with pytest.raises(ClockDriftError):
        clock.update(Timestamp(5000, 0, "b"))


def test_node_id_validation() -> None:
    with pytest.raises(ValueError):
        HybridLogicalClock("bad:id")
