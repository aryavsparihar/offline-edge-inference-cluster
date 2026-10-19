"""Uplink scheduler, gossip membership and configuration loading."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from edgecluster.cluster.gossip import Membership
from edgecluster.cluster.scheduler import (
    ManualClock,
    ScaledClock,
    UplinkSchedule,
    UplinkWindow,
    parse_duration,
    parse_offset,
)
from edgecluster.config import load_config


def test_parse_helpers() -> None:
    assert parse_duration("2h") == 7200
    assert parse_duration("90s") == 90
    assert parse_duration("1.5m") == 90
    assert parse_duration(5) == 5
    assert parse_offset("06:30") == 6 * 3600 + 1800
    assert parse_offset("00:00:10") == 10


def test_daily_window() -> None:
    s = UplinkSchedule([UplinkWindow(6 * 3600, 2 * 3600)])
    day = 86400 * 10
    assert not s.is_open(day + 5 * 3600)
    assert s.is_open(day + 6 * 3600)
    assert s.time_left(day + 7 * 3600) == 3600
    assert not s.is_open(day + 8 * 3600)
    assert s.time_until_open(day + 5 * 3600) == 3600
    assert s.time_until_open(day + 9 * 3600) == 21 * 3600


def test_window_wrapping_midnight_and_utc_offset() -> None:
    s = UplinkSchedule([UplinkWindow(23 * 3600, 2 * 3600)])
    assert s.is_open(23.5 * 3600) and s.is_open(86400 + 0.5 * 3600)
    assert not s.is_open(86400 + 1.5 * 3600)
    ist = UplinkSchedule([UplinkWindow(6 * 3600, 3600)], utc_offset_s=19800)  # 06:00 IST = 00:30 UTC
    assert ist.is_open(0.5 * 3600 + 60) and not ist.is_open(6 * 3600)


def test_always_open_and_validation() -> None:
    assert UplinkSchedule([], always_open=True).is_open(123)
    with pytest.raises(ValueError):
        UplinkSchedule([UplinkWindow(0, 0)])


def test_scaled_clock_runs_faster() -> None:
    c = ScaledClock(scale=1000, origin=0)
    import time

    time.sleep(0.01)
    assert c.now() >= 5


def test_membership_liveness_and_peer_choice() -> None:
    clock = ManualClock(0)
    a = Membership("a", ["a", "b", "c", "d"], clock.now, suspect_after_s=10, probe_suspected=0.0)
    b = Membership("b", ["a", "b", "c", "d"], clock.now, suspect_after_s=10)
    b.tick()
    a.merge(b.heartbeats())
    assert a.is_alive("b") and not a.is_alive("c")
    rng = random.Random(0)
    assert a.choose_peers(1, rng) == ["b"]
    assert set(a.choose_peers(3, rng)) == {"b", "c", "d"}  # fills up with suspected peers
    clock.advance(11)
    assert not a.is_alive("b")
    a.merge({"zzz": 5})  # unknown nodes are ignored (static membership)
    assert "zzz" not in a.state


def test_config_from_yaml_with_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        "cluster:\n"
        "  peers: [{id: node1, url: 'http://a:8000'}, {id: node2, url: 'http://b:8000'}]\n"
        "  uplink: {windows: [{start: '06:00', duration: 2h}]}\n"
        "data_dir: /tmp/x\n"
    )
    monkeypatch.setenv("EDGE_NODE_ID", "node2")
    monkeypatch.setenv("EDGE_UPLINK_ALWAYS_OPEN", "true")
    cfg = load_config(p)
    assert cfg.node_id == "node2"
    assert cfg.peer_urls() == {"node1": "http://a:8000"}
    assert cfg.uplink.always_open is True
    assert cfg.uplink.schedule().windows[0].duration_s == 7200
    with pytest.raises(ValueError):
        load_config(None, node_id="bad:id")
