"""The in-process simulator converges and its metric detects divergence."""

from __future__ import annotations

from sim.simulator import SimConfig, Simulator, simulate


def test_converges_to_100_percent_under_faults() -> None:
    cfg = SimConfig(nodes=6, windows=4, rounds_per_window=4, loss=0.3, partition_prob=0.5,
                    partition_windows=2, restart_prob=0.2, writes_per_node=10, seed=5)
    res = simulate(cfg)
    assert res.windows_to_full_convergence is not None
    final = res.windows[-1]
    assert final.identical_pct == final.correct_pct == final.counters_identical_pct == 100.0
    assert res.total_writes == 6 * 4 * 10


def test_metric_detects_divergence_without_sync() -> None:
    cfg = SimConfig(nodes=4, windows=1, rounds_per_window=1, loss=1.0, partition_windows=0,
                    restart_prob=0.0, writes_per_node=5, quiesce_windows=0, seed=1)
    res = simulate(cfg)
    assert res.windows[0].identical_pct == 0.0  # every record exists on one replica only
    assert res.windows_to_full_convergence is None


def test_deterministic_for_seed() -> None:
    cfg = SimConfig(nodes=4, windows=2, writes_per_node=5, seed=3)
    a, b = simulate(cfg), simulate(cfg)
    assert [w.identical_pct for w in a.windows] == [w.identical_pct for w in b.windows]


def test_restart_preserves_data(tmp_path) -> None:  # noqa: ANN001
    sim = Simulator(SimConfig(nodes=2, windows=1, writes_per_node=3, seed=0), workdir=tmp_path)
    try:
        sim.write_phase()
        before = sim.net.nodes["node1"].store.snapshot()
        sim.restart("node1")
        assert sim.net.nodes["node1"].store.snapshot() == before
    finally:
        sim.close()
