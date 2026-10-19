"""Fast in-process cluster simulator with partitions, message loss and restarts.

Every simulated node runs the real production code: ``ReplicatedStore``
(SQLite on disk), ``AntiEntropy``, ``Membership`` and ``SyncNode``. Only the
transport and the clock are simulated:

* **Transport**: requests are JSON round-tripped like on the wire. Each request
  and each response is dropped independently with probability ``loss``.
  A dropped response means the server already applied the request, which is
  the hard case for replication.
* **Partitions**: when no partition is active, a new one starts with probability
  ``partition_prob`` at the beginning of a window and lasts
  ``partition_windows`` windows. Nodes are split at random into
  ``partition_groups`` groups that cannot talk to each other.
* **Restarts**: each node restarts with probability ``restart_prob`` per window,
  at a random round. It closes its SQLite store, reopens it from disk, and loses
  all in-memory state (membership view, RNG).
* **Clocks**: each node's physical clock has a random fixed skew of up to
  ``max_skew_s``, the way an RTC-less Pi drifts without NTP. Ordering relies on HLCs.

Workload: before each window (the offline period), every node writes new
query-log records, updates shared case records (so concurrent writes to the
same key conflict across nodes), deletes a few cases, and bumps counters.

Metrics, computed after each window over the union of all record keys:

* ``identical_pct``: % of records whose (timestamp, tombstone, value) is
  byte-identical on all replicas. This is the convergence / data-integrity metric.
* ``correct_pct``: % of records where every replica holds the true LWW winner
  among all versions ever written (no lost or stale writes).
* ``counters_identical_pct``: the same for PN-counter slots.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from edgecluster.cluster.gossip import Membership
from edgecluster.cluster.node import SyncNode
from edgecluster.cluster.scheduler import ManualClock, UplinkSchedule, UplinkWindow
from edgecluster.sync.antientropy import TransportError
from edgecluster.sync.crdt import LWWRegister
from edgecluster.sync.store import Entry, ReplicatedStore

LABELS = ("pension", "ration_card", "housing", "health_scheme", "education_scholarship", "documents_id", "other")


@dataclass
class SimConfig:
    """Simulation parameters (all reported alongside results)."""

    nodes: int = 12
    windows: int = 10
    rounds_per_window: int = 6
    fanout: int = 2
    loss: float = 0.2
    partition_prob: float = 0.3
    partition_windows: int = 1
    partition_groups: int = 2
    restart_prob: float = 0.05
    writes_per_node: int = 20
    update_frac: float = 0.3
    delete_frac: float = 0.05
    case_pool: int = 200
    write_windows: int | None = None  # writes before windows 1..write_windows (None = every window)
    quiesce_windows: int = 20  # extra windows without writes to measure time-to-100%
    page_size: int = 200
    max_pages: int = 5
    max_skew_s: float = 300.0
    period_s: float = 86400.0
    window_s: float = 7200.0
    seed: int = 0


@dataclass
class WindowStats:
    """Metrics after one window."""

    window: int
    phase: str  # "write" or "quiesce"
    records: int
    identical_pct: float
    correct_pct: float
    counters_identical_pct: float
    partitioned: bool
    restarts: int
    messages_sent: int
    messages_lost: int


@dataclass
class SimResult:
    """Full output of one simulation run."""

    config: SimConfig
    windows: list[WindowStats] = field(default_factory=list)
    windows_to_full_convergence: int | None = None
    total_writes: int = 0
    wall_s: float = 0.0

    @property
    def final_write_phase(self) -> WindowStats:
        """Stats after the last window that had writes before it (window K)."""
        return [w for w in self.windows if w.phase == "write"][-1]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly result."""
        return {
            "config": asdict(self.config),
            "windows": [asdict(w) for w in self.windows],
            "windows_to_full_convergence": self.windows_to_full_convergence,
            "total_writes": self.total_writes,
            "wall_s": round(self.wall_s, 2),
        }


class SimNetwork:
    """In-process message bus with loss and partitions."""

    def __init__(self, rng: random.Random, loss: float) -> None:
        self.rng = rng
        self.loss = loss
        self.nodes: dict[str, SyncNode] = {}
        self.group: dict[str, int] = {}
        self.sent = 0
        self.lost = 0

    def reachable(self, a: str, b: str) -> bool:
        """Whether a and b are on the same side of the current partition."""
        return self.group.get(a, 0) == self.group.get(b, 0)

    def deliver(self, src: str, dst: str, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a request and return the response, or raise TransportError."""
        self.sent += 1
        if not self.reachable(src, dst):
            raise TransportError(f"{src} and {dst} are partitioned")
        if self.rng.random() < self.loss:
            self.lost += 1
            raise TransportError("request lost")
        target = self.nodes[dst]
        resp = target.handle(op, json.loads(json.dumps(payload)))
        if self.rng.random() < self.loss:
            self.lost += 1
            raise TransportError("response lost")
        return json.loads(json.dumps(resp))


class SimTransport:
    """``PeerTransport`` bound to one source node."""

    def __init__(self, net: SimNetwork, src: str) -> None:
        self.net = net
        self.src = src

    def request(self, peer_id: str, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Route through the simulated network."""
        return self.net.deliver(self.src, peer_id, op, payload)


class Simulator:
    """Drives N nodes through K uplink windows."""

    def __init__(self, cfg: SimConfig, workdir: str | Path | None = None) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self._own_dir = workdir is None
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="edgesim-"))
        self.clock = ManualClock(0.0)
        self.net = SimNetwork(random.Random(cfg.seed + 1), cfg.loss)
        self.ids = [f"node{i + 1}" for i in range(cfg.nodes)]
        self.skew_ms = {n: int(self.rng.uniform(-cfg.max_skew_s, cfg.max_skew_s) * 1000) for n in self.ids}
        # Every window opens at offset 0 of the period, for window_s seconds.
        self.schedule = UplinkSchedule([UplinkWindow(0.0, cfg.window_s)], period_s=cfg.period_s)
        self.truth: dict[tuple[str, str], LWWRegister] = {}
        self.writes = 0
        self.partition_left = 0
        for n in self.ids:
            self._start(n)

    # ------------------------------------------------------------------ node lifecycle
    def _physical_ms(self, node_id: str) -> int:
        # Simulated epoch starts at 2026-01-01 so timestamps look realistic.
        return int(1_767_225_600_000 + self.clock.now() * 1000) + self.skew_ms[node_id]

    def _start(self, node_id: str) -> None:
        store = ReplicatedStore(
            self.workdir / f"{node_id}.db", node_id, physical_ms=lambda n=node_id: self._physical_ms(n)
        )
        membership = Membership(node_id, self.ids, self.clock.now, suspect_after_s=self.cfg.window_s)
        self.net.nodes[node_id] = SyncNode(
            store,
            SimTransport(self.net, node_id),
            membership,
            self.schedule,
            self.clock,
            fanout=self.cfg.fanout,
            rng=random.Random(self.rng.getrandbits(32)),
            page_size=self.cfg.page_size,
            max_pages=self.cfg.max_pages,
        )

    def restart(self, node_id: str) -> None:
        """Crash-restart: drop all in-memory state, reopen the store from disk."""
        self.net.nodes[node_id].store.close()
        self._start(node_id)

    def close(self) -> None:
        """Close stores and remove the temporary directory."""
        for node in self.net.nodes.values():
            node.store.close()
        if self._own_dir:
            shutil.rmtree(self.workdir, ignore_errors=True)

    # ------------------------------------------------------------------ workload
    def _record(self, entry: Entry) -> None:
        key = (entry.collection, entry.key)
        reg = self.truth.get(key, LWWRegister())
        self.truth[key] = reg.merge(entry.register())
        self.writes += 1

    def write_phase(self) -> None:
        """Offline period: every node writes locally."""
        cfg = self.cfg
        for node_id in self.ids:
            store = self.net.nodes[node_id].store
            for _ in range(cfg.writes_per_node):
                r = self.rng.random()
                case_key = f"case-{self.rng.randrange(cfg.case_pool):04d}"
                if r < cfg.delete_frac:
                    self._record(store.delete("cases", case_key))
                elif r < cfg.delete_frac + cfg.update_frac:
                    value = {
                        "status": self.rng.choice(["open", "documents_pending", "submitted", "approved"]),
                        "scheme": self.rng.choice(LABELS[:-1]),
                        "updated_by": node_id,
                    }
                    self._record(store.put("cases", case_key, value))
                else:
                    label = self.rng.choice(LABELS)
                    key = f"{node_id}-{self.rng.getrandbits(48):012x}"
                    value = {"label": label, "node": node_id, "lang": self.rng.choice(["hi", "ur", "doi"])}
                    self._record(store.put("queries", key, value))
                    store.increment(f"queries:{label}", 1)

    # ------------------------------------------------------------------ faults
    def _update_partition(self) -> bool:
        cfg = self.cfg
        if self.partition_left > 0:
            self.partition_left -= 1
        elif cfg.partition_windows > 0 and self.rng.random() < cfg.partition_prob:
            self.partition_left = cfg.partition_windows - 1
            groups = list(range(cfg.partition_groups))
            ids = self.ids[:]
            self.rng.shuffle(ids)
            # Guarantee every group is non-empty, then assign the rest at random.
            self.net.group = {n: groups[i] if i < len(groups) else self.rng.choice(groups) for i, n in enumerate(ids)}
            return True
        else:
            self.net.group = {}
        return bool(self.net.group)

    # ------------------------------------------------------------------ metrics
    def measure(self) -> tuple[int, float, float, float]:
        """(records, identical %, correct %, counters identical %)."""
        snaps = [self.net.nodes[n].store.snapshot() for n in self.ids]
        keys = set().union(*[s.keys() for s in snaps]) | set(self.truth)
        identical = correct = 0
        for k in keys:
            versions = [s.get(k) for s in snaps]
            if all(v is not None and v == versions[0] for v in versions):
                identical += 1
                truth = self.truth.get(k)
                v0 = versions[0]
                assert v0 is not None
                if truth is not None and v0[0] == truth.timestamp.encode():
                    correct += 1
        csnaps = [self.net.nodes[n].store.counter_snapshot() for n in self.ids]
        ckeys = set().union(*[s.keys() for s in csnaps])
        cident = sum(1 for k in ckeys if all(s.get(k) == csnaps[0].get(k) for s in csnaps))
        total = len(keys) or 1
        return (
            len(keys),
            100.0 * identical / total,
            100.0 * correct / total,
            100.0 * cident / (len(ckeys) or 1),
        )

    # ------------------------------------------------------------------ main loop
    def run_window(self, index: int, phase: str) -> WindowStats:
        """Offline writes (in the write phase), then one uplink window of gossip rounds."""
        cfg = self.cfg
        if phase == "write":
            # Writes happen in the offline gap before this window.
            self.clock.t = index * cfg.period_s - (cfg.period_s - cfg.window_s) / 2
            self.write_phase()
        # Jump to the start of the next window.
        self.clock.t = index * cfg.period_s
        partitioned = self._update_partition()
        restart_round = {
            n: self.rng.randrange(cfg.rounds_per_window) for n in self.ids if self.rng.random() < cfg.restart_prob
        }
        sent0, lost0 = self.net.sent, self.net.lost
        step = cfg.window_s / cfg.rounds_per_window
        for r in range(cfg.rounds_per_window):
            for n, rr in restart_round.items():
                if rr == r:
                    self.restart(n)
            order = self.ids[:]
            self.rng.shuffle(order)
            for n in order:
                self.net.nodes[n].run_round()
            self.clock.advance(step * 0.999)  # stay inside the window
        records, ident, correct, cident = self.measure()
        return WindowStats(
            index + 1,
            phase,
            records,
            round(ident, 3),
            round(correct, 3),
            round(cident, 3),
            partitioned,
            len(restart_round),
            self.net.sent - sent0,
            self.net.lost - lost0,
        )

    def run(self) -> SimResult:
        """Run the write phase (K windows) then quiesce until fully converged."""
        t0 = time.perf_counter()
        res = SimResult(self.cfg)
        write_windows = self.cfg.windows if self.cfg.write_windows is None else self.cfg.write_windows
        for w in range(self.cfg.windows):
            res.windows.append(self.run_window(w, "write" if w < write_windows else "quiesce"))
        for extra in range(self.cfg.quiesce_windows):
            last = res.windows[-1]
            if last.identical_pct == 100.0 and last.correct_pct == 100.0 and last.counters_identical_pct == 100.0:
                break
            res.windows.append(self.run_window(self.cfg.windows + extra, "quiesce"))
        last = res.windows[-1]
        if last.identical_pct == 100.0 and last.correct_pct == 100.0 and last.counters_identical_pct == 100.0:
            res.windows_to_full_convergence = len(res.windows)
        res.total_writes = self.writes
        res.wall_s = time.perf_counter() - t0
        return res


def simulate(cfg: SimConfig) -> SimResult:
    """Run one simulation in a temporary directory."""
    sim = Simulator(cfg)
    try:
        return sim.run()
    finally:
        sim.close()


def print_result(res: SimResult) -> None:
    """Human-readable per-window table."""
    cfg = res.config
    print(
        f"nodes={cfg.nodes} K={cfg.windows} rounds/window={cfg.rounds_per_window} fanout={cfg.fanout} "
        f"loss={cfg.loss} partition(p={cfg.partition_prob}, len={cfg.partition_windows}w, "
        f"groups={cfg.partition_groups}) restart_p={cfg.restart_prob} writes/node/window={cfg.writes_per_node} "
        f"seed={cfg.seed}"
    )
    print(
        f"{'win':>4} {'phase':>8} {'records':>8} {'identical%':>11} {'correct%':>9} {'counters%':>10} "
        f"{'part':>5} {'rst':>4} {'msgs':>6} {'lost':>6}"
    )
    for w in res.windows:
        print(
            f"{w.window:>4} {w.phase:>8} {w.records:>8} {w.identical_pct:>11.3f} {w.correct_pct:>9.3f} "
            f"{w.counters_identical_pct:>10.3f} {'yes' if w.partitioned else '-':>5} {w.restarts:>4} "
            f"{w.messages_sent:>6} {w.messages_lost:>6}"
        )
    k = res.final_write_phase
    print(
        f"\nAfter K={k.window} windows (writes still arriving): {k.identical_pct:.3f}% records identical on all "
        f"{cfg.nodes} replicas; {k.correct_pct:.3f}% hold the true latest version."
    )
    if res.windows_to_full_convergence is not None:
        print(
            f"100% identical and correct after {res.windows_to_full_convergence} windows "
            f"({res.windows_to_full_convergence - k.window} quiet windows after the last writes)."
        )
    else:
        print(f"Did NOT reach 100% within {cfg.quiesce_windows} quiet windows.")
    print(f"total writes={res.total_writes} wall={res.wall_s:.1f}s")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = SimConfig()
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        if name == "write_windows":
            ap.add_argument(flag, type=int, default=None)
        else:
            ap.add_argument(flag, type=type(value), default=value)
    ap.add_argument("--json", help="write the full result as JSON to this path")
    ap.add_argument("--headline", help="write a one-line summary JSON (read by bench/latency.py)")
    args = ap.parse_args()
    cfg = SimConfig(**{k: getattr(args, k) for k in asdict(defaults)})
    res = simulate(cfg)
    print_result(res)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res.to_dict(), indent=2))
    if args.headline:
        k = res.final_write_phase
        summary = {
            "k": k.window,
            "identical_pct_at_k": k.identical_pct,
            "correct_pct_at_k": k.correct_pct,
            "windows_to_100": res.windows_to_full_convergence,
            "settings": (
                f"in-process sim: {cfg.nodes} replicas, {cfg.rounds_per_window} rounds/window, fanout {cfg.fanout}, "
                f"{cfg.loss:.0%} message loss, partitions p={cfg.partition_prob} x {cfg.partition_windows} window(s), "
                f"restart p={cfg.restart_prob}, {cfg.writes_per_node} writes/node/window, seed {cfg.seed}"
            ),
        }
        Path(args.headline).parent.mkdir(parents=True, exist_ok=True)
        Path(args.headline).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
