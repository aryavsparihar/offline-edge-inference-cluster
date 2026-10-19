"""Start a real N-node cluster on localhost, partition it, heal it, watch it converge.

    python -m sim.cluster_demo --nodes 3

Each node is a separate ``python -m edgecluster.serving`` process with its own
SQLite replica, syncing over HTTP inside a compressed uplink schedule (a
``--window``-second window every ``--period`` seconds). The partition is
emulated with ``POST /admin/links`` on each node, which drops traffic to and
from the listed peers. That is the same failure the sync layer sees when a
site's radio link is down.

The demo:
1. writes records on every node and waits for all digests to match;
2. partitions the cluster ({node1..node_{N-1}} | {node_N}) and makes
   conflicting updates to the same case record on both sides, plus new
   records on each side;
3. shows that the replicas have diverged; then heals the partition;
4. waits for convergence and checks every replica holds the identical,
   newest version of every record.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import yaml


def write_config(path: Path, n: int, base_port: int, data_dir: Path, period: float, window: float,
                 interval: float, model_dir: str | None) -> None:
    """Shared cluster config used by every node process."""
    cfg: dict[str, Any] = {
        "cluster": {
            "peers": [{"id": f"node{i}", "url": f"http://127.0.0.1:{base_port + i}"} for i in range(1, n + 1)],
            "uplink": {"period": period, "utc_offset": 0, "windows": [{"start": 0, "duration": window}]},
            "sync": {"interval": interval, "fanout": 2, "timeout": 2, "suspect_after": period},
        },
        "host": "127.0.0.1",
        "data_dir": str(data_dir),
        "admin_enabled": True,
    }
    if model_dir:
        cfg["classifier"] = {"model_dir": model_dir, "model_file": "model.pruned.int8.onnx"}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def start_nodes(n: int, base_port: int, config: Path, log_dir: Path) -> list[subprocess.Popen[bytes]]:
    """Launch node processes."""
    procs = []
    for i in range(1, n + 1):
        log = open(log_dir / f"node{i}.log", "wb")  # noqa: SIM115 - closed with the process
        procs.append(
            subprocess.Popen(
                [sys.executable, "-m", "edgecluster.serving", "--config", str(config),
                 "--node-id", f"node{i}", "--port", str(base_port + i), "--log-level", "warning"],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        )
    return procs


def wait_healthy(client: httpx.Client, urls: list[str], timeout: float = 60) -> None:
    """Block until every node answers /health."""
    deadline = time.time() + timeout
    pending = set(urls)
    while pending and time.time() < deadline:
        for u in list(pending):
            try:
                if client.get(f"{u}/health").status_code == 200:
                    pending.discard(u)
            except httpx.HTTPError:
                pass
        time.sleep(0.3)
    if pending:
        raise RuntimeError(f"nodes not healthy: {sorted(pending)}")


def digests(client: httpx.Client, urls: list[str]) -> list[dict[str, Any]]:
    """Current state fingerprint of every node."""
    return [client.get(f"{u}/sync/digest").json() for u in urls]


def show(label: str, ds: list[dict[str, Any]], t0: float) -> bool:
    """Print one status line; returns True if all digests match."""
    same = len({d["digest"] for d in ds}) == 1
    cells = "  ".join(f"{d['node_id']}:{d['records']:>3} rec {d['digest'][:8]}" for d in ds)
    print(f"[{time.time() - t0:6.1f}s] {label:<26} {cells}  {'CONVERGED' if same else 'diverged'}")
    return same


def wait_converged(client: httpx.Client, urls: list[str], t0: float, label: str, timeout: float) -> float:
    """Poll until all digests match; returns seconds waited."""
    start = time.time()
    while time.time() - start < timeout:
        if show(label, digests(client, urls), t0):
            return time.time() - start
        time.sleep(1.0)
    raise RuntimeError(f"no convergence within {timeout}s")


def main() -> None:
    """Run the demo."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, default=3)
    ap.add_argument("--base-port", type=int, default=8100)
    ap.add_argument("--period", type=float, default=20.0, help="uplink period (s)")
    ap.add_argument("--window", type=float, default=12.0, help="window length within each period (s)")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between sync rounds in a window")
    ap.add_argument("--records", type=int, default=20, help="records written per node per phase")
    ap.add_argument("--partition-seconds", type=float, default=25.0)
    ap.add_argument("--model-dir", help="optional ONNX classifier dir to also serve /classify")
    ap.add_argument("--workdir", help="keep node data/logs here (default: temp dir)")
    args = ap.parse_args()
    n = args.nodes
    work = Path(args.workdir or tempfile.mkdtemp(prefix="edge-cluster-"))
    work.mkdir(parents=True, exist_ok=True)
    config = work / "cluster.yaml"
    write_config(config, n, args.base_port, work / "data", args.period, args.window, args.interval, args.model_dir)
    urls = [f"http://127.0.0.1:{args.base_port + i}" for i in range(1, n + 1)]
    print(f"starting {n} nodes (logs in {work}); uplink window {args.window:.0f}s every {args.period:.0f}s")
    procs = start_nodes(n, args.base_port, config, work)
    client = httpx.Client(timeout=5)
    t0 = time.time()
    try:
        wait_healthy(client, urls)
        print("all nodes healthy\n")

        print("phase 1: every node writes records locally")
        for i, u in enumerate(urls, 1):
            for j in range(args.records):
                client.post(f"{u}/records/queries", json={"key": f"q-node{i}-{j}", "value": {"label": "pension", "node": i}})
            client.post(f"{u}/records/cases", json={"key": "case-0001", "value": {"status": "open", "by": f"node{i}"}})
        took = wait_converged(client, urls, t0, "phase 1 sync", 120)
        print(f"-> converged {took:.1f}s after the writes\n")

        minority = [f"node{n}"]
        majority = [f"node{i}" for i in range(1, n)]
        print(f"phase 2: partition {majority} | {minority} for {args.partition_seconds:.0f}s, conflicting writes on both sides")
        for i, u in enumerate(urls, 1):
            me = f"node{i}"
            client.post(f"{u}/admin/links", json={"blocked": minority if me in majority else majority})
        client.post(f"{urls[0]}/records/cases", json={"key": "case-0001", "value": {"status": "documents_pending", "by": "node1"}})
        time.sleep(0.05)
        client.post(f"{urls[-1]}/records/cases", json={"key": "case-0001", "value": {"status": "approved", "by": f"node{n}"}})
        for i, u in enumerate(urls, 1):
            for j in range(args.records):
                client.post(f"{u}/records/queries", json={"key": f"p-node{i}-{j}", "value": {"label": "housing", "node": i}})
        end = time.time() + args.partition_seconds
        while time.time() < end:
            show("partitioned", digests(client, urls), t0)
            time.sleep(min(5.0, max(end - time.time(), 0)))
        print("\nphase 3: heal the partition")
        for u in urls:
            client.post(f"{u}/admin/links", json={"blocked": []})
        took = wait_converged(client, urls, t0, "healing", 180)
        print(f"-> converged {took:.1f}s after healing\n")

        cases = [client.get(f"{u}/records/cases/case-0001").json()["value"] for u in urls]
        counts = [client.get(f"{u}/records/queries", params={"limit": 1}).json()["total"] for u in urls]
        expected = n * args.records * 2
        print(f"case-0001 on every node: {cases[0]}  (identical everywhere: {all(c == cases[0] for c in cases)})")
        print(f"query records per node: {counts} (expected {expected} each)")
        ok = all(c == cases[0] for c in cases) and all(c == expected for c in counts)
        print("RESULT:", "PASS - all replicas identical after partition" if ok else "FAIL")
        stats = [client.get(f"{u}/cluster").json()["stats"] for u in urls]
        for i, s in enumerate(stats, 1):
            print(f"  node{i}: rounds={s['rounds']} ok={s['exchanges_ok']} failed={s['exchanges_failed']} "
                  f"pulled={s['entries_pulled']} pushed={s['entries_pushed']}")
        if not ok:
            sys.exit(1)
    finally:
        for p in procs:
            p.send_signal(signal.SIGINT)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
