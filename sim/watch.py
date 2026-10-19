"""Poll nodes' ``/sync/digest`` and print convergence status until all match.

python -m sim.watch --nodes 12 --base-port 8000      # docker-compose cluster
python -m sim.watch --urls http://127.0.0.1:8101 http://127.0.0.1:8102
"""

from __future__ import annotations

import argparse
import time

import httpx


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urls", nargs="*")
    ap.add_argument("--nodes", type=int, default=12)
    ap.add_argument("--base-port", type=int, default=8000)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--until-converged", action="store_true")
    args = ap.parse_args()
    urls = args.urls or [f"http://127.0.0.1:{args.base_port + i}" for i in range(1, args.nodes + 1)]
    client = httpx.Client(timeout=3)
    t0 = time.time()
    while time.time() - t0 < args.timeout:
        rows = []
        for u in urls:
            try:
                d = client.get(f"{u}/sync/digest").json()
                rows.append((d["node_id"], d["records"], d["digest"][:10]))
            except httpx.HTTPError:
                rows.append((u, -1, "down"))
        digests = {r[2] for r in rows if r[1] >= 0}
        groups = len(digests)
        print(
            f"[{time.time() - t0:6.1f}s] {len(rows)} nodes, {groups} distinct state(s): "
            + " ".join(f"{n}={c}" for n, c, _ in rows)
        )
        if args.until_converged and groups == 1 and all(r[1] >= 0 for r in rows):
            print("converged: all replicas identical")
            return
        time.sleep(args.interval)
    raise SystemExit("timeout")


if __name__ == "__main__":
    main()
