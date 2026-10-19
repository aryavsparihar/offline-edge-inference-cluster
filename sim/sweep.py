"""Convergence sweep over message loss, partition length, sync rounds and restarts.

Every cell is the mean (and minimum) over several seeds. Nothing is
cherry-picked: the full grid is written to ``results/sim/sweep.md`` and
``sweep.json``.

    python -m sim.sweep --seeds 3 --out results/sim
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from sim.simulator import SimConfig, simulate

K_POINTS = (1, 3, 5, 10)


def run_one(cfg: SimConfig) -> dict[str, Any]:
    """Run a single simulation and keep the numbers the report needs."""
    res = simulate(cfg)
    curve = {w.window: w.identical_pct for w in res.windows if w.phase == "write"}
    correct = {w.window: w.correct_pct for w in res.windows if w.phase == "write"}
    return {
        "config": asdict(cfg),
        "identical_at_k": {k: curve[k] for k in K_POINTS if k in curve},
        "correct_at_k": {k: correct[k] for k in K_POINTS if k in correct},
        "windows_to_100": res.windows_to_full_convergence,
        "quiet_windows_to_100": None
        if res.windows_to_full_convergence is None
        else res.windows_to_full_convergence - cfg.windows,
        "records": res.windows[-1].records,
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/min over seeds for one grid cell."""
    out: dict[str, Any] = {}
    for k in K_POINTS:
        vals = [r["identical_at_k"][k] for r in runs if k in r["identical_at_k"]]
        if vals:
            out[f"k{k}_mean"] = round(statistics.fmean(vals), 2)
            out[f"k{k}_min"] = round(min(vals), 2)
    q = [r["quiet_windows_to_100"] for r in runs]
    out["converged_runs"] = sum(v is not None for v in q)
    out["runs"] = len(runs)
    done = [v for v in q if v is not None]
    out["quiet_windows_to_100_max"] = max(done) if done else None
    out["all_correct_at_end"] = all(r["windows_to_100"] is not None for r in runs)
    return out


def grid(base: SimConfig) -> dict[str, list[tuple[str, SimConfig]]]:
    """The sweeps reported in the README."""
    sweeps: dict[str, list[tuple[str, SimConfig]]] = {
        "loss_x_partition": [],
        "rounds": [],
        "restarts": [],
        "retries": [],
    }
    for loss in (0.0, 0.1, 0.2, 0.3, 0.5):
        for plen in (0, 1, 3):
            label = f"loss={loss:.1f} partition_len={plen}w"
            sweeps["loss_x_partition"].append((label, replace(base, loss=loss, partition_windows=plen)))
    for rounds in (1, 2, 3, 6, 12):
        sweeps["rounds"].append((f"rounds/window={rounds}", replace(base, rounds_per_window=rounds)))
    for rp in (0.0, 0.05, 0.2, 0.5):
        sweeps["restarts"].append((f"restart_p={rp}", replace(base, restart_prob=rp)))
    # Protocol ablation: bounded per-request retries vs none, without partitions.
    for loss in (0.2, 0.5):
        for retries in (0, 2):
            sweeps["retries"].append(
                (
                    f"loss={loss:.1f} retries={retries} (no partitions)",
                    replace(base, loss=loss, retries=retries, partition_windows=0),
                )
            )
    return sweeps


def to_markdown(results: dict[str, list[dict[str, Any]]], base: SimConfig, seeds: int) -> str:
    """Render the sweep as markdown tables."""
    lines = [
        "# Sync convergence sweep",
        "",
        f"Base settings: {base.nodes} nodes, K={base.windows} windows, {base.rounds_per_window} rounds/window, "
        f"fanout {base.fanout}, loss {base.loss} per message, partitions p={base.partition_prob} "
        f"lasting {base.partition_windows} window(s) into {base.partition_groups} groups, restart p={base.restart_prob}/node/window, "
        f"{base.writes_per_node} writes/node before each window ({base.update_frac:.0%} updates to "
        f"{base.case_pool} shared case records, {base.delete_frac:.0%} deletes), {base.retries} retries per request, "
        f"clock skew up to ±{base.max_skew_s:.0f}s. "
        f"Each cell: mean (min) over {seeds} seeds.",
        "",
        "`k=N`: % of records identical on all replicas right after window N, with new writes still arriving "
        "before every window. `quiet→100%`: extra windows without writes until every replica holds the identical, "
        "correct latest version of every record and counter (max over seeds).",
        "",
    ]
    for name, rows in results.items():
        lines += [
            f"## {name}",
            "",
            "| setting | k=1 | k=3 | k=5 | k=10 | converged | quiet→100% |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            a = row["agg"]
            cells = [
                f"{a.get(f'k{k}_mean', float('nan')):.2f} ({a.get(f'k{k}_min', float('nan')):.2f})" for k in K_POINTS
            ]
            q = a["quiet_windows_to_100_max"]
            lines.append(
                f"| {row['label']} | {' | '.join(cells)} | {a['converged_runs']}/{a['runs']} | "
                f"{'n/a' if q is None else q} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/sim")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nodes", type=int, default=SimConfig.nodes)
    ap.add_argument("--windows", type=int, default=SimConfig.windows)
    args = ap.parse_args()
    base = SimConfig(nodes=args.nodes, windows=args.windows)
    sweeps = grid(base)
    jobs = [
        (name, label, replace(cfg, seed=s))
        for name, cells in sweeps.items()
        for label, cfg in cells
        for s in range(args.seeds)
    ]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        outs = list(pool.map(run_one, [j[2] for j in jobs]))
    results: dict[str, list[dict[str, Any]]] = {}
    for name, cells in sweeps.items():
        results[name] = []
        for label, _ in cells:
            runs = [o for (n, lab, _), o in zip(jobs, outs, strict=True) if n == name and lab == label]
            results[name].append({"label": label, "agg": aggregate(runs), "runs": runs})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(
        json.dumps({"base": asdict(base), "seeds": args.seeds, "results": results}, indent=1)
    )
    # Headline = the base-settings cell (no cherry-picking), read by bench/latency.py.
    base_label = f"loss={base.loss:.1f} partition_len={base.partition_windows}w"
    cell = next(r for r in results["loss_x_partition"] if r["label"] == base_label)
    k = max(K_POINTS)
    headline = {
        "k": k,
        "identical_pct_at_k": cell["agg"][f"k{k}_mean"],
        "identical_pct_at_k_min": cell["agg"][f"k{k}_min"],
        "windows_to_100": None
        if cell["agg"]["quiet_windows_to_100_max"] is None
        else k + cell["agg"]["quiet_windows_to_100_max"],
        "settings": (
            f"in-process sim, mean of {args.seeds} seeds: {base.nodes} replicas, {base.rounds_per_window} rounds/window, "
            f"fanout {base.fanout}, {base.loss:.0%} loss per message, partitions p={base.partition_prob} x "
            f"{base.partition_windows} window, restart p={base.restart_prob}, {base.writes_per_node} writes/node/window"
        ),
    }
    (out / "headline.json").write_text(json.dumps(headline, indent=2))
    md = to_markdown(results, base, args.seeds)
    (out / "sweep.md").write_text(md + "\n")
    print(md)
    print(f"\n{len(jobs)} simulations in {time.perf_counter() - t0:.0f}s -> {out}/sweep.md")


if __name__ == "__main__":
    main()
