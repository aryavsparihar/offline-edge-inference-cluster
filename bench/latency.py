"""Latency benchmark under continuous load, meant to run on the Raspberry Pi.

    python -m bench.latency --model-dir runs/xlmr-bbca/onnx --model-file model.pruned.int8.onnx

Classification: closed-loop continuous load (each worker sends the next
request as soon as the previous one returns) for ``--duration`` seconds after
a warm-up. Batch size 1, every input padded/truncated to a fixed
``--max-seq-len`` (default 128), so the latency does not depend on how short
the probe texts are. Latency is tokenize + ONNX Runtime ``run`` (in-process),
or the full HTTP round trip with ``--url``.

Generation (``--generator-dir``): wall time to produce one response of
exactly ``--gen-tokens`` new tokens (EOS suppressed until then) from a fixed
chat prompt, greedy decoding, batch 1. Time-to-first-token and ms/token are
also reported.

The final table has the same rows as the project's target table. Size and
accuracy are read from the compression report JSON if given. Sync integrity
is read from the simulator's headline JSON if present. Every value printed
is measured, either here or by those scripts.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil

TARGETS = {
    "size": "1.1GB fp32 → ~260MB (4.2x)",
    "accuracy": "≥91% of fp32 accuracy retained",
    "cls_latency": "~380ms",
    "gen_latency": "~1.2s",
    "sync": "~99.2%",
}


# ---------------------------------------------------------------------- hardware
def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip("\x00 \n")
    except OSError:
        return None


def hardware_info() -> dict[str, Any]:
    """Describe the machine the benchmark ran on (Pi model, CPU, RAM, governor, temperature)."""
    info: dict[str, Any] = {
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 2),
    }
    model = _read("/proc/device-tree/model")
    if model:
        info["board"] = model
    cpuinfo = _read("/proc/cpuinfo") or ""
    for line in cpuinfo.splitlines():
        if line.lower().startswith(("model name", "hardware")):
            info["cpu"] = line.split(":", 1)[1].strip()
            break
    if "cpu" not in info:
        info["cpu"] = platform.processor() or platform.machine()
        if platform.system() == "Darwin":
            with contextlib.suppress(OSError, subprocess.CalledProcessError):
                info["cpu"] = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
                ).stdout.strip()
    gov = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov:
        info["cpu_governor"] = gov
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
    except ImportError:
        pass
    return info


def cpu_temp_c() -> float | None:
    """SoC temperature (Linux thermal zone), if available."""
    raw = _read("/sys/class/thermal/thermal_zone0/temp")
    return round(int(raw) / 1000, 1) if raw and raw.isdigit() else None


def throttled_flags() -> str | None:
    """Raspberry Pi firmware throttling flags (``vcgencmd get_throttled``)."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2)
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


class RssSampler:
    """Samples a process's RSS in the background."""

    def __init__(self, pid: int | None = None, interval: float = 0.25) -> None:
        self.proc = psutil.Process(pid)
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self.proc.memory_info().rss)
            except psutil.Error:
                return
            self._stop.wait(self.interval)

    def __enter__(self) -> RssSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict[str, float]:
        """Peak and mean RSS in MB."""
        if not self.samples:
            return {}
        return {"rss_peak_mb": max(self.samples) / 2**20, "rss_mean_mb": float(np.mean(self.samples)) / 2**20}


# ---------------------------------------------------------------------- load generation
def run_load(call: Callable[[int], None], duration: float, warmup: float, concurrency: int) -> dict[str, Any]:
    """Closed-loop load: ``concurrency`` workers call ``call(i)`` back-to-back."""
    lat: list[list[float]] = [[] for _ in range(concurrency)]
    errors = [0] * concurrency
    stop_at = [0.0]
    measuring = threading.Event()

    def worker(w: int) -> None:
        i = w
        while True:
            now = time.perf_counter()
            if stop_at[0] and now >= stop_at[0]:
                return
            t0 = time.perf_counter()
            try:
                call(i)
            except Exception:  # count, keep going: the benchmark must survive transient errors
                errors[w] += 1
                continue
            if measuring.is_set():
                lat[w].append((time.perf_counter() - t0) * 1000)
            i += concurrency

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(concurrency)]
    start = time.perf_counter()
    stop_at[0] = start + warmup + duration
    for t in threads:
        t.start()
    time.sleep(warmup)
    measuring.set()
    t_meas = time.perf_counter()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t_meas
    flat = np.asarray([x for w in lat for x in w])
    if flat.size == 0:
        raise RuntimeError("no successful requests during the measurement window")
    return {
        "requests": int(flat.size),
        "errors": int(sum(errors)),
        "duration_s": round(elapsed, 2),
        "throughput_rps": round(flat.size / elapsed, 2),
        "p50_ms": round(float(np.percentile(flat, 50)), 1),
        "p95_ms": round(float(np.percentile(flat, 95)), 1),
        "p99_ms": round(float(np.percentile(flat, 99)), 1),
        "mean_ms": round(float(flat.mean()), 1),
        "max_ms": round(float(flat.max()), 1),
    }


def load_texts(path: str | None) -> list[str]:
    """Probe texts: the demo sample by default."""
    from edgecluster.data.schema import load_many

    paths = [path] if path else ["data/sample/hi.jsonl", "data/sample/ur.jsonl"]
    return [ex.text for ex in load_many(paths)]


def bench_classify_inprocess(args: argparse.Namespace, texts: list[str]) -> dict[str, Any]:
    """Benchmark the classifier inside this process."""
    from edgecluster.serving.classifier import OnnxClassifier

    rss_before = psutil.Process().memory_info().rss / 2**20
    clf = OnnxClassifier(
        args.model_dir,
        args.model_file,
        args.max_seq_len,
        pad_to_max=True,
        intra_op_threads=args.threads,
        graph_optimization=args.graph_opt,
    )
    rss_loaded = psutil.Process().memory_info().rss / 2**20

    def call(i: int) -> None:
        clf.run(clf.encode([texts[i % len(texts)]]))

    with RssSampler() as rss:
        res = run_load(call, args.duration, args.warmup, args.concurrency)
    res |= rss.summary()
    res["rss_model_load_mb"] = round(rss_loaded - rss_before, 1)
    res["model_file_mb"] = round(os.path.getsize(Path(args.model_dir) / args.model_file) / 1e6, 1)
    res["model"] = f"{clf.model_name}/{args.model_file}"
    return res


def bench_classify_http(args: argparse.Namespace, texts: list[str]) -> dict[str, Any]:
    """Benchmark a running node over HTTP."""
    import httpx

    local = threading.local()

    def call(i: int) -> None:
        client = getattr(local, "client", None)
        if client is None:
            client = local.client = httpx.Client(timeout=30)
        r = client.post(f"{args.url.rstrip('/')}/classify", json={"text": texts[i % len(texts)], "log": False})
        r.raise_for_status()

    sampler = RssSampler(args.server_pid) if args.server_pid else None
    if sampler:
        sampler.__enter__()
    try:
        res = run_load(call, args.duration, args.warmup, args.concurrency)
    finally:
        if sampler:
            sampler.__exit__(None, None, None)
    if sampler:
        res |= sampler.summary()
    res["model"] = args.url
    return res


def bench_generate(args: argparse.Namespace) -> dict[str, Any]:
    """Benchmark the generator: responses of exactly ``gen_tokens`` new tokens."""
    from edgecluster.serving.generator import OnnxGenerator, chatml_prompt

    rss_before = psutil.Process().memory_info().rss / 2**20
    gen = OnnxGenerator(args.generator_dir, args.generator_file, intra_op_threads=args.threads)
    rss_loaded = psutil.Process().memory_info().rss / 2**20
    prompts = [
        "What documents do I need to apply for a new ration card?",
        "How can an elderly person apply for an old-age pension?",
        "Which scheme gives free hospital treatment to poor families?",
    ]
    system = "You are a helpful assistant for welfare scheme information. Answer briefly."
    ids = [gen.tokenizer.encode(chatml_prompt(system, p), add_special_tokens=False).ids for p in prompts]
    gen.generate_ids(ids[0], 4, min_new_tokens=4)  # warm-up
    totals, ttfts, per_tok = [], [], []
    with RssSampler() as rss:
        for i in range(args.gen_runs):
            t0 = time.perf_counter()
            out, ttft = gen.generate_ids(ids[i % len(ids)], args.gen_tokens, min_new_tokens=args.gen_tokens)
            total = (time.perf_counter() - t0) * 1000
            totals.append(total)
            ttfts.append(ttft)
            per_tok.append((total - ttft) / max(len(out) - 1, 1))
    return {
        "runs": args.gen_runs,
        "new_tokens": args.gen_tokens,
        "prompt_tokens": [len(x) for x in ids],
        "p50_ms": round(float(np.percentile(totals, 50)), 1),
        "p95_ms": round(float(np.percentile(totals, 95)), 1),
        "ttft_p50_ms": round(float(np.percentile(ttfts, 50)), 1),
        "decode_ms_per_token_p50": round(float(np.percentile(per_tok, 50)), 1),
        "rss_model_load_mb": round(rss_loaded - rss_before, 1),
        **rss.summary(),
    }


# ---------------------------------------------------------------------- report
def _load_json(path: str | None) -> dict[str, Any] | None:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    return None


def _mb(x: float) -> str:
    return f"{x:.0f}MB" if x >= 10 else f"{x:.2f}MB"


def results_table(res: dict[str, Any]) -> str:
    """Markdown table in the same shape as the project's target table."""
    rows = []
    rep = res.get("compression_report")
    if rep:
        by = {r["variant"]: r for r in rep["rows"]}
        fp32, best = by.get("fp32"), by.get("pruned+int8")
        if fp32 and best:
            rows.append(
                (
                    "Model size",
                    TARGETS["size"],
                    f"{_mb(fp32['size_mb'])} fp32 → {_mb(best['size_mb'])} ({best['size_ratio_vs_fp32']:.2f}x)",
                    f"ONNX file size on disk, {rep['model']}",
                )
            )
            rows.append(
                (
                    "Accuracy",
                    TARGETS["accuracy"],
                    f"{best['accuracy_retained_pct']:.1f}% retained ({100 * best['accuracy']:.2f}% vs {100 * fp32['accuracy']:.2f}% fp32)",
                    f"{rep['dataset']} test split (n={best['n_test']})",
                )
            )
    else:
        rows.append(
            (
                "Model size",
                TARGETS["size"],
                f"{res['classify']['model_file_mb']:.0f}MB (served file)",
                "pass --report for the fp32 comparison",
            )
        )
        rows.append(("Accuracy", TARGETS["accuracy"], "not measured", "pass --report runs/<name>/report.json"))
    c = res["classify"]
    rows.append(
        (
            "Classification latency",
            TARGETS["cls_latency"],
            f"p50 {c['p50_ms']:.0f}ms / p95 {c['p95_ms']:.0f}ms (p99 {c['p99_ms']:.0f}ms, {c['throughput_rps']:.2f} req/s, "
            f"peak RSS {c.get('rss_peak_mb', float('nan')):.0f}MB)",
            f"max_seq_len={res['settings']['max_seq_len']} (padded), batch 1, concurrency {res['settings']['concurrency']}, "
            f"{res['settings']['duration_s']}s continuous load, {c['model']}",
        )
    )
    g = res.get("generate")
    if g:
        rows.append(
            (
                "Generative latency",
                TARGETS["gen_latency"],
                f"p50 {g['p50_ms'] / 1000:.2f}s / p95 {g['p95_ms'] / 1000:.2f}s (TTFT {g['ttft_p50_ms']:.0f}ms, "
                f"{g['decode_ms_per_token_p50']:.0f}ms/token)",
                f"one response of exactly {g['new_tokens']} new tokens, greedy, batch 1",
            )
        )
    else:
        rows.append(("Generative latency", TARGETS["gen_latency"], "not measured", "pass --generator-dir"))
    s = res.get("sync")
    if s:
        rows.append(
            (
                "Sync integrity",
                TARGETS["sync"],
                f"{s['identical_pct_at_k']:.2f}% after K={s['k']} windows; 100% after {s['windows_to_100']} windows",
                s["settings"],
            )
        )
    else:
        rows.append(("Sync integrity", TARGETS["sync"], "not measured", "run `make sim` first"))
    hw = res["hardware"]
    hw_line = ", ".join(f"{k}={v}" for k, v in hw.items())
    lines = ["| Metric | Target | Measured | How measured |", "|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c_} | {d} |" for a, b, c_, d in rows]
    lines += ["", f"Hardware: {hw_line}", f"Run at {res['timestamp']}"]
    if res.get("thermal"):
        lines.append(f"Thermal: {res['thermal']}")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", help="classifier ONNX directory (in-process mode)")
    ap.add_argument("--model-file", default="model.pruned.int8.onnx")
    ap.add_argument("--url", help="benchmark a running node over HTTP instead")
    ap.add_argument("--server-pid", type=int, help="sample this PID's RSS in HTTP mode")
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--graph-opt", default="all")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--texts", help="JSONL with probe texts (default: demo sample)")
    ap.add_argument("--generator-dir")
    ap.add_argument("--generator-file", default="onnx/model_quantized.onnx")
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--gen-runs", type=int, default=10)
    ap.add_argument("--report", help="compression report.json for the size/accuracy rows")
    ap.add_argument("--sync", default="results/sim/headline.json", help="simulator headline JSON")
    ap.add_argument("--out", default="results/bench")
    args = ap.parse_args()
    if not args.model_dir and not args.url:
        ap.error("pass --model-dir or --url")
    texts = load_texts(args.texts)
    temp0 = cpu_temp_c()
    res: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": hardware_info(),
        "settings": {
            "max_seq_len": args.max_seq_len,
            "threads": args.threads,
            "concurrency": args.concurrency,
            "duration_s": args.duration,
            "warmup_s": args.warmup,
            "graph_opt": args.graph_opt,
        },
    }
    res["classify"] = bench_classify_http(args, texts) if args.url else bench_classify_inprocess(args, texts)
    if args.generator_dir:
        res["generate"] = bench_generate(args)
    temp1 = cpu_temp_c()
    if temp0 is not None:
        res["thermal"] = f"{temp0}°C before, {temp1}°C after; throttled={throttled_flags()}"
    res["compression_report"] = _load_json(args.report)
    res["sync"] = _load_json(args.sync)
    table = results_table(res)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"bench-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    (out / f"{stem}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    (out / f"{stem}.md").write_text(table + "\n")
    print(json.dumps({k: res[k] for k in ("classify", "generate") if k in res}, indent=2))
    print()
    print(table)
    print(f"\nSaved to {out / stem}.json/.md")


if __name__ == "__main__":
    main()
