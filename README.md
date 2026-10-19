# offline-edge-inference-cluster

An offline-first distributed inference system for a cluster of Raspberry Pi 4
nodes that are only connected to each other and to the internet during short
**uplink windows** (about two hours a day). It serves multilingual
(Hindi / Urdu / Dogri) intent classification of welfare-scheme queries, and
optionally a small generative model. Nodes keep working fully offline and
reconcile their data with CRDTs when a window opens.

This repository is a **rebuild** of *Nida-e-Watan*, a digital-inclusion
project that aimed to help displaced families in the Chenab Valley (Jammu & Kashmir)
find and access government welfare schemes such as pensions, ration cards,
housing, health cover, scholarships and identity documents. The project's
original code and data were lost in 2026. Everything here was rewritten from
scratch, and every number in this README was produced by running this code.

**Contents:** [Problem](#problem) · [Architecture](#architecture) · [Quickstart](#quickstart) ·
[Data](#data) · [Compression](#compression) · [Serving](#serving) · [Sync](#sync) ·
[Results](#results) · [Repository layout](#repository-layout) · [Limitations](#limitations)

## Problem

* **Hardware:** 12 × Raspberry Pi 4 (4 GB RAM, 4× Cortex-A72, no GPU, no RTC),
  one per site (camps, panchayat offices, schools).
* **Connectivity:** no reliable internet. Sites can reach each other (and the
  internet) only inside a scheduled window of roughly two hours a day, over
  links that drop packets.
* **Workload:** short queries typed or transcribed in Hindi, Urdu or Dogri,
  such as "my old-age pension hasn't come for three months" or "how do I get
  a duplicate ration card", have to be routed to the right scheme category.
  The answer is needed immediately, not at the next uplink window.
* **Data:** query logs and case records (which family needs which document,
  application status) are created at every site and must eventually agree
  everywhere, with no central server and no lost updates.

That gives three pillars: (1) shrink a multilingual transformer until it
runs well on a Pi, (2) serve it locally with ONNX Runtime, (3) replicate
state with CRDTs and anti-entropy that tolerate partitions, message loss and
reboots.

## Architecture

```mermaid
flowchart LR
  subgraph WS["Workstation / CI (PyTorch)"]
    D[("IndicGLUE bbca.hi<br/>or JSONL")] --> FT[fine-tune]
    FT --> IMP["head + FFN importance<br/>(validation split)"]
    IMP --> PR["structured pruning<br/>+ recovery fine-tune"]
    PR --> EX["ONNX export<br/>(verified vs PyTorch)"]
    EX --> Q["INT8 dynamic quantization<br/>(ONNX Runtime)"]
    Q --> REP["report: size + accuracy"]
  end
  Q -- "model.pruned.int8.onnx<br/>tokenizer.json, meta.json" --> N1

  subgraph CL["12 × Raspberry Pi 4 (no internet outside uplink windows)"]
    N1["node1"]
    N2["node2"]
    N3["node3"]
    N12["node12"]
  end
  N1 <-. "delta anti-entropy<br/>(only inside uplink windows)" .-> N2
  N2 <-.-> N3
  N3 <-.-> N12
  N12 <-.-> N1
```

Inside one node:

```mermaid
flowchart TB
  C["client: field worker app / kiosk"] -->|"POST /classify"| API
  subgraph NODE["edgecluster node (one process)"]
    API["FastAPI<br/>/classify /generate /records /health /metrics"]
    API --> TOK["tokenizers (Rust)<br/>tokenizer.json"]
    TOK --> ORT["ONNX Runtime CPU<br/>4 intra-op threads"]
    API -->|"query log + counters"| STORE[("SQLite<br/>LWW records + PN-counters<br/>HLC timestamps, change seq")]
    SYNC["sync loop<br/>uplink scheduler + gossip"] --> AE["anti-entropy<br/>pull / push deltas"]
    AE <--> STORE
    API -->|"/sync/pull /sync/push"| AE
  end
  AE <-->|"HTTP, piggybacked heartbeats"| PEERS["peers (static config)"]
```

## Quickstart

Requires Python 3.10+ (3.11 recommended, the Raspberry Pi OS Bookworm default).

```bash
git clone https://github.com/aryavsparihar/offline-edge-inference-cluster.git
cd offline-edge-inference-cluster
make install          # .venv with pinned training + serving + test deps
```

**1. Run the tests** (CRDT property tests, store, protocol, API, a real 3-node
HTTP cluster, pruning on tiny encoders, and the full compression pipeline on
a tiny Hub model):

```bash
make test
```

**2. Compress a model.** The fast smoke version uses a tiny random BERT on the
demo sample and finishes in about ten seconds on CPU. It checks the mechanics
only, and its accuracies are meaningless:

```bash
make compress-tiny
```

The real run fine-tunes XLM-R base on IndicGLUE `bbca.hi`, prunes it, quantizes
it and prints the size/accuracy report. It downloads about 1.1 GB, and a GPU or
Apple-silicon MPS helps:

```bash
make compress                                        # configs/compress/xlmr-bbca.yaml
make compress CONFIG=configs/compress/mbert-bbca.yaml
```

**3. Start a 3-node local cluster and watch it converge after a partition.**
These are real HTTP node processes with a compressed uplink schedule (a 12 s
window every 20 s):

```bash
make cluster-demo
```

**4. Simulate the 12-node deployment** with partitions, message loss and
restarts:

```bash
make sim      # one run, default fault settings, per-window table
make sweep    # full grid over loss / partition length / rounds / restarts
```

**5. docker-compose, 12 nodes by default:**

```bash
make cluster-up NODES=12      # builds the image, starts node1..node12 on :8001..:8012, waits for convergence
make cluster-down
```

**6. Serve one node with a compressed model:**

```bash
make serve NODE_CONFIG=configs/node.local.yaml
curl -s -X POST localhost:8000/classify -H 'content-type: application/json' \
     -d '{"text": "राशन कार्ड खो गया है, डुप्लीकेट कैसे मिलेगा", "lang": "hi"}'
```

**7. Benchmark on a Raspberry Pi:** see [deploy/raspberry-pi.md](deploy/raspberry-pi.md).

```bash
make bench MODEL_DIR=runs/xlmr-bbca/onnx RUN=runs/xlmr-bbca
```

## Data

**The original regional-language corpus is not included in this repository.
The project's original code and data were lost in 2026, and this repository
is a rebuild of the system.** To run it, use either:

* the **hand-made demo sample** in [`data/sample/`](data/sample/): 42 Hindi and
  42 Urdu queries written for this rebuild (6 per intent), plus 7 Dogri
  placeholder drafts marked `# TODO: native-speaker review` that the pipeline
  skips until reviewed. These are demo examples, not original data and not
  real user queries. They are enough to exercise every component, not to
  train a useful model.
* the **public-dataset loader** for AI4Bharat's IndicGLUE
  (`python -m edgecluster.data.public --config bbca.hi`), which gives the
  pipeline real Indic-language training and test data. `bbca.hi` is Hindi
  news-topic classification, used as a proxy task for measuring compression.
  It is not the welfare-intent task.

Format, label set and dataset caveats: [data/README.md](data/README.md).

## Compression

`edgecluster/compression/`, driven by `python -m edgecluster.compression.pipeline --config ...`:

| stage | module | notes |
|---|---|---|
| fine-tune | `finetune.py` | plain PyTorch loop (AdamW, linear warmup/decay); best checkpoint chosen on the validation split |
| head importance | `pruning.py` | Michel et al. 2019: gradient of the loss w.r.t. a per-head gate (forward pre-hook on the attention output projection), L2-normalised per layer, global ranking |
| FFN importance | `pruning.py` | first-order Taylor score \|a·∂L/∂a\| per intermediate neuron |
| pruning | `pruning.py` | heads via `model.prune_heads` when the installed `transformers` has it, otherwise by slicing Q/K/V/O directly. FFN neurons by slicing the up/down projections |
| recovery | `finetune.py` | short fine-tune of the pruned model |
| export | `export.py` | `torch.onnx.export` (opset 17, dynamic batch/sequence axes). ORT logits are checked against PyTorch on validation texts |
| INT8 | `quantize.py` | `onnxruntime.quantization.quantize_dynamic` after `quant_pre_process`. `MatMul` + `Gather` (embeddings), per-tensor by default, per-channel optional |
| report | `report.py` | size on disk and test accuracy for fp32, fp32+INT8, pruned and pruned+INT8 |
| ablation | `ablation.py` | re-quantizes the pruned model per-tensor/per-channel, with/without embedding quantization |

`model.prune_heads` was removed in transformers 5, so the requirements pin
**transformers 4.57.6**, the last 4.x release. With transformers 5 the code
falls back to slicing the projections itself, which gives the same shapes and
the same function. The offline tests check both the shapes and that the
pruned model saves and reloads to identical logits.

**Model support:**

* `xlm-roberta-base` (278M parameters, 1.11 GB fp32 ONNX) matches the original
  project's 1.1 GB figure. 69% of its parameters are the 250k-token
  embedding matrix, which head/FFN pruning can't touch but INT8 `Gather`
  quantization does.
* `bert-base-multilingual-cased` (178M parameters, 0.71 GB fp32 ONNX) is
  smaller, so the 1.1 GB → 260 MB target doesn't apply to it.
* `ai4bharat/indic-bert` is ALBERT. **Per-layer head pruning is not
  meaningful for ALBERT**: all 12 layers share a single attention block, so
  removing head *h* removes it from every layer at once. The pruning code
  supports this. It scores heads over all 12 applications of the shared block
  and prunes the shared block, and the offline tests cover it on a tiny ALBERT.
  The real `ai4bharat/indic-bert` checkpoint is **gated** on the Hub (you must
  accept its terms and log in), so it was not run for this README. Use
  `configs/compress/indicbert-bbca.yaml` after `huggingface-cli login`.

## Serving

`edgecluster/serving/` is a FastAPI app with one ONNX Runtime session and a
Rust `tokenizers` tokenizer. There is no torch or transformers on the Pi.

| endpoint | purpose |
|---|---|
| `POST /classify` | `{"text", "lang"}` → label, confidence, per-label scores, tokenize/inference/total ms. Also writes a replicated query-log record (label, language, node, SHA-256 of the text, but **not** the raw text unless `store_query_text: true`) and bumps a per-label PN-counter |
| `POST /generate` | only if `generator.enabled`: Qwen2.5-0.5B-Instruct (INT8 ONNX) with a hand-written KV-cache greedy decoder |
| `GET /health`, `GET /metrics` | Prometheus histograms for classify / inference / generate latency, sync counters, records, RSS |
| `POST/GET/DELETE /records/{cases,queries}` | the replicated record store |
| `POST /sync/pull`, `POST /sync/push`, `GET /sync/digest`, `GET /cluster` | the anti-entropy protocol and cluster view |
| `POST /admin/links`, `POST /admin/sync-now` | only with `admin_enabled`: emulate a link failure, force a round (demos/tests) |

Thread settings for a 4-core Pi: `intra_op_num_threads=4`,
`inter_op_num_threads=1`, sequential execution, `ORT_ENABLE_ALL` graph
optimizations, inference calls serialized by a lock so concurrent requests
queue instead of oversubscribing the cores.

**Generative model:** Qwen2.5-0.5B-Instruct (494M params, Apache-2.0), using
the INT8 ONNX export published at `onnx-community/Qwen2.5-0.5B-Instruct`
(`onnx/model_quantized.onnx`, 512 MB on disk). Larger models (1.5B+) don't
leave room for the classifier and OS on a 4 GB Pi. The decoder is plain numpy
around `session.run`. Memory use and latency are in [Results](#results).

## Sync

`edgecluster/sync/` (everything below is implemented from scratch):

* **Hybrid logical clocks** (`hlc.py`): timestamps `(wall_ms, logical, node_id)`.
  They stay monotonic when an RTC-less Pi's wall clock jumps backwards, and the
  clock state is persisted so they survive reboots.
* **CRDTs** (`crdt/`): `GCounter`, `PNCounter`, `LWWRegister` (HLC-ordered),
  `ORSet` (add-wins, observed-remove) and `LWWMap` (per-key LWW registers with
  tombstones). Hypothesis property tests check that every merge is
  commutative, associative and idempotent, that merge is an upper bound, and
  that merge order doesn't matter (`tests/test_crdt_properties.py`).
* **Replicated store** (`store.py`): SQLite (WAL). Each `(collection, key)`
  is an LWW register and each counter is stored as per-node PN slots. Every
  change gets a node-local sequence number and remembers which peer it came
  from.
* **Anti-entropy** (`antientropy.py`): delta-based push-pull. A pulls B's
  changes after its cursor and B's changes from A are never echoed back. A then
  pushes its own unacknowledged changes. Cursors only advance after a
  successful merge, so a lost response just means a resend, which is
  idempotent. Requests are paginated and retried with a bound. If a peer's
  database is wiped, its new `store_id` resets the cursors and triggers a
  full resend.

`edgecluster/cluster/`:

* **static peer config** (`config.py`, one YAML shared by all nodes),
* **gossip** (`gossip.py`): heartbeat maps merged by per-node max,
  piggybacked on pull requests/responses. Peer selection prefers live peers
  and still probes suspected ones, since in this deployment silence usually
  means a closed window rather than a dead node,
* **uplink-window scheduler** (`scheduler.py`): repeating windows in local
  time (e.g. `06:00` + `2h`, IST), with real, compressed (`time_scale`) or
  manual clocks so the same code runs in production, demos and the simulator.

## Results

Every number in this section was produced by this repository's code. Raw
outputs are in [`results/`](results/). **No Raspberry Pi was available while
building this rebuild**, so the two latency rows are *not measured on the
target hardware*. The numbers shown next to them come from the build machine
and are not comparable to a Pi 4. Run `make bench` on a Pi and paste its
table here.

Build machine: MacBook Air (Apple M4, 10 cores, 16 GB, fanless), macOS,
Python 3.11.17, onnxruntime 1.31.0, PyTorch 2.14.1 (MPS for training),
transformers 4.57.6. Benchmarks ran with only normal desktop apps open.

### Against the targets

| Metric | Target | Measured | Gap to target | How measured |
|---|---|---|---|---|
| Model size | 1.1 GB fp32 → ~260 MB (4.2×) | **1112.4 MB → 258.9 MB (4.30×)** | met (1.1 MB under, +0.10×) | ONNX file size on disk. XLM-R base, 30% of heads + 20% of FFN neurons pruned, 1-epoch recovery fine-tune, dynamic INT8 (`MatMul`+`Gather`, per-tensor). [report](results/compression/xlmr-bbca/report.md) |
| Accuracy | ≥ 91% of fp32 retained | **100.3% retained**: 76.21% vs 75.98% fp32 | met (+9.3 pts above the floor) | IndicGLUE `bbca.hi` held-out test split, n=866, 14 topics. The 95% CI at this n is about ±2.8 pts, so fp32 and pruned+INT8 are statistically tied. Majority-class baseline: 41.2% |
| Classification latency | ~380 ms on a Raspberry Pi 4 | **not yet measured on a Pi** (original project reported ~380 ms†) | unknown | Build machine only (not comparable): p50 15 ms / p95 17 ms / p99 22 ms, 64.2 req/s, peak RSS 1066 MB. Same model, max_seq_len=128 (padded), batch 1, concurrency 1, 4 intra-op threads, 60 s continuous load after 10 s warm-up. [bench](results/bench/bench-20261020-013534.md) |
| Generative latency | ~1.2 s | **not yet measured on a Pi** (original project reported ~1.2 s†) | unknown | Definition: wall time for one response of exactly 32 new tokens (greedy, batch 1, ~40-token chat prompt), Qwen2.5-0.5B-Instruct INT8 ONNX. Build machine only: p50 1.89 s / p95 2.17 s, time-to-first-token 86 ms, 58 ms/token |
| Sync integrity | ~99.2% | **96.14%** mean over 10 seeds (min 86.82%) after K=10 windows; **100% in every seed** after one more window without new writes | −3.06 pts at K=10 | % of records identical on all 12 replicas right after window 10, with new writes before every window. Settings: 20% loss per message, a random 2-way partition in 30% of windows (1 window long), 5% node restart per window, 6 sync rounds per window, fanout 2, 240 writes per window, clock skew up to ±5 min. [sweep](results/sim/sweep.md) |

† **Figures from the original Nida-e-Watan deployment.** That system's code,
data and benchmark logs were lost in 2026, so these numbers can't be checked
or reproduced, and its exact model, sequence length and measurement setup are
unknown. They are kept for reference only and are **not results of this
code**. Replace them with the table from `make bench` once it has been run on
a Raspberry Pi 4.

**Why sync integrity is below 99.2% at K=10:** in the seeds where a random
partition happens to be open during window 10, records written just before
it exist on only one side, and the metric counts them as not identical. The
lower values all come from those seeds. Without partitions the cluster is at
100% after every window for loss up to 30% (and 98.4% at 50% loss).
Convergence itself never failed: all 280 simulated runs reached 100%
identical *and* correct (every replica holds the true latest write of every
record and counter) within at most 4 quiet windows, or 13 at 50% loss without
retries. The simulator parameters were chosen before the first run and were
not adjusted to approach the target.

### Compression trade-offs (XLM-R base and mBERT, `bbca.hi` test, n=866)

| model / pruning | heads removed | params | pruned+INT8 size | compression | test acc. (fp32 → pruned+INT8) | retained |
|---|---|---|---|---|---|---|
| XLM-R base, heads 30%, FFN 20% (default) | 43 / 144 | 278.1M → 258.3M | 258.9 MB | 4.30× | 75.98% → 76.21% | 100.3% |
| XLM-R base, heads 50%, FFN 40% | 72 / 144 | 278.1M → 241.2M | 241.8 MB | 4.60× | 75.98% → 74.94% | 98.6% |
| XLM-R base, heads 60%, FFN 50% | 86 / 144 | 278.1M → 232.8M | 233.4 MB | 4.77× | 75.98% → 76.10% | 100.2% |
| mBERT cased, heads 30%, FFN 20% | 43 / 144 | 177.9M → 158.1M | 158.7 MB | 4.48× (from 711.7 MB) | 72.63% → 71.13% | 97.9% |
| XLM-R base, INT8 only (no pruning) | 0 | 278.1M | 278.7 MB | 3.99× | 75.98% → 75.87% | 99.8% |

Observations:

* Most of the size reduction comes from INT8 on the **embedding matrix**.
  XLM-R's 250k-token vocabulary is 69% of its parameters and head/FFN pruning
  can't touch it. Quantizing `MatMul` only leaves the model at 836 MB.
* Head importance is very uneven across layers. With 30% pruning, layers 0-1
  keep all 12 heads while layers 6, 9 and 10 keep 3-4 heads
  (`heads_remaining` in [run.json](results/compression/xlmr-bbca/run.json)).
* Pruning alone costs little before recovery (validation 76.30% → 75.43% at
  30%, → 72.83% at 50%), and one epoch of recovery fine-tuning restores it.
* Exported ONNX graphs match PyTorch logits to ≤ 1.2e-5 with 100% argmax
  agreement.
* Fine-tuning takes about 21 minutes for 4 epochs (784 steps, batch 16, seq
  128) on the M4 GPU for either model. Each pruning variant adds about 5
  minutes of recovery.

INT8 settings on the 30%-pruned XLM-R ([ablation](results/compression/xlmr-bbca/ablation.md)):

| INT8 setting | size | test accuracy | build-machine p50 |
|---|---|---|---|
| per-tensor, `MatMul` + `Gather` (default) | 258.9 MB | 76.21% | 31 ms |
| per-channel, `MatMul` + `Gather` | 259.2 MB | 75.29% | 200 ms |
| per-tensor, `MatMul` only | 836.1 MB | 76.33% | 50 ms |
| per-channel, `MatMul` only | 836.4 MB | 75.52% | 152 ms |

Per-channel scales were slower *and* slightly less accurate here, so
per-tensor is the default. Measure again on the Pi: ARM kernels differ. The
ablation latencies were taken while other jobs were running and are only
comparable within the table. The idle benchmark gave 15 ms for the default.

Build-machine classification latency of the other XLM-R variants under the
same 60 s benchmark: fp32 p50 29 ms / p95 34 ms (peak RSS 2966 MB),
fp32+INT8 p50 26 ms / p95 30 ms (1085 MB), pruned+INT8 p50 15 ms / p95 17 ms
(1066 MB).

**What was tried and what wasn't:** pruning ratios (three settings above),
recovery fine-tuning (1 epoch), per-channel vs per-tensor INT8, quantized op
set (with/without `Gather`). Not tried, because their payoff is Pi latency and
that can't be measured here: shorter `max_seq_len`, graph optimization
levels, and intra/inter-op thread counts. All of these are flags on
`make bench` / `bench.latency` (`--max-seq-len`, `--graph-opt`, `--threads`,
`--concurrency`). Vocabulary trimming of the embedding matrix would shrink
XLM-R far more than pruning, but it changes the tokenizer and wasn't in scope.

### Serving and the generator

* The pruned+INT8 XLM-R also ran inside the `linux/arm64` Docker image (the
  same OS/arch/onnxruntime stack as a 64-bit Pi). It classified Hindi
  headlines correctly (cricket → `sport` 0.97, budget → `india` 0.98) using
  636 MiB of container memory.
* Qwen2.5-0.5B-Instruct INT8 adds 880-920 MB RSS on load (two measurements). A test checks
  that KV-cache decoding yields exactly the tokens of a full recompute.
  Quality: English answers are coherent, but Hindi output mostly repeated the
  question and Urdu output degenerated in the sample prompts
  ([details](deploy/raspberry-pi.md#7-memory-budget-4-gb)). Treat `/generate`
  as experimental.

### Sync sweep (12 replicas, 10 seeds per cell)

Full grid: [results/sim/sweep.md](results/sim/sweep.md). Mean (min) % of
records identical on all replicas after window *k*:

| setting | k=1 | k=3 | k=5 | k=10 | windows to 100% after writes stop (max) |
|---|---|---|---|---|---|
| loss 0–30%, no partitions | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 0 |
| loss 50%, no partitions | 100.00 (100.00) | 100.00 (100.00) | 99.68 (96.75) | 98.38 (95.79) | 1 |
| loss 20%, 1-window partitions (default) | 80.00 (0.00) | 83.56 (45.41) | 92.72 (75.56) | 96.14 (86.82) | 1 |
| loss 20%, 3-window partitions | 80.00 (0.00) | 58.68 (0.00) | 81.49 (35.74) | 88.96 (65.07) | 3 |
| loss 50%, 3-window partitions | 80.00 (0.00) | 58.68 (0.00) | 78.74 (35.74) | 87.19 (65.07) | 3 |
| default, 1 round per window | 9.28 (0.00) | 51.35 (15.11) | 64.15 (47.11) | 74.36 (61.44) | 4 |
| default, 12 rounds per window | 80.00 (0.00) | 85.20 (62.38) | 92.97 (76.48) | 98.70 (87.04) | 1 |
| loss 50%, **no retries**, no partitions | 90.60 (68.30) | 57.99 (52.37) | 52.74 (45.92) | 53.83 (46.36) | 13 |

The loss levels from 0 to 30% give identical numbers because a partition
schedule depends only on the seed, and with retries every exchange that is
not partitioned eventually completes within the window. Two protocol changes
were made after the very first simulator run showed slow convergence under
loss: heartbeats piggybacked on pull messages instead of a separate gossip
round trip, and up to 2 retries per request. The last row shows what the
retries are worth.

Live clusters (real HTTP between processes, not simulated):

* `make cluster-demo` (3 processes, 12 s window every 20 s): converged 6.1 s
  after the first writes (it waited for the window to open). During a 25 s
  partition the majority side converged and the minority diverged. 1.0 s after
  healing all three were identical, and a concurrently edited case record
  resolved to the same version everywhere.
  [log](results/cluster_demo_3nodes.txt)
* docker-compose, **12 containers**, partitioned 8 | 4 for 40 s with
  conflicting writes: all 12 replicas identical (240/240 records) 3.5 s after
  healing. [log](results/compose_demo_12nodes.txt)

## Repository layout

```
edgecluster/
  compression/   finetune, pruning (importance + heads + FFN), export, quantize, evaluate, report, pipeline
  serving/       FastAPI app, ONNX classifier, ONNX generator, ORT session tuning, Prometheus metrics
  sync/          hlc, crdt/ (GCounter, PNCounter, LWWRegister, ORSet, LWWMap), SQLite store, anti-entropy
  cluster/       uplink scheduler, gossip membership, HTTP transport, SyncNode runtime
  data/          JSONL schema, IndicGLUE loader
  config.py      YAML + EDGE_* env configuration
sim/             in-process simulator, sweep, localhost cluster demo, docker-compose generator + watcher
bench/           continuous-load latency benchmark (run on the Pi)
deploy/          Raspberry Pi guide, systemd unit, install script, multi-arch Dockerfile
configs/         compression configs, cluster/node examples
data/            format docs, hand-made demo sample
tests/           pytest + Hypothesis
results/         outputs of the runs quoted in this README
```

## Limitations

* No Raspberry Pi was available while building this rebuild, so **edge latency
  and memory are not reported here**. `make bench` prints them in the target
  table format for pasting into this README. The build-machine numbers above
  say nothing about Pi performance.
* `bbca.hi` is a stand-in task (Hindi news topics). Accuracy on the real
  welfare-intent task, and on Urdu and Dogri, needs real labeled queries.
* The sync layer trusts its peers: there is no authentication or encryption
  on the sync endpoints. Run it on a private network or VPN (e.g. WireGuard)
  between sites.
* LWW resolves concurrent edits to the same case record by HLC order, so one
  edit wins and the other is overwritten (it is not merged field by field).
  That is acceptable for status fields but not for free-text notes.
* Tombstones are kept forever. There is no garbage collection yet.

## License

Apache-2.0
