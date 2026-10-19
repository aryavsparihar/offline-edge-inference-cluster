# Raspberry Pi 4 (4 GB) deployment guide

The edge node needs only the **runtime** dependencies: ONNX Runtime, the Rust
`tokenizers` library, FastAPI/uvicorn and a few small packages. There is no
PyTorch or `transformers` on the Pi. Train and compress on a workstation,
then copy the ONNX directory over.

## 1. Operating system

* Flash **Raspberry Pi OS Lite (64-bit)**, Bookworm, with Raspberry Pi Imager.
  The 64-bit OS is required: ONNX Runtime publishes `aarch64` (manylinux)
  wheels on PyPI but no 32-bit `armv7l` wheels.
* Check: `uname -m` must print `aarch64`, and `python3 --version` should be
  3.11 (Bookworm's default).
* Use a good A2 SD card or, better, boot from USB SSD. SQLite writes and model
  loading are much faster, and SSDs survive power cuts better.
* Cooling: a heatsink and fan are needed for sustained load. Without them the
  SoC throttles at 80-85 °C and latency gets worse during long benchmarks.
  `make bench` records the temperature and `vcgencmd get_throttled` flags so
  you can tell.
* No RTC: the Pi's clock is restored from `fake-hwclock` at boot and only
  corrected by NTP during an uplink window. The sync layer uses hybrid logical
  clocks, so a wrong wall clock doesn't break ordering. Keep
  `sync.max_drift` unset (the default) unless NTP is reliable.

## 2. Install a node

```bash
sudo apt update && sudo apt install -y git
git clone https://github.com/aryavsparihar/offline-edge-inference-cluster.git
cd offline-edge-inference-cluster
sudo NODE_ID=node3 ./deploy/install-pi.sh
```

The script creates an `edge` system user, installs into `/opt/edgecluster/.venv`
with `pip install -r requirements/runtime.txt` (this pulls the official
`onnxruntime` aarch64 wheel from PyPI), writes `/etc/edgecluster/node.env`,
copies `configs/cluster.example.yaml` to `/etc/edgecluster/cluster.yaml` and
enables `deploy/systemd/edgecluster.service`.

Manual equivalent:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements/runtime.txt
.venv/bin/pip install --no-deps -e .
.venv/bin/python -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
# -> 1.x.y ['CPUExecutionProvider']
```

## 3. Copy the model

On the workstation, after `make compress`:

```bash
rsync -av runs/xlmr-bbca/onnx/ pi@node3.local:/tmp/xlmr-bbca/
ssh pi@node3.local 'sudo mkdir -p /opt/edgecluster/models && sudo mv /tmp/xlmr-bbca /opt/edgecluster/models/'
```

The directory must contain `model.pruned.int8.onnx` (or whichever variant
`classifier.model_file` names), `tokenizer.json` and `meta.json`.

## 4. Configure

Edit `/etc/edgecluster/cluster.yaml` on every node. It is the same file
everywhere: peers, uplink windows (local time via `utc_offset`) and
classifier settings. The node id comes from `EDGE_NODE_ID` in
`/etc/edgecluster/node.env`.

```bash
sudo systemctl restart edgecluster
journalctl -u edgecluster -f
curl -s localhost:8000/health | python3 -m json.tool
curl -s -X POST localhost:8000/classify -H 'content-type: application/json' \
     -d '{"text": "मेरी पेंशन तीन महीने से नहीं आई", "lang": "hi"}'
```

## 5. Thread settings

`classifier.intra_op_threads: 4` and `inter_op_threads: 1` match the Pi 4's
four Cortex-A72 cores. The service serializes `session.run` calls with a lock.
Running two 4-thread inferences at once on 4 cores is slower than queueing
them. To trade latency for throughput, run with `--concurrency 2 --threads 2`
in the benchmark and compare.

Optional, for stable benchmark numbers:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## 6. Benchmark on the Pi

```bash
make bench MODEL_DIR=/opt/edgecluster/models/xlmr-bbca RUN=runs/xlmr-bbca DURATION=120
# with the generative model:
make generator                                   # downloads ~520 MB into models/qwen2.5-0.5b-instruct
make bench MODEL_DIR=... GEN_DIR=models/qwen2.5-0.5b-instruct
```

Stop the systemd service first (`sudo systemctl stop edgecluster`) so it does
not compete for the cores. The table it prints has the same rows as the
project's target table. Paste it into the README's Results section with the
hardware line.

## 7. Memory budget (4 GB)

| component | resident memory |
|---|---|
| Raspberry Pi OS Lite, idle | ~150-250 MB |
| node process without a model (FastAPI + SQLite) | measured by `/metrics` → `edge_process_rss_bytes` |
| classifier: XLM-R pruned+INT8 | reported as `rss_model_load_mb` by `make bench` |
| generator: Qwen2.5-0.5B-Instruct INT8 (512 MB file) | reported as `rss_model_load_mb` by `make bench` |

The systemd unit caps the node at `MemoryMax=2G`. Larger generative models
(1.5B+ parameters) don't fit next to the classifier with useful headroom and
aren't supported.

## 8. Docker instead of systemd (optional)

```bash
docker buildx build --platform linux/arm64 -f deploy/Dockerfile -t edgecluster:latest --load .
docker run -d --name node3 -p 8000:8000 -e EDGE_NODE_ID=node3 -e EDGE_MODEL_DIR=/models/xlmr-bbca \
  -v /etc/edgecluster:/config:ro -v /opt/edgecluster/models:/models:ro -v edge-data:/data \
  edgecluster:latest --config /config/cluster.yaml
```

Native systemd uses less memory and starts faster, so prefer it on the Pi.
