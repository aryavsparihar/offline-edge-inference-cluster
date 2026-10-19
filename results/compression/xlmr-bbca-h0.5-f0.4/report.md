### FacebookAI/xlm-roberta-base on bbca.hi (max_seq_len=128)

Pruning: 72 of 144 attention heads removed (ratio 0.5, via model.prune_heads), FFN ratio 0.4, recovery fine-tune 1.0 epoch(s). Measured on Darwin 27.0.0 / arm64, onnxruntime 1.31.0.

| variant | size on disk (MB) | compression vs fp32 | test accuracy | accuracy retained | macro-F1 | local p50 / p95 (ms, bs=1, padded) |
|---|---|---|---|---|---|---|
| fp32 | 1112.4 | 1.00x | 75.98% | 100.0% | 0.3050 | 46.1 / 51.8 |
| fp32+int8 | 278.7 | 3.99x | 75.87% | 99.8% | 0.3026 | 49.6 / 70.4 |
| pruned | 965.1 | 1.15x | 75.29% | 99.1% | 0.3046 | 39.9 / 87.1 |
| pruned+int8 | 241.8 | 4.60x | 74.94% | 98.6% | 0.3025 | 39.2 / 43.4 |

Local latency is measured on the machine that ran the report, *not* on a Raspberry Pi. Use `make bench` on the Pi for edge numbers.
