### FacebookAI/xlm-roberta-base on bbca.hi (max_seq_len=128)

Pruning: 86 of 144 attention heads removed (ratio 0.6, via model.prune_heads), FFN ratio 0.5, recovery fine-tune 1.0 epoch(s). Measured on Darwin 27.0.0 / arm64, onnxruntime 1.31.0.

| variant | size on disk (MB) | compression vs fp32 | test accuracy | accuracy retained | macro-F1 | local p50 / p95 (ms, bs=1, padded) |
|---|---|---|---|---|---|---|
| fp32 | 1112.4 | 1.00x | 75.98% | 100.0% | 0.3050 | 60.6 / 69.7 |
| fp32+int8 | 278.7 | 3.99x | 75.87% | 99.8% | 0.3026 | 44.1 / 49.1 |
| pruned | 931.4 | 1.19x | 76.33% | 100.5% | 0.3100 | 29.8 / 38.2 |
| pruned+int8 | 233.4 | 4.77x | 76.10% | 100.2% | 0.3063 | 27.8 / 30.3 |

Local latency is measured on the machine that ran the report, *not* on a Raspberry Pi. Use `make bench` on the Pi for edge numbers.
