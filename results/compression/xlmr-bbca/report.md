### FacebookAI/xlm-roberta-base on bbca.hi (max_seq_len=128)

Pruning: 43 of 144 attention heads removed (ratio 0.3, via model.prune_heads), FFN ratio 0.2, recovery fine-tune 1.0 epoch(s). Measured on Darwin 27.0.0 / arm64, onnxruntime 1.31.0.

| variant | size on disk (MB) | compression vs fp32 | test accuracy | accuracy retained | macro-F1 | local p50 / p95 (ms, bs=1, padded) |
|---|---|---|---|---|---|---|
| fp32 | 1112.4 | 1.00x | 75.98% | 100.0% | 0.3050 | 47.3 / 52.4 |
| fp32+int8 | 278.7 | 3.99x | 75.87% | 99.8% | 0.3026 | 44.1 / 60.0 |
| pruned | 1033.3 | 1.08x | 75.75% | 99.7% | 0.3145 | 38.7 / 46.5 |
| pruned+int8 | 258.9 | 4.30x | 76.21% | 100.3% | 0.3150 | 34.7 / 36.9 |

Local latency is measured on the machine that ran the report, *not* on a Raspberry Pi. Use `make bench` on the Pi for edge numbers.
