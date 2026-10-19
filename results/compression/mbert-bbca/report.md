### google-bert/bert-base-multilingual-cased on bbca.hi (max_seq_len=128)

Pruning: 43 of 144 attention heads removed (ratio 0.3, via model.prune_heads), FFN ratio 0.2, recovery fine-tune 1.0 epoch(s). Measured on Darwin 27.0.0 / arm64, onnxruntime 1.31.0.

| variant | size on disk (MB) | compression vs fp32 | test accuracy | accuracy retained | macro-F1 | local p50 / p95 (ms, bs=1, padded) |
|---|---|---|---|---|---|---|
| fp32 | 711.7 | 1.00x | 72.63% | 100.0% | 0.2684 | 55.4 / 99.4 |
| fp32+int8 | 178.5 | 3.99x | 72.75% | 100.2% | 0.2682 | 29.1 / 31.7 |
| pruned | 632.5 | 1.13x | 71.36% | 98.3% | 0.2647 | 91.6 / 124.0 |
| pruned+int8 | 158.7 | 4.48x | 71.13% | 97.9% | 0.2634 | 56.5 / 71.7 |

Local latency is measured on the machine that ran the report, *not* on a Raspberry Pi. Use `make bench` on the Pi for edge numbers.
