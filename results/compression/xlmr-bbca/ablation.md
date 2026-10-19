### INT8 ablation on xlmr-bbca (model.pruned.onnx, test n=866)

| setting | size (MB) | test accuracy | macro-F1 | local p50 (ms) |
|---|---|---|---|---|
| per-tensor, MatMul+Gather | 258.9 | 76.21% | 0.3150 | 31.1 |
| per-channel, MatMul+Gather | 259.2 | 75.29% | 0.3112 | 200.0 |
| per-tensor, MatMul only | 836.1 | 76.33% | 0.3160 | 49.6 |
| per-channel, MatMul only | 836.4 | 75.52% | 0.3140 | 151.5 |
