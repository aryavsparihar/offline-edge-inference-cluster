# Sync convergence sweep

Base settings: 12 nodes, K=10 windows, 6 rounds/window, fanout 2, loss 0.2 per message, partitions p=0.3 lasting 1 window(s) into 2 groups, restart p=0.05/node/window, 20 writes/node before each window (30% updates to 200 shared case records, 5% deletes), clock skew up to ±300s. Each cell: mean (min) over 5 seeds.

`k=N`: % of records identical on all replicas right after window N, with new writes still arriving before every window. `quiet→100%`: extra windows without writes until every replica holds the identical, correct latest version of every record and counter (max over seeds).

## loss_x_partition

| setting | k=1 | k=3 | k=5 | k=10 | converged | quiet→100% |
|---|---|---|---|---|---|---|
| loss=0.0 partition_len=0w | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 5/5 | 0 |
| loss=0.0 partition_len=1w | 80.00 (0.00) | 81.79 (45.41) | 90.30 (75.56) | 94.87 (86.82) | 5/5 | 1 |
| loss=0.0 partition_len=3w | 80.00 (0.00) | 58.76 (0.00) | 82.47 (36.37) | 85.33 (65.07) | 5/5 | 3 |
| loss=0.1 partition_len=0w | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 5/5 | 0 |
| loss=0.1 partition_len=1w | 80.00 (0.00) | 81.79 (45.41) | 90.30 (75.56) | 94.87 (86.82) | 5/5 | 1 |
| loss=0.1 partition_len=3w | 80.00 (0.00) | 58.76 (0.00) | 82.47 (36.37) | 85.33 (65.07) | 5/5 | 3 |
| loss=0.2 partition_len=0w | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 5/5 | 0 |
| loss=0.2 partition_len=1w | 80.00 (0.00) | 81.79 (45.41) | 90.30 (75.56) | 94.87 (86.82) | 5/5 | 1 |
| loss=0.2 partition_len=3w | 80.00 (0.00) | 58.76 (0.00) | 82.47 (36.37) | 85.33 (65.07) | 5/5 | 3 |
| loss=0.3 partition_len=0w | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 5/5 | 0 |
| loss=0.3 partition_len=1w | 80.00 (0.00) | 81.79 (45.41) | 90.30 (75.56) | 94.87 (86.82) | 5/5 | 1 |
| loss=0.3 partition_len=3w | 80.00 (0.00) | 58.76 (0.00) | 82.47 (36.37) | 85.33 (65.07) | 5/5 | 3 |
| loss=0.5 partition_len=0w | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 99.11 (98.05) | 5/5 | 1 |
| loss=0.5 partition_len=1w | 80.00 (0.00) | 81.79 (45.41) | 89.56 (73.75) | 92.66 (86.82) | 5/5 | 2 |
| loss=0.5 partition_len=3w | 80.00 (0.00) | 58.76 (0.00) | 82.10 (36.37) | 84.93 (65.07) | 5/5 | 3 |

## rounds

| setting | k=1 | k=3 | k=5 | k=10 | converged | quiet→100% |
|---|---|---|---|---|---|---|
| rounds/window=1 | 0.00 (0.00) | 50.54 (15.11) | 63.61 (47.11) | 76.54 (65.12) | 5/5 | 4 |
| rounds/window=2 | 80.00 (0.00) | 91.43 (63.76) | 93.95 (75.72) | 87.17 (77.85) | 5/5 | 2 |
| rounds/window=3 | 80.00 (0.00) | 74.55 (45.55) | 85.28 (74.61) | 94.71 (86.92) | 5/5 | 1 |
| rounds/window=6 | 80.00 (0.00) | 81.79 (45.41) | 90.30 (75.56) | 94.87 (86.82) | 5/5 | 1 |
| rounds/window=12 | 80.00 (0.00) | 92.62 (63.10) | 95.32 (76.61) | 97.41 (87.04) | 5/5 | 1 |

## restarts

| setting | k=1 | k=3 | k=5 | k=10 | converged | quiet→100% |
|---|---|---|---|---|---|---|
| restart_p=0.0 | 80.00 (0.00) | 85.19 (62.40) | 90.44 (75.93) | 97.39 (86.93) | 5/5 | 1 |
| restart_p=0.05 | 80.00 (0.00) | 81.79 (45.41) | 90.30 (75.56) | 94.87 (86.82) | 5/5 | 1 |
| restart_p=0.2 | 80.00 (0.00) | 85.58 (63.56) | 90.57 (76.17) | 96.61 (83.07) | 5/5 | 1 |
| restart_p=0.5 | 80.00 (0.00) | 77.91 (62.60) | 90.35 (75.53) | 89.69 (86.80) | 5/5 | 1 |

