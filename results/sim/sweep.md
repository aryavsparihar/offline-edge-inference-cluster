# Sync convergence sweep

Base settings: 12 nodes, K=10 windows, 6 rounds/window, fanout 2, loss 0.2 per message, partitions p=0.3 lasting 1 window(s) into 2 groups, restart p=0.05/node/window, 20 writes/node before each window (30% updates to 200 shared case records, 5% deletes), 2 retries per request, clock skew up to ±300s. Each cell: mean (min) over 10 seeds.

`k=N`: % of records identical on all replicas right after window N, with new writes still arriving before every window. `quiet→100%`: extra windows without writes until every replica holds the identical, correct latest version of every record and counter (max over seeds).

## loss_x_partition

| setting | k=1 | k=3 | k=5 | k=10 | converged | quiet→100% |
|---|---|---|---|---|---|---|
| loss=0.0 partition_len=0w | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 10/10 | 0 |
| loss=0.0 partition_len=1w | 80.00 (0.00) | 83.56 (45.41) | 92.72 (75.56) | 96.14 (86.82) | 10/10 | 1 |
| loss=0.0 partition_len=3w | 80.00 (0.00) | 58.68 (0.00) | 81.49 (35.74) | 88.96 (65.07) | 10/10 | 3 |
| loss=0.1 partition_len=0w | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 10/10 | 0 |
| loss=0.1 partition_len=1w | 80.00 (0.00) | 83.56 (45.41) | 92.72 (75.56) | 96.14 (86.82) | 10/10 | 1 |
| loss=0.1 partition_len=3w | 80.00 (0.00) | 58.68 (0.00) | 81.49 (35.74) | 88.96 (65.07) | 10/10 | 3 |
| loss=0.2 partition_len=0w | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 10/10 | 0 |
| loss=0.2 partition_len=1w | 80.00 (0.00) | 83.56 (45.41) | 92.72 (75.56) | 96.14 (86.82) | 10/10 | 1 |
| loss=0.2 partition_len=3w | 80.00 (0.00) | 58.68 (0.00) | 81.49 (35.74) | 88.96 (65.07) | 10/10 | 3 |
| loss=0.3 partition_len=0w | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 10/10 | 0 |
| loss=0.3 partition_len=1w | 80.00 (0.00) | 83.56 (45.41) | 92.72 (75.56) | 96.14 (86.82) | 10/10 | 1 |
| loss=0.3 partition_len=3w | 80.00 (0.00) | 58.68 (0.00) | 81.49 (35.74) | 88.96 (65.07) | 10/10 | 3 |
| loss=0.5 partition_len=0w | 100.00 (100.00) | 100.00 (100.00) | 99.68 (96.75) | 98.38 (95.79) | 10/10 | 1 |
| loss=0.5 partition_len=1w | 80.00 (0.00) | 83.56 (45.41) | 91.95 (73.75) | 94.01 (82.37) | 10/10 | 2 |
| loss=0.5 partition_len=3w | 80.00 (0.00) | 58.68 (0.00) | 78.74 (35.74) | 87.19 (65.07) | 10/10 | 3 |

## rounds

| setting | k=1 | k=3 | k=5 | k=10 | converged | quiet→100% |
|---|---|---|---|---|---|---|
| rounds/window=1 | 9.28 (0.00) | 51.35 (15.11) | 64.15 (47.11) | 74.36 (61.44) | 10/10 | 4 |
| rounds/window=2 | 77.75 (0.00) | 95.42 (63.76) | 87.00 (73.46) | 89.48 (77.85) | 10/10 | 2 |
| rounds/window=3 | 80.00 (0.00) | 79.97 (45.55) | 85.44 (74.61) | 94.81 (86.92) | 10/10 | 1 |
| rounds/window=6 | 80.00 (0.00) | 83.56 (45.41) | 92.72 (75.56) | 96.14 (86.82) | 10/10 | 1 |
| rounds/window=12 | 80.00 (0.00) | 85.20 (62.38) | 92.97 (76.48) | 98.70 (87.04) | 10/10 | 1 |

## restarts

| setting | k=1 | k=3 | k=5 | k=10 | converged | quiet→100% |
|---|---|---|---|---|---|---|
| restart_p=0.0 | 80.00 (0.00) | 85.26 (62.13) | 92.79 (75.73) | 94.82 (86.77) | 10/10 | 1 |
| restart_p=0.05 | 80.00 (0.00) | 83.56 (45.41) | 92.72 (75.56) | 96.14 (86.82) | 10/10 | 1 |
| restart_p=0.2 | 80.00 (0.00) | 89.00 (62.13) | 88.14 (75.81) | 98.31 (83.07) | 10/10 | 1 |
| restart_p=0.5 | 80.00 (0.00) | 81.65 (62.40) | 88.01 (75.53) | 90.64 (83.32) | 10/10 | 1 |

## retries

| setting | k=1 | k=3 | k=5 | k=10 | converged | quiet→100% |
|---|---|---|---|---|---|---|
| loss=0.2 retries=0 (no partitions) | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 99.89 (99.26) | 10/10 | 1 |
| loss=0.2 retries=2 (no partitions) | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 100.00 (100.00) | 10/10 | 0 |
| loss=0.5 retries=0 (no partitions) | 90.60 (68.30) | 57.99 (52.37) | 52.74 (45.92) | 53.83 (46.36) | 10/10 | 13 |
| loss=0.5 retries=2 (no partitions) | 100.00 (100.00) | 100.00 (100.00) | 99.68 (96.75) | 98.38 (95.79) | 10/10 | 1 |

