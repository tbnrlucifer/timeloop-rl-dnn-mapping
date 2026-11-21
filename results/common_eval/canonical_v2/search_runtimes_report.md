# Wall-Clock Search Runtime Summary (Optimization Runs Only)

*(Excludes canonical Accelergy 0.4 reevaluation side runs. Values are wall-clock elapsed seconds from `time.perf_counter()`.)*

## 1. Same-Machine Apples-to-Apples Comparison (~420 Evals, Canonical Laptop)

| Workload | Arm | Runs | Mean (s) | Std (s) | Mean (min) | Min – Max (min) |
|---|---|---:|---:|---:|---:|---:|
| baseline | Hybrid Iter<=3 (400-420 evals) | 5 | 2285.5 | 281.7 | 38.1 | 32.3 – 45.2 |
| baseline | RL-only (420 evals) | 5 | 2198.9 | 201.9 | 36.6 | 34.6 – 42.4 |
| baseline | GAMMA-only (420 evals) | 5 | 1905.7 | 87.1 | 31.8 | 30.1 – 33.3 |
| channel-heavy | Hybrid Iter<=3 (400-420 evals) | 5 | 3493.3 | 659.9 | 58.2 | 46.6 – 72.3 |
| channel-heavy | RL-only (420 evals) | 5 | 2298.3 | 138.9 | 38.3 | 35.9 – 41.4 |
| channel-heavy | GAMMA-only (420 evals) | 5 | 2551.0 | 178.4 | 42.5 | 39.1 – 46.7 |

## 2. All 5 Workloads: Original Runs (Note: Different Machines for 120-Eval Baselines)

| Workload | Arm | Machine / Setup | Runs | Mean (s) | Std (s) | Mean (min) |
|---|---|---|---:|---:|---:|---:|
| small | Hybrid Iter<=3 (400-420) | Canonical laptop (GPU+CPU, seq) | 5 | 1418.1 | 197.4 | 23.6 |
| small | RL-only (120) | RL junior (CPU-only) | 5 | 1098.6 | 186.6 | 18.3 |
| small | GAMMA-only (120) | GAMMA junior (WSL2, 4x parallel) | 5 | 554.0 | 32.2 | 9.2 |
| baseline | Hybrid Iter<=3 (400-420) | Canonical laptop (GPU+CPU, seq) | 5 | 2285.5 | 281.7 | 38.1 |
| baseline | RL-only (120) | RL junior (CPU-only) | 5 | 1209.2 | 42.3 | 20.2 |
| baseline | GAMMA-only (120) | GAMMA junior (WSL2, 4x parallel) | 5 | 974.5 | 113.5 | 16.2 |
| spatial-heavy | Hybrid Iter<=3 (400-420) | Canonical laptop (GPU+CPU, seq) | 5 | 2812.0 | 338.1 | 46.9 |
| spatial-heavy | RL-only (120) | RL junior (CPU-only) | 5 | 1234.0 | 70.0 | 20.6 |
| spatial-heavy | GAMMA-only (120) | GAMMA junior (WSL2, 4x parallel) | 5 | 1326.8 | 58.6 | 22.1 |
| channel-heavy | Hybrid Iter<=3 (400-420) | Canonical laptop (GPU+CPU, seq) | 5 | 3493.3 | 659.9 | 58.2 |
| channel-heavy | RL-only (120) | RL junior (CPU-only) | 5 | 1252.5 | 32.6 | 20.9 |
| channel-heavy | GAMMA-only (120) | GAMMA junior (WSL2, 4x parallel) | 5 | 1167.6 | 110.9 | 19.5 |
| pointwise | Hybrid Iter<=3 (400-420) | Canonical laptop (GPU+CPU, seq) | 5 | 386.8 | 29.0 | 6.4 |
| pointwise | RL-only (120) | RL junior (CPU-only) | 5 | 1577.1 | 230.6 | 26.3 |
| pointwise | GAMMA-only (120) | GAMMA junior (WSL2, 4x parallel) | 5 | 142.4 | 12.0 | 2.4 |

## 3. Hybrid Stage-by-Stage Runtime Breakdown (Canonical Laptop, n=5 Seeds)

| Workload | Initial RL (s) | GAMMA Total 3-Iter (s) | PPO Feedback Total (s) | Iter<=1 to Winner (s) | Full 3-Iter Total (s / min) |
|---|---:|---:|---:|---:|---:|
| small | 245.2 | 976.5 | 131.4 | 606.2 | 1418.1s (23.6m) |
| baseline | 200.5 | 1795.4 | 247.7 | 807.3 | 2285.5s (38.1m) |
| spatial-heavy | 208.1 | 2218.4 | 338.4 | 976.7 | 2812.0s (46.9m) |
| channel-heavy | 213.8 | 2793.7 | 435.5 | 1087.3 | 3493.3s (58.2m) |
| pointwise | 162.1 | 162.2 | 29.1 | 210.0 | 386.8s (6.4m) |
