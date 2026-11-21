# Canonical Top-5 Debiased Comparison Report (Accelergy 0.4)

- Runtime: `1290.4s` | Cache hits: `84` | New Timeloop evals: `241` | Failures: `0`

## Selection-Bias Audit (Accelergy 0.3 -> 0.4 Top-5 Rank Flips)

| Arm | Runs with Rank-1 Flip | Max Flip Gain (%) | Median Flip Gain on Flipped Runs (%) |
|---|---:|---:|---:|
| rl_only | 2/25 | 1.041% | 1.041% |
| gamma_only | 1/25 | 0.487% | 0.487% |

## Iso-Budget (120 Evals): Hybrid (Iter<=1) vs RL-only (120, Top-5 Debiased)

- **Pairs:** 25/25 | **hybrid_iter1 W/T/L:** `10 / 0 / 15`
- **Overall Geomean EDP Ratio:** `1.0015` (`-0.15%` reduction) | **Energy Ratio:** `1.0015` (`-0.15%`) | **Cycles Ratio:** `1.0000` (`-0.00%`)
- **Two-sided Wilcoxon signed-rank:** `W=148.0`, `p=0.711161`

| Workload | hybrid_iter1 Geomean EDP | rl_only Geomean EDP | EDP Ratio | EDP Red. | Energy Red. | Cycles Red. | Wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 1.5731e+08 | 1.5591e+08 | 1.0090 | -0.90% | -0.90% | +0.00% | 1/5 (0t) |
| baseline | 6.6292e+12 | 6.6167e+12 | 1.0019 | -0.19% | -0.19% | +0.00% | 3/5 (0t) |
| spatial-heavy | 2.7156e+14 | 2.7200e+14 | 0.9984 | +0.16% | +0.16% | +0.00% | 2/5 (0t) |
| channel-heavy | 1.6244e+13 | 1.6346e+13 | 0.9937 | +0.63% | +0.63% | +0.00% | 3/5 (0t) |
| pointwise | 2.9566e+11 | 2.9434e+11 | 1.0045 | -0.45% | -0.45% | -0.00% | 1/5 (0t) |

## Iso-Budget (120 Evals): Hybrid (Iter<=1) vs GAMMA-only (120, Top-5 Debiased)

- **Pairs:** 25/25 | **hybrid_iter1 W/T/L:** `10 / 0 / 15`
- **Overall Geomean EDP Ratio:** `1.0099` (`-0.99%` reduction) | **Energy Ratio:** `1.0073` (`-0.73%`) | **Cycles Ratio:** `1.0026` (`-0.26%`)
- **Two-sided Wilcoxon signed-rank:** `W=133.0`, `p=0.441837`

| Workload | hybrid_iter1 Geomean EDP | gamma_only Geomean EDP | EDP Ratio | EDP Red. | Energy Red. | Cycles Red. | Wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 1.5731e+08 | 1.5776e+08 | 0.9971 | +0.29% | +0.29% | +0.00% | 2/5 (0t) |
| baseline | 6.6292e+12 | 6.6021e+12 | 1.0041 | -0.41% | +0.88% | -1.30% | 3/5 (0t) |
| spatial-heavy | 2.7156e+14 | 2.6693e+14 | 1.0173 | -1.73% | -1.73% | +0.00% | 2/5 (0t) |
| channel-heavy | 1.6244e+13 | 1.5831e+13 | 1.0261 | -2.61% | -2.61% | +0.00% | 2/5 (0t) |
| pointwise | 2.9566e+11 | 2.9412e+11 | 1.0052 | -0.52% | -0.52% | -0.00% | 1/5 (0t) |

## Full Hybrid (Iter<=3, 400-420 Evals) vs RL-only (120, Top-5 Debiased)

- **Pairs:** 25/25 | **hybrid_iter3 W/T/L:** `23 / 0 / 2`
- **Overall Geomean EDP Ratio:** `0.9676` (`+3.24%` reduction) | **Energy Ratio:** `0.9726` (`+2.74%`) | **Cycles Ratio:** `0.9949` (`+0.51%`)
- **Two-sided Wilcoxon signed-rank:** `W=18.0`, `p=1.508e-05`

| Workload | hybrid_iter3 Geomean EDP | rl_only Geomean EDP | EDP Ratio | EDP Red. | Energy Red. | Cycles Red. | Wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 1.5516e+08 | 1.5591e+08 | 0.9952 | +0.48% | +0.48% | +0.00% | 5/5 (0t) |
| baseline | 6.2739e+12 | 6.6167e+12 | 0.9482 | +5.18% | +2.70% | +2.55% | 4/5 (0t) |
| spatial-heavy | 2.6014e+14 | 2.7200e+14 | 0.9564 | +4.36% | +4.36% | +0.00% | 5/5 (0t) |
| channel-heavy | 1.5390e+13 | 1.6346e+13 | 0.9415 | +5.85% | +5.85% | +0.00% | 5/5 (0t) |
| pointwise | 2.9388e+11 | 2.9434e+11 | 0.9984 | +0.16% | +0.16% | -0.00% | 4/5 (0t) |

## Full Hybrid (Iter<=3, 400-420 Evals) vs GAMMA-only (120, Top-5 Debiased)

- **Pairs:** 25/25 | **hybrid_iter3 W/T/L:** `20 / 1 / 4`
- **Overall Geomean EDP Ratio:** `0.9758` (`+2.42%` reduction) | **Energy Ratio:** `0.9783` (`+2.17%`) | **Cycles Ratio:** `0.9974` (`+0.26%`)
- **Two-sided Wilcoxon signed-rank:** `W=68.0`, `p=0.00963503`

| Workload | hybrid_iter3 Geomean EDP | gamma_only Geomean EDP | EDP Ratio | EDP Red. | Energy Red. | Cycles Red. | Wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 1.5516e+08 | 1.5776e+08 | 0.9835 | +1.65% | +1.65% | +0.00% | 5/5 (0t) |
| baseline | 6.2739e+12 | 6.6021e+12 | 0.9503 | +4.97% | +3.74% | +1.28% | 4/5 (0t) |
| spatial-heavy | 2.6014e+14 | 2.6693e+14 | 0.9745 | +2.55% | +2.55% | +0.00% | 3/5 (0t) |
| channel-heavy | 1.5390e+13 | 1.5831e+13 | 0.9721 | +2.79% | +2.79% | +0.00% | 5/5 (0t) |
| pointwise | 2.9388e+11 | 2.9412e+11 | 0.9992 | +0.08% | +0.08% | -0.00% | 3/5 (1t) |

## Baselines Head-to-Head (120 Evals): RL-only vs GAMMA-only (Top-5 Debiased)

- **Pairs:** 25/25 | **rl_only W/T/L:** `10 / 1 / 14`
- **Overall Geomean EDP Ratio:** `1.0084` (`-0.84%` reduction) | **Energy Ratio:** `1.0058` (`-0.58%`) | **Cycles Ratio:** `1.0026` (`-0.26%`)
- **Two-sided Wilcoxon signed-rank:** `W=93.0`, `p=0.103404`

| Workload | rl_only Geomean EDP | gamma_only Geomean EDP | EDP Ratio | EDP Red. | Energy Red. | Cycles Red. | Wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 1.5591e+08 | 1.5776e+08 | 0.9882 | +1.18% | +1.18% | +0.00% | 4/5 (1t) |
| baseline | 6.6167e+12 | 6.6021e+12 | 1.0022 | -0.22% | +1.06% | -1.30% | 3/5 (0t) |
| spatial-heavy | 2.7200e+14 | 2.6693e+14 | 1.0190 | -1.90% | -1.90% | +0.00% | 1/5 (0t) |
| channel-heavy | 1.6346e+13 | 1.5831e+13 | 1.0325 | -3.25% | -3.25% | +0.00% | 1/5 (0t) |
| pointwise | 2.9434e+11 | 2.9412e+11 | 1.0008 | -0.08% | -0.08% | +0.00% | 1/5 (0t) |

## Seed Robustness Across Workloads (Canonical Accelergy 0.4 EDP)

| Workload | Arm | Mean EDP | Std EDP | CV (%) | Min EDP | Max EDP |
|---|---|---:|---:|---:|---:|---:|
| small | Hybrid (Iter<=3, 400-420) | 1.5516e+08 | 6.57e+05 | 0.42% | 1.5454e+08 | 1.5609e+08 |
| small | Hybrid (Iter<=1, 120) | 1.5732e+08 | 1.67e+06 | 1.06% | 1.5491e+08 | 1.5935e+08 |
| small | RL-only (120, top-5 debiased) | 1.5591e+08 | 1.98e+05 | 0.13% | 1.5567e+08 | 1.5613e+08 |
| small | GAMMA-only (120, top-5 debiased) | 1.5777e+08 | 1.58e+06 | 1.00% | 1.5575e+08 | 1.5961e+08 |
| baseline | Hybrid (Iter<=3, 400-420) | 6.2786e+12 | 2.74e+11 | 4.36% | 6.0243e+12 | 6.7015e+12 |
| baseline | Hybrid (Iter<=1, 120) | 6.6346e+12 | 2.97e+11 | 4.47% | 6.1702e+12 | 6.9694e+12 |
| baseline | RL-only (120, top-5 debiased) | 6.6177e+12 | 1.34e+11 | 2.03% | 6.4814e+12 | 6.8158e+12 |
| baseline | GAMMA-only (120, top-5 debiased) | 6.6124e+12 | 4.15e+11 | 6.27% | 6.1595e+12 | 7.1959e+12 |
| spatial-heavy | Hybrid (Iter<=3, 400-420) | 2.6018e+14 | 5.36e+12 | 2.06% | 2.5454e+14 | 2.6587e+14 |
| spatial-heavy | Hybrid (Iter<=1, 120) | 2.7160e+14 | 5.16e+12 | 1.90% | 2.6536e+14 | 2.7947e+14 |
| spatial-heavy | RL-only (120, top-5 debiased) | 2.7202e+14 | 3.64e+12 | 1.34% | 2.6702e+14 | 2.7680e+14 |
| spatial-heavy | GAMMA-only (120, top-5 debiased) | 2.6706e+14 | 9.11e+12 | 3.41% | 2.5792e+14 | 2.7778e+14 |
| channel-heavy | Hybrid (Iter<=3, 400-420) | 1.5390e+13 | 1.67e+11 | 1.08% | 1.5225e+13 | 1.5605e+13 |
| channel-heavy | Hybrid (Iter<=1, 120) | 1.6272e+13 | 1.09e+12 | 6.72% | 1.5420e+13 | 1.8082e+13 |
| channel-heavy | RL-only (120, top-5 debiased) | 1.6358e+13 | 7.03e+11 | 4.30% | 1.5830e+13 | 1.7575e+13 |
| channel-heavy | GAMMA-only (120, top-5 debiased) | 1.5832e+13 | 2.11e+11 | 1.33% | 1.5624e+13 | 1.6111e+13 |
| pointwise | Hybrid (Iter<=3, 400-420) | 2.9388e+11 | 4.49e+08 | 0.15% | 2.9353e+11 | 2.9463e+11 |
| pointwise | Hybrid (Iter<=1, 120) | 2.9566e+11 | 1.72e+09 | 0.58% | 2.9436e+11 | 2.9856e+11 |
| pointwise | RL-only (120, top-5 debiased) | 2.9434e+11 | 2.89e+08 | 0.10% | 2.9398e+11 | 2.9463e+11 |
| pointwise | GAMMA-only (120, top-5 debiased) | 2.9412e+11 | 3.74e+08 | 0.13% | 2.9385e+11 | 2.9463e+11 |
