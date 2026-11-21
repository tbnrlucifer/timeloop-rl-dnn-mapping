# Canonical Reevaluation & Paired Comparison Summary

- Evaluated `75/75` winners in `387.9s`.
- Canonical evaluator: local `TimeloopEnv` (`in_config_3d`, `use_sparse=False`, `use_IO=True`, Accelergy `0.4`).

## Self-Consistency & Cross-Environment Scale Audit

| Arm | Evaluated | Scale (Canon / Hist) Min | Scale Median | Scale Max |
|---|---:|---:|---:|---:|
| hybrid | 25/25 | 1.000000 | 1.000000 | 1.000000 |
| rl_only | 25/25 | 0.098103 | 0.103022 | 0.145832 |
| gamma_only | 25/25 | 0.096748 | 0.102169 | 0.145777 |

## Paired Comparison: `hybrid` vs `rl_only` (Canonical Accelergy 0.4 EDP)

- **Pairs compared:** 25/25
- **hybrid wins / ties / losses:** 23 / 0 / 2
- **Overall geomean EDP ratio (hybrid / rl_only):** `0.9671` (`+3.29%` EDP reduction; >0% means `hybrid` is better)
- **Overall median EDP ratio (hybrid / rl_only):** `0.9809`
- **Two-sided Wilcoxon signed-rank test:** W=18.00, p=1.508e-05

| Workload | hybrid Geomean EDP | rl_only Geomean EDP | Geomean Ratio (hybrid/rl_only) | EDP Reduction | hybrid Wins |
|---|---:|---:|---:|---:|---:|
| small | 1.5516e+08 | 1.5636e+08 | 0.9924 | +0.76% | 5/5 (0 tie) |
| baseline | 6.2739e+12 | 6.6167e+12 | 0.9482 | +5.18% | 4/5 (0 tie) |
| spatial-heavy | 2.6014e+14 | 2.7200e+14 | 0.9564 | +4.36% | 5/5 (0 tie) |
| channel-heavy | 1.5390e+13 | 1.6346e+13 | 0.9415 | +5.85% | 5/5 (0 tie) |
| pointwise | 2.9388e+11 | 2.9434e+11 | 0.9984 | +0.16% | 4/5 (0 tie) |

## Paired Comparison: `hybrid` vs `gamma_only` (Canonical Accelergy 0.4 EDP)

- **Pairs compared:** 25/25
- **hybrid wins / ties / losses:** 20 / 1 / 4
- **Overall geomean EDP ratio (hybrid / gamma_only):** `0.9756` (`+2.44%` EDP reduction; >0% means `hybrid` is better)
- **Overall median EDP ratio (hybrid / gamma_only):** `0.9841`
- **Two-sided Wilcoxon signed-rank test:** W=68.00, p=0.00963503

| Workload | hybrid Geomean EDP | gamma_only Geomean EDP | Geomean Ratio (hybrid/gamma_only) | EDP Reduction | hybrid Wins |
|---|---:|---:|---:|---:|---:|
| small | 1.5516e+08 | 1.5792e+08 | 0.9826 | +1.74% | 5/5 (0 tie) |
| baseline | 6.2739e+12 | 6.6021e+12 | 0.9503 | +4.97% | 4/5 (0 tie) |
| spatial-heavy | 2.6014e+14 | 2.6693e+14 | 0.9745 | +2.55% | 3/5 (0 tie) |
| channel-heavy | 1.5390e+13 | 1.5831e+13 | 0.9721 | +2.79% | 5/5 (0 tie) |
| pointwise | 2.9388e+11 | 2.9412e+11 | 0.9992 | +0.08% | 3/5 (1 tie) |

## Paired Comparison: `rl_only` vs `gamma_only` (Canonical Accelergy 0.4 EDP)

- **Pairs compared:** 25/25
- **rl_only wins / ties / losses:** 10 / 1 / 14
- **Overall geomean EDP ratio (rl_only / gamma_only):** `1.0088` (`-0.88%` EDP reduction; >0% means `rl_only` is better)
- **Overall median EDP ratio (rl_only / gamma_only):** `1.0010`
- **Two-sided Wilcoxon signed-rank test:** W=93.00, p=0.103404

| Workload | rl_only Geomean EDP | gamma_only Geomean EDP | Geomean Ratio (rl_only/gamma_only) | EDP Reduction | rl_only Wins |
|---|---:|---:|---:|---:|---:|
| small | 1.5636e+08 | 1.5792e+08 | 0.9901 | +0.99% | 4/5 (1 tie) |
| baseline | 6.6167e+12 | 6.6021e+12 | 1.0022 | -0.22% | 3/5 (0 tie) |
| spatial-heavy | 2.7200e+14 | 2.6693e+14 | 1.0190 | -1.90% | 1/5 (0 tie) |
| channel-heavy | 1.6346e+13 | 1.5831e+13 | 1.0325 | -3.25% | 1/5 (0 tie) |
| pointwise | 2.9434e+11 | 2.9412e+11 | 1.0008 | -0.08% | 1/5 (0 tie) |
