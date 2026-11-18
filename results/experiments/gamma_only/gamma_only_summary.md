# GAMMA-only baseline results

Script: `Computationally-Efficient-DNN-Mapping/gamma_lpf_only.py` (unmodified), default settings `LPF_GAMMA_POP=20`, `LPF_GAMMA_GENS=5` (nominal budget 20 x (1+5) = 120 Timeloop evaluations, matching `RL_ONLY_EVAL_BUDGET=120`). Fitness objective: EDP (lower is better).

Timeloop evaluation counts come from the per-run logs (`results/logs/gamma_only/`), since the JSON does not record them.

## Per-run results

| Workload | Seed | Best EDP | Runtime (s) | Timeloop evals | Successful | Failed | Valid final candidates | Notes |
|---|---|---|---|---|---|---|---|---|
| small | 1 | 1.4936e+09 | 522.8 | 120 | 120 | 0 | 20 | — |
| small | 2 | 1.4886e+09 | 604.1 | 120 | 120 | 0 | 20 | — |
| small | 3 | 1.4895e+09 | 546.0 | 120 | 120 | 0 | 20 | — |
| small | 4 | 1.4697e+09 | 532.4 | 120 | 118 | 2 | 20 | 2 non-zero Timeloop return codes; 2 candidates got -inf fitness |
| small | 42 | 1.4851e+09 | 564.9 | 120 | 120 | 0 | 20 | — |
| baseline | 1 | 6.6205e+13 | 1079.3 | 120 | 120 | 0 | 20 | — |
| baseline | 2 | 6.2615e+13 | 1035.8 | 120 | 119 | 1 | 20 | 1 candidates got -inf fitness |
| baseline | 3 | 6.1842e+13 | 926.9 | 120 | 120 | 0 | 20 | — |
| baseline | 4 | 6.6860e+13 | 1032.5 | 120 | 120 | 0 | 20 | — |
| baseline | 42 | 6.9117e+13 | 798.1 | 120 | 119 | 1 | 20 | 1 non-zero Timeloop return codes; 1 candidates got -inf fitness |
| spatial-heavy | 1 | 2.6918e+15 | 1256.3 | 120 | 119 | 1 | 20 | 1 non-zero Timeloop return codes; 1 candidates got -inf fitness |
| spatial-heavy | 2 | 2.6059e+15 | 1376.6 | 120 | 119 | 1 | 20 | 1 non-zero Timeloop return codes; 1 candidates got -inf fitness |
| spatial-heavy | 3 | 2.6235e+15 | 1273.8 | 120 | 120 | 0 | 20 | — |
| spatial-heavy | 4 | 2.6200e+15 | 1344.3 | 120 | 119 | 1 | 20 | 1 non-zero Timeloop return codes; 1 candidates got -inf fitness |
| spatial-heavy | 42 | 2.7008e+15 | 1382.9 | 120 | 120 | 0 | 20 | — |
| channel-heavy | 1 | 1.6240e+14 | 1137.9 | 120 | 120 | 0 | 20 | — |
| channel-heavy | 2 | 1.6149e+14 | 1281.3 | 120 | 120 | 0 | 20 | — |
| channel-heavy | 3 | 1.6094e+14 | 991.9 | 120 | 120 | 0 | 20 | — |
| channel-heavy | 4 | 1.6267e+14 | 1226.6 | 120 | 119 | 1 | 20 | 1 non-zero Timeloop return codes; 1 candidates got -inf fitness |
| channel-heavy | 42 | 1.6103e+14 | 1200.3 | 120 | 120 | 0 | 20 | — |
| pointwise | 1 | 2.0209e+12 | 145.3 | 120 | 119 | 1 | 20 | 1 non-zero Timeloop return codes; 1 candidates got -inf fitness |
| pointwise | 2 | 2.0157e+12 | 137.4 | 120 | 119 | 1 | 20 | 1 non-zero Timeloop return codes; 1 candidates got -inf fitness |
| pointwise | 3 | 2.0237e+12 | 162.1 | 120 | 118 | 2 | 20 | 2 non-zero Timeloop return codes; 2 candidates got -inf fitness |
| pointwise | 4 | 2.0157e+12 | 133.8 | 120 | 117 | 3 | 20 | 3 non-zero Timeloop return codes; 3 candidates got -inf fitness |
| pointwise | 42 | 2.0157e+12 | 133.3 | 120 | 119 | 1 | 20 | 1 non-zero Timeloop return codes; 1 candidates got -inf fitness |

## Per-workload aggregate (over seeds 1, 2, 3, 4, 42)

| Workload | Runs OK | Best EDP (min) | Mean EDP | Std EDP | Mean runtime (s) |
|---|---|---|---|---|---|
| small | 5/5 | 1.4697e+09 | 1.4853e+09 | 9.2538e+06 | 554.0 |
| baseline | 5/5 | 6.1842e+13 | 6.5328e+13 | 3.0409e+12 | 974.5 |
| spatial-heavy | 5/5 | 2.6059e+15 | 2.6484e+15 | 4.4328e+13 | 1326.8 |
| channel-heavy | 5/5 | 1.6094e+14 | 1.6171e+14 | 7.9002e+11 | 1167.6 |
| pointwise | 5/5 | 2.0157e+12 | 2.0184e+12 | 3.7035e+09 | 142.4 |

## Run environment and caveats

- Machine: WSL2 Ubuntu 24.04, Python 3.10 (uv), `numpy<2`, CPU-only torch. Runs executed 4 in parallel (18 cores), so runtimes may be slightly inflated versus running alone.
- Timeloop: `pytimeloop/lib/timeloop` at the pinned submodule commit, built with `--accelergy`; GCC 13 needed `-include cstdint` (injected via a compiler wrapper, no source changes).
- Accelergy: not covered by SETUP.md. Installed Accelergy v0.3-era (`accelergy` @ pre-2023-08 master), `accelergy-aladdin-plug-in` 7fe4102, `accelergy-cacti-plug-in` dece954 (CACTI binary copied next to the plug-in). Estimators used: Aladdin_table for RegisterFile/MACC, CACTI for GlobalBuffer.
- Failed Timeloop evaluations are reported by the script as `-inf` fitness ("Timeloop evaluation did not return an EDP result"); no run aborted.

### Cost-model mismatch with earlier committed results (IMPORTANT)

The earlier committed `gamma_lpf_only_pointwise_seed1_20261021_110503.json` reports best EDP 2.938e11. Re-evaluating *that exact best mapping* on this machine gives EDP 2.016e12 (Energy 143,475 uJ x 14,049,408 cycles) — 6.86x higher. This equals the best EDP found here for pointwise seeds 2/4/42, so GAMMA finds equally good mappings; the difference is the Timeloop/Accelergy energy model (the RegisterFile read energy of ~6.1 pJ from Aladdin dominates at 24.85 pJ/compute).

**Absolute EDPs in this table are therefore not directly comparable to RL-only/hybrid results produced on a different Accelergy installation.** RL-only and GAMMA-only must be evaluated with the same Accelergy version and plug-ins (same ERT) before comparing.
