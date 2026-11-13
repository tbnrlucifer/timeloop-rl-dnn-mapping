# RL-only baseline results

## Per run

| Workload | Seed | Best EDP | Runtime (s) | Evaluated | Succeeded | Failed | Unique PPO candidates |
|---|---|---|---|---|---|---|---|
| small | 1 | 1.4763e+09 | 1431 | 120 | 120 | 0 | 13237 |
| small | 2 | 1.4753e+09 | 994 | 120 | 120 | 0 | 14424 |
| small | 3 | 1.4761e+09 | 1021 | 120 | 120 | 0 | 11890 |
| small | 4 | 1.4697e+09 | 1036 | 120 | 120 | 0 | 12302 |
| small | 42 | 1.4759e+09 | 1010 | 120 | 120 | 0 | 17800 |
| baseline | 1 | 6.5642e+13 | 1245 | 120 | 120 | 0 | 6651 |
| baseline | 2 | 6.6257e+13 | 1198 | 120 | 120 | 0 | 5849 |
| baseline | 3 | 6.5395e+13 | 1141 | 120 | 120 | 0 | 6232 |
| baseline | 4 | 6.3773e+13 | 1241 | 120 | 120 | 0 | 6529 |
| baseline | 42 | 6.7132e+13 | 1221 | 120 | 120 | 0 | 5848 |
| spatial-heavy | 1 | 2.6621e+15 | 1243 | 120 | 120 | 0 | 6903 |
| spatial-heavy | 2 | 2.6728e+15 | 1218 | 120 | 120 | 0 | 5825 |
| spatial-heavy | 3 | 2.6343e+15 | 1125 | 120 | 120 | 0 | 8189 |
| spatial-heavy | 4 | 2.6708e+15 | 1310 | 120 | 120 | 0 | 7283 |
| spatial-heavy | 42 | 2.6868e+15 | 1274 | 120 | 120 | 0 | 6009 |
| channel-heavy | 1 | 1.6221e+14 | 1220 | 120 | 120 | 0 | 8125 |
| channel-heavy | 2 | 1.7046e+14 | 1303 | 120 | 120 | 0 | 7495 |
| channel-heavy | 3 | 1.6136e+14 | 1247 | 120 | 120 | 0 | 7037 |
| channel-heavy | 4 | 1.6347e+14 | 1230 | 120 | 120 | 0 | 7707 |
| channel-heavy | 42 | 1.6230e+14 | 1262 | 120 | 120 | 0 | 6665 |
| pointwise | 1 | 2.0236e+12 | 1640 | 120 | 116 | 4 | 13514 |
| pointwise | 2 | 2.0178e+12 | 1734 | 120 | 110 | 10 | 13068 |
| pointwise | 3 | 2.0158e+12 | 1608 | 120 | 112 | 8 | 12880 |
| pointwise | 4 | 2.0237e+12 | 1728 | 120 | 107 | 13 | 14521 |
| pointwise | 42 | 2.0203e+12 | 1176 | 120 | 110 | 10 | 14351 |

## Per workload (across seeds)

| Workload | Seeds done | Mean best EDP | Std | Min | Max | Mean runtime (s) |
|---|---|---|---|---|---|---|
| small | 5/5 | 1.4747e+09 | 2.816e+06 | 1.4697e+09 | 1.4763e+09 | 1099 |
| baseline | 5/5 | 6.5640e+13 | 1.240e+12 | 6.3773e+13 | 6.7132e+13 | 1209 |
| spatial-heavy | 5/5 | 2.6653e+15 | 1.948e+13 | 2.6343e+15 | 2.6868e+15 | 1234 |
| channel-heavy | 5/5 | 1.6396e+14 | 3.713e+12 | 1.6136e+14 | 1.7046e+14 | 1252 |
| pointwise | 5/5 | 2.0203e+12 | 3.480e+09 | 2.0158e+12 | 2.0237e+12 | 1577 |

## Notes

- Script: `Computationally-Efficient-DNN-Mapping/rl_only_baseline.py`, unmodified. Seeds 1, 2, 3, 4, 42.
- Settings (defaults): `RL_ONLY_EVAL_BUDGET=120`, `LPF_PPO_INITIAL_EPISODES=30`.
- Thresholds: `LPF_WT_THRESHOLD=375`, `LPF_OT_THRESHOLD=216` (defaults) for small, baseline, spatial-heavy and channel-heavy.
  **Pointwise used `LPF_OT_THRESHOLD=32`**: with R=S=T=1, the maximum output reuse is C=32, so the default of 216 keeps 0 candidates.
  The JSON files do not record thresholds. Hybrid pointwise runs should use the same 375/32 thresholds so the results are comparable.
- "Failed" means Timeloop returned an invalid mapping or no EDP for that candidate. It only happened on pointwise (4–13 of 120 per run). No run crashed or timed out.
- `rl_only_pointwise_seed1_20261021_111634.json` is the team's earlier run (older code, thresholds unknown), so it is not comparable.
- Environment: WSL Ubuntu, Python 3.10, numpy<2. Accelergy 0.3 (commit da08b20) is needed because Timeloop rejects Accelergy 0.4 ERTs.
  Extra build fixes not covered in SETUP.md:
  - `#include <cstdint>` in the Timeloop sources (g++ 13)
  - build `libdefine_gterminate.so`
  - `conda tos accept`
  - CACTI copied into the plug-in folder
