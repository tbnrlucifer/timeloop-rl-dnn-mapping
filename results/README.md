# Experiment Results

This directory contains the experimental outputs generated during the development and evaluation of the RL-GAMMA DNN mapping framework.

The results are organized into current experiments, historical experiments, test artifacts, and execution logs.

## Directory Structure

```text
results/
├── experiments/
│   ├── hybrid/
│   ├── rl_only/
│   └── gamma_only/
├── historical/
│   ├── composite_gamma/
│   ├── direct_lpf/
│   ├── incomplete_runs/
│   └── repeated_runs/
├── tests/
└── logs/
```

### Current experiments

The `experiments/` directory contains the selected results from the current LPF-based implementation.

| Directory | Method |
|---|---|
| `hybrid/` | PPO + GAMMA |
| `rl_only/` | PPO only |
| `gamma_only/` | GAMMA only |

The current results include:

| Workload | RL-only | GAMMA-only | RL + GAMMA |
|---|---|---|---|
| Small | — | — | Seed 1 |
| Baseline | — | — | Seed 1 |
| Spatial-heavy | — | — | Seed 1 |
| Channel-heavy | — | — | Seed 1 |
| Pointwise | Seed 1 | Seed 1 | Seed 1 |

**Important:** Only the pointwise workload currently has results for all three methods. The other workloads have hybrid results only. Therefore, the current dataset does not yet support a complete three-way comparison across all workloads.

## Workload Configurations

The workloads use the following dimension order:

`[N, K, C, F, P, Q, R, S, T]`

| Workload | Configuration |
|---|---|
| Small | `[1, 16, 8, 32, 32, 32, 3, 3, 3]` |
| Baseline | `[1, 32, 16, 120, 120, 120, 3, 3, 3]` |
| Spatial-heavy | `[1, 32, 16, 224, 224, 224, 3, 3, 3]` |
| Channel-heavy | `[1, 128, 64, 56, 56, 56, 3, 3, 3]` |
| Pointwise | `[1, 64, 32, 112, 112, 112, 1, 1, 1]` |

All listed workloads are dense experiments. Sparse mapping evaluation has not been included in this result set.

## Historical Experiments

Historical results are retained to preserve the development and experimental history of the project.

### `historical/composite_gamma/`

Contains experiments using the earlier composite-GAMMA implementation.

These experiments use an earlier mapping representation and evaluation pipeline. They should not be directly pooled with results from the current LPF-based implementation.

### `historical/direct_lpf/`

Contains earlier experiments using the direct LPF-based approach, conducted during September and October 2026.

These results predate the selected current experiment runs and may use different configurations or implementation versions.

One historical result, `rl_ga_lpf_full_run_20260928_061948.json`, contains an inconsistent fitness/EDP sign. It should be excluded from quantitative comparisons.

### `historical/repeated_runs/`

Contains repeated executions of an experiment with the same reported configuration.

For example, the baseline seed-1 hybrid experiment was executed twice. Both runs reported the same best EDP and iteration results, but their runtimes differed.

Keeping both allows execution variability to be examined.

### `historical/incomplete_runs/`

Contains shorter or incomplete experiments that are not part of the selected current comparison.

These files are retained for reference but should not be treated as equivalent to the selected full runs.

## Tests

The `tests/` directory contains JSON outputs from development and smoke tests.

These artifacts are useful for checking implementation behavior but are not formal comparative experiment results.

The corresponding test scripts are located in `Computationally-Efficient-DNN-Mapping/`.

## Execution Logs

The `logs/` directory contains console output and iteration logs from experiment executions.

These logs are retained for debugging and traceability. They are excluded from Git tracking by the repository's `.gitignore` rules.

## Evaluation Metric

The primary optimization objective is Energy-Delay Product (EDP):

$$
EDP = Energy \times Cycles
$$

Lower EDP indicates a better mapping.

The GAMMA fitness objective is the negative EDP, so higher fitness corresponds to lower EDP.

## Experimental Limitations

- Only pointwise, seed 1 currently has a complete RL-only, GAMMA-only, and RL+GAMMA comparison.
- The other four workloads currently have hybrid results only.
- Multiple random seeds are needed to establish whether observed improvements are consistent.
- Historical experiments use different implementation versions and should be analyzed separately.
- Runtime comparisons should account for the different evaluation budgets and computational overhead of each method.
- All current listed experiments use dense workloads; sparse workloads remain future work.

The results should therefore be interpreted as preliminary experimental evidence rather than a definitive demonstration of the hybrid method's superiority.