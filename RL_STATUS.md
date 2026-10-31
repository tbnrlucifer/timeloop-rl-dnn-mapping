# RL Implementation and Research Status

## 1. Overview

The project initially explored reinforcement-learning-based DNN mapping using the approach described in:

**Computationally Efficient DNN Mapping Search Heuristic using Deep Reinforcement Learning**

The implementation has since evolved into a hybrid framework combining Proximal Policy Optimization (PPO), GAMMA's genetic algorithm, and Timeloop-based hardware evaluation.

The current implementation supports three experimental approaches:

- RL-only mapping search.
- GAMMA-only mapping search.
- Hybrid RL-GAMMA mapping search with evolutionary feedback to PPO.

---

## 2. Implementation Status

### Reinforcement Learning (PPO)

Implemented:

- State representation for LPF-based mapping generation.
- Candidate mapping generation.
- Actor-Critic network.
- PPO policy optimization and training loop.
- Reward computation using Timeloop evaluation.
- Mapping candidate storage and serialization.
- Continued PPO training using feedback mappings.

The shared PPO implementation is located in:

`Computationally-Efficient-DNN-Mapping/rl_script_fixed.py`

### GAMMA Integration

Implemented:

- LPF-based chromosome representation.
- Conversion of PPO-generated mappings into GAMMA-compatible individuals.
- GAMMA population initialization using RL-generated candidates.
- Genetic algorithm-based mapping evolution.
- Mapping validation and repair.
- Timeloop-based fitness evaluation using EDP.
- Elite mapping selection.

The integration is implemented in the `gamma-timeloop` submodule.

### RL-GAMMA Feedback

Implemented:

- Initial PPO mapping generation.
- RL-guided GAMMA initialization.
- Evolutionary search using GAMMA.
- Selection of elite GAMMA mappings.
- Feedback of selected mappings into PPO.
- Continued PPO training across successive iterations.

The main hybrid implementation is:

`Computationally-Efficient-DNN-Mapping/rl_ga_loop_lpf.py`

---

## 3. Experimental Baselines

The repository provides three independent experiment entry points:

| Approach | Script |
|---|---|
| RL-only | `rl_only_baseline.py` |
| GAMMA-only | `gamma_lpf_only.py` |
| RL-GAMMA | `rl_ga_loop_lpf.py` |

These implementations allow comparison of mapping quality, runtime, and search efficiency under different workload configurations and random seeds.

---

## 4. Testing and Validation

The following tests have been executed successfully:

- LPF representation and operator tests.
- Timeloop evaluation smoke tests.
- GAMMA-LPF integration smoke tests.
- Timeloop-backed operator tests.

These tests validate important parts of the implementation and its integration with the evaluation environment.

They do not, by themselves, establish that the hybrid approach outperforms either standalone baseline.

---

## 5. Known Limitations

### Mapping Representation Conversion

The conversion between PPO's LPF representation and GAMMA's chromosome representation is not necessarily lossless.

Some spatial mapping information may be represented temporally when it cannot be expressed directly in GAMMA's chromosome format. Mapping validation and repair can also modify candidates.

### Experimental Evaluation

A working implementation does not guarantee convergence to an optimal mapping.

The effectiveness of the hybrid approach still needs to be assessed through systematic comparisons across multiple workloads and random seeds.

### Computational Cost

The hybrid approach performs additional evolutionary search and PPO feedback operations. Any improvement in mapping quality must therefore be considered alongside its additional runtime and evaluation cost.

---

## 6. Ongoing Research

The main research objectives are:

1. Compare RL-only, GAMMA-only, and RL-GAMMA under consistent experimental conditions.
2. Evaluate performance across different workload characteristics.
3. Repeat experiments across multiple random seeds.
4. Analyze the consistency of mapping quality improvements.
5. Compare search runtime and evaluation cost.
6. Investigate the effect of RL-guided initialization and evolutionary feedback independently.

The central research question remains whether combining reinforcement learning with genetic algorithm-based search provides a measurable advantage over either approach independently.

---

## 7. Legacy Implementation

Earlier development versions used `modified_script.py` and other experimental scripts for candidate generation and Timeloop reevaluation.

These should be treated as legacy development artifacts rather than the primary implementation described in this document.