# Timeloop RL-GA DNN Mapping

This branch extends the reinforcement-learning-based DNN mapping
workflow with a custom Genetic Algorithm (GA) feedback loop and
Timeloop-based evaluation.

The project is based on the approach described in:

**Computationally Efficient DNN Mapping Search Heuristic using Deep
Reinforcement Learning**

The current research direction combines:

- Reinforcement Learning (RL) for generating DNN mappings
- A custom Genetic Algorithm for evolutionary refinement
- Timeloop for hardware-aware mapping evaluation
- PPO-based feedback from GA solutions to the RL mapper

---

## Research Workflow

The current hybrid workflow is:

```text
                 ┌─────────────────────┐
                 │   RL Mapping Search │
                 └──────────┬──────────┘
                            │
                     RL candidate
                       mappings
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Custom GA        │
                 │  mapping refinement │
                 └──────────┬──────────┘
                            │
                       elite mappings
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Timeloop Evaluator │
                 │     Energy × Cycles │
                 └──────────┬──────────┘
                            │
                         EDP/reward
                            │
                            ▼
                 ┌─────────────────────┐
                 │   PPO Feedback      │
                 │   to RL network     │
                 └──────────┬──────────┘
                            │
                            ▼
                    Subsequent RL search
                            │
                            └──────► repeat
