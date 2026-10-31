"""RL-only baseline: PPO candidate generation followed by Timeloop EDP evaluation."""

import json
import os
import random
import time
from datetime import datetime

import numpy as np

from rl_ga_loop_lpf import (
    WORKLOAD_NAME,
    DIMENSION,
    EXPERIMENT_SEED,
    IN_CONFIG_DIR,
    REPORT_DIR,
    RESULTS_DIR,
    set_experiment_seed,
    _jsonify,
    create_workload_timeloop_evaluator,
)
from rl_script_fixed import train_network


def main():
    seed = set_experiment_seed(EXPERIMENT_SEED)
    start = time.perf_counter()

    # Match GAMMA-only's nominal evaluation budget: 20 + 5 * 20 = 120.
    budget = int(os.environ.get("RL_ONLY_EVAL_BUDGET", "120"))
    episodes = int(os.environ.get("LPF_PPO_INITIAL_EPISODES", "30"))

    if budget < 1:
        raise ValueError("RL_ONLY_EVAL_BUDGET must be positive")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = RESULTS_DIR / (
        f"rl_only_{WORKLOAD_NAME}_seed{seed}_{stamp}.json"
    )

    print("RL-only PPO baseline", flush=True)
    print(f"Workload: {WORKLOAD_NAME}; dimension: {DIMENSION}", flush=True)
    print(f"Seed: {seed}; PPO episodes per initialization: {episodes}", flush=True)
    print(f"Timeloop evaluation budget: {budget}", flush=True)

    # Same PPO training code and initializations as the hybrid, without GAMMA.
    _, _, candidates = train_network(
        [
            DIMENSION[6], DIMENSION[7], DIMENSION[8],
            DIMENSION[4], DIMENSION[5], DIMENSION[3],
            DIMENSION[2], DIMENSION[1],
        ],
        episodes=episodes,
    )
    print(f"Retained PPO candidates: {len(candidates)}", flush=True)

    # Deduplicate mappings while preserving the first record for each LPF tuple list.
    unique = {}
    for candidate in candidates:
        key = tuple(
            (int(t[0]), int(t[1]))
            for t in candidate["tuples"]
        )
        if key not in unique:
            unique[key] = candidate

    unique_candidates = list(unique.values())
    print(f"Unique PPO candidates: {len(unique_candidates)}", flush=True)

    if not unique_candidates:
        raise RuntimeError(
            "PPO retained no candidates. Check LPF_WT_THRESHOLD and "
            "LPF_OT_THRESHOLD for this workload."
        )

    # Seeded sampling avoids evaluating only the earliest PPO trajectories.
    rng = random.Random(seed)
    selected = rng.sample(
        unique_candidates,
        min(budget, len(unique_candidates)),
    )

    _, evaluator = create_workload_timeloop_evaluator()
    evaluated = []
    failures = 0

    for i, candidate in enumerate(selected, start=1):
        try:
            result = evaluator(candidate["tuples"], candidate["mapping"])
            edp = None if result is None else result.get("edp")

            if edp is None or not np.isfinite(float(edp)) or float(edp) <= 0:
                failures += 1
                print(f"[{i}/{len(selected)}] invalid EDP", flush=True)
                continue

            edp = float(edp)
            evaluated.append({
                "edp": edp,
                "rl_reward": float(candidate["rl_reward"]),
                "tuples": [
                    [int(t[0]), int(t[1])]
                    for t in candidate["tuples"]
                ],
                "mapping": candidate["mapping"],
                "episode": int(candidate["episode"]),
            })
            print(
                f"[{i}/{len(selected)}] EDP={edp:.8e}",
                flush=True,
            )

        except Exception as exc:
            failures += 1
            print(
                f"[{i}/{len(selected)}] Timeloop evaluation failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    evaluated.sort(key=lambda x: x["edp"])
    elapsed = time.perf_counter() - start

    result = {
        "experiment": {
            "name": "RL-only PPO baseline",
            "workload_name": WORKLOAD_NAME,
            "dimension": _jsonify(DIMENSION),
            "seed": seed,
            "ppo_episodes_per_initialization": episodes,
            "initializations": 10,
            "evaluation_budget": budget,
            "unique_candidates_available": len(unique_candidates),
            "unique_candidates_selected": len(selected),
            "timeloop_evaluations_succeeded": len(evaluated),
            "timeloop_evaluations_failed": failures,
            "selection": "seeded random sample of unique retained PPO candidates",
            "fitness_objective": "Timeloop EDP",
            "gamma_used": False,
        },
        "final": {
            "best_edp": evaluated[0]["edp"] if evaluated else None,
            "best_mapping": evaluated[0]["mapping"] if evaluated else None,
            "best_tuples": evaluated[0]["tuples"] if evaluated else None,
        },
        "evaluated_candidates": evaluated,
        "runtime_seconds": elapsed,
    }

    output.write_text(json.dumps(result, indent=2))
    print(f"Best RL-only EDP: {result['final']['best_edp']}", flush=True)
    print(f"Successful Timeloop evaluations: {len(evaluated)}", flush=True)
    print(f"Failed evaluations: {failures}", flush=True)
    print(f"Runtime: {elapsed:.2f} s", flush=True)
    print(f"Results saved to: {output}", flush=True)


if __name__ == "__main__":
    main()
