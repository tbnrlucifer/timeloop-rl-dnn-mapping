import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch

from rl_ga_loop_lpf import (
    WORKLOADS,
    WORKLOAD_NAME,
    DIMENSION,
    EXPERIMENT_SEED,
    GAMMA_POP,
    GAMMA_GENS,
    IN_CONFIG_DIR,
    REPORT_DIR,
    RESULTS_DIR,
    set_experiment_seed,
    _jsonify,
    create_gamma_lpf_env,
)


def main():
    seed = set_experiment_seed(EXPERIMENT_SEED)
    start = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = RESULTS_DIR / (
        f"gamma_lpf_only_{WORKLOAD_NAME}_seed{seed}_{stamp}.json"
    )

    print("GAMMA-only baseline", flush=True)
    print(f"Workload: {WORKLOAD_NAME}", flush=True)
    print(f"Dimension: {DIMENSION}", flush=True)
    print(f"Seed: {seed}", flush=True)
    print(f"Population: {GAMMA_POP}; generations: {GAMMA_GENS}", flush=True)
    print("Initialization: GAMMA-native random population", flush=True)

    env = create_gamma_lpf_env()

    # No PPO population is passed: GAMMA creates its own population.
    final_pops, final_fitness = env.run(
        dimension=DIMENSION,
        num_pops=GAMMA_POP,
        num_gens=GAMMA_GENS,
        initial_population=None,
    )

    ranked = []
    for individual, fitness in zip(final_pops, final_fitness):
        score = float(np.asarray(fitness).reshape(-1)[0])
        ranked.append({
            "fitness": score,
            "edp": -score,
            "tuples": [
                [int(gene[0]), int(gene[1])]
                for gene in individual
            ],
        })

    ranked.sort(key=lambda item: item["edp"])
    valid = [item for item in ranked if np.isfinite(item["edp"]) and item["edp"] >= 0]

    result = {
        "experiment": {
            "name": "GAMMA-only LPF baseline",
            "workload_name": WORKLOAD_NAME,
            "dimension": _jsonify(DIMENSION),
            "seed": seed,
            "population": GAMMA_POP,
            "generations": GAMMA_GENS,
            "initialization": "GAMMA-native",
            "fitness_objective": "EDP",
        },
        "final": {
            "best_edp": valid[0]["edp"] if valid else None,
            "best_mapping": valid[0]["tuples"] if valid else None,
            "valid_final_candidates": len(valid),
        },
        "final_population": ranked,
        "runtime_seconds": time.perf_counter() - start,
    }

    output.write_text(json.dumps(result, indent=2))
    print(f"Best GAMMA-only EDP: {result['final']['best_edp']}", flush=True)
    print(f"Runtime: {result['runtime_seconds']:.2f} s", flush=True)
    print(f"Results saved to: {output}", flush=True)


if __name__ == "__main__":
    main()
