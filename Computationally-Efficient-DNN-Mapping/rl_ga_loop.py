import sys
import json
import time
import os
import random
from datetime import datetime
import numpy as np
import torch
from pathlib import Path

# ============================================================
# Import paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAMMA_SRC = PROJECT_ROOT / "gamma-timeloop" / "src"

sys.path.insert(0, str(GAMMA_SRC))


from rl_script_fixed import (
    train_network,
    ppo_timeloop_feedback,
    getmapping,
    format_mapping_dict,
)
from gamma_timeloop_env import GammaTimeloopEnv
from mapping_validation import validate_ppo_tuples, validate_gamma_individual
from timeloop_env import TimeloopEnv


# ============================================================
# Configuration
# ============================================================

DIMENSION = [1, 32, 16, 120, 120, 120, 3, 3, 3]

IN_CONFIG_DIR = str(
    PROJECT_ROOT / "gamma-timeloop" / "src" / "in_config_3d"
)

REPORT_DIR = str(
    PROJECT_ROOT / "gamma-timeloop" / "src" / "report_rl_ga_3d"
)

RESULTS_DIR = PROJECT_ROOT / "results"

INITIAL_INPUT_LIST = [3, 3, 3, 120, 120, 120, 16, 32]

# Controlled-experiment seed. Override per run, for example:
#   RL_GA_SEED=1 python ...
#
# Environment-variable configuration avoids changing the source file for
# every benchmark seed while keeping the chosen seed in the JSON output.
EXPERIMENT_SEED = int(os.environ.get("RL_GA_SEED", "42"))


# ============================================================
# Reproducibility
# ============================================================

def set_experiment_seed(seed):
    """
    Seed the RNGs used by the RL-GAMMA hybrid pipeline.

    This establishes identical initial stochastic conditions for runs that
    use the same seed. It does not force strict CUDA determinism; the goal
    is controlled statistical comparison without changing the algorithm or
    risking unsupported deterministic CUDA operations.
    """
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    return seed


# ============================================================
# Utilities
# ============================================================

def tuples_to_state(tuples):
    mapped = assign_increasing_values(tuples)
    norm = Normalizer().normalize(mapped)
    return norm


def _jsonify(value):
    """Convert tuples, NumPy values and nested containers to JSON-safe values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(_jsonify(v) for v in value)
    return value


def evaluate_projected_tuples(tuples, timeloop_evaluator):
    """
    Evaluate the PPO interpretation of a GAMMA-projected tuple sequence.

    Timeloop remains the evaluator.  This path is:
        tuples -> getmapping -> format_mapping_dict -> Timeloop.
    """

    validate_ppo_tuples(
        tuples,
        DIMENSION,
        getmapping_fn=getmapping,
    )

    memmap = getmapping(list(tuples))
    mapping_dict = format_mapping_dict(memmap)
    result = timeloop_evaluator(list(tuples), mapping_dict)

    if not isinstance(result, dict) or "edp" not in result:
        raise RuntimeError(
            f"Projected Timeloop evaluation did not return an EDP: {result!r}"
        )

    return {
        "edp": float(result["edp"]),
        "energy_pJ": (
            float(result["energy_pJ"])
            if result.get("energy_pJ") is not None else None
        ),
        "cycles": (
            float(result["cycles"])
            if result.get("cycles") is not None else None
        ),
        "utilization": (
            float(result["utilization"])
            if result.get("utilization") is not None else None
        ),
        "memmap": _jsonify(result.get("memmap")),
    }


# ============================================================
# Timeloop evaluator
# ============================================================

def create_timeloop_evaluator():
    """
    Create a direct Timeloop evaluator for RL/GA mappings.

    The RL/GA side works with tuple-lists.
    getmapping() converts those into a Timeloop mapping dictionary.
    TimeloopEnv.run_mapping_dict() evaluates that mapping.
    """

    timeloop_env = TimeloopEnv(
        config_path=REPORT_DIR,
        in_config_dir=IN_CONFIG_DIR,
        debug=True,
        use_sparse=False,
        use_IO=True,
    )

    # Create IO-backed pool directories.
    timeloop_env.create_pool_env(
        num_pools=1,
        dimension=DIMENSION,
        indv=None,
        use_IO=True,
    )

    def evaluator(tuples, memmap):

        return timeloop_env.run_mapping_dict(
            dimension=DIMENSION,
            mapping=memmap,
            pool_idx=0,
            fitness_obj=["edp"],
        )

    return timeloop_env, evaluator


# ============================================================
# Native GAMMA integration
# ============================================================

def create_gamma_env():
    return GammaTimeloopEnv(
        in_config_dir=IN_CONFIG_DIR,
        fitness_obj=["edp"],
        report_dir=REPORT_DIR,
        use_pool=False,
        use_IO=True,
        debug=True,
        use_sparse=False,
    )


def gamma_refine(gamma_env, rl_population, generations=5,
                 pop_size=20, elite_size=5):
    """
    Run native GAMMA evolution and return the complete final population.

    GAMMA itself still evolves and ranks candidates using its native fitness.
    We deliberately retain all final individuals because the GAMMA -> PPO
    conversion is a projection, and native GAMMA rank is not guaranteed to be
    preserved after PPO projection and Timeloop evaluation.
    """
    # rl_tuples_to_individual() validates against dimension_dict, so GAMMA
    # dimensions must exist before converting PPO seeds.
    gamma_env.set_dimension(DIMENSION)

    seeds = [
        gamma_env.rl_tuples_to_individual(tuples)
        for tuples in rl_population
    ]

    final_pops, final_fitness = gamma_env.run(
        dimension=DIMENSION,
        num_pops=pop_size,
        num_gens=generations,
        elite_ratio=elite_size / float(pop_size),
        parents_ratio=0.5,
        inject_ratio=0.1,
        initial_population=seeds,
    )

    ranked = []
    for indv, fit in zip(final_pops, final_fitness):
        # GAMMA's Timeloop objective is maximized; EDP fitness is negative
        # EDP, therefore larger fitness means smaller positive EDP.
        fitness_value = float(np.asarray(fit).reshape(-1)[0])
        ranked.append((fitness_value, indv))

    ranked.sort(key=lambda x: x[0], reverse=True)

    if len(ranked) != pop_size:
        raise RuntimeError(
            f"Expected {pop_size} final GAMMA candidates, got {len(ranked)}"
        )

    return ranked


def gamma_individual_to_tuples(gamma_env, indv):
    """Project a native GAMMA individual into PPO's fixed LPF tuple space.

    This is deliberately a projection, not a lossless inverse.  GAMMA stores
    explicit per-level tile factors and spatial dimensions; PPO represents a
    mapping as one ordered LPF list whose spatial part is inferred by
    getmapping() from a contiguous prefix and the 256-PE limit.

    Projection convention:
      1. Traverse GAMMA levels l1 -> lN and dimensions in each level's
         loop_order.
      2. Expand every tile product into prime LPFs.
      3. Emit LPFs from GAMMA par_dims first (spatial preference), then emit
         every remaining LPF in the same canonical level/order traversal.
      4. Validate the resulting tuple list with the PPO parser.

    The spatial-first ordering is only a projection heuristic.  getmapping()
    remains authoritative for the actual PPO spatial/RF/Shared/DRAM split.
    """
    gamma_to_ppo = {
        'R': 1,
        'S': 2,
        'T': 3,
        'P': 4,
        'Q': 5,
        'F': 6,
        'C': 7,
        'K': 8,
    }

    if int(gamma_env.dimension_dict.get('N', 1)) != 1:
        raise ValueError(
            'Current PPO representation has no LPFs for N; '
            'GAMMA->PPO projection requires N == 1.'
        )

    spatial_tokens = []
    temporal_tokens = []

    for level in range(1, gamma_env.num_buf_levels + 1):
        level_data = indv[f'l{level}']
        par_dims = set(level_data.get('par_dims', set()))

        for dim in level_data['loop_order']:
            # PPO has no N dimension for the current workload.
            if dim == 'N':
                continue
            if dim not in gamma_to_ppo:
                raise ValueError(f'Unsupported GAMMA dimension {dim!r}')

            factor = int(gamma_env.get_prod(level_data['tile_size'][dim]))
            if factor < 1:
                raise ValueError(
                    f'Invalid GAMMA factor {factor} for l{level}/{dim}'
                )
            if factor == 1:
                continue

            # Emit prime LPFs in deterministic ascending-prime order.
            prime_counts = gamma_env.get_prime_factors(factor)
            tokens = []
            for prime in sorted(prime_counts):
                # IMPORTANT: emit a distinct tuple object for every LPF.
                # getmapping() tracks placement by object identity, so list
                # multiplication such as [(id, prime)] * count would make
                # repeated LPFs aliases of the same tuple object and cause
                # the parser to drop occurrences.
                for _ in range(int(prime_counts[prime])):
                    tokens.append((gamma_to_ppo[dim], int(prime)))

            if dim in par_dims:
                spatial_tokens.extend(tokens)
            else:
                temporal_tokens.extend(tokens)

    tuples = spatial_tokens + temporal_tokens

    expected_products = {
        1: int(DIMENSION[6]),  # R
        2: int(DIMENSION[7]),  # S
        3: int(DIMENSION[8]),  # T
        4: int(DIMENSION[4]),  # P
        5: int(DIMENSION[5]),  # Q
        6: int(DIMENSION[3]),  # F
        7: int(DIMENSION[2]),  # C
        8: int(DIMENSION[1]),  # K
    }
    products = {loop_id: 1 for loop_id in expected_products}
    for loop_id, lpf in tuples:
        if loop_id not in products:
            raise ValueError(f'Invalid PPO loop ID {loop_id}')
        products[loop_id] *= int(lpf)

    if products != expected_products:
        raise RuntimeError(
            f'GAMMA->PPO factor loss: {products} != {expected_products}'
        )

    expected_lpf_count = sum(
        sum(gamma_env.get_prime_factors(value).values())
        for value in expected_products.values()
    )
    if len(tuples) != expected_lpf_count:
        raise RuntimeError(
            f'GAMMA->PPO LPF count mismatch: {len(tuples)} '
            f'!= {expected_lpf_count}'
        )

    # Validate with PPO's actual parser.  The resulting memmap is not required
    # to reproduce GAMMA exactly; it is the definitive PPO interpretation of
    # this projected tuple sequence.
    memmap = getmapping(list(tuples))
    flattened = [tup for level in memmap for tup in level]
    if sorted(flattened) != sorted(tuples):
        raise RuntimeError('PPO parser lost or duplicated LPFs')

    spatial_product = 1
    for _, lpf in memmap[0]:
        spatial_product *= int(lpf)
    if spatial_product > 256:
        raise RuntimeError(
            f'PPO spatial PE product exceeds 256: {spatial_product}'
        )

    return tuples


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Seed before constructing the Timeloop/GAMMA environments or the PPO
    # network so same-seed experiments start from the same RNG state.
    experiment_seed = set_experiment_seed(EXPERIMENT_SEED)

    experiment_start = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"rl_ga_full_run_{run_stamp}.json"

    results = {
        "experiment": {
            "name": "RL-GAMMA-RL 3-iteration hybrid run",
            "timestamp": run_stamp,
            "experiment_seed": experiment_seed,
            "selection_policy": "projected_timeloop_top5",
            "dimension": _jsonify(DIMENSION),
            "initial_input_list": _jsonify(INITIAL_INPUT_LIST),
            "gamma": {
                "population": 20,
                "generations": 5,
                "parents_ratio": 0.5,
                "native_elite_count": 5,
                "ppo_feedback_selection_count": 5,
                "ppo_feedback_selection":
                    "projected Timeloop top-5 across all final candidates",
                "inject_ratio": 0.1,
            },
            "ppo": {
                "feedback_steps_per_elite": 4,
                "continued_training_episodes": 4,
            },
        },
        "timing_seconds": {},
        "initial_rl": {},
        "iterations": [],
        "final": {},
    }

    print("\n==============================================")
    print("       RL-GA DNN MAPPING PIPELINE")
    print("==============================================\n")
    print(f"Experiment seed: {experiment_seed}")
    print("Selection policy: projected Timeloop top-5")
    print(f"Experiment JSON: {results_path}")

    # --------------------------------------------------------
    # Step 1: Create Timeloop evaluator
    # --------------------------------------------------------

    print("Creating Gamma/Timeloop evaluator...")
    timeloop_env, timeloop_evaluator = create_timeloop_evaluator()
    print("Timeloop evaluator ready.\n")

    # --------------------------------------------------------
    # Step 2: Initial PPO training
    # --------------------------------------------------------

    print("Running initial RL training...\n")
    initial_start = time.perf_counter()
    acnet, optimizer, mappings = train_network(INITIAL_INPUT_LIST)
    initial_elapsed = time.perf_counter() - initial_start

    initial_rewards = [float(item["rl_reward"]) for item in mappings]
    initial_best = (
        max(mappings, key=lambda item: float(item["rl_reward"]))
        if mappings else None
    )

    results["timing_seconds"]["initial_rl"] = initial_elapsed
    results["initial_rl"] = {
        "mapping_count": len(mappings),
        "best_rl_reward": (
            float(initial_best["rl_reward"]) if initial_best else None
        ),
        "best_rl_tuples": (
            _jsonify(initial_best["tuples"]) if initial_best else None
        ),
        "mean_rl_reward": (
            float(np.mean(initial_rewards)) if initial_rewards else None
        ),
        "top_rl_rewards": sorted(initial_rewards, reverse=True)[:20],
    }

    print("\nInitial RL training completed.")
    print(f"RL mappings obtained: {len(mappings)}")
    print(f"Best initial RL reward: {results['initial_rl']['best_rl_reward']}")

    # --------------------------------------------------------
    # Step 3: RL -> GAMMA -> Timeloop -> PPO
    # --------------------------------------------------------

    best_ga = []

    for iteration in range(3):
        iteration_start = time.perf_counter()
        iteration_result = {
            "iteration": iteration + 1,
            "population": {},
            "gamma": {},
            "feedback": {},
            "continued_rl": {},
            "timing_seconds": {},
        }

        print(
            f"\n=============================================="
            f"\n        HYBRID ITERATION {iteration + 1}/3"
            f"\n=============================================="
        )

        # ----------------------------------------------------
        # Select best RL candidates
        # ----------------------------------------------------

        top = sorted(
            mappings,
            key=lambda x: x["rl_reward"],
            reverse=True,
        )[:20]

        rl_population = [item["tuples"] for item in top]
        selected_rewards = [float(item["rl_reward"]) for item in top]
        selected_best = top[0] if top else None

        print(
            f"\nSelected {len(rl_population)} RL candidates "
            f"for GAMMA refinement."
        )

        # ----------------------------------------------------
        # Add previous GAMMA elites
        # ----------------------------------------------------

        if iteration == 0:
            population = rl_population[:20]
            num_ga = 0
            num_rl = len(population)
        else:
            # Preserve previous projected GAMMA elites explicitly because
            # GammaTimeloopEnv.run() truncates initial_population[:num_pops].
            num_ga = min(len(best_ga), 5)
            num_rl = 20 - num_ga
            population = best_ga[:num_ga] + rl_population[:num_rl]

        iteration_result["population"] = {
            "size": len(population),
            "previous_gamma_elites": num_ga,
            "new_rl_candidates": num_rl,
            "selected_rl_best_reward": (
                float(selected_best["rl_reward"])
                if selected_best else None
            ),
            "selected_rl_best_tuples": (
                _jsonify(selected_best["tuples"])
                if selected_best else None
            ),
            "selected_rl_mean_reward": (
                float(np.mean(selected_rewards))
                if selected_rewards else None
            ),
        }

        print(f"GAMMA initial population: {len(population)}")

        # ----------------------------------------------------
        # PPO -> native GAMMA -> Timeloop
        # ----------------------------------------------------

        print("\nSeeding native GAMMA with PPO mappings...")
        gamma_start = time.perf_counter()
        gamma_env = create_gamma_env()

        ranked_gamma = gamma_refine(
            gamma_env=gamma_env,
            rl_population=population,
            generations=5,
            pop_size=20,
            elite_size=5,
        )
        gamma_elapsed = time.perf_counter() - gamma_start
        iteration_result["timing_seconds"]["gamma"] = gamma_elapsed

        print(
            "\nProjecting all final GAMMA candidates and evaluating their "
            "PPO interpretations..."
        )

        # IMPORTANT:
        # GAMMA evolution remains native-GAMMA-driven.  Only the selection
        # used for GAMMA -> PPO feedback is projection-aware.
        #
        # For every final GAMMA individual:
        #   native GAMMA individual
        #       -> gamma_individual_to_tuples()
        #       -> PPO getmapping()
        #       -> Timeloop
        #
        # We then select the best five PPO-representable mappings by the
        # projected Timeloop EDP.
        all_gamma_candidates = []

        for native_rank, (fitness_value, indv) in enumerate(
            ranked_gamma, start=1
        ):
            native_edp = -float(fitness_value)

            validate_gamma_individual(
                indv,
                DIMENSION,
                gamma_env.num_buf_levels,
                gamma_env.buf_spmap_cstr,
            )

            projected_tuples = gamma_individual_to_tuples(
                gamma_env, indv
            )
            projected_metrics = evaluate_projected_tuples(
                projected_tuples, timeloop_evaluator
            )

            candidate = {
                "native_gamma_rank": native_rank,
                "native_gamma_edp": native_edp,
                "projected_timeloop_edp": projected_metrics["edp"],
                "projected_timeloop_energy_pJ":
                    projected_metrics["energy_pJ"],
                "projected_timeloop_cycles":
                    projected_metrics["cycles"],
                "projected_timeloop_utilization":
                    projected_metrics["utilization"],
                "projected_tuples": _jsonify(projected_tuples),
                "projected_memmap": projected_metrics["memmap"],
            }
            all_gamma_candidates.append(candidate)

            print(
                f"  native #{native_rank:2d}/"
                f"{len(ranked_gamma):2d}: "
                f"native GAMMA EDP={native_edp:.6e} | "
                f"projected Timeloop EDP="
                f"{projected_metrics['edp']:.6e}"
            )

        # Assign the projected rank over the entire final GAMMA population.
        projected_sorted = sorted(
            all_gamma_candidates,
            key=lambda item: item["projected_timeloop_edp"],
        )
        for projected_rank, candidate in enumerate(
            projected_sorted, start=1
        ):
            candidate["projected_timeloop_rank"] = projected_rank

        # Projection-aware selection.
        # PPO receives the five GAMMA candidates with the best
        # PPO-projected Timeloop EDP.
        selected_gamma = projected_sorted[:5]

        print("\nProjection-aware top-5 PPO feedback selection:")
        for candidate in selected_gamma:
            print(
                f"  projected #{candidate['projected_timeloop_rank']}: "
                f"native rank #{candidate['native_gamma_rank']} | "
                f"native EDP={candidate['native_gamma_edp']:.6e} | "
                f"projected EDP="
                f"{candidate['projected_timeloop_edp']:.6e}"
            )

        native_top5 = all_gamma_candidates[:5]
        selected_native_ranks = [
            candidate["native_gamma_rank"]
            for candidate in selected_gamma
        ]
        selected_outside_native_top5 = [
            candidate["native_gamma_rank"]
            for candidate in selected_gamma
            if candidate["native_gamma_rank"] > 5
        ]

        iteration_result["gamma"] = {
            "final_population_count": len(all_gamma_candidates),
            "native_top5_count": len(native_top5),
            "projection_selected_count": len(selected_gamma),
            # Complete final GAMMA population, in native GAMMA rank order.
            # This makes the native-vs-projected ranking auditable.
            "candidates": all_gamma_candidates,
            # Native GAMMA top five retained explicitly for comparison.
            "native_top5": native_top5,
            # Actual PPO feedback selection, in projected Timeloop rank order.
            "selected_for_ppo_feedback": selected_gamma,
            "selected_native_gamma_ranks": selected_native_ranks,
            "selected_outside_native_top5":
                selected_outside_native_top5,
            "best_native_gamma_edp": (
                all_gamma_candidates[0]["native_gamma_edp"]
                if all_gamma_candidates else None
            ),
            "best_native_gamma_rank": 1 if all_gamma_candidates else None,
            "best_native_gamma_tuples": (
                all_gamma_candidates[0]["projected_tuples"]
                if all_gamma_candidates else None
            ),
            "best_projected_timeloop_edp": (
                projected_sorted[0]["projected_timeloop_edp"]
                if projected_sorted else None
            ),
            "best_projected_native_gamma_rank": (
                projected_sorted[0]["native_gamma_rank"]
                if projected_sorted else None
            ),
        }

        # Projection-aware top-5 elites become the next PPO starting states.
        # Reuse the exact projected tuples that were evaluated and selected.
        best_ga = [
            [tuple(item) for item in candidate["projected_tuples"]]
            for candidate in selected_gamma
        ]

        # ----------------------------------------------------
        # Timeloop -> PPO: genuine on-policy feedback
        # ----------------------------------------------------

        feedback_start = time.perf_counter()
        feedback_steps = ppo_timeloop_feedback(
            acnet=acnet,
            optimizer=optimizer,
            ga_elites=best_ga,
            timeloop_evaluator=timeloop_evaluator,
            steps_per_elite=4,
        )
        feedback_elapsed = time.perf_counter() - feedback_start

        iteration_result["feedback"] = {
            "elite_count": len(best_ga),
            "steps_per_elite": 4,
            "feedback_steps": int(feedback_steps),
        }
        iteration_result["timing_seconds"]["feedback"] = feedback_elapsed

        print(f"Timeloop -> PPO feedback steps: {feedback_steps}")

        # ----------------------------------------------------
        # Continue PPO training
        # ----------------------------------------------------

        print("\nContinuing PPO training...")
        continued_start = time.perf_counter()
        acnet, optimizer, mappings = train_network(
            INITIAL_INPUT_LIST,
            acnet,
            optimizer,
            episodes=4,
            seed_tuples=best_ga,
        )
        continued_elapsed = time.perf_counter() - continued_start
        iteration_result["timing_seconds"]["continued_rl"] = (
            continued_elapsed
        )

        continued_rewards = [
            float(item["rl_reward"]) for item in mappings
        ]
        continued_best = (
            max(mappings, key=lambda item: float(item["rl_reward"]))
            if mappings else None
        )

        iteration_result["continued_rl"] = {
            "mapping_count": len(mappings),
            "best_rl_reward": (
                float(continued_best["rl_reward"])
                if continued_best else None
            ),
            "best_rl_tuples": (
                _jsonify(continued_best["tuples"])
                if continued_best else None
            ),
            "mean_rl_reward": (
                float(np.mean(continued_rewards))
                if continued_rewards else None
            ),
            "top_rl_rewards": sorted(
                continued_rewards, reverse=True
            )[:20],
        }

        iteration_result["timing_seconds"]["iteration_total"] = (
            time.perf_counter() - iteration_start
        )
        results["iterations"].append(iteration_result)

        # Persist after every completed iteration so a long run still leaves
        # recoverable data if a later stage fails.
        with results_path.open("w") as f:
            json.dump(_jsonify(results), f, indent=2)

        print(
            f"Iteration {iteration + 1} results saved to: "
            f"{results_path}"
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    all_native = []
    all_projected = []

    for iteration_item in results["iterations"]:
        for elite in iteration_item["gamma"]["candidates"]:
            candidate = {
                "iteration": int(iteration_item["iteration"]),
                "native_gamma_rank": int(
                    elite["native_gamma_rank"]
                ),
                "projected_timeloop_rank": int(
                    elite["projected_timeloop_rank"]
                ),
                "native_gamma_edp": float(
                    elite["native_gamma_edp"]
                ),
                "projected_timeloop_edp": float(
                    elite["projected_timeloop_edp"]
                ),
                "projected_tuples": elite["projected_tuples"],
            }
            all_native.append(candidate)
            all_projected.append(candidate)

    best_native = (
        min(all_native, key=lambda item: item["native_gamma_edp"])
        if all_native else None
    )
    best_projected = (
        min(
            all_projected,
            key=lambda item: item["projected_timeloop_edp"],
        )
        if all_projected else None
    )

    final_rewards = [float(item["rl_reward"]) for item in mappings]
    final_best = (
        max(mappings, key=lambda item: float(item["rl_reward"]))
        if mappings else None
    )

    results["timing_seconds"]["total"] = (
        time.perf_counter() - experiment_start
    )
    results["final"] = {
        "final_mapping_count": len(mappings),
        "best_final_rl_reward": (
            float(final_best["rl_reward"]) if final_best else None
        ),
        "best_final_rl_tuples": (
            _jsonify(final_best["tuples"]) if final_best else None
        ),
        "mean_final_rl_reward": (
            float(np.mean(final_rewards))
            if final_rewards else None
        ),
        "best_native_gamma": best_native,
        "best_projected_timeloop": best_projected,
    }

    with results_path.open("w") as f:
        json.dump(_jsonify(results), f, indent=2)

    print(
        "\n=============================================="
        "\n              PIPELINE COMPLETE"
        "\n=============================================="
    )
    print(f"Final experiment JSON: {results_path}")
    print(f"Total time: {results['timing_seconds']['total']:.2f} s")
