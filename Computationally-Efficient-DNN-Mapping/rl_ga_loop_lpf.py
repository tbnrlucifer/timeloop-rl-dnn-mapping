"""RL -> GAMMA(LPF) -> RL hybrid experiment.

This is the direct-LPF experimental branch.  The existing rl_ga_loop.py remains
untouched and is the composite-GAMMA baseline.

Key difference:
    PPO tuple list == GAMMA chromosome == Timeloop input (after getmapping).

There is no GAMMA->PPO projection or composite tile reconstruction in this
branch. GAMMA's population/selection/generation framework is retained, while
its chromosome/operators are adapted to LPF genes.
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GAMMA_SRC = ROOT / 'gamma-timeloop' / 'src'
sys.path.insert(0, str(GAMMA_SRC))

# Shared PPO implementation. Importing rl_ga_loop is safe because its main
# experiment is guarded by if __name__ == '__main__'.
from rl_ga_loop import (  # noqa: E402
    IN_CONFIG_DIR,
    REPORT_DIR,
    RESULTS_DIR,
    _jsonify,
    set_experiment_seed,
)
from timeloop_env import TimeloopEnv  # noqa: E402
from rl_script_fixed import train_network, ppo_timeloop_feedback  # noqa: E402
from gamma_lpf_env import GammaLPFEnv  # noqa: E402

EXPERIMENT_SEED = int(os.environ.get('RL_GA_SEED', '42'))
GAMMA_POP = int(os.environ.get('LPF_GAMMA_POP', '20'))
GAMMA_GENS = int(os.environ.get('LPF_GAMMA_GENS', '5'))
GAMMA_ELITES = int(os.environ.get('LPF_GAMMA_ELITES', '5'))
ITERATIONS = int(os.environ.get('LPF_ITERATIONS', '3'))
PPO_FEEDBACK_STEPS = int(os.environ.get('LPF_PPO_FEEDBACK_STEPS', '4'))
PPO_CONTINUED_EPISODES = int(os.environ.get('LPF_PPO_CONTINUED_EPISODES', '4'))

# Select a workload with RL_GA_WORKLOAD.  All workloads keep the same hardware
# architecture; only Timeloop problem dimensions and PPO's initial factors vary.
WORKLOADS = {
    'baseline':       [1, 32, 16, 120, 120, 120, 3, 3, 3],
    'small':          [1, 16, 8, 32, 32, 32, 3, 3, 3],
    'spatial-heavy':  [1, 32, 16, 224, 224, 224, 3, 3, 3],
    'channel-heavy':  [1, 128, 64, 56, 56, 56, 3, 3, 3],
    'pointwise':      [1, 64, 32, 112, 112, 112, 1, 1, 1],
}
WORKLOAD_NAME = os.environ.get('RL_GA_WORKLOAD', 'baseline').strip().lower()
if WORKLOAD_NAME not in WORKLOADS:
    raise ValueError(
        f"Unknown RL_GA_WORKLOAD={WORKLOAD_NAME!r}; choose one of {sorted(WORKLOADS)}"
    )
DIMENSION = WORKLOADS[WORKLOAD_NAME]
# PPO's factor list uses [R,S,T,P,Q,F,C,K], unlike Timeloop's dimension order.
INITIAL_INPUT_LIST = [
    DIMENSION[6], DIMENSION[7], DIMENSION[8],
    DIMENSION[4], DIMENSION[5], DIMENSION[3],
    DIMENSION[2], DIMENSION[1],
]


def create_workload_timeloop_evaluator():
    """Create a Timeloop evaluator bound to the selected workload dimensions."""
    timeloop_env = TimeloopEnv(
        config_path=REPORT_DIR,
        in_config_dir=IN_CONFIG_DIR,
        debug=True,
        use_sparse=False,
        use_IO=True,
    )
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
            fitness_obj=['edp'],
        )

    return timeloop_env, evaluator


def create_gamma_lpf_env():
    return GammaLPFEnv(
        in_config_dir=IN_CONFIG_DIR,
        fitness_obj=['edp'],
        report_dir=REPORT_DIR,
        use_pool=False,
        use_IO=True,
        debug=True,
        use_sparse=False,
    )


def rank_gamma_lpf(pops, fitness):
    ranked = []

    for indv, fit in zip(pops, fitness):
        gamma_fitness = float(np.asarray(fit).reshape(-1)[0])
        edp = -gamma_fitness

        ranked.append({
            'fitness': gamma_fitness,
            'edp': edp,
            'tuples': [
                (int(x[0]), int(x[1]))
                for x in indv
            ],
        })

    ranked.sort(key=lambda x: x['fitness'], reverse=True)
    return ranked


def run():
    seed = set_experiment_seed(EXPERIMENT_SEED)
    experiment_start = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = RESULTS_DIR / f'rl_ga_lpf_{WORKLOAD_NAME}_seed{seed}_full_run_{stamp}.json'

    results = {
        'experiment': {
            'name': 'RL-GAMMA-LPF-RL hybrid run',
            'workload_name': WORKLOAD_NAME,
            'timestamp': stamp,
            'experiment_seed': seed,
            'representation': 'canonical PPO LPF tuple chromosome',
            'gamma_search_framework': 'GAMMA population/selection/crossover/mutation adapted to LPFs',
            'dimension': _jsonify(DIMENSION),
            'initial_input_list': _jsonify(INITIAL_INPUT_LIST),
            'gamma': {
                'population': GAMMA_POP,
                'generations': GAMMA_GENS,
                'elite_count': GAMMA_ELITES,
                'parents_ratio': 0.5,
                'inject_ratio': 0.1,
            },
            'ppo': {
                'feedback_steps_per_elite': PPO_FEEDBACK_STEPS,
                'continued_training_episodes': PPO_CONTINUED_EPISODES,
            },
        },
        'timing_seconds': {},
        'initial_rl': {},
        'iterations': [],
        'final': {},
    }

    print('\n==============================================')
    print('       RL-GAMMA-LPF DNN MAPPING PIPELINE')
    print('==============================================\n')
    print(f'Experiment seed: {seed}')
    print(f'Canonical chromosome: PPO LPF tuple list')
    print(f'Results: {results_path}')

    # Reuse the project's existing PPO Timeloop evaluator for feedback.
    # GAMMA-LPF itself evaluates through the direct mapping path in its own
    # TimeloopEnv, so this is only the Timeloop -> PPO feedback evaluator.
    _, timeloop_evaluator = create_workload_timeloop_evaluator()

    print('\nRunning initial PPO training...')
    t0 = time.perf_counter()
    acnet, optimizer, mappings = train_network(
        INITIAL_INPUT_LIST,
        episodes=int(os.environ.get("LPF_PPO_INITIAL_EPISODES", "30")),
    )
    initial_elapsed = time.perf_counter() - t0
    initial_rewards = [float(x['rl_reward']) for x in mappings]
    initial_best = max(mappings, key=lambda x: float(x['rl_reward'])) if mappings else None
    results['timing_seconds']['initial_rl'] = initial_elapsed
    results['initial_rl'] = {
        'mapping_count': len(mappings),
        'best_rl_reward': float(initial_best['rl_reward']) if initial_best else None,
        'best_rl_tuples': _jsonify(initial_best['tuples']) if initial_best else None,
        'mean_rl_reward': float(np.mean(initial_rewards)) if initial_rewards else None,
    }
    print(f'Initial PPO mappings: {len(mappings)}')

    best_ga = []

    for iteration in range(ITERATIONS):
        iteration_start = time.perf_counter()
        top = sorted(mappings, key=lambda x: float(x['rl_reward']), reverse=True)[:20]
        rl_population = [
            [(int(t[0]), int(t[1])) for t in x['tuples']]
            for x in top
        ]

        if iteration == 0:
            population = rl_population[:GAMMA_POP]
            previous_ga = 0
        else:
            previous_ga = min(len(best_ga), GAMMA_ELITES)
            population = best_ga[:previous_ga] + rl_population[:max(0, GAMMA_POP - previous_ga)]

        if not population:
            raise RuntimeError('No PPO mappings available to seed GAMMA-LPF')

        # Defensive LPF validation happens inside GammaLPFEnv.run() before
        # every Timeloop evaluation.
        gamma_env = create_gamma_lpf_env()
        gamma_start = time.perf_counter()

        final_pops, final_fitness = gamma_env.run(
            dimension=DIMENSION,
            num_pops=GAMMA_POP,
            num_gens=GAMMA_GENS,
            elite_ratio=GAMMA_ELITES / float(GAMMA_POP),
            parents_ratio=0.5,
            inject_ratio=0.1,
            initial_population=population,
        )

        gamma_elapsed = time.perf_counter() - gamma_start
        ranked = rank_gamma_lpf(final_pops, final_fitness)

        selected = ranked[:GAMMA_ELITES]
        best_ga = [
            [(int(g[0]), int(g[1])) for g in x['tuples']]
            for x in selected
        ]

        feedback_start = time.perf_counter()
        feedback_steps = ppo_timeloop_feedback(
            acnet=acnet,
            optimizer=optimizer,
            ga_elites=best_ga,
            timeloop_evaluator=timeloop_evaluator,
            steps_per_elite=PPO_FEEDBACK_STEPS,
            dimension=DIMENSION,
        )
        feedback_elapsed = time.perf_counter() - feedback_start

        acnet, optimizer, mappings = train_network(
            INITIAL_INPUT_LIST,
            acnet,
            optimizer,
            episodes=PPO_CONTINUED_EPISODES,
            seed_tuples=best_ga,
        )

        print(f'Continued PPO mappings: {len(mappings)}')

        if mappings:
            print(
                f"Best continued RL reward: "
                f"{max(float(x['rl_reward']) for x in mappings):.6f}"
            )

            results['iterations'].append({
            'iteration': iteration + 1,
            'population': {
                'size': len(population),
                'previous_lpf_gamma_elites': previous_ga,
                'new_rl_candidates': len(population) - previous_ga,
            },
            'gamma': {
                'final_population_count': len(ranked),
                'selected_for_ppo_feedback': selected,
            },
            'feedback': {
                'feedback_steps': int(feedback_steps)
            },
            'continued_rl': {
                'mapping_count': len(mappings),
                'best_rl_reward': (
                    max(float(x['rl_reward']) for x in mappings)
                    if mappings else None
                ),
            },
            'timing_seconds': {
                'gamma': gamma_elapsed,
                'feedback': feedback_elapsed,
                'iteration_total': time.perf_counter() - iteration_start,
            },
        })

        with results_path.open('w') as f:
            json.dump(_jsonify(results), f, indent=2)

    results['timing_seconds']['total'] = time.perf_counter() - experiment_start
    results['final']['best_lpf_gamma_edp'] = min(
        (x['edp'] for item in results['iterations'] for x in item['gamma']['selected_for_ppo_feedback']),
        default=None,
    )
    with results_path.open('w') as f:
        json.dump(_jsonify(results), f, indent=2)
    print('\n==============================================')
    print('              PIPELINE COMPLETE')
    print('==============================================')
    print(f"Best GAMMA EDP seen: {results['final']['best_lpf_gamma_edp']}")
    print(f"Total experiment time: {results['timing_seconds']['total']:.2f} s")
    print(f"Results saved to: {results_path}")
    return results_path


if __name__ == '__main__':
    run()
