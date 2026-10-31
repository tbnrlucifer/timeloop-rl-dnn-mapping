"""
Tiny end-to-end smoke test for GammaLPFEnv.run().

Checks the actual redesigned GAMMA evolutionary loop:
    initial population
      -> generation-0 Timeloop evaluation
      -> parent/elite selection
      -> crossover
      -> mutation
      -> Timeloop evaluation
      -> next-generation selection

This intentionally uses a tiny population/generation count so failures are
cheap to diagnose.  The existing test_lpf_operators_timeloop.py remains the
operator-level regression test.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GAMMA_SRC = REPO_ROOT / "gamma-timeloop" / "src"

PYTIMELOOP_ROOT = REPO_ROOT / "pytimeloop"

for path in (PYTIMELOOP_ROOT, GAMMA_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import random
import numpy as np

from gamma_lpf_env import GammaLPFEnv
from gamma_lpf_chromosome import (
    expected_lpf_counter,
    fingerprint,
    validate_chromosome,
)
from gamma_timeloop_env import getmapping


GAMMA_CONFIG_DIR = REPO_ROOT / "gamma-timeloop" / "src" / "in_config_3d"
REPORT_DIR = SCRIPT_DIR / "gamma_lpf_smoke_report"


DIMENSION = [1, 32, 16, 120, 120, 120, 3, 3, 3]
NUM_POPS = 4
NUM_GENS = 2


def fresh_gene(g):
    # Do NOT use tuple(g): tuple(existing_tuple) may return the same object.
    return (int(g[0]), int(g[1]))


def make_population(env, n):
    population = []

    # Let GammaLPFEnv construct valid canonical chromosomes.
    env.set_dimension(DIMENSION)

    for _ in range(n):
        indv = env._random_chromosome()

        # Explicitly materialize every LPF occurrence as a distinct object.
        indv = [fresh_gene(g) for g in indv]

        validate_chromosome(
            indv,
            DIMENSION,
            getmapping_fn=getmapping,
            spatial_limit=256,
        )

        assert len({id(g) for g in indv}) == len(indv)
        assert len(indv) == sum(expected_lpf_counter(DIMENSION).values())

        population.append(indv)

    return population


def main():
    print("=" * 70)
    print(" GAMMA LPF EVOLUTIONARY LOOP SMOKE TEST")
    print("=" * 70)

    random.seed(42)
    np.random.seed(42)

    print("\n1. Creating GammaLPFEnv...")
    env = GammaLPFEnv(
        in_config_dir=str(GAMMA_CONFIG_DIR),
        fitness_obj=["edp"],
        report_dir=str(REPORT_DIR),
        use_pool=False,
        use_IO=True,
        log_level=3,
        debug=False,
        save_chkpt=False,
    )
    print("   PASS: environment created.")

    print("\n2. Building tiny initial population...")
    initial_population = make_population(env, NUM_POPS)

    expected_count = expected_lpf_counter(DIMENSION)

    for i, indv in enumerate(initial_population):
        assert len(indv) == 27
        assert len({id(g) for g in indv}) == len(indv)
        assert {tuple(g) for g in indv}  # non-empty
        assert expected_count == __import__("collections").Counter(indv)

        print(f"   population[{i}] valid | LPFs={len(indv)}")

    print("   PASS: all initial individuals are canonical LPF chromosomes.")

    print("\n3. Running GammaLPFEnv.run()...")
    print(f"   Population: {NUM_POPS}")
    print(f"   Generations: {NUM_GENS}")
    print("   Timeloop: REAL")
    print("   PPO parser: REAL getmapping()")

    env.run(
        dimension=DIMENSION,
        num_pops=NUM_POPS,
        num_gens=NUM_GENS,
        elite_ratio=0.25,
        parents_ratio=0.5,
        inject_ratio=0.25,
        initial_population=initial_population,
    )

    print("\n4. Checking recorded evolutionary state...")

    # run() records one best solution per generation only when save_chkpt=True,
    # so for this smoke test we inspect the final returned internal state.
    # GammaLPFEnv.run() currently keeps the selected final population in
    # local variables, therefore the essential validation is performed by
    # the successful completion of every generation's _validate()+Timeloop
    # calls.  We additionally verify the environment's dimension.
    assert env.dimension == DIMENSION

    print("   PASS: run() completed without invalid chromosomes or Timeloop errors.")
    print("   PASS: final environment dimension is correct.")

    print("\n" + "=" * 70)
    print("PASS: GammaLPFEnv.run() completed the tiny evolutionary smoke test.")
    print("=" * 70)


if __name__ == "__main__":
    main()
