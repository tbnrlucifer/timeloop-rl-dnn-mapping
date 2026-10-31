"""
End-to-end validation of GAMMA-style LPF operators.

Each operator must:
  1. preserve the canonical LPF multiset,
  2. produce a PPO-parser-valid chromosome,
  3. produce a Timeloop-valid mapping,
  4. receive a finite EDP from the real timeloop-model.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAMMA_SRC = PROJECT_ROOT / "gamma-timeloop" / "src"
PYTIMELOOP_ROOT = PROJECT_ROOT / "pytimeloop"

for path in (PYTIMELOOP_ROOT, GAMMA_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from gamma_lpf_chromosome import (
    validate_chromosome,
    multiset_fingerprint,
)
from gamma_timeloop_env import getmapping
from gamma_lpf_env import GammaLPFEnv
from timeloop_env import TimeloopEnv


DIMENSION = [1, 32, 16, 120, 120, 120, 3, 3, 3]

IN_CONFIG_DIR = str(GAMMA_SRC / "in_config_3d")
REPORT_DIR = str(Path(__file__).resolve().parent / "report_lpf_operator_timeloop")


LOOP_ID_TO_NAME = {
    1: 'R',
    2: 'S',
    3: 'T',
    4: 'P',
    5: 'Q',
    6: 'F',
    7: 'C',
    8: 'K',
    9: 'N',
}

ALL_DIMS = [
    'R', 'S', 'T', 'P', 'Q',
    'F', 'C', 'K', 'N'
]


def format_mapping_dict(memmap):
    """Same formatter used by the PPO implementation."""
    level_map = {
        0: ('GlobalBuffer', 'spatial'),
        1: ('RegisterFile', 'temporal'),
        2: ('GlobalBuffer', 'temporal'),
        3: ('MainMemory', 'temporal'),
    }

    out = []

    for li, (target, mtype) in level_map.items():
        fd, perm = {}, []

        if li < len(memmap):
            for lid, fac in memmap[li]:
                nm = LOOP_ID_TO_NAME.get(lid, f'L{lid}')
                fd[nm] = fd.get(nm, 1) * fac

                if nm not in perm:
                    perm.append(nm)

        for d in ALL_DIMS:
            fd.setdefault(d, 1)

            if d not in perm:
                perm.append(d)

        out.append({
            'target': target,
            'type': mtype,
            'factors': ' '.join(
                f"{d}={fd[d]}" for d in ALL_DIMS
            ),
            'permutation': ''.join(perm),
        })

    return {'mapping': out}


def make_seed():
    """
    Construct every repeated LPF occurrence independently.

    This is important because getmapping() tracks tuple occurrences
    using Python object identity.
    """
    factors_by_loop = {
        1: [3],
        2: [3],
        3: [3],
        4: [2, 2, 2, 3, 5],
        5: [2, 2, 2, 3, 5],
        6: [2, 2, 2, 3, 5],
        7: [2, 2, 2, 2],
        8: [2, 2, 2, 2, 2],
    }

    seed = []

    for loop_id in [1, 2, 3, 4, 5, 6, 7, 8]:
        for factor in factors_by_loop[loop_id]:
            seed.append(tuple([loop_id, factor]))

    return seed


def evaluate_timeloop(tuples, timeloop_env):
    """Validate, map, and execute one canonical chromosome."""

    validate_chromosome(
        tuples,
        DIMENSION,
        getmapping_fn=getmapping,
    )

    memmap = getmapping(list(tuples))
    mapping_dict = format_mapping_dict(memmap)

    result = timeloop_env.run_mapping_dict(
        dimension=DIMENSION,
        mapping=mapping_dict,
        pool_idx=0,
        fitness_obj=["edp"],
    )

    if result is None:
        raise RuntimeError("Timeloop returned None")

    edp = result.get("edp")

    if edp is None:
        raise RuntimeError("Timeloop result has no EDP")

    if not isinstance(edp, (int, float)):
        raise RuntimeError(f"Non-numeric EDP: {edp}")

    if edp <= 0:
        raise RuntimeError(f"Invalid EDP: {edp}")

    return result


def test_operator(name, chromosome, original_fp, timeloop_env):
    print(f"\n{'=' * 70}")
    print(f"TEST: {name}")
    print(f"{'=' * 70}")

    print(f"Chromosome length: {len(chromosome)}")

    # 1. LPF multiset preservation.
    fingerprint = multiset_fingerprint(chromosome)

    if fingerprint != original_fp:
        raise AssertionError(
            f"{name}: LPF multiset changed"
        )

    print("  [PASS] LPF multiset preserved")

    # 2. Full canonical validation.
    validate_chromosome(
        chromosome,
        DIMENSION,
        getmapping_fn=getmapping,
    )

    print("  [PASS] Canonical LPF validation")

    # 3. Real Timeloop.
    result = evaluate_timeloop(
        chromosome,
        timeloop_env,
    )

    print("  [PASS] PPO getmapping()")

    print(
        f"  [PASS] Timeloop | "
        f"EDP={result['edp']:.6e}"
    )

    return result


def main():

    print("=" * 70)
    print(" GAMMA LPF OPERATORS -> PPO -> TIMELOOP TEST")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------------

    seed = make_seed()

    print("\n1. Validating seed...")

    validate_chromosome(
        seed,
        DIMENSION,
        getmapping_fn=getmapping,
    )

    original_fp = multiset_fingerprint(seed)

    print("   PASS: seed is a valid canonical LPF chromosome.")
    print(f"   LPF count: {len(seed)}")

    # ------------------------------------------------------------------
    # Construct GAMMA LPF environment without running its evolutionary
    # loop. We only need the actual operator implementations.
    # ------------------------------------------------------------------

    env = object.__new__(GammaLPFEnv)

    env.dimension = DIMENSION
    env.parallizable_dim_note = [
        'N', 'K', 'C', 'F', 'P', 'Q'
    ]

    # ------------------------------------------------------------------
    # Timeloop environment
    # ------------------------------------------------------------------

    print("\n2. Creating Timeloop environment...")

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

    print("   PASS: Timeloop environment created.")

    # ------------------------------------------------------------------
    # Operator tests
    # ------------------------------------------------------------------

    results = {}

    print("\n3. Running GAMMA-style operators...")

    random.seed(42)
    x = env.mutate_order(
        seed[:],
        alpha=1.0,
    )

    results["mutate_order"] = test_operator(
        "mutate_order",
        x,
        original_fp,
        timeloop_env,
    )

    random.seed(43)
    x = env.mutate_level(
        x,
        alpha=1.0,
    )

    results["mutate_level"] = test_operator(
        "mutate_level",
        x,
        original_fp,
        timeloop_env,
    )

    random.seed(44)
    x = env.mutate_tiles(
        x,
        alpha=1.0,
        num_mu_loc=2,
    )

    results["mutate_tiles"] = test_operator(
        "mutate_tiles",
        x,
        original_fp,
        timeloop_env,
    )

    random.seed(45)
    x = env.mutate_par(
        x,
        alpha=1.0,
    )

    results["mutate_par"] = test_operator(
        "mutate_par",
        x,
        original_fp,
        timeloop_env,
    )

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    random.seed(46)

    children = env.crossover(
        [seed[:], x[:]],
        [seed, x],
        num_injects=0,
        alpha=1.0,
    )

    print(
        f"\nCrossover produced {len(children)} child(ren)."
    )

    for i, child in enumerate(children):
        results[f"crossover_{i + 1}"] = test_operator(
            f"crossover child {i + 1}",
            child,
            original_fp,
            timeloop_env,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(" FINAL SUMMARY")
    print("=" * 70)

    for name, result in results.items():
        print(
            f"{name:20s} "
            f"EDP={result['edp']:.6e}"
        )

    print("\n" + "=" * 70)
    print("PASS: every tested GAMMA-style operator produced a")
    print("      canonical LPF chromosome that executed successfully")
    print("      on the real Timeloop runtime.")
    print("=" * 70)


if __name__ == "__main__":
    main()