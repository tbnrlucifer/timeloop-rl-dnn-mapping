"""Real Timeloop smoke test for the canonical LPF chromosome."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAMMA_SRC = PROJECT_ROOT / "gamma-timeloop" / "src"
PYTIMELOOP_ROOT = PROJECT_ROOT / "pytimeloop"

for path in (PYTIMELOOP_ROOT, GAMMA_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from gamma_lpf_chromosome import validate_chromosome
from gamma_timeloop_env import getmapping
from timeloop_env import TimeloopEnv

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

ALL_DIMS = ['R', 'S', 'T', 'P', 'Q', 'F', 'C', 'K', 'N']


def format_mapping_dict(memmap):
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

DIMENSION = [1, 32, 16, 120, 120, 120, 3, 3, 3]

IN_CONFIG_DIR = str(
    GAMMA_SRC / "in_config_3d"
)

REPORT_DIR = str(Path(__file__).resolve().parent / "report_lpf_timeloop_smoke")


def make_seed():
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
            # Explicitly construct a fresh tuple object.
            seed.append(tuple([loop_id, factor]))

    return seed


def main():
    tuples = make_seed()

    print("==============================================")
    print("       LPF -> PPO PARSER -> TIMELOOP TEST")
    print("==============================================")

    print("\n1. Validating canonical LPF chromosome...")

    validate_chromosome(
        tuples,
        DIMENSION,
        getmapping_fn=getmapping,
    )

    print("   PASS: LPF chromosome is valid.")

    print("\n2. Converting LPF chromosome with PPO getmapping()...")

    memmap = getmapping(list(tuples))

    print(f"   PPO mapping levels: {len(memmap)}")
    print("   PASS: getmapping() accepted the chromosome.")

    print("\n3. Converting PPO mapping to Timeloop mapping dictionary...")

    mapping_dict = format_mapping_dict(memmap)

    print("\n========== DEBUG MEMMAP ==========")
    for i, level in enumerate(memmap):
        print(f"Level {i}: {level}")

    print("\n========== DEBUG TIMELOOP MAPPING ==========")
    for entry in mapping_dict["mapping"]:
        print(entry)
    print("============================================\n")

    if not isinstance(mapping_dict, dict):
        raise RuntimeError(
            f"format_mapping_dict() returned {type(mapping_dict)}"
        )

    if "mapping" not in mapping_dict:
        raise RuntimeError(
            "Timeloop mapping dictionary has no 'mapping' key."
        )

    print(
        f"   Timeloop mapping entries: "
        f"{len(mapping_dict['mapping'])}"
    )
    print("   PASS: mapping dictionary constructed.")

    print("\n4. Creating Timeloop environment...")

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

    print("\n5. Running actual Timeloop evaluation...")
    print("   This should launch timeloop-model.\n")

    result = timeloop_env.run_mapping_dict(
        dimension=DIMENSION,
        mapping=mapping_dict,
        pool_idx=0,
        fitness_obj=["edp"],
    )

    print("\n==============================================")
    print("              TIMELOOP RESULT")
    print("==============================================")

    print(f"Result: {result}")

    if result is None:
        raise RuntimeError(
            "Timeloop returned None. Evaluation failed."
        )

    required = [
        "energy_pJ",
        "cycles",
        "utilization",
        "edp",
    ]

    for key in required:
        if key not in result:
            raise RuntimeError(
                f"Timeloop result is missing '{key}'."
            )

    edp = float(result["edp"])
    energy = float(result["energy_pJ"])
    cycles = float(result["cycles"])

    if not all(value > 0 for value in [edp, energy, cycles]):
        raise RuntimeError(
            f"Invalid Timeloop metrics: "
            f"energy={energy}, cycles={cycles}, edp={edp}"
        )

    print(f"\nEnergy:       {energy:.6e} pJ")
    print(f"Cycles:       {cycles:.6e}")
    print(f"Utilization:  {float(result['utilization']):.6f}")
    print(f"EDP:          {edp:.6e}")

    print("\n==============================================")
    print("PASS: LPF chromosome successfully evaluated by")
    print("      the real Timeloop runtime.")
    print("==============================================")


if __name__ == "__main__":
    main()