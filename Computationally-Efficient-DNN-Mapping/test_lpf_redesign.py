"""Representation/operator tests for the canonical LPF GAMMA experiment."""
from __future__ import annotations

import random
import sys
from pathlib import Path

GAMMA_SRC = Path(__file__).resolve().parent.parent / "gamma-timeloop" / "src"
sys.path.insert(0, str(GAMMA_SRC))

from gamma_lpf_chromosome import validate_chromosome, multiset_fingerprint
import types
import importlib.util

# Load the exact local PPO parser source without requiring the native Timeloop
# runtime. This test is intentionally representation/operator-only.
parser_path = GAMMA_SRC / "gamma_timeloop_env.py"
parser_src = parser_path.read_text()
start = parser_src.index("def calculateMem")
end = parser_src.index("from multiprocessing.pool import Pool")
parser_mod = types.ModuleType("gamma_timeloop_env")
exec(parser_src[start:end], parser_mod.__dict__)
sys.modules["gamma_timeloop_env"] = parser_mod

class _Base: pass
parser_mod.GammaTimeloopEnv = _Base

from gamma_timeloop_env import getmapping
from gamma_lpf_env import GammaLPFEnv

DIMENSION = [1, 32, 16, 120, 120, 120, 3, 3, 3]


def make_seed():
    # Deterministic valid LPF ordering; deliberately starts with a feasible
    # spatial prefix and then uses all remaining factors.
    # loop IDs: R,S,T,P,Q,F,C,K
    seed = []
    for lid, factor in [
        (1, 3), (2, 3), (3, 3),
        (4, 2), (4, 2), (4, 2), (4, 3), (4, 5),
        (5, 2), (5, 2), (5, 2), (5, 3), (5, 5),
        (6, 2), (6, 2), (6, 2), (6, 3), (6, 5),
        (7, 2), (7, 2), (7, 2), (7, 2),
        (8, 2), (8, 2), (8, 2), (8, 2), (8, 2),
    ]:
        seed.append((lid, factor))
    return seed


def main():
    seed = make_seed()
    validate_chromosome(seed, DIMENSION, getmapping_fn=getmapping)
    original = multiset_fingerprint(seed)

    env = object.__new__(GammaLPFEnv)
    env.dimension = DIMENSION
    env.parallizable_dim_note = ['N', 'K', 'C', 'F', 'P', 'Q']

    random.seed(42)
    x = env.mutate_order(seed[:], alpha=1.0)
    assert multiset_fingerprint(x) == original
    validate_chromosome(x, DIMENSION, getmapping_fn=getmapping)

    random.seed(43)
    x = env.mutate_level(x, alpha=1.0)
    assert multiset_fingerprint(x) == original
    validate_chromosome(x, DIMENSION, getmapping_fn=getmapping)

    random.seed(44)
    x = env.mutate_tiles(x, alpha=1.0, num_mu_loc=2)
    assert multiset_fingerprint(x) == original
    validate_chromosome(x, DIMENSION, getmapping_fn=getmapping)

    random.seed(45)
    x = env.mutate_par(x, alpha=1.0)
    assert multiset_fingerprint(x) == original
    validate_chromosome(x, DIMENSION, getmapping_fn=getmapping)

    random.seed(46)
    a = env.crossover([seed[:], x[:]], [seed, x], num_injects=0, alpha=1.0)
    for child in a:
        assert multiset_fingerprint(child) == original
        validate_chromosome(child, DIMENSION, getmapping_fn=getmapping)

    print('PASS: canonical LPF representation and all four GAMMA-style operators preserve validity.')


if __name__ == '__main__':
    main()
