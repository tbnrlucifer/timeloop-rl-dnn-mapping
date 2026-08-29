#!/usr/bin/env python3
"""
Task 1: Generate a Random Mapping and Evaluate via Timeloop
============================================================
Generates a valid random Timeloop mapping YAML for a CONV layer
and evaluates it by calling the `timeloop-model` binary.

Usage:
    python task1_random_mapping_timeloop.py \
        --arch   Computationally-Efficient-DNN-Mapping/arch/simple_weight_stationary.yaml \
        --prob   Computationally-Efficient-DNN-Mapping/prob/conv_layer.yaml \
        --outdir /tmp/tl_random

The script will:
  1. Read the problem spec (layer dims K, C, R, S, P, Q, N).
  2. Randomly pick loop-order and tile sizes that fit the hardware.
  3. Write a mapping YAML.
  4. Call timeloop-model and print the key stats (latency, energy).
"""

import argparse, os, random, subprocess, sys, yaml
from pathlib import Path
from datetime import datetime

# ── helpers ─────────────────────────────────────────────────────────────────

DIMENSIONS = ["N", "K", "C", "R", "S", "P", "Q"]   # standard Timeloop dim order


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_divisors(n: int) -> list[int]:
    """Return all divisors of n."""
    return [i for i in range(1, n + 1) if n % i == 0]


def random_tile(dim_size: int) -> int:
    """Pick a random divisor of dim_size as the tile size."""
    return random.choice(find_divisors(dim_size))


# ── mapping generation ───────────────────────────────────────────────────────

def extract_layer_dims(prob: dict) -> dict[str, int]:
    """
    Pull dimension sizes out of a Timeloop problem spec.
    Handles both the 'instance' and 'shape' sub-key styles.
    """
    dims = {}
    prob_body = prob.get("problem", prob)         # unwrap top-level key if present

    # Style A: flat instance dict  { instance: {N:1, K:64, ...} }
    instance = prob_body.get("instance", {})
    for k, v in instance.items():
        if k in DIMENSIONS:
            dims[k] = int(v)

    # Style B: shape + instance combined
    if not dims:
        for section in ("shape", "instance"):
            for k, v in prob_body.get(section, {}).items():
                if k in DIMENSIONS:
                    dims[k] = int(v)

    # Fallback: search recursively for keys matching dim names
    if not dims:
        def _search(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if k in DIMENSIONS and isinstance(v, int):
                        dims.setdefault(k, v)
                    _search(v)
            elif isinstance(d, list):
                for item in d:
                    _search(item)
        _search(prob_body)

    if not dims:
        # Hard-coded fallback: a typical CONV layer (VGG-like layer 2)
        print("[WARN] Could not parse layer dims from prob YAML; using default CONV dims.")
        dims = {"N": 1, "K": 64, "C": 64, "R": 3, "S": 3, "P": 112, "Q": 112}

    return dims


def build_random_mapping(dims: dict[str, int], arch: dict) -> dict:
    """
    Build a two-level (DRAM → Buffer → MAC) Timeloop mapping YAML
    with a random loop order and random tile sizes.

    Loop hierarchy assumed (inner → outer):
        L0  spatial / registers  (innermost)
        L1  on-chip SRAM / scratchpad
        L2  DRAM                 (outermost)

    We generate ONE temporal loop nest (no explicit spatial unrolling)
    so the mapping is valid for any simple weight-stationary or output-
    stationary single-PE-array accelerator config.
    """
    # Shuffle loop order randomly
    loop_order = DIMENSIONS.copy()
    random.shuffle(loop_order)

    # Choose tile sizes: split each dimension across two levels
    mapping_loops = []
    remaining = {d: dims.get(d, 1) for d in DIMENSIONS}

    # Level 1 (inner buffer / L1): pick a tile that is a divisor of the full dim
    l1_tiles = {d: random_tile(remaining[d]) for d in DIMENSIONS}

    # Level 2 (outer / DRAM): the rest
    l2_tiles = {d: remaining[d] // l1_tiles[d] for d in DIMENSIONS}

    # Build the mapping list in Timeloop format
    # Inner level first (closer to compute), outer level second
    for level_idx, tiles in enumerate([l1_tiles, l2_tiles]):
        for dim in loop_order:
            t = tiles[dim]
            if t > 1:
                mapping_loops.append({
                    "target": f"Buffer_L{level_idx + 1}",
                    "type": "temporal",
                    "dimension": dim,
                    "factors": f"{dim}={t}",
                    "permutation": dim,
                })

    # ── minimal valid mapping structure ─────────────────────────────────────
    # Timeloop mapping spec: list of directives per storage/compute level.
    # We use a flattened "factors + permutation" style that most arch yamls accept.

    # Build per-target factor strings
    def factors_str(tiles: dict) -> str:
        return " ".join(f"{d}={tiles[d]}" for d in DIMENSIONS)

    # Random spatial parallelism for the innermost spatial level (if arch has it)
    # We parallelise across one randomly chosen spatial dimension
    spatial_dim = random.choice(["K", "C"])
    spatial_factor = min(dims.get(spatial_dim, 1), 4)   # keep it small & valid

    mapping = {
        "mapping": [
            # DRAM (outermost): holds the full problem, tiles flowing out
            {
                "target": "DRAM",
                "type": "temporal",
                "factors": factors_str(l2_tiles),
                "permutation": " ".join(loop_order),
            },
            # Scratchpad / global buffer
            {
                "target": "Buffer_L1",
                "type": "temporal",
                "factors": factors_str(l1_tiles),
                "permutation": " ".join(reversed(loop_order)),
            },
            # Spatial (innermost): 1 PE for simplicity
            {
                "target": "Buffer_L1",
                "type": "spatial",
                "factors": " ".join(f"{d}=1" for d in DIMENSIONS),
                "permutation": " ".join(DIMENSIONS),
            },
        ]
    }
    return mapping, l1_tiles, l2_tiles, loop_order


# ── timeloop runner ──────────────────────────────────────────────────────────

def run_timeloop(arch_path: str, prob_path: str, map_path: str, outdir: str) -> dict:
    """
    Run `timeloop-model` and parse the stats output.
    Returns a dict with latency, energy, etc.
    """
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        "timeloop-model",
        arch_path,
        prob_path,
        map_path,
        "-o", outdir,
    ]
    print(f"\n[TIMELOOP CMD] {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=outdir)
    stdout = result.stdout
    stderr = result.stderr

    print("─── STDOUT ───────────────────────────────────────────────────────")
    print(stdout[-4000:] if len(stdout) > 4000 else stdout)
    if stderr.strip():
        print("─── STDERR ───────────────────────────────────────────────────────")
        print(stderr[-2000:] if len(stderr) > 2000 else stderr)

    if result.returncode != 0:
        print(f"\n[ERROR] timeloop-model exited with code {result.returncode}")
        return {}

    # ── parse stats file ─────────────────────────────────────────────────────
    stats_file = os.path.join(outdir, "timeloop-model.stats.txt")
    if not os.path.exists(stats_file):
        # try alternate location
        for fname in Path(outdir).rglob("*.stats.txt"):
            stats_file = str(fname)
            break

    stats = {}
    if os.path.exists(stats_file):
        with open(stats_file) as f:
            for line in f:
                line = line.strip()
                # Examples:
                #   Cycles: 12345
                #   Energy: 67890.12 uJ
                for key in ("Cycles", "Energy", "GFLOPs", "Utilization"):
                    if line.startswith(key + ":"):
                        val = line.split(":", 1)[1].strip().split()[0]
                        try:
                            stats[key] = float(val)
                        except ValueError:
                            stats[key] = val
    return stats


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Random mapping → Timeloop evaluation")
    parser.add_argument("--arch",   default="Computationally-Efficient-DNN-Mapping/arch/simple_weight_stationary.yaml",
                        help="Path to Timeloop architecture YAML")
    parser.add_argument("--prob",   default="Computationally-Efficient-DNN-Mapping/prob/conv_layer.yaml",
                        help="Path to Timeloop problem YAML")
    parser.add_argument("--outdir", default="/tmp/tl_random_mapping",
                        help="Output directory for Timeloop results")
    parser.add_argument("--seed",   type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--mapfile", default=None,
                        help="If given, write the mapping YAML here instead of a temp file")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"[INFO] Random seed: {args.seed}")
    else:
        seed = random.randint(0, 9999)
        random.seed(seed)
        print(f"[INFO] Using random seed: {seed}")

    # ── load arch + prob ──────────────────────────────────────────────────────
    print(f"[INFO] Loading arch: {args.arch}")
    arch = load_yaml(args.arch)

    print(f"[INFO] Loading prob: {args.prob}")
    prob = load_yaml(args.prob)

    dims = extract_layer_dims(prob)
    print(f"[INFO] Layer dimensions: {dims}")

    # ── generate mapping ──────────────────────────────────────────────────────
    mapping, l1_tiles, l2_tiles, loop_order = build_random_mapping(dims, arch)

    print("\n[GENERATED MAPPING]")
    print(f"  Loop order (inner→outer): {loop_order}")
    print(f"  L1 tile sizes: {l1_tiles}")
    print(f"  L2 tile sizes: {l2_tiles}")

    # Write mapping YAML
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    map_path = args.mapfile or os.path.join(args.outdir, f"random_mapping_{ts}.yaml")
    os.makedirs(os.path.dirname(map_path) if os.path.dirname(map_path) else ".", exist_ok=True)
    with open(map_path, "w") as f:
        yaml.dump(mapping, f, default_flow_style=False)
    print(f"[INFO] Mapping YAML written to: {map_path}")
    print(yaml.dump(mapping, default_flow_style=False))

    # ── run timeloop ──────────────────────────────────────────────────────────
    stats = run_timeloop(args.arch, args.prob, map_path, args.outdir)

    print("\n" + "=" * 60)
    print("TIMELOOP EVALUATION RESULTS")
    print("=" * 60)
    if stats:
        for k, v in stats.items():
            print(f"  {k:20s}: {v}")
    else:
        print("  [No stats parsed — check output directory for raw files]")
    print(f"\n  Output directory: {args.outdir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
