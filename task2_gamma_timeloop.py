#!/usr/bin/env python3
"""
Task 2: GAMMA → Timeloop Evaluation
=====================================
Runs the GAMMA genetic-algorithm mapper to find an optimised mapping
for a given DNN layer / hardware config, then evaluates the best
mapping with Timeloop.

GAMMA originally uses MAESTRO as its cost model.  This script bridges
GAMMA → Timeloop by:
  1. Running GAMMA (via its CLI) to produce a best mapping in MAESTRO
     dataflow notation.
  2. Converting that notation to a Timeloop mapping YAML.
  3. Calling timeloop-model and reporting results.

Directory layout assumed:
    timeloop-rl-dnn-mapping/
    ├── gamma-timeloop/          ← GAMMA repo cloned here
    │   ├── main.py              ← GAMMA entry point
    │   └── ...
    ├── Computationally-Efficient-DNN-Mapping/
    │   ├── arch/
    │   └── prob/
    └── pytimeloop/

Usage (from the project root):
    python task2_gamma_timeloop.py \
        --gamma_dir  gamma-timeloop \
        --arch       Computationally-Efficient-DNN-Mapping/arch/simple_weight_stationary.yaml \
        --prob       Computationally-Efficient-DNN-Mapping/prob/conv_layer.yaml \
        --outdir     /tmp/gamma_tl_results \
        --epochs     10 \
        --num_pe     168 \
        --l1_size    512 \
        --l2_size    108000

If GAMMA is not installed / not in gamma_dir, the script falls back to
a greedy heuristic mapping (channel-first weight-stationary) so you can
still verify Timeloop is working.
"""

import argparse, os, re, subprocess, sys, yaml
from pathlib import Path

# ── GAMMA dimension / MAESTRO notation helpers ───────────────────────────────

DIM_MAP = {
    "K": "K",   # output channels
    "C": "C",   # input channels
    "R": "R",   # filter height
    "S": "S",   # filter width
    "Y": "P",   # output height  (GAMMA uses Y/Y', Timeloop uses P)
    "X": "Q",   # output width
    "N": "N",   # batch
}

TL_DIMS = ["N", "K", "C", "R", "S", "P", "Q"]


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def extract_layer_dims(prob: dict) -> dict[str, int]:
    dims = {}
    prob_body = prob.get("problem", prob)
    instance = prob_body.get("instance", {})
    for k, v in instance.items():
        if k in TL_DIMS:
            dims[k] = int(v)
    if not dims:
        def _search(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if k in TL_DIMS and isinstance(v, int):
                        dims.setdefault(k, v)
                    _search(v)
            elif isinstance(d, list):
                for item in d:
                    _search(item)
        _search(prob_body)
    if not dims:
        print("[WARN] Could not parse dims; using default VGG-like layer.")
        dims = {"N": 1, "K": 64, "C": 64, "R": 3, "S": 3, "P": 112, "Q": 112}
    return dims


# ── GAMMA runner ─────────────────────────────────────────────────────────────

def run_gamma(gamma_dir: str, args) -> str | None:
    """
    Invoke GAMMA's main.py.  Returns the path to the best-mapping output
    file if found, else None.
    """
    main_py = os.path.join(gamma_dir, "main.py")
    if not os.path.exists(main_py):
        print(f"[WARN] GAMMA main.py not found at {main_py}.")
        return None

    outdir = os.path.join(args.outdir, "gamma_out")
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        sys.executable, main_py,
        "--fitness1", args.fitness,
        "--stages",   "1",
        "--num_pe",   str(args.num_pe),
        "--l1_size",  str(args.l1_size),
        "--l2_size",  str(args.l2_size),
        "--NocBW",    str(args.noc_bw),
        "--slevel_min", "1",
        "--slevel_max", "2",
        "--epochs",   str(args.epochs),
        "--model",    args.model,
        "--outdir",   outdir,
    ]
    if args.singlelayer is not None:
        cmd += ["--singlelayer", str(args.singlelayer)]

    print(f"\n[GAMMA CMD] {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=gamma_dir)
    print("─── GAMMA STDOUT ─────────────────────────────────────────────────")
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.stderr.strip():
        print("─── GAMMA STDERR ─────────────────────────────────────────────────")
        print(result.stderr[-1000:])

    # GAMMA writes result files: look for a .m or best_map file
    for pattern in ["*best*.yaml", "*best*.m", "*result*.yaml", "*.yaml"]:
        hits = sorted(Path(outdir).rglob(pattern))
        if hits:
            print(f"[INFO] GAMMA output file: {hits[0]}")
            return str(hits[0])

    # Try the CWD as well
    for pattern in ["best_map.yaml", "best_map.m"]:
        p = os.path.join(gamma_dir, pattern)
        if os.path.exists(p):
            return p

    return None


# ── MAESTRO → Timeloop mapping converter ─────────────────────────────────────

def maestro_to_timeloop_mapping(maestro_file: str, dims: dict) -> dict:
    """
    Parse a MAESTRO dataflow .m file and produce a Timeloop mapping dict.

    MAESTRO example (from GAMMA slides):
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(3,3) R;
        ...
        Cluster(64);
        SpatialMap(1,1) C;
        ...

    We convert each TemporalMap → Timeloop temporal loop factor.
    SpatialMap → Timeloop spatial factor (parallel PEs).
    """
    with open(maestro_file) as f:
        content = f.read()

    # Extract directives
    temporal_pat = re.compile(r'TemporalMap\s*\((\d+),\s*(\d+)\)\s+(\w+)\s*;')
    spatial_pat  = re.compile(r'SpatialMap\s*\((\d+),\s*(\d+)\)\s+(\w+)\s*;')

    # Collect loops per level (split on Cluster(...))
    levels = content.split("Cluster(")
    if len(levels) < 2:
        levels = [content, ""]  # single level

    mapping_directives = []

    for lvl_idx, block in enumerate(levels[:2]):    # at most 2 levels
        target = "DRAM" if lvl_idx == 0 else "Buffer_L1"

        temporal_factors = {}
        for m in temporal_pat.finditer(block):
            tile, stride, dim_name = int(m.group(1)), int(m.group(2)), m.group(3)
            tl_dim = DIM_MAP.get(dim_name, dim_name)
            temporal_factors[tl_dim] = tile

        spatial_factors = {}
        for m in spatial_pat.finditer(block):
            tile, stride, dim_name = int(m.group(1)), int(m.group(2)), m.group(3)
            tl_dim = DIM_MAP.get(dim_name, dim_name)
            spatial_factors[tl_dim] = tile

        # Fill missing dims with 1
        for d in TL_DIMS:
            temporal_factors.setdefault(d, 1)
            spatial_factors.setdefault(d, 1)

        factors_str = lambda fd: " ".join(f"{d}={fd[d]}" for d in TL_DIMS)

        if any(v > 1 for v in temporal_factors.values()):
            mapping_directives.append({
                "target": target,
                "type": "temporal",
                "factors": factors_str(temporal_factors),
                "permutation": " ".join(TL_DIMS),
            })
        if any(v > 1 for v in spatial_factors.values()):
            mapping_directives.append({
                "target": target,
                "type": "spatial",
                "factors": factors_str(spatial_factors),
                "permutation": " ".join(TL_DIMS),
            })

    return {"mapping": mapping_directives}


# ── fallback heuristic mapping ────────────────────────────────────────────────

def weight_stationary_heuristic(dims: dict, num_pe: int = 168) -> dict:
    """
    Simple weight-stationary heuristic as a fallback when GAMMA is unavailable.
    Parallelises across K (output channels) up to num_pe PEs.
    """
    import math

    k_spatial = min(dims.get("K", 1), num_pe)
    # Find largest divisor of K that fits in num_pe
    for k in range(k_spatial, 0, -1):
        if dims.get("K", 1) % k == 0:
            k_spatial = k
            break

    mapping = {
        "mapping": [
            {
                "target": "DRAM",
                "type": "temporal",
                "factors": " ".join(
                    f"{d}={dims.get(d, 1) // (k_spatial if d == 'K' else 1)}"
                    for d in TL_DIMS
                ),
                "permutation": "N K C R S P Q",
            },
            {
                "target": "Buffer_L1",
                "type": "temporal",
                "factors": " ".join(f"{d}=1" for d in TL_DIMS),
                "permutation": "K C R S P Q N",
            },
            {
                "target": "Buffer_L1",
                "type": "spatial",
                "factors": " ".join(
                    f"{d}={k_spatial if d == 'K' else 1}" for d in TL_DIMS
                ),
                "permutation": " ".join(TL_DIMS),
            },
        ]
    }
    return mapping


# ── Timeloop runner ───────────────────────────────────────────────────────────

def run_timeloop(arch_path: str, prob_path: str, map_path: str, outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    cmd = [
        "timeloop-model",
        os.path.abspath(arch_path),
        os.path.abspath(prob_path),
        os.path.abspath(map_path),
        "-o", outdir,
    ]
    print(f"\n[TIMELOOP CMD] {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=outdir)

    print("─── TIMELOOP STDOUT ──────────────────────────────────────────────")
    print(result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout)
    if result.stderr.strip():
        print("─── TIMELOOP STDERR ──────────────────────────────────────────────")
        print(result.stderr[-2000:])

    if result.returncode != 0:
        print(f"\n[ERROR] timeloop-model exited with code {result.returncode}")
        return {}

    stats = {}
    stats_file = os.path.join(outdir, "timeloop-model.stats.txt")
    if not os.path.exists(stats_file):
        for fname in Path(outdir).rglob("*.stats.txt"):
            stats_file = str(fname)
            break

    if os.path.exists(stats_file):
        with open(stats_file) as f:
            for line in f:
                line = line.strip()
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
    parser = argparse.ArgumentParser(description="GAMMA → Timeloop mapping evaluation")
    parser.add_argument("--gamma_dir",   default="gamma-timeloop",
                        help="Path to GAMMA repo directory")
    parser.add_argument("--arch",        default="Computationally-Efficient-DNN-Mapping/arch/simple_weight_stationary.yaml")
    parser.add_argument("--prob",        default="Computationally-Efficient-DNN-Mapping/prob/conv_layer.yaml")
    parser.add_argument("--outdir",      default="/tmp/gamma_timeloop_results")
    parser.add_argument("--fitness",     default="latency",
                        choices=["latency", "energy", "power"],
                        help="GAMMA optimisation objective")
    parser.add_argument("--epochs",      type=int, default=10,
                        help="GAMMA evolution epochs")
    parser.add_argument("--num_pe",      type=int, default=168)
    parser.add_argument("--l1_size",     type=int, default=512,   help="L1 SRAM (Bytes)")
    parser.add_argument("--l2_size",     type=int, default=108000, help="L2 SRAM (Bytes)")
    parser.add_argument("--noc_bw",      type=int, default=81920000, help="NoC BW (Bytes/s)")
    parser.add_argument("--model",       default="vgg16",
                        help="DNN model name for GAMMA (vgg16 | resnet18 | mobilenetv2)")
    parser.add_argument("--singlelayer", type=int, default=1,
                        help="Layer index (1-based) to optimise. Use None for all layers.")
    parser.add_argument("--gamma_map",   default=None,
                        help="Skip GAMMA, use this pre-generated MAESTRO .m file directly")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── load problem ──────────────────────────────────────────────────────────
    print(f"[INFO] Loading prob: {args.prob}")
    prob = load_yaml(args.prob)
    dims = extract_layer_dims(prob)
    print(f"[INFO] Layer dimensions: {dims}")

    # ── Step 1: get GAMMA best mapping ─────────────────────────────────────────
    tl_mapping = None
    mapping_source = "unknown"

    if args.gamma_map and os.path.exists(args.gamma_map):
        # User supplied a pre-generated MAESTRO file
        print(f"\n[INFO] Using supplied GAMMA mapping: {args.gamma_map}")
        tl_mapping = maestro_to_timeloop_mapping(args.gamma_map, dims)
        mapping_source = f"GAMMA (pre-generated): {args.gamma_map}"

    else:
        gamma_out = run_gamma(args.gamma_dir, args)
        if gamma_out:
            print(f"\n[INFO] Converting GAMMA mapping → Timeloop: {gamma_out}")
            if gamma_out.endswith(".m"):
                tl_mapping = maestro_to_timeloop_mapping(gamma_out, dims)
                mapping_source = f"GAMMA (genetic algorithm): {gamma_out}"
            else:
                # Assume it is already a Timeloop-format YAML
                tl_mapping = load_yaml(gamma_out)
                mapping_source = f"GAMMA YAML: {gamma_out}"
        else:
            print("\n[WARN] GAMMA not available / produced no output.")
            print("[INFO] Falling back to weight-stationary heuristic mapping.")
            tl_mapping = weight_stationary_heuristic(dims, args.num_pe)
            mapping_source = "Heuristic weight-stationary fallback"

    # ── Step 2: write Timeloop mapping YAML ──────────────────────────────────
    map_path = os.path.join(args.outdir, "gamma_best_mapping.yaml")
    with open(map_path, "w") as f:
        yaml.dump(tl_mapping, f, default_flow_style=False)
    print(f"\n[INFO] Timeloop mapping written to: {map_path}")
    print(yaml.dump(tl_mapping, default_flow_style=False))

    # ── Step 3: evaluate with Timeloop ────────────────────────────────────────
    tl_outdir = os.path.join(args.outdir, "timeloop_eval")
    stats = run_timeloop(args.arch, args.prob, map_path, tl_outdir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GAMMA → TIMELOOP EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Mapping source : {mapping_source}")
    print(f"  Fitness target : {args.fitness}")
    print(f"  GA epochs      : {args.epochs}")
    print(f"  HW (PEs/L1/L2) : {args.num_pe} / {args.l1_size}B / {args.l2_size}B")
    print()
    if stats:
        for k, v in stats.items():
            print(f"  {k:20s}: {v}")
    else:
        print("  [No stats parsed — check Timeloop output directory]")
    print(f"\n  Output directory: {args.outdir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
