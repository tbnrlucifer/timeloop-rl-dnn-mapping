#!/usr/bin/env python3
"""
Task 3: RL Mapping → Timeloop Evaluation
=========================================
Runs the RL training (senior's fixed approach), picks the best mapping
from the saved pkl, writes a Timeloop-compatible mapping YAML, and
evaluates it with timeloop-model.

Usage (from Computationally-Efficient-DNN-Mapping/):
    python task3_rl_mapping_timeloop.py \
        --arch arch/arch.yaml \
        --prob prob/prob_3d_12.yaml \
        --outdir /tmp/tl_rl
"""

import argparse, os, subprocess, sys, yaml, pickle, importlib.util
from pathlib import Path

# ── Load rl_script_fixed without executing its __main__ ──────────────────────
def load_rl_module(path="rl_script_fixed.py"):
    spec = importlib.util.spec_from_file_location("rl", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── Convert format_mapping_dict output → Timeloop YAML dict ──────────────────
def rl_mapping_to_timeloop_yaml(fmt: dict) -> dict:
    """
    fmt is the dict returned by format_mapping_dict(), e.g.:
      {'mapping': [
        {'target': 'GlobalBuffer', 'type': 'spatial',   'factors': 'R=3 S=3', 'permutation': 'RS'},
        {'target': 'RegisterFile', 'type': 'temporal',  'factors': 'P=15 ..', 'permutation': '..'},
        {'target': 'GlobalBuffer', 'type': 'temporal',  'factors': '..',       'permutation': '..'},
        {'target': 'MainMemory',   'type': 'temporal',  'factors': '...',      'permutation': '...'},
      ]}

    Timeloop expects each entry to have:
      target, type, factors, permutation
    — which is exactly what we already have. Just need to ensure
    factors includes ALL 9 dims (missing ones default to =1).
    """
    ALL_DIMS = ['R', 'S', 'T', 'P', 'Q', 'F', 'C', 'K', 'N']
    tl_mapping = []

    for entry in fmt['mapping']:
        # Parse existing factors into dict
        factor_dict = {}
        for token in entry['factors'].split():
            if '=' in token:
                d, v = token.split('=')
                factor_dict[d] = int(v)

        # Fill missing dims with 1
        for d in ALL_DIMS:
            factor_dict.setdefault(d, 1)

        # Rebuild factors string (all 9 dims, sorted canonically)
        factors_str = ' '.join(f"{d}={factor_dict[d]}" for d in ALL_DIMS)

        # Permutation: append any dims not already listed
        existing_perm = list(entry['permutation'])
        for d in ALL_DIMS:
            if d not in existing_perm:
                existing_perm.append(d)
        permutation_str = ''.join(existing_perm)

        tl_mapping.append({
            'target':      entry['target'],
            'type':        entry['type'],
            'factors':     factors_str,
            'permutation': permutation_str,
        })

    return {'mapping': tl_mapping}


# ── Timeloop binary + lib paths ──────────────────────────────────────────────
TIMELOOP_BIN = os.path.expanduser("~/timeloop/build/timeloop-model")
TIMELOOP_LIB = os.path.expanduser("~/timeloop/lib")

# ── Run timeloop-model ────────────────────────────────────────────────────────
def run_timeloop(arch, prob, mapfile, outdir):
    os.makedirs(outdir, exist_ok=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = TIMELOOP_LIB + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [TIMELOOP_BIN, arch, prob, mapfile]
    print(f"\n[CMD] {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=outdir, env=env)

    print("─── STDOUT ───────────────────────────────────────────────────")
    print(result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout)
    if result.stderr.strip():
        print("─── STDERR ───────────────────────────────────────────────────")
        print(result.stderr[-2000:])

    if result.returncode != 0:
        print(f"[ERROR] timeloop-model exited with code {result.returncode}")
        return {}

    # Parse stats — covers stdout summary line + stats file
    stats = {}

    # Fast parse: Timeloop prints summary to stdout
    # e.g. "Utilization = 0.23 | pJ/Algorithmic-Compute = 8.485 | pJ/Compute = 8.485"
    for line in result.stdout.splitlines():
        if "Utilization" in line and "pJ" in line:
            for token in line.split("|"):
                token = token.strip()
                if "=" in token:
                    k, v = token.split("=", 1)
                    k = k.strip().replace(" ", "_")
                    try:    stats[k] = float(v.strip())
                    except: pass

    # Deep parse: stats file for Cycles, Energy, etc.
    STAT_KEYS = {
        "Cycles":                   "Cycles",
        "Energy":                   "Energy_uJ",
        "GFLOPs":                   "GFLOPs",
        "pJ/Compute":               "pJ_per_Compute",
        "pJ/Algorithmic-Compute":   "pJ_per_Algo_Compute",
        "Utilization":              "Utilization",
    }
    for fname in Path(outdir).rglob("*.stats.txt"):
        with open(fname) as f:
            for line in f:
                line = line.strip()
                for raw_key, stat_name in STAT_KEYS.items():
                    if line.startswith(raw_key):
                        val = line.split(":", 1)[-1].strip().split()[0]
                        try:    stats[stat_name] = float(val)
                        except: stats[stat_name] = val
        break
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",      default="arch/arch.yaml")
    parser.add_argument("--prob",      default="prob/prob_3d_12.yaml")
    parser.add_argument("--outdir",    default="/tmp/tl_rl_mapping")
    parser.add_argument("--episodes",  type=int, default=20)
    parser.add_argument("--rl_script", default="rl_script_fixed.py")
    parser.add_argument("--pkl",       default=None,
                        help="Skip training; load existing pkl instead")
    args = parser.parse_args()

    # ── 1. Get mappings ───────────────────────────────────────────────────────
    if args.pkl and os.path.exists(args.pkl):
        print(f"[INFO] Loading existing pkl: {args.pkl}")
        with open(args.pkl, "rb") as f:
            mappings = pickle.load(f)
    else:
        print("[INFO] Running RL training...")
        rl = load_rl_module(args.rl_script)
        input_list = [3, 3, 3, 120, 120, 120, 16, 32]   # matches prob_3d_12
        acnet, mappings = rl.train_network(input_list, episodes=args.episodes)
        with open("output_mappings.pkl", "wb") as f:
            pickle.dump(mappings, f)
        print(f"[INFO] Saved {len(mappings)} mappings to output_mappings.pkl")

    # ── 2. Pick best mapping ──────────────────────────────────────────────────
    best = max(mappings, key=lambda e: e[0])
    best_reward, best_fmt, best_tuples, best_memmap = best[0], best[1], best[3], best[4]
    print(f"\n[INFO] Best reward: {best_reward:.4f}")

    # The pkl already stores the formatted mapping and the raw memmap.
    # Use those directly instead of attempting to reconstruct the state.
    tl_yaml  = rl_mapping_to_timeloop_yaml(best_fmt)

    print("\n[RL MAPPING (internal)]")
    import pprint; pprint.pprint(best_fmt)
    print("\n[TIMELOOP MAPPING YAML]")
    print(yaml.dump(tl_yaml, default_flow_style=False))

    # ── 3. Write mapping YAML ─────────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)
    map_path = os.path.join(args.outdir, "rl_best_mapping.yaml")
    with open(map_path, "w") as f:
        yaml.dump(tl_yaml, f, default_flow_style=False)
    print(f"[INFO] Mapping YAML written to: {map_path}")

    # ── 4. Run Timeloop ───────────────────────────────────────────────────────
    stats = run_timeloop(args.arch, args.prob, map_path, args.outdir)

    print("\n" + "=" * 60)
    print("TIMELOOP RESULTS")
    print("=" * 60)
    if stats:
        for k, v in stats.items():
            print(f"  {k:20s}: {v}")
    else:
        print("  [No stats — check raw files in outdir]")
    print(f"\n  Output dir: {args.outdir}")
    print("=" * 60)


if __name__ == "__main__":
    main()