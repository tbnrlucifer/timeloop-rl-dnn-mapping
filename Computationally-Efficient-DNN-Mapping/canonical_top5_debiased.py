#!/usr/bin/env python3
"""Top-5 canonical debiased reevaluation, Energy/Cycles decomposition,
seed robustness, and iso-budget (Iter<=1) vs full (Iter<=3) comparison.

Read-only on existing files; writes outputs to results/common_eval/canonical_v2/.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "gamma-timeloop").exists():
    ROOT = Path.home() / "timeloop-rl-dnn-mapping"

IN_CONFIG_DIR = str(ROOT / "gamma-timeloop" / "src" / "in_config_3d")
REPORT_BASE_DIR = ROOT / "gamma-timeloop" / "src" / "report_canonical_eval_v1"
V1_JSON = ROOT / "results" / "common_eval" / "canonical_v1" / "canonical_75_winners.json"
OUT_DIR = ROOT / "results" / "common_eval" / "canonical_v2"

WORKLOADS = {
    "small":         [1, 16, 8, 32, 32, 32, 3, 3, 3],
    "baseline":      [1, 32, 16, 120, 120, 120, 3, 3, 3],
    "spatial-heavy": [1, 32, 16, 224, 224, 224, 3, 3, 3],
    "channel-heavy": [1, 128, 64, 56, 56, 56, 3, 3, 3],
    "pointwise":     [1, 64, 32, 112, 112, 112, 1, 1, 1],
}
WORKLOAD_ORDER = ["small", "baseline", "spatial-heavy", "channel-heavy", "pointwise"]
SEEDS = [1, 2, 3, 4, 42]
TOP_K = 5

EXCLUDED_FILENAMES = {
    "rl_only_pointwise_seed1_20261021_111634.json",
    "rl_ga_lpf_pointwise_seed1_full_run_20261021_103707.json",
    "gamma_lpf_only_pointwise_seed1_20261021_110503.json",
}


def parse_tuples(val: Any) -> list[list[int]]:
    if isinstance(val, str):
        val = json.loads(val)
    return [[int(x[0]), int(x[1])] for x in val]


def safe_float(val: Any) -> float | None:
    try:
        x = float(val)
        return x if math.isfinite(x) and x > 0 else None
    except (TypeError, ValueError):
        return None


def tuple_sig(tuples: list[list[int]]) -> str:
    return json.dumps(tuples, separators=(",", ":"))


def geomean(vals: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def load_v1_cache() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    """Pre-seed evaluator cache from canonical_v1/canonical_75_winners.json."""
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    v1_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    if not V1_JSON.exists():
        return cache, v1_by_key
    data = json.loads(V1_JSON.read_text())
    for r in data:
        wl = r["workload"]
        sig = tuple_sig(parse_tuples(r["tuples"]))
        if safe_float(r.get("canonical_edp")) is not None:
            cache[(wl, sig)] = {
                "edp": float(r["canonical_edp"]),
                "energy_pJ": float(r["canonical_energy_pJ"]),
                "cycles": float(r["canonical_cycles"]),
                "utilization": float(r["canonical_utilization"]),
            }
            v1_by_key[(r["arm"], wl, int(r["seed"]))] = r
    return cache, v1_by_key


def make_cached_evaluator(cache: dict[tuple[str, str], dict[str, Any]]):
    sys.path.insert(0, str(ROOT / "gamma-timeloop" / "src"))
    sys.path.insert(0, str(ROOT / "Computationally-Efficient-DNN-Mapping"))
    from rl_script_fixed import getmapping, format_mapping_dict
    from timeloop_env import TimeloopEnv

    envs: dict[str, Any] = {}
    stats = {"hits": 0, "misses": 0, "failures": 0}

    def get_env(wl: str):
        if wl not in envs:
            rdir = REPORT_BASE_DIR / wl
            rdir.mkdir(parents=True, exist_ok=True)
            env = TimeloopEnv(
                config_path=str(rdir),
                in_config_dir=IN_CONFIG_DIR,
                debug=False,
                use_sparse=False,
                use_IO=True,
            )
            env.create_pool_env(num_pools=1, dimension=WORKLOADS[wl], indv=None, use_IO=True)
            envs[wl] = env
        return envs[wl]

    def evaluate(wl: str, tuples: list[list[int]], mapping_dict: dict | None = None) -> dict[str, Any] | None:
        sig = tuple_sig(tuples)
        if (wl, sig) in cache:
            stats["hits"] += 1
            return cache[(wl, sig)]
        stats["misses"] += 1
        env = get_env(wl)
        if mapping_dict is None:
            memmap = getmapping([tuple(x) for x in tuples])
            mapping_dict = format_mapping_dict(memmap)
        res = env.run_mapping_dict(
            dimension=WORKLOADS[wl],
            mapping=mapping_dict,
            pool_idx=0,
            fitness_obj=["edp"],
        )
        if not res or safe_float(res.get("edp")) is None:
            stats["failures"] += 1
            return None
        entry = {
            "edp": float(res["edp"]),
            "energy_pJ": float(res["energy_pJ"]),
            "cycles": float(res["cycles"]),
            "utilization": float(res["utilization"]),
        }
        cache[(wl, sig)] = entry
        return entry

    return evaluate, stats


def load_hybrid_iter_winners(evaluate) -> dict[tuple[int, str, int], dict[str, Any]]:
    """Extract Hybrid cumulative winners at iter<=1, iter<=2, iter<=3 with canonical energy/cycles."""
    out: dict[tuple[int, str, int], dict[str, Any]] = {}
    folder = ROOT / "results/experiments/hybrid"
    for path in sorted(folder.glob("*.json")):
        if path.name in EXCLUDED_FILENAMES:
            continue
        obj = json.loads(path.read_text())
        exp = obj["experiment"]
        wl = exp["workload_name"]
        seed = int(exp.get("experiment_seed", exp.get("seed")))
        best_so_far: dict[str, Any] | None = None
        for idx, it in enumerate(obj.get("iterations") or [], 1):
            sel = (it.get("gamma") or {}).get("selected_for_ppo_feedback") or []
            for c in sel:
                edp = safe_float(c.get("edp"))
                if edp is None:
                    continue
                if best_so_far is None or edp < best_so_far["historical_edp"]:
                    best_so_far = {
                        "historical_edp": edp,
                        "tuples": parse_tuples(c["tuples"]),
                        "source_file": str(path.relative_to(ROOT)),
                    }
            if best_so_far is not None:
                res = evaluate(wl, best_so_far["tuples"], None)
                out[(idx, wl, seed)] = {
                    "arm": f"hybrid_iter{idx}",
                    "workload": wl,
                    "seed": seed,
                    "historical_edp": best_so_far["historical_edp"],
                    "canonical_edp": res["edp"] if res else best_so_far["historical_edp"],
                    "canonical_energy_pJ": res["energy_pJ"] if res else None,
                    "canonical_cycles": res["cycles"] if res else None,
                    "canonical_utilization": res["utilization"] if res else None,
                    "chosen_hist_rank": 1,
                    "hist1_canonical_edp": res["edp"] if res else best_so_far["historical_edp"],
                    "rank_flip_gain_pct": 0.0,
                    "source_file": best_so_far["source_file"],
                }
    return out


def process_baseline_arm(arm: str, folder: Path, list_key: str, evaluate) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    winners = []
    all_top5_rows = []
    files = sorted(p for p in folder.glob("*.json") if p.name not in EXCLUDED_FILENAMES)
    for f_idx, path in enumerate(files, 1):
        obj = json.loads(path.read_text())
        exp = obj["experiment"]
        wl = exp["workload_name"]
        seed = int(exp["seed"])

        seen = set()
        uniq = []
        for c in obj.get(list_key) or []:
            edp = safe_float(c.get("edp"))
            if edp is None:
                continue
            t = parse_tuples(c["tuples"])
            sig = tuple_sig(t)
            if sig in seen:
                continue
            seen.add(sig)
            uniq.append({"edp": edp, "tuples": t, "mapping": c.get("mapping")})
        uniq.sort(key=lambda x: x["edp"])
        top_cands = uniq[:TOP_K]

        evaluated_cands = []
        for rank, c in enumerate(top_cands, 1):
            res = evaluate(wl, c["tuples"], c["mapping"])
            if not res:
                continue
            row = {
                "arm": arm,
                "workload": wl,
                "seed": seed,
                "hist_rank": rank,
                "historical_edp": c["edp"],
                "canonical_edp": res["edp"],
                "canonical_energy_pJ": res["energy_pJ"],
                "canonical_cycles": res["cycles"],
                "canonical_utilization": res["utilization"],
                "scale_ratio": res["edp"] / c["edp"],
                "source_file": str(path.relative_to(ROOT)),
            }
            all_top5_rows.append(row)
            evaluated_cands.append(row)

        hist1 = evaluated_cands[0]
        best = min(evaluated_cands, key=lambda r: r["canonical_edp"])
        gain_pct = (hist1["canonical_edp"] - best["canonical_edp"]) / hist1["canonical_edp"] * 100.0
        winners.append({
            "arm": arm,
            "workload": wl,
            "seed": seed,
            "historical_edp": best["historical_edp"],
            "canonical_edp": best["canonical_edp"],
            "canonical_energy_pJ": best["canonical_energy_pJ"],
            "canonical_cycles": best["canonical_cycles"],
            "canonical_utilization": best["canonical_utilization"],
            "chosen_hist_rank": best["hist_rank"],
            "hist1_canonical_edp": hist1["canonical_edp"],
            "rank_flip_gain_pct": gain_pct,
            "source_file": best["source_file"],
        })
        flag = f" [FLIP: hist#{best['hist_rank']} saved {gain_pct:.2f}%]" if best["hist_rank"] != 1 else ""
        print(
            f"  [{f_idx:02d}/{len(files):02d}] {arm:10s} {wl:13s} seed={seed:<2d} "
            f"hist#1={hist1['canonical_edp']:.4e} -> best_top5={best['canonical_edp']:.4e}{flag}",
            flush=True,
        )
    return winners, all_top5_rows


def paired_table(by_key: dict[tuple[str, str, int], dict[str, Any]], arm_a: str, arm_b: str, title: str) -> str:
    lines = [f"## {title}", ""]
    ratios, e_ratios, c_ratios = [], [], []
    a_vals, b_vals = [], []
    per_wl = defaultdict(list)
    wins = ties = losses = 0

    for wl in WORKLOAD_ORDER:
        for seed in SEEDS:
            ra = by_key.get((arm_a, wl, seed))
            rb = by_key.get((arm_b, wl, seed))
            if not ra or not rb:
                continue
            ea, eb = ra["canonical_edp"], rb["canonical_edp"]
            r = ea / eb
            ratios.append(r); a_vals.append(ea); b_vals.append(eb)
            er = ra["canonical_energy_pJ"] / rb["canonical_energy_pJ"]
            cr = ra["canonical_cycles"] / rb["canonical_cycles"]
            e_ratios.append(er); c_ratios.append(cr)
            per_wl[wl].append((ea, eb, r, er, cr))
            if abs(r - 1.0) <= 1e-6:
                ties += 1
            elif r < 1.0:
                wins += 1
            else:
                losses += 1

    gm = geomean(ratios)
    egm = geomean(e_ratios)
    cgm = geomean(c_ratios)
    stat, pval = wilcoxon(a_vals, b_vals)
    lines.append(f"- **Pairs:** {len(ratios)}/25 | **{arm_a} W/T/L:** `{wins} / {ties} / {losses}`")
    lines.append(
        f"- **Overall Geomean EDP Ratio:** `{gm:.4f}` (`{(1-gm)*100:+.2f}%` reduction) "
        f"| **Energy Ratio:** `{egm:.4f}` (`{(1-egm)*100:+.2f}%`) "
        f"| **Cycles Ratio:** `{cgm:.4f}` (`{(1-cgm)*100:+.2f}%`)"
    )
    lines.append(f"- **Two-sided Wilcoxon signed-rank:** `W={stat:.1f}`, `p={pval:.6g}`")
    lines.append("")
    lines.append(f"| Workload | {arm_a} Geomean EDP | {arm_b} Geomean EDP | EDP Ratio | EDP Red. | Energy Red. | Cycles Red. | Wins |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for wl in WORKLOAD_ORDER:
        items = per_wl[wl]
        ga = geomean([x[0] for x in items])
        gb = geomean([x[1] for x in items])
        gr = geomean([x[2] for x in items])
        ger = geomean([x[3] for x in items])
        gcr = geomean([x[4] for x in items])
        w = sum(1 for x in items if x[2] < 1.0 - 1e-6)
        t = sum(1 for x in items if abs(x[2] - 1.0) <= 1e-6)
        lines.append(
            f"| {wl} | {ga:.4e} | {gb:.4e} | {gr:.4f} | {(1-gr)*100:+.2f}% | "
            f"{(1-ger)*100:+.2f}% | {(1-gcr)*100:+.2f}% | {w}/{len(items)} ({t}t) |"
        )
    lines.append("")
    return "\n".join(lines)


def robustness_table(by_key: dict[tuple[str, str, int], dict[str, Any]]) -> str:
    lines = ["## Seed Robustness Across Workloads (Canonical Accelergy 0.4 EDP)", ""]
    lines.append("| Workload | Arm | Mean EDP | Std EDP | CV (%) | Min EDP | Max EDP |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for wl in WORKLOAD_ORDER:
        for arm, label in [
            ("hybrid_iter3", "Hybrid (Iter<=3, 400-420)"),
            ("hybrid_iter1", "Hybrid (Iter<=1, 120)"),
            ("rl_only", "RL-only (120, top-5 debiased)"),
            ("gamma_only", "GAMMA-only (120, top-5 debiased)"),
        ]:
            vals = [by_key[(arm, wl, s)]["canonical_edp"] for s in SEEDS if (arm, wl, s) in by_key]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
            std = math.sqrt(var)
            cv = (std / mean * 100.0) if mean > 0 else 0.0
            lines.append(f"| {wl} | {label} | {mean:.4e} | {std:.2e} | {cv:.2f}% | {min(vals):.4e} | {max(vals):.4e} |")
    lines.append("")
    return "\n".join(lines)


def main():
    t0 = time.perf_counter()
    cache, _ = load_v1_cache()
    print(f"Loaded {len(cache)} cached canonical evaluations from canonical_v1.", flush=True)
    evaluate, stats = make_cached_evaluator(cache)

    print("\n=== 1. EVALUATING HYBRID ITER<=1,2,3 (MOSTLY CACHED) ===", flush=True)
    hyb_iters = load_hybrid_iter_winners(evaluate)

    print("\n=== 2. TOP-5 DEBIASED REEVALUATION: RL-ONLY (25 RUNS) ===", flush=True)
    rl_winners, rl_top5 = process_baseline_arm(
        "rl_only", ROOT / "results/experiments/rl_only", "evaluated_candidates", evaluate
    )

    print("\n=== 3. TOP-5 DEBIASED REEVALUATION: GAMMA-ONLY (25 RUNS) ===", flush=True)
    ga_winners, ga_top5 = process_baseline_arm(
        "gamma_only", ROOT / "results/experiments/gamma_only", "final_population", evaluate
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_winners = list(hyb_iters.values()) + rl_winners + ga_winners
    by_key = {(r["arm"], r["workload"], r["seed"]): r for r in all_winners}

    # Save CSVs
    with (OUT_DIR / "debiased_winners_v2.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_winners[0].keys()))
        w.writeheader()
        w.writerows(all_winners)

    all_top5 = rl_top5 + ga_top5
    with (OUT_DIR / "baseline_top5_audit_v2.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_top5[0].keys()))
        w.writeheader()
        w.writerows(all_top5)

    # Selection-bias summary
    md = [
        "# Canonical Top-5 Debiased Comparison Report (Accelergy 0.4)",
        "",
        f"- Runtime: `{time.perf_counter() - t0:.1f}s` | Cache hits: `{stats['hits']}` | New Timeloop evals: `{stats['misses']}` | Failures: `{stats['failures']}`",
        "",
        "## Selection-Bias Audit (Accelergy 0.3 -> 0.4 Top-5 Rank Flips)",
        "",
        "| Arm | Runs with Rank-1 Flip | Max Flip Gain (%) | Median Flip Gain on Flipped Runs (%) |",
        "|---|---:|---:|---:|",
    ]
    for arm, wlist in [("rl_only", rl_winners), ("gamma_only", ga_winners)]:
        flips = [r["rank_flip_gain_pct"] for r in wlist if r["chosen_hist_rank"] != 1]
        med_f = sorted(flips)[len(flips) // 2] if flips else 0.0
        max_f = max(flips) if flips else 0.0
        md.append(f"| {arm} | {len(flips)}/25 | {max_f:.3f}% | {med_f:.3f}% |")
    md.append("")

    md.append(paired_table(by_key, "hybrid_iter1", "rl_only", "Iso-Budget (120 Evals): Hybrid (Iter<=1) vs RL-only (120, Top-5 Debiased)"))
    md.append(paired_table(by_key, "hybrid_iter1", "gamma_only", "Iso-Budget (120 Evals): Hybrid (Iter<=1) vs GAMMA-only (120, Top-5 Debiased)"))
    md.append(paired_table(by_key, "hybrid_iter3", "rl_only", "Full Hybrid (Iter<=3, 400-420 Evals) vs RL-only (120, Top-5 Debiased)"))
    md.append(paired_table(by_key, "hybrid_iter3", "gamma_only", "Full Hybrid (Iter<=3, 400-420 Evals) vs GAMMA-only (120, Top-5 Debiased)"))
    md.append(paired_table(by_key, "rl_only", "gamma_only", "Baselines Head-to-Head (120 Evals): RL-only vs GAMMA-only (Top-5 Debiased)"))
    md.append(robustness_table(by_key))

    report = "\n".join(md)
    (OUT_DIR / "debiased_comparison_report_v2.md").write_text(report)
    print("\n" + report, flush=True)
    print(f"Saved v2 artifacts under: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
