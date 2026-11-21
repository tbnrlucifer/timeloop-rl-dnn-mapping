#!/usr/bin/env python3
"""Canonical Timeloop/Accelergy reevaluation and paired comparison across
RL-only, GAMMA-only, and RL+GAMMA Hybrid.

- Read-only on all existing experiment files.
- Writes outputs to results/common_eval/canonical_v1/.
- Reevaluates each workload in an isolated TimeloopEnv directory to prevent
  any cross-workload state leakage.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "gamma-timeloop").exists():
    ROOT = Path.home() / "timeloop-rl-dnn-mapping"

IN_CONFIG_DIR = str(ROOT / "gamma-timeloop" / "src" / "in_config_3d")
REPORT_BASE_DIR = ROOT / "gamma-timeloop" / "src" / "report_canonical_eval_v1"
OUT_DIR = ROOT / "results" / "common_eval" / "canonical_v1"

WORKLOADS = {
    "small":         [1, 16, 8, 32, 32, 32, 3, 3, 3],
    "baseline":      [1, 32, 16, 120, 120, 120, 3, 3, 3],
    "spatial-heavy": [1, 32, 16, 224, 224, 224, 3, 3, 3],
    "channel-heavy": [1, 128, 64, 56, 56, 56, 3, 3, 3],
    "pointwise":     [1, 64, 32, 112, 112, 112, 1, 1, 1],
}
WORKLOAD_ORDER = ["small", "baseline", "spatial-heavy", "channel-heavy", "pointwise"]
SEEDS = [1, 2, 3, 4, 42]

# Excluded historical Oct-21 duplicate pointwise seed-1 runs across all three arms
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


def load_gamma_eval_counts() -> dict[tuple[str, int], int]:
    """Read logged Timeloop evaluation counts from gamma_only_summary.csv if present."""
    csv_path = ROOT / "results/experiments/gamma_only/gamma_only_summary.csv"
    counts: dict[tuple[str, int], int] = {}
    if not csv_path.exists():
        return counts
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            norm = {k.strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}
            wl = norm.get("workload")
            seed_raw = norm.get("seed")
            evals_raw = norm.get("timeloop_evals") or norm.get("timeloop evals") or norm.get("evals")
            if wl in WORKLOADS and seed_raw and evals_raw:
                try:
                    counts[(wl, int(seed_raw))] = int(evals_raw)
                except ValueError:
                    pass
    return counts


def load_hybrid_winners() -> list[dict[str, Any]]:
    v4_path = ROOT / "results/common_eval/prepared_v4/hybrid_winners_v4.json"
    data = json.loads(v4_path.read_text())
    out = []
    for rec in data["records"]:
        wl = rec["workload"]
        seed = int(rec["seed"])
        out.append({
            "arm": "hybrid",
            "workload": wl,
            "seed": seed,
            "historical_edp": float(rec["original_edp"]),
            "tuples": parse_tuples(rec["mapping_tuples"]),
            "mapping_dict": None,
            "eval_count": None,
            "source_file": rec["source_file"],
        })
    return out


def load_rl_only_winners() -> list[dict[str, Any]]:
    folder = ROOT / "results/experiments/rl_only"
    chosen: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(folder.glob("*.json")):
        if path.name in EXCLUDED_FILENAMES:
            continue
        obj = json.loads(path.read_text())
        exp = obj.get("experiment") or {}
        final = obj.get("final") or {}
        wl = exp["workload_name"]
        seed = int(exp["seed"])
        succ = exp.get("timeloop_evaluations_succeeded")
        fail = exp.get("timeloop_evaluations_failed", 0)
        eval_count = int(succ + fail) if isinstance(succ, int) else None
        chosen[(wl, seed)] = {
            "arm": "rl_only",
            "workload": wl,
            "seed": seed,
            "historical_edp": float(final["best_edp"]),
            "tuples": parse_tuples(final["best_tuples"]),
            "mapping_dict": final.get("best_mapping"),
            "eval_count": eval_count,
            "source_file": str(path.relative_to(ROOT)),
        }
    return list(chosen.values())


def load_gamma_only_winners() -> list[dict[str, Any]]:
    folder = ROOT / "results/experiments/gamma_only"
    gamma_counts = load_gamma_eval_counts()
    chosen: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(folder.glob("*.json")):
        if path.name in EXCLUDED_FILENAMES:
            continue
        obj = json.loads(path.read_text())
        exp = obj.get("experiment") or {}
        final = obj.get("final") or {}
        wl = exp["workload_name"]
        seed = int(exp["seed"])
        chosen[(wl, seed)] = {
            "arm": "gamma_only",
            "workload": wl,
            "seed": seed,
            "historical_edp": float(final["best_edp"]),
            "tuples": parse_tuples(final["best_mapping"]),
            "mapping_dict": None,
            "eval_count": gamma_counts.get((wl, seed), 120),
            "source_file": str(path.relative_to(ROOT)),
        }
    return list(chosen.values())


def make_env_pool():
    sys.path.insert(0, str(ROOT / "gamma-timeloop" / "src"))
    sys.path.insert(0, str(ROOT / "Computationally-Efficient-DNN-Mapping"))
    from rl_script_fixed import getmapping, format_mapping_dict
    from timeloop_env import TimeloopEnv

    envs: dict[str, Any] = {}

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
            env.create_pool_env(
                num_pools=1,
                dimension=WORKLOADS[wl],
                indv=None,
                use_IO=True,
            )
            envs[wl] = env
        return envs[wl]

    def evaluate(wl: str, tuples: list[list[int]], mapping_dict: dict | None = None) -> dict[str, Any] | None:
        env = get_env(wl)
        if mapping_dict is None:
            memmap = getmapping([tuple(x) for x in tuples])
            mapping_dict = format_mapping_dict(memmap)
        return env.run_mapping_dict(
            dimension=WORKLOADS[wl],
            mapping=mapping_dict,
            pool_idx=0,
            fitness_obj=["edp"],
        )

    return evaluate, getmapping, format_mapping_dict


def average_ranks(vals: list[float]) -> list[float]:
    """Tie-aware average ranks for Spearman correlation."""
    indexed = sorted(enumerate(vals), key=lambda kv: kv[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_r = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_r
        i = j + 1
    return ranks


def spearman_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    rx, ry = average_ranks(xs), average_ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = sum((a - mx) ** 2 for a in rx)
    den_y = sum((b - my) ** 2 for b in ry)
    if den_x <= 0 or den_y <= 0:
        return None
    return num / math.sqrt(den_x * den_y)


def geomean(vals: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def build_pairwise_report(rows: list[dict[str, Any]], arm_a: str, arm_b: str) -> str:
    by_key = {(r["arm"], r["workload"], r["seed"]): r for r in rows if r.get("canonical_edp")}
    lines = [f"## Paired Comparison: `{arm_a}` vs `{arm_b}` (Canonical Accelergy 0.4 EDP)", ""]
    ratios_all = []
    a_all, b_all = [], []
    wins = ties = losses = 0
    per_wl: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

    for wl in WORKLOAD_ORDER:
        for seed in SEEDS:
            ra = by_key.get((arm_a, wl, seed))
            rb = by_key.get((arm_b, wl, seed))
            if not ra or not rb:
                continue
            ea, eb = ra["canonical_edp"], rb["canonical_edp"]
            ratio = ea / eb
            ratios_all.append(ratio)
            a_all.append(ea)
            b_all.append(eb)
            per_wl[wl].append((ea, eb, ratio))
            if abs(ratio - 1.0) <= 1e-6:
                ties += 1
            elif ratio < 1.0:
                wins += 1
            else:
                losses += 1

    if not ratios_all:
        lines.append("No overlapping evaluated pairs found.\n")
        return "\n".join(lines)

    gm = geomean(ratios_all)
    med = sorted(ratios_all)[len(ratios_all) // 2]
    wilcoxon_str = "n/a"
    try:
        from scipy.stats import wilcoxon
        stat, pval = wilcoxon(a_all, b_all)
        wilcoxon_str = f"W={stat:.2f}, p={pval:.6g}"
    except Exception as e:
        wilcoxon_str = f"skipped ({e})"

    lines.append(f"- **Pairs compared:** {len(ratios_all)}/25")
    lines.append(f"- **{arm_a} wins / ties / losses:** {wins} / {ties} / {losses}")
    lines.append(f"- **Overall geomean EDP ratio ({arm_a} / {arm_b}):** `{gm:.4f}` (`{(1.0 - gm) * 100:+.2f}%` EDP reduction; >0% means `{arm_a}` is better)")
    lines.append(f"- **Overall median EDP ratio ({arm_a} / {arm_b}):** `{med:.4f}`")
    lines.append(f"- **Two-sided Wilcoxon signed-rank test:** {wilcoxon_str}")
    lines.append("")
    lines.append(f"| Workload | {arm_a} Geomean EDP | {arm_b} Geomean EDP | Geomean Ratio ({arm_a}/{arm_b}) | EDP Reduction | {arm_a} Wins |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for wl in WORKLOAD_ORDER:
        items = per_wl.get(wl, [])
        if not items:
            continue
        ga = geomean([x[0] for x in items])
        gb = geomean([x[1] for x in items])
        gr = geomean([x[2] for x in items])
        w = sum(1 for x in items if x[2] < 1.0 - 1e-6)
        t = sum(1 for x in items if abs(x[2] - 1.0) <= 1e-6)
        lines.append(
            f"| {wl} | {ga:.4e} | {gb:.4e} | {gr:.4f} | {(1.0 - gr) * 100:+.2f}% | {w}/{len(items)} ({t} tie) |"
        )
    lines.append("")
    return "\n".join(lines)


def run_rank_check(evaluate, seed: int = 1, max_per_run: int = 20):
    """Within-workload Spearman rank correlation check between junior's Accelergy 0.3
    and canonical Accelergy 0.4 on saved candidate pools."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rank_rows = []
    print(f"\n=== WITHIN-WORKLOAD RANK-CORRELATION CHECK (seed={seed}, up to {max_per_run} candidates/run) ===", flush=True)

    for arm, folder, key_name in [
        ("gamma_only", ROOT / "results/experiments/gamma_only", "final_population"),
        ("rl_only", ROOT / "results/experiments/rl_only", "evaluated_candidates"),
    ]:
        for path in sorted(folder.glob(f"*_seed{seed}_*.json")):
            if path.name in EXCLUDED_FILENAMES:
                continue
            obj = json.loads(path.read_text())
            wl = obj["experiment"]["workload_name"]
            cands = obj.get(key_name) or []
            # Deduplicate by tuple list and take stratified sample including top-5
            seen = set()
            uniq = []
            for c in cands:
                t = parse_tuples(c["tuples"])
                edp = safe_float(c.get("edp"))
                if edp is None:
                    continue
                sig = json.dumps(t)
                if sig in seen:
                    continue
                seen.add(sig)
                uniq.append({"edp": edp, "tuples": t, "mapping": c.get("mapping")})
            uniq.sort(key=lambda x: x["edp"])
            if len(uniq) > max_per_run:
                # Keep top 5 + evenly spaced remainder
                rest = uniq[5:]
                step = max(1, len(rest) // (max_per_run - 5))
                sample = uniq[:5] + rest[::step][: max_per_run - 5]
            else:
                sample = uniq

            hist_vals, canon_vals = [], []
            for idx, c in enumerate(sample):
                res = evaluate(wl, c["tuples"], c["mapping"])
                if not res or safe_float(res.get("edp")) is None:
                    continue
                ce = float(res["edp"])
                hist_vals.append(c["edp"])
                canon_vals.append(ce)
                rank_rows.append({
                    "arm": arm,
                    "workload": wl,
                    "seed": seed,
                    "hist_rank": idx + 1,
                    "historical_edp": c["edp"],
                    "canonical_edp": ce,
                    "canonical_energy_pJ": float(res["energy_pJ"]),
                    "canonical_cycles": float(res["cycles"]),
                    "scale_ratio": ce / c["edp"],
                })

            rho = spearman_corr(hist_vals, canon_vals)
            best_hist_canon = canon_vals[0] if canon_vals else None
            best_overall_canon = min(canon_vals) if canon_vals else None
            winner_gap = ((best_hist_canon - best_overall_canon) / best_overall_canon * 100.0) if (best_hist_canon and best_overall_canon) else 0.0
            scales = [c / h for h, c in zip(hist_vals, canon_vals)]
            print(
                f"  {arm:10s} {wl:13s}: n={len(hist_vals):2d}, "
                f"Spearman rho={rho:.4f}" if rho is not None else f"  {arm:10s} {wl:13s}: n={len(hist_vals):2d}, rho=n/a",
                f"| scale=[{min(scales):.4f}, {max(scales):.4f}] | top1_gap={winner_gap:.3f}%",
                flush=True,
            )

    out_csv = OUT_DIR / f"rank_check_seed{seed}.csv"
    if rank_rows:
        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rank_rows[0].keys()))
            w.writeheader()
            w.writerows(rank_rows)
        print(f"Saved rank-check details to: {out_csv}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true", help="Evaluate baseline seed 1 for all 3 arms + conversion equivalence check")
    ap.add_argument("--rank-check", action="store_true", help="Run within-workload Spearman rank correlation check on saved candidate pools")
    ap.add_argument("--rank-seed", type=int, default=1, help="Seed for --rank-check (default: 1)")
    ap.add_argument("--skip-75", action="store_true", help="Skip the 75-winner run (use with --rank-check)")
    args = ap.parse_args()

    all_records = load_hybrid_winners() + load_rl_only_winners() + load_gamma_only_winners()
    all_records.sort(key=lambda r: (WORKLOAD_ORDER.index(r["workload"]), SEEDS.index(r["seed"]), r["arm"]))

    counts = defaultdict(int)
    for r in all_records:
        counts[r["arm"]] += 1

    print("=== COVERAGE CHECK ===", flush=True)
    for arm in ("hybrid", "rl_only", "gamma_only"):
        print(f"  {arm:10s}: {counts[arm]}/25", flush=True)

    evaluate, getmapping, format_mapping_dict = make_env_pool()

    if args.smoke_test:
        print("\n=== SMOKE TEST (baseline, seed=1) ===", flush=True)
        for arm in ("hybrid", "rl_only", "gamma_only"):
            rec = next(r for r in all_records if r["arm"] == arm and r["workload"] == "baseline" and r["seed"] == 1)
            res = evaluate(rec["workload"], rec["tuples"], rec["mapping_dict"])
            assert res is not None, f"Evaluation failed for {arm}"
            ratio = res["edp"] / rec["historical_edp"]
            print(
                f"  {arm:10s}: hist_edp={rec['historical_edp']:.6e} -> "
                f"canon_edp={res['edp']:.6e} (scale={ratio:.6f}) "
                f"[cycles={res['cycles']:.0f}, energy_pJ={res['energy_pJ']:.4e}]",
                flush=True,
            )
            if arm == "rl_only":
                res_from_tuples = evaluate(rec["workload"], rec["tuples"], None)
                assert res_from_tuples is not None
                diff = abs(res_from_tuples["edp"] - res["edp"]) / res["edp"]
                print(
                    f"  rl_only tuple-vs-dict check: rel_diff={diff:.3e} "
                    f"({'EXACT MATCH' if diff < 1e-9 else 'DIFFERS!'})",
                    flush=True,
                )
        print("\nExpected: hybrid scale ~= 1.000000; rl_only & gamma_only scale ~= 0.09-0.15.", flush=True)
        return

    if not args.skip_75:
        t0 = time.perf_counter()
        print(f"\n=== EVALUATING ALL {len(all_records)} WINNERS IN CANONICAL ENV ===", flush=True)
        results_rows = []
        for idx, rec in enumerate(all_records, 1):
            res = evaluate(rec["workload"], rec["tuples"], rec["mapping_dict"])
            row = {
                "arm": rec["arm"],
                "workload": rec["workload"],
                "seed": rec["seed"],
                "historical_edp": rec["historical_edp"],
                "canonical_edp": float(res["edp"]) if res else None,
                "canonical_energy_pJ": float(res["energy_pJ"]) if res else None,
                "canonical_cycles": float(res["cycles"]) if res else None,
                "canonical_utilization": float(res["utilization"]) if res else None,
                "scale_ratio": (float(res["edp"]) / rec["historical_edp"]) if res else None,
                "eval_count": rec["eval_count"],
                "source_file": rec["source_file"],
            }
            results_rows.append(row)
            if res:
                print(
                    f"[{idx:02d}/{len(all_records):02d}] {row['arm']:10s} {row['workload']:13s} "
                    f"seed={row['seed']:<2d} hist={row['historical_edp']:.4e} -> "
                    f"canon={row['canonical_edp']:.4e} (scale={row['scale_ratio']:.4f})",
                    flush=True,
                )
            else:
                print(
                    f"[{idx:02d}/{len(all_records):02d}] {row['arm']:10s} {row['workload']:13s} "
                    f"seed={row['seed']:<2d} FAILED!",
                    flush=True,
                )

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = OUT_DIR / "canonical_75_winners.csv"
        out_json = OUT_DIR / "canonical_75_winners.json"
        out_md = OUT_DIR / "comparison_summary.md"

        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results_rows[0].keys()))
            w.writeheader()
            w.writerows(results_rows)

        json_payload = [
            {**row, "tuples": rec["tuples"]}
            for row, rec in zip(results_rows, all_records)
        ]
        out_json.write_text(json.dumps(json_payload, indent=2))

        # Build summary markdown
        md_parts = [
            "# Canonical Reevaluation & Paired Comparison Summary",
            "",
            f"- Evaluated `{sum(1 for r in results_rows if r['canonical_edp'])}/{len(results_rows)}` winners in `{time.perf_counter() - t0:.1f}s`.",
            "- Canonical evaluator: local `TimeloopEnv` (`in_config_3d`, `use_sparse=False`, `use_IO=True`, Accelergy `0.4`).",
            "",
            "## Self-Consistency & Cross-Environment Scale Audit",
            "",
            "| Arm | Evaluated | Scale (Canon / Hist) Min | Scale Median | Scale Max |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm in ("hybrid", "rl_only", "gamma_only"):
            scales = sorted(r["scale_ratio"] for r in results_rows if r["arm"] == arm and r["scale_ratio"])
            if scales:
                md_parts.append(
                    f"| {arm} | {len(scales)}/25 | {scales[0]:.6f} | {scales[len(scales)//2]:.6f} | {scales[-1]:.6f} |"
                )
        md_parts.append("")
        md_parts.append(build_pairwise_report(results_rows, "hybrid", "rl_only"))
        md_parts.append(build_pairwise_report(results_rows, "hybrid", "gamma_only"))
        md_parts.append(build_pairwise_report(results_rows, "rl_only", "gamma_only"))

        report_text = "\n".join(md_parts)
        out_md.write_text(report_text)
        print("\n" + report_text, flush=True)
        print(f"Saved outputs to:\n  {out_csv}\n  {out_json}\n  {out_md}", flush=True)

    if args.rank_check:
        run_rank_check(evaluate, seed=args.rank_seed)


if __name__ == "__main__":
    main()
