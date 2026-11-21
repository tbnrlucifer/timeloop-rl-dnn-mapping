#!/usr/bin/env python3
"""Extract wall-clock search runtimes from original optimization JSONs
(RL-only 120, GAMMA-only 120, Hybrid Iter<=1 & Iter<=3, and Iso-Budget 420).

Excludes reevaluation runs and historical Oct-21 duplicate pointwise seed-1 runs.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "results").exists():
    ROOT = Path.home() / "timeloop-rl-dnn-mapping"

OUT_DIR = ROOT / "results" / "common_eval" / "canonical_v2"
WORKLOAD_ORDER = ["small", "baseline", "spatial-heavy", "channel-heavy", "pointwise"]
SEEDS = [1, 2, 3, 4, 42]

EXCLUDED = {
    "rl_only_pointwise_seed1_20261021_111634.json",
    "rl_ga_lpf_pointwise_seed1_full_run_20261021_103707.json",
    "gamma_lpf_only_pointwise_seed1_20261021_110503.json",
}


def num(val: Any) -> float | None:
    try:
        x = float(val)
        return x if math.isfinite(x) and x >= 0 else None
    except (TypeError, ValueError):
        return None


def parse_hybrid_timing(obj: dict) -> dict[str, float | None]:
    ts = obj.get("timing_seconds")
    iters = obj.get("iterations") or []

    total = None
    initial_rl = None
    if isinstance(ts, (int, float)):
        total = num(ts)
    elif isinstance(ts, dict):
        for k in ("total", "total_seconds", "experiment_total", "full_run", "overall"):
            if k in ts and num(ts[k]) is not None:
                total = num(ts[k])
                break
        for k in ("initial_rl", "initial_ppo", "rl_init"):
            if k in ts and num(ts[k]) is not None:
                initial_rl = num(ts[k])
                break

    gamma_sum = 0.0
    feedback_sum = 0.0
    iter_total_sum = 0.0
    iter1_gamma = None
    iter1_total = None

    for idx, it in enumerate(iters, 1):
        its = it.get("timing_seconds") or {}
        if isinstance(its, dict):
            g = num(its.get("gamma")) or 0.0
            fb = num(its.get("feedback")) or 0.0
            itot = num(its.get("iteration_total")) or (g + fb)
            gamma_sum += g
            feedback_sum += fb
            iter_total_sum += itot
            if idx == 1:
                iter1_gamma = g
                iter1_total = itot

    # If top-level total wasn't named with a known key, sum top-level numeric parts or fallback
    if total is None and isinstance(ts, dict):
        numeric_vals = [num(v) for v in ts.values() if num(v) is not None]
        if numeric_vals:
            total = sum(numeric_vals)
    if total is None:
        total = (initial_rl or 0.0) + iter_total_sum

    # Time to reach Iter<=1 winner = initial_rl + iter1_gamma (before iter1 feedback)
    iter1_to_winner = (initial_rl + iter1_gamma) if (initial_rl is not None and iter1_gamma is not None) else None

    return {
        "total_seconds": total,
        "initial_rl_seconds": initial_rl,
        "gamma_total_seconds": gamma_sum if iters else None,
        "feedback_total_seconds": feedback_sum if iters else None,
        "iter1_to_winner_seconds": iter1_to_winner,
        "iter1_total_seconds": (initial_rl + iter1_total) if (initial_rl is not None and iter1_total is not None) else None,
    }


def collect_arm(arm: str, folder: Path, budget: int, machine_note: str) -> list[dict[str, Any]]:
    rows = []
    if not folder.exists():
        return rows
    for path in sorted(folder.glob("*.json")):
        if path.name in EXCLUDED:
            continue
        obj = json.loads(path.read_text())
        exp = obj.get("experiment") or {}
        wl = exp.get("workload_name")
        seed = int(exp.get("experiment_seed", exp.get("seed")))
        if wl not in WORKLOAD_ORDER or seed not in SEEDS:
            continue

        if arm == "hybrid":
            t = parse_hybrid_timing(obj)
            rows.append({
                "arm": "hybrid_iter3",
                "budget": budget,
                "workload": wl,
                "seed": seed,
                "runtime_seconds": t["total_seconds"],
                "initial_rl_seconds": t["initial_rl_seconds"],
                "gamma_total_seconds": t["gamma_total_seconds"],
                "feedback_total_seconds": t["feedback_total_seconds"],
                "iter1_to_winner_seconds": t["iter1_to_winner_seconds"],
                "machine": machine_note,
                "source_file": str(path.relative_to(ROOT)),
            })
        else:
            rows.append({
                "arm": arm,
                "budget": budget,
                "workload": wl,
                "seed": seed,
                "runtime_seconds": num(obj.get("runtime_seconds")),
                "initial_rl_seconds": None,
                "gamma_total_seconds": None,
                "feedback_total_seconds": None,
                "iter1_to_winner_seconds": None,
                "machine": machine_note,
                "source_file": str(path.relative_to(ROOT)),
            })
    return rows


def agg_stats(vals: list[float]) -> tuple[float, float, float, float]:
    n = len(vals)
    mean = sum(vals) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return mean, std, min(vals), max(vals)


def main():
    # Peek at exact keys of Hybrid timing_seconds first
    hyb_files = sorted(
        p for p in (ROOT / "results/experiments/hybrid").glob("*.json")
        if p.name not in EXCLUDED
    )
    if hyb_files:
        sample = json.loads(hyb_files[0].read_text())
        print("=== SAMPLE HYBRID timing_seconds KEYS ===", flush=True)
        print("  top-level:", sample.get("timing_seconds"), flush=True)
        for idx, it in enumerate(sample.get("iterations") or [], 1):
            print(f"  iter[{idx}]:", it.get("timing_seconds"), flush=True)
        print()

    rows = []
    rows += collect_arm("hybrid", ROOT / "results/experiments/hybrid", 420, "Canonical laptop (RTX 4060 GPU + CPU, sequential)")
    rows += collect_arm("rl_only_120", ROOT / "results/experiments/rl_only", 120, "RL junior machine (CPU-only PyTorch)")
    rows += collect_arm("gamma_only_120", ROOT / "results/experiments/gamma_only", 120, "GAMMA junior WSL2 (18-core, 4 parallel)")
    rows += collect_arm("rl_only_420", ROOT / "results/experiments/iso_budget_420/rl_only", 420, "Canonical laptop (RTX 4060 GPU + CPU, sequential)")
    rows += collect_arm("gamma_only_420", ROOT / "results/experiments/iso_budget_420/gamma_only", 420, "Canonical laptop (CPU, sequential)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "search_runtimes_all.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    grouped = defaultdict(list)
    hyb_breakdown = defaultdict(list)
    for r in rows:
        if r["runtime_seconds"] is not None:
            grouped[(r["arm"], r["workload"])].append(r["runtime_seconds"])
        if r["arm"] == "hybrid_iter3":
            hyb_breakdown[r["workload"]].append(r)

    md = [
        "# Wall-Clock Search Runtime Summary (Optimization Runs Only)",
        "",
        "*(Excludes canonical Accelergy 0.4 reevaluation side runs. Values are wall-clock elapsed seconds from `time.perf_counter()`.)*",
        "",
        "## 1. Same-Machine Apples-to-Apples Comparison (~420 Evals, Canonical Laptop)",
        "",
        "| Workload | Arm | Runs | Mean (s) | Std (s) | Mean (min) | Min – Max (min) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for wl in ["baseline", "channel-heavy"]:
        for arm, label in [
            ("hybrid_iter3", "Hybrid Iter<=3 (400-420 evals)"),
            ("rl_only_420", "RL-only (420 evals)"),
            ("gamma_only_420", "GAMMA-only (420 evals)"),
        ]:
            vals = grouped.get((arm, wl), [])
            if vals:
                m, s, lo, hi = agg_stats(vals)
                md.append(f"| {wl} | {label} | {len(vals)} | {m:.1f} | {s:.1f} | {m/60:.1f} | {lo/60:.1f} – {hi/60:.1f} |")

    md += [
        "",
        "## 2. All 5 Workloads: Original Runs (Note: Different Machines for 120-Eval Baselines)",
        "",
        "| Workload | Arm | Machine / Setup | Runs | Mean (s) | Std (s) | Mean (min) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    arm_meta = [
        ("hybrid_iter3", "Hybrid Iter<=3 (400-420)", "Canonical laptop (GPU+CPU, seq)"),
        ("rl_only_120", "RL-only (120)", "RL junior (CPU-only)"),
        ("gamma_only_120", "GAMMA-only (120)", "GAMMA junior (WSL2, 4x parallel)"),
    ]
    for wl in WORKLOAD_ORDER:
        for arm, label, mach in arm_meta:
            vals = grouped.get((arm, wl), [])
            if vals:
                m, s, _, _ = agg_stats(vals)
                md.append(f"| {wl} | {label} | {mach} | {len(vals)} | {m:.1f} | {s:.1f} | {m/60:.1f} |")

    md += [
        "",
        "## 3. Hybrid Stage-by-Stage Runtime Breakdown (Canonical Laptop, n=5 Seeds)",
        "",
        "| Workload | Initial RL (s) | GAMMA Total 3-Iter (s) | PPO Feedback Total (s) | Iter<=1 to Winner (s) | Full 3-Iter Total (s / min) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for wl in WORKLOAD_ORDER:
        recs = hyb_breakdown.get(wl, [])
        if not recs:
            continue
        def avg_field(k):
            vs = [r[k] for r in recs if r.get(k) is not None]
            return f"{sum(vs)/len(vs):.1f}" if vs else "n/a"
        tot_vs = [r["runtime_seconds"] for r in recs if r.get("runtime_seconds") is not None]
        tot_str = f"{sum(tot_vs)/len(tot_vs):.1f}s ({sum(tot_vs)/len(tot_vs)/60:.1f}m)" if tot_vs else "n/a"
        md.append(
            f"| {wl} | {avg_field('initial_rl_seconds')} | {avg_field('gamma_total_seconds')} | "
            f"{avg_field('feedback_total_seconds')} | {avg_field('iter1_to_winner_seconds')} | {tot_str} |"
        )

    report = "\n".join(md) + "\n"
    md_path = OUT_DIR / "search_runtimes_report.md"
    md_path.write_text(report)
    print(report, flush=True)
    print(f"Saved:\n  {csv_path}\n  {md_path}", flush=True)


if __name__ == "__main__":
    main()
