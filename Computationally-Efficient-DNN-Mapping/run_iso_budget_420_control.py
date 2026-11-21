#!/usr/bin/env python3
"""Isolated 420-eval iso-budget baseline control runner (canonical Accelergy 0.4).

Runs rl_only_baseline.py (RL_ONLY_EVAL_BUDGET=420) and gamma_lpf_only.py
(LPF_GAMMA_POP=20, LPF_GAMMA_GENS=20 -> 420 nominal evals), moving ONLY newly
created JSON outputs into results/experiments/iso_budget_420/{arm}/ so existing
120-eval datasets remain untouched.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ISO_ROOT = ROOT / "results" / "experiments" / "iso_budget_420"
LOG_ROOT = ROOT / "results" / "logs" / "iso_budget_420"


def already_done(arm: str, workload: str, seed: int) -> bool:
    arm_dir = ISO_ROOT / arm
    if not arm_dir.exists():
        return False
    return any(arm_dir.glob(f"*{workload}_seed{seed}_*.json"))


def run_one(arm: str, workload: str, seed: int, budget: int):
    if already_done(arm, workload, seed):
        print(f"[SKIP] {arm} {workload} seed={seed} already exists in {ISO_ROOT / arm}", flush=True)
        return

    (ISO_ROOT / arm).mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'pytimeloop'}:{env.get('PYTHONPATH', '')}"
    env["RL_GA_WORKLOAD"] = workload
    env["RL_GA_SEED"] = str(seed)

    if arm == "rl_only":
        script = ROOT / "Computationally-Efficient-DNN-Mapping" / "rl_only_baseline.py"
        env["RL_ONLY_EVAL_BUDGET"] = str(budget)
    elif arm == "gamma_only":
        script = ROOT / "Computationally-Efficient-DNN-Mapping" / "gamma_lpf_only.py"
        pop = 20
        gens = (budget // pop) - 1  # 20 * (20 + 1) = 420
        env["LPF_GAMMA_POP"] = str(pop)
        env["LPF_GAMMA_GENS"] = str(gens)
    else:
        raise ValueError(arm)

    log_path = LOG_ROOT / f"{arm}_{workload}_seed{seed}_b{budget}.log"
    print(f"\n[RUN] {arm} | workload={workload} | seed={seed} | budget={budget}", flush=True)
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    if proc.returncode != 0:
        print(f"[ERROR] Return code {proc.returncode}; see {log_path}", flush=True)
        return

    m = re.search(r"Results saved to:\s*(.+)", proc.stdout)
    if not m:
        print(f"[WARN] Could not parse output path from stdout; see {log_path}", flush=True)
        return
    src = Path(m.group(1).strip())
    if src.exists():
        dst = ISO_ROOT / arm / src.name
        shutil.move(str(src), str(dst))
        print(f"[OK] Moved result -> {dst}", flush=True)
    else:
        print(f"[WARN] Reported result file not found: {src}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", nargs="+", default=["baseline", "channel-heavy"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 42])
    ap.add_argument("--arms", nargs="+", default=["rl_only", "gamma_only"])
    ap.add_argument("--budget", type=int, default=420)
    args = ap.parse_args()

    for wl in args.workloads:
        for seed in args.seeds:
            for arm in args.arms:
                run_one(arm, wl, seed, args.budget)


if __name__ == "__main__":
    main()
