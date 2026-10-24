"""Roadmap Section 9: "analysis/: plotting, rliable statistics and diagnostic
dashboards." Reads TensorBoard event files from outputs/<algo>/<timestamp>/ (written by
train.py) and produces a comparison plot + summary table across algos.

Usage:
    python analysis/plot_comparison.py --outputs-dir outputs --metric success_rate
    python analysis/plot_comparison.py --metric group_return_std_mean --out analysis/std.png
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_runs(outputs_dir: str) -> dict[str, list[str]]:
    """algo -> list of run dirs (outputs/<algo>/<timestamp>/)."""
    runs = {}
    for algo_dir in sorted(glob.glob(os.path.join(outputs_dir, "*"))):
        if not os.path.isdir(algo_dir):
            continue
        algo = os.path.basename(algo_dir)
        run_dirs = sorted(glob.glob(os.path.join(algo_dir, "*")))
        run_dirs = [d for d in run_dirs if os.path.isdir(d)]
        if run_dirs:
            runs[algo] = run_dirs
    return runs


def read_scalar(run_dir: str, tag: str):
    """Returns (steps, values) for a scalar tag from the (single) event file in run_dir."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    full_tag = f"train/{tag}"
    if full_tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(full_tag)
    return [e.step for e in events], [e.value for e in events]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--metric", default="success_rate")
    ap.add_argument("--out", default=None, help="save PNG here instead of showing")
    ap.add_argument("--latest-only", action="store_true", default=True)
    args = ap.parse_args()

    runs = load_runs(args.outputs_dir)
    if not runs:
        print(f"no runs found under {args.outputs_dir}/ -- run scripts/run_month2_comparison.sh first")
        return

    import matplotlib

    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    print(f"metric: {args.metric}")
    print(f"{'algo':<10} {'last_value':>12} {'best_value':>12} {'n_points':>10}")

    for algo, run_dirs in runs.items():
        run_dir = run_dirs[-1] if args.latest_only else run_dirs[0]
        steps, values = read_scalar(run_dir, args.metric)
        if not steps:
            print(f"{algo:<10} {'(no data)':>12}")
            continue
        ax.plot(steps, values, label=algo)
        print(f"{algo:<10} {values[-1]:>12.4f} {max(values):>12.4f} {len(values):>10d}")

    ax.set_xlabel("env steps")
    ax.set_ylabel(args.metric)
    ax.set_title(f"{args.metric} vs env steps (pick-place-v3, warm-started from BC)")
    ax.legend()
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"saved plot -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
