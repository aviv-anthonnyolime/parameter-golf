#!/usr/bin/env python3
"""
Read JSONL results and display a ranked leaderboard + matplotlib charts.

Usage:
    python scripts/ranking.py                                  # global results
    python scripts/ranking.py experiments/phase1_UT/results.jsonl
    python scripts/ranking.py --chart                          # also save PNG charts
    python scripts/ranking.py --chart --out charts/            # custom output dir
    python scripts/ranking.py --sort val_bpb                  # sort by one score
    python scripts/ranking.py --sort val_bpb_ttt val_bpb      # sort by two scores
    python scripts/ranking.py --normalize                      # project to max steps (α=0.35)
    python scripts/ranking.py --normalize 3500                 # project to 3500 steps
    python scripts/ranking.py --normalize --alpha 0.3         # custom scaling exponent
    python scripts/ranking.py --normalize --sort proj_bpb     # sort by projected bpb
    python scripts/ranking.py --wandb                         # fetch real curves from W&B, compare at min steps
    python scripts/ranking.py --wandb --target-steps 2000     # compare at step 2000 using W&B data
    python scripts/ranking.py --wandb --sort wb_bpb           # sort by W&B-interpolated bpb

Sortable fields: val_bpb, val_bpb_int8, val_bpb_ttt, val_loss,
                 avg_ms_per_step, total_steps, compressed_size_mb,
                 proj_loss, proj_bpb  (only when --normalize is set)
                 wb_loss, wb_bpb     (only when --wandb is set)

--wandb fetches the actual per-step val_loss curves from Weights & Biases and
interpolates each run at a common step count — no scaling-law assumptions needed.
Requires `pip install wandb` and WANDB_API_KEY set (or `wandb login`).

Normalization uses the power-law scaling: L_proj = L × (steps / target)^α
This makes short runs comparable to long ones by extrapolating to a common horizon.
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(path: Path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


NUMERIC_FIELDS = [
    "val_bpb", "val_bpb_int8", "val_bpb_ttt",
    "val_loss", "avg_ms_per_step", "total_steps", "compressed_size_mb",
    "proj_loss", "proj_bpb",
    "wb_loss", "wb_bpb",
]

DEFAULT_SORT = ["val_bpb_ttt", "val_bpb_int8", "val_bpb"]

BPB_SCALE = 1.0 / 0.6931471805599453  # 1/ln(2): nats-per-token → bits-per-byte


def add_projected_scores(entries, target_steps, alpha):
    """Add proj_loss and proj_bpb to each entry using power-law extrapolation."""
    for e in entries:
        steps = e.get("total_steps") or 0
        loss = e.get("val_loss")
        if steps > 0 and loss is not None:
            e["proj_loss"] = loss * (steps / target_steps) ** alpha
            e["proj_bpb"] = e["proj_loss"] * BPB_SCALE
        else:
            e["proj_loss"] = None
            e["proj_bpb"] = None


def _interpolate(steps, values, target):
    """Linear interpolation of `values` at `target` step."""
    if not steps:
        return None
    if target <= steps[0]:
        return values[0]
    if target >= steps[-1]:
        return values[-1]
    for i in range(len(steps) - 1):
        s0, s1 = steps[i], steps[i + 1]
        if s0 <= target <= s1:
            t = (target - s0) / (s1 - s0)
            return values[i] + t * (values[i + 1] - values[i])
    return None


def add_wandb_scores(entries, target_steps, api_key=None):
    """
    Fetch val_loss history from W&B for each entry and interpolate at target_steps.
    Adds wb_loss and wb_bpb fields in-place.
    Run name in W&B is "{docker_name}_{params_tag}".
    """
    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed. pip install wandb")
        sys.exit(1)

    if api_key:
        wandb.login(key=api_key, relogin=False)

    api = wandb.Api()

    # Collect entity/project from entries (fall back to known defaults)
    hp0 = (entries[0].get("hyperparams") or {}) if entries else {}
    entity = hp0.get("wandb_entity", "citaman")
    project = hp0.get("wandb_project", "Openai-challenge-parameter-golf")

    # Build a name→run map for the whole project (one API call)
    print(f"  Fetching W&B runs from {entity}/{project} …")
    try:
        wb_runs = {r.name: r for r in api.runs(f"{entity}/{project}")}
    except Exception as exc:
        print(f"ERROR fetching W&B runs: {exc}")
        sys.exit(1)
    print(f"  Found {len(wb_runs)} runs in W&B")

    for e in entries:
        run_name = f"{e.get('docker_name', '')}_{e.get('params_tag', '')}"
        wb_run = wb_runs.get(run_name)
        if wb_run is None:
            print(f"  WARN: no W&B run found for '{run_name}'")
            e["wb_loss"] = None
            e["wb_bpb"] = None
            continue

        history = wb_run.history(keys=["val_loss"], pandas=False)
        pts = [(row["_step"], row["val_loss"]) for row in history
               if row.get("val_loss") is not None]
        if not pts:
            e["wb_loss"] = None
            e["wb_bpb"] = None
            continue

        pts.sort()
        steps_list, loss_list = zip(*pts)
        wb_loss = _interpolate(list(steps_list), list(loss_list), target_steps)
        e["wb_loss"] = wb_loss
        e["wb_bpb"] = wb_loss * BPB_SCALE if wb_loss is not None else None


def print_table(entries, sort_by=None, show_proj=False, show_wb=False):
    if not entries:
        print("No results found.")
        return

    keys = sort_by if sort_by else DEFAULT_SORT

    def sort_key(e):
        def v(k):
            val = e.get(k)
            return 99 if val is None else val
        return tuple(v(k) for k in keys)

    entries.sort(key=sort_key)
    print(f"  Sorted by: {', '.join(keys)}")

    # Header
    extra_cols = ""
    if show_proj:
        extra_cols += f" {'proj_loss':>10} {'proj_bpb':>9}"
    if show_wb:
        extra_cols += f" {'wb_loss':>8} {'wb_bpb':>8}"
    hdr = (
        f"{'Rank':>4}  {'Docker Name':<20} {'Experiment':<12} "
        f"{'val_bpb':>8} {'bpb_int8':>9} {'bpb_ttt':>8} "
        f"{'loss':>7} {'ms/step':>8} {'steps':>6} {'size_MB':>8} {'Params':>18}"
        f"{extra_cols}"
    )
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)

    for i, e in enumerate(entries, 1):
        ptag = e.get("params_tag", "?")
        extra_str = ""
        if show_proj:
            pl, pb = e.get("proj_loss"), e.get("proj_bpb")
            extra_str += f" {pl:>10.4f} {pb:>9.4f}" if pl is not None else f" {'N/A':>10} {'N/A':>9}"
        if show_wb:
            wl, wb = e.get("wb_loss"), e.get("wb_bpb")
            extra_str += f" {wl:>8.4f} {wb:>8.4f}" if wl is not None else f" {'N/A':>8} {'N/A':>8}"
        print(
            f"{i:>4}  {e.get('docker_name', '?'):<20} {e.get('experiment', '?'):<12} "
            f"{e.get('val_bpb') or 0:>8.4f} {e.get('val_bpb_int8') or 0:>9.4f} {e.get('val_bpb_ttt') or 0:>8.4f} "
            f"{e.get('val_loss') or 0:>7.4f} {e.get('avg_ms_per_step') or 0:>8.1f} {e.get('total_steps') or 0:>6} "
            f"{e.get('compressed_size_mb') or 0:>8.2f} {ptag:>18}"
            f"{extra_str}"
        )
    print(sep)
    print(f"  Total runs: {len(entries)}\n")


def make_charts(entries, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not installed. pip install matplotlib")
        return

    if not entries:
        print("No entries to chart.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort chronologically for line charts
    entries_chrono = sorted(entries, key=lambda e: e.get("timestamp", ""))
    names = [e.get("docker_name", "?") for e in entries_chrono]
    x = list(range(len(entries_chrono)))

    # --- Bar chart: val_bpb ranked (best at left) ---
    entries_ranked = sorted(entries, key=lambda e: e.get("val_bpb_ttt") or 99)
    ranked_names = [e.get("docker_name", "?") for e in entries_ranked]
    ranked_bpb = [e.get("val_bpb_ttt") or 0 for e in entries_ranked]

    fig, ax = plt.subplots(figsize=(max(8, len(entries) * 0.8), 5))
    colors = plt.cm.viridis([i / max(len(entries_ranked) - 1, 1) for i in range(len(entries_ranked))])
    bars = ax.bar(range(len(ranked_names)), ranked_bpb, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(ranked_names)))
    ax.set_xticklabels(ranked_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("val_bpb (TTT)")
    ax.set_title("Runs Ranked by val_bpb (TTT) — Lower is Better")
    # Add value labels on bars
    for bar, val in zip(bars, ranked_bpb):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)
    # Add breathing room to y axis
    if ranked_bpb:
        ymin = min(ranked_bpb) - 0.02
        ymax = max(ranked_bpb) + 0.02
        ax.set_ylim(ymin, ymax)
    plt.tight_layout()
    bar_path = out_dir / "ranking_bar.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {bar_path}")

    # --- Line+dot chart: training loss over runs (chronological) ---
    train_losses = [e.get("val_loss") or 0 for e in entries_chrono]
    fig, ax = plt.subplots(figsize=(max(8, len(entries) * 0.8), 5))
    ax.plot(x, train_losses, "o-", color="#2196F3", markersize=8, linewidth=2, markeredgecolor="black", markeredgewidth=0.5)
    for xi, (name, val) in enumerate(zip(names, train_losses)):
        ax.annotate(f"{name}\n{val:.4f}", (xi, val), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=7, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("val_loss (final)")
    ax.set_title("Validation Loss Over Runs (Chronological)")
    if train_losses:
        margin = max((max(train_losses) - min(train_losses)) * 0.15, 0.01)
        ax.set_ylim(min(train_losses) - margin, max(train_losses) + margin)
    # Extra x margin
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    loss_path = out_dir / "val_loss_line.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {loss_path}")

    # --- Line+dot chart: val_bpb over runs (chronological) ---
    bpb_values = [e.get("val_bpb_ttt") or e.get("val_bpb") or 0 for e in entries_chrono]
    bpb_int8 = [e.get("val_bpb_int8") or 0 for e in entries_chrono]
    fig, ax = plt.subplots(figsize=(max(8, len(entries) * 0.8), 5))
    ax.plot(x, bpb_values, "o-", color="#E91E63", markersize=8, linewidth=2, label="bpb_ttt",
            markeredgecolor="black", markeredgewidth=0.5)
    ax.plot(x, bpb_int8, "s--", color="#9C27B0", markersize=6, linewidth=1.5, label="bpb_int8",
            markeredgecolor="black", markeredgewidth=0.5, alpha=0.7)
    for xi, (name, val) in enumerate(zip(names, bpb_values)):
        ax.annotate(f"{name}\n{val:.4f}", (xi, val), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=7, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("val_bpb")
    ax.set_title("val_bpb Over Runs (Chronological) — Lower is Better")
    all_bpb = bpb_values + bpb_int8
    if all_bpb:
        margin = max((max(all_bpb) - min(all_bpb)) * 0.15, 0.01)
        ax.set_ylim(min(all_bpb) - margin, max(all_bpb) + margin)
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    bpb_path = out_dir / "val_bpb_line.png"
    fig.savefig(bpb_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {bpb_path}")

    # --- Scatter: val_bpb vs compressed size ---
    sizes = [e.get("compressed_size_mb") or 0 for e in entries_chrono]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sizes, bpb_values, s=80, c="#FF5722", edgecolors="black", linewidth=0.5, zorder=3)
    for xi, (name, sz, bpb) in enumerate(zip(names, sizes, bpb_values)):
        ax.annotate(name, (sz, bpb), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color="#333")
    ax.axvline(x=16.0, color="red", linestyle="--", alpha=0.5, label="16MB limit")
    ax.set_xlabel("Compressed Size (MB)")
    ax.set_ylabel("val_bpb (TTT)")
    ax.set_title("val_bpb vs Compressed Size")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    scatter_path = out_dir / "bpb_vs_size.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {scatter_path}")


def main():
    parser = argparse.ArgumentParser(description="Display run rankings from JSONL results")
    parser.add_argument("results_file", nargs="?", default=None, help="Path to results.jsonl")
    parser.add_argument("--chart", action="store_true", help="Generate matplotlib charts")
    parser.add_argument("--out", default="charts", help="Output directory for charts (default: charts/)")
    parser.add_argument(
        "--sort", nargs="+", metavar="FIELD", default=None,
        help=f"One or more fields to sort by (default: {' '.join(DEFAULT_SORT)}). "
             f"Available: {', '.join(NUMERIC_FIELDS)}",
    )
    parser.add_argument(
        "--normalize", nargs="?", const=-1, type=int, metavar="STEPS", default=None,
        help="Project all runs to a common step count for fair comparison. "
             "Omit the value to use the max steps seen in the run set. "
             "Uses power-law scaling: L_proj = L × (steps/target)^α",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.35, metavar="FLOAT",
        help="Scaling exponent for --normalize (default: 0.35, typical for LMs)",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Fetch real val_loss curves from W&B and interpolate at a common step count",
    )
    parser.add_argument(
        "--target-steps", type=int, default=None, metavar="N",
        help="Step count to compare at with --wandb (default: min total_steps across all runs)",
    )
    args = parser.parse_args()

    # Default: global results file
    if args.results_file is None:
        default_path = Path(__file__).resolve().parent.parent / "results" / "all_runs.jsonl"
        if not default_path.exists():
            print(f"No results file found at {default_path}")
            print("Run a training first, or pass a path: python scripts/ranking.py <path>")
            sys.exit(1)
        results_path = default_path
    else:
        results_path = Path(args.results_file)

    if not results_path.exists():
        print(f"File not found: {results_path}")
        sys.exit(1)

    entries = load_results(results_path)

    if args.sort:
        invalid = [f for f in args.sort if f not in NUMERIC_FIELDS]
        if invalid:
            print(f"ERROR: unknown sort field(s): {', '.join(invalid)}")
            print(f"Available: {', '.join(NUMERIC_FIELDS)}")
            sys.exit(1)

    show_proj = args.normalize is not None
    if show_proj:
        all_steps = [e.get("total_steps") or 0 for e in entries]
        target = max(all_steps) if args.normalize == -1 else args.normalize
        add_projected_scores(entries, target_steps=target, alpha=args.alpha)
        print(f"\n  Normalization: projecting to {target} steps  (α={args.alpha})")

    show_wb = args.wandb
    if show_wb:
        all_steps = [e.get("total_steps") or 0 for e in entries if e.get("total_steps")]
        target_steps = args.target_steps or min(all_steps)
        print(f"\n  W&B comparison at step {target_steps} (interpolated from real curves)")
        add_wandb_scores(entries, target_steps=target_steps)

    # Default sort: wb_bpb > proj_bpb > standard when the relevant mode is active
    sort_by = args.sort
    if sort_by is None:
        if show_wb:
            sort_by = ["wb_bpb"]
        elif show_proj:
            sort_by = ["proj_bpb"]

    print_table(entries, sort_by=sort_by, show_proj=show_proj, show_wb=show_wb)

    if args.chart:
        print("Generating charts...")
        make_charts(entries, Path(args.out))


if __name__ == "__main__":
    main()
