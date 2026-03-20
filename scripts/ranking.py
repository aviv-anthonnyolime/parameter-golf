#!/usr/bin/env python3
"""
Read JSONL results and display a ranked leaderboard + matplotlib charts.

Usage:
    python scripts/ranking.py                                  # global results
    python scripts/ranking.py experiments/phase1_UT/results.jsonl
    python scripts/ranking.py --chart                          # also save PNG charts
    python scripts/ranking.py --chart --out charts/            # custom output dir
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


def print_table(entries):
    if not entries:
        print("No results found.")
        return

    # Sort by val_bpb_ttt (competition score), then val_bpb_int8, then val_bpb
    entries.sort(key=lambda e: (e.get("val_bpb_ttt", 99), e.get("val_bpb_int8", 99), e.get("val_bpb", 99)))

    # Header
    hdr = (
        f"{'Rank':>4}  {'Docker Name':<20} {'Experiment':<12} "
        f"{'val_bpb':>8} {'bpb_int8':>9} {'bpb_ttt':>8} "
        f"{'loss':>7} {'ms/step':>8} {'steps':>6} {'size_MB':>8} {'Params':>18}"
    )
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)

    for i, e in enumerate(entries, 1):
        hp = e.get("hyperparams", {})
        ptag = e.get("params_tag", "?")
        print(
            f"{i:>4}  {e.get('docker_name', '?'):<20} {e.get('experiment', '?'):<12} "
            f"{e.get('val_bpb', 0):>8.4f} {e.get('val_bpb_int8', 0):>9.4f} {e.get('val_bpb_ttt', 0):>8.4f} "
            f"{e.get('val_loss', 0):>7.4f} {e.get('avg_ms_per_step', 0):>8.1f} {e.get('total_steps', 0):>6} "
            f"{e.get('compressed_size_mb', 0):>8.2f} {ptag:>18}"
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
    entries_ranked = sorted(entries, key=lambda e: e.get("val_bpb_ttt", 99))
    ranked_names = [e.get("docker_name", "?") for e in entries_ranked]
    ranked_bpb = [e.get("val_bpb_ttt", 0) for e in entries_ranked]

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
    train_losses = [e.get("val_loss", 0) for e in entries_chrono]
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
    bpb_values = [e.get("val_bpb_ttt", e.get("val_bpb", 0)) for e in entries_chrono]
    bpb_int8 = [e.get("val_bpb_int8", 0) for e in entries_chrono]
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
    sizes = [e.get("compressed_size_mb", 0) for e in entries_chrono]
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
    print_table(entries)

    if args.chart:
        print("Generating charts...")
        make_charts(entries, Path(args.out))


if __name__ == "__main__":
    main()
