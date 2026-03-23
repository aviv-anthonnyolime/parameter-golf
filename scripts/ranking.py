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
    python scripts/ranking.py --project-h100                  # project runs to 8xH100 (10 min)
    python scripts/ranking.py --project-h100 --h100-time 600  # custom time budget
    python scripts/ranking.py --project-h100 --speed-ratio 12.6  # custom speed ratio

Sortable fields: val_bpb, val_bpb_int8, val_bpb_ttt, val_loss,
                 avg_ms_per_step, total_steps, compressed_size_mb,
                 proj_loss, proj_bpb  (only when --normalize is set)
                 wb_loss, wb_bpb     (only when --wandb is set)
                 h100_steps, h100_ms  (only when --project-h100 is set)

--wandb fetches the actual per-step val_loss curves from Weights & Biases and
interpolates each run at a common step count — no scaling-law assumptions needed.
Requires `pip install wandb` and WANDB_API_KEY set (or `wandb login`).

Normalization uses the power-law scaling: L_proj = L × (steps / target)^α
This makes short runs comparable to long ones by extrapolating to a common horizon.

--project-h100 estimates what each run would achieve on 8×H100 in the given time
budget. Uses an empirical speed ratio derived from comparing A10G and H100 baselines.
"""

import argparse
import json
import re
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
    "h100_steps", "h100_ms", "h100_loss",
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


def add_h100_projection(entries, speed_ratio, time_budget_s, alpha):
    """
    Project each run to 8×H100 by estimating ms/step on H100 and computing
    how many steps would fit in the time budget.

    Uses W&B curve interpolation (via _wb_val_pts) when available and in-range.
    Falls back to power-law extrapolation (L_proj = L × (observed_steps / target_steps)^α).

    NOTE: the default α=0.35 is a chinchilla-scale scaling law. For the 2-4×
    extrapolation ranges typical here, the *empirical* α from baseline data is
    ~0.06. Use --alpha 0.06 for conservative estimates.

    Adds h100_ms, h100_steps, h100_loss, h100_bpb fields.
    """
    time_budget_ms = time_budget_s * 1000
    extrapolation_ratios = []
    for e in entries:
        ms = e.get("avg_ms_per_step")
        if ms is None or ms <= 0:
            e["h100_ms"] = None
            e["h100_steps"] = None
            e["h100_loss"] = None
            e["h100_bpb"] = None
            continue

        h100_ms = ms / speed_ratio
        h100_steps = int(time_budget_ms / h100_ms)
        e["h100_ms"] = round(h100_ms, 1)
        e["h100_steps"] = h100_steps

        # Try W&B curve interpolation first
        pts = e.get("_wb_val_pts")
        loss = None
        if pts and len(pts) >= 2:
            steps_list, loss_list = zip(*[(s, l) for s, l, *_ in pts])
            loss = _interpolate(list(steps_list), list(loss_list), h100_steps)

        # Fall back to power-law extrapolation
        if loss is None:
            obs_steps = e.get("total_steps") or 0
            obs_loss = e.get("val_loss")
            if obs_steps > 0 and obs_loss is not None:
                if h100_steps <= obs_steps:
                    loss = obs_loss  # within observed range, use final
                else:
                    loss = obs_loss * (obs_steps / h100_steps) ** alpha

        e["h100_loss"] = loss
        e["h100_bpb"] = loss * BPB_SCALE if loss is not None else None

        obs_steps = e.get("total_steps") or 0
        if obs_steps > 0 and h100_steps > obs_steps:
            extrapolation_ratios.append(h100_steps / obs_steps)

    if extrapolation_ratios:
        avg_ratio = sum(extrapolation_ratios) / len(extrapolation_ratios)
        if avg_ratio > 1.5:
            print(f"  WARNING: avg extrapolation ratio is {avg_ratio:.1f}× beyond "
                  f"observed data. α={alpha} may be too aggressive.")
            print(f"  Empirical α from baseline: ~0.06. Try --alpha 0.06 for "
                  f"conservative estimates.")


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


def load_from_wandb(entity, project, jsonl_entries=None):
    """
    Build entry list from W&B (primary source) + optional JSONL supplement.
    Includes crashed runs that never wrote to the JSONL.
    Stores raw val history in _wb_val_pts for later interpolation.
    """
    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed. pip install wandb")
        sys.exit(1)

    api = wandb.Api()
    print(f"  Fetching W&B runs from {entity}/{project} …")
    try:
        wb_runs = list(api.runs(f"{entity}/{project}"))
    except Exception as exc:
        print(f"ERROR fetching W&B runs: {exc}")
        sys.exit(1)
    print(f"  Found {len(wb_runs)} runs in W&B")

    # JSONL lookup keyed by (docker_name, params_tag) for supplementary fields
    jsonl_map = {}
    for e in (jsonl_entries or []):
        name = e.get("docker_name", "") or e.get("run_name", "")
        key = (name, e.get("params_tag", ""))
        jsonl_map[key] = e

    SUPPLEMENTARY = (
        "val_bpb_int8", "val_bpb_ttt", "val_loss_int8", "val_loss_ttt",
        "compressed_size_mb", "compressed_size_bytes",
        "avg_ms_per_step", "total_time_s", "peak_memory_mib",
        "timestamp", "hyperparams", "logfile", "run_id",
    )

    entries = []
    for wb_run in wb_runs:
        # Tags are [experiment, params_tag, docker_name]
        tags = wb_run.tags or []
        docker_name = next((t for t in tags if re.match(r'^[a-z]+(-[a-z]+)?$', t)), None)
        params_tag  = next((t for t in tags if re.match(r'^\d+L-', t)), None)
        experiment  = next((t for t in tags
                            if t not in (docker_name, params_tag)), "?")

        if not docker_name or not params_tag:
            # Fall back to parsing "{name}_{params_tag}" from run name
            m = re.match(r'^([a-z]+(?:-[a-z]+)?)_(.+)$', wb_run.name)
            if not m:
                print(f"  WARN: cannot parse run '{wb_run.name}', skipping")
                continue
            docker_name = docker_name or m.group(1)
            params_tag  = params_tag  or m.group(2)

        # Fetch val_loss + val_bpb history (one call per run)
        history = wb_run.history(keys=["val_loss", "val_bpb"], pandas=False)
        val_pts = [
            (int(row["_step"]), row["val_loss"], row.get("val_bpb"))
            for row in history if row.get("val_loss") is not None
        ]
        if not val_pts:
            print(f"  WARN: no val checkpoints for '{wb_run.name}', skipping")
            continue

        val_pts.sort()
        last_step, last_loss, last_bpb = val_pts[-1]
        if last_bpb is None:
            last_bpb = last_loss * BPB_SCALE

        # avg_ms_per_step from W&B summary if not in JSONL
        wb_step_ms = None
        summary = wb_run.summary or {}
        if "step_avg_ms" in summary:
            wb_step_ms = summary["step_avg_ms"]

        entry = {
            "docker_name":  docker_name,
            "params_tag":   params_tag,
            "experiment":   experiment,
            "total_steps":  last_step,
            "val_loss":     last_loss,
            "val_bpb":      last_bpb,
            "run_state":    wb_run.state,   # "finished" | "crashed" | "running"
            "_wb_val_pts":  val_pts,        # raw curve kept for interpolation
        }
        if wb_step_ms is not None:
            entry["avg_ms_per_step"] = wb_step_ms

        # Merge supplementary JSONL fields
        jsonl = jsonl_map.get((docker_name, params_tag), {})
        for field in SUPPLEMENTARY:
            if field in jsonl:
                entry[field] = jsonl[field]
        # If JSONL has a later/richer val_loss (clean exit), prefer it
        if jsonl.get("val_loss") is not None:
            entry["val_loss"] = jsonl["val_loss"]
            entry["val_bpb"]  = jsonl.get("val_bpb", last_bpb)

        entries.append(entry)

    return entries


def apply_target_steps(entries, target_steps):
    """
    Interpolate val_loss/val_bpb at target_steps from stored W&B curves.
    Overwrites val_loss/val_bpb in-place so the main table uses fair values.
    """
    for e in entries:
        pts = e.get("_wb_val_pts")
        if not pts:
            continue
        steps_list, loss_list, bpb_list = zip(*pts)
        loss = _interpolate(list(steps_list), list(loss_list), target_steps)
        bpb  = _interpolate(list(steps_list),
                            [b if b is not None else l * BPB_SCALE
                             for l, b in zip(loss_list, bpb_list)],
                            target_steps)
        if loss is not None:
            e["val_loss"] = loss
            e["val_bpb"]  = bpb
            e["total_steps"] = target_steps  # reflect the horizon we're comparing at


def print_table(entries, sort_by=None, show_proj=False, show_wb=False,
                show_state=False, show_h100=False):
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

    STATE_ICON = {"finished": " ", "crashed": "!", "running": "~"}

    # Header
    state_col = " S" if show_state else ""
    extra_cols = ""
    if show_proj:
        extra_cols += f" {'proj_loss':>10} {'proj_bpb':>9}"
    if show_wb:
        extra_cols += f" {'wb_loss':>8} {'wb_bpb':>8}"
    if show_h100:
        extra_cols += f" {'H100ms':>7} {'H100stp':>8} {'H100loss':>9}"
    hdr = (
        f"{'Rank':>4}  {'Docker Name':<20} {'Experiment':<12} "
        f"{'val_bpb':>8} {'bpb_int8':>9} {'bpb_ttt':>8} "
        f"{'loss':>7} {'ms/step':>8} {'steps':>6} {'size_MB':>8} {'Params':>18}"
        f"{state_col}{extra_cols}"
    )
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)

    for i, e in enumerate(entries, 1):
        ptag = e.get("params_tag", "?")
        state_str = f" {STATE_ICON.get(e.get('run_state', ''), '?')}" if show_state else ""
        extra_str = ""
        if show_proj:
            pl, pb = e.get("proj_loss"), e.get("proj_bpb")
            extra_str += f" {pl:>10.4f} {pb:>9.4f}" if pl is not None else f" {'N/A':>10} {'N/A':>9}"
        if show_wb:
            wl, wb = e.get("wb_loss"), e.get("wb_bpb")
            extra_str += f" {wl:>8.4f} {wb:>8.4f}" if wl is not None else f" {'N/A':>8} {'N/A':>8}"
        if show_h100:
            hms = e.get("h100_ms")
            hstp = e.get("h100_steps")
            hloss = e.get("h100_loss")
            hms_s = f"{hms:>7.1f}" if hms is not None else f"{'N/A':>7}"
            hstp_s = f"{hstp:>8}" if hstp is not None else f"{'N/A':>8}"
            hloss_s = f"{hloss:>9.4f}" if hloss is not None else f"{'N/A':>9}"
            extra_str += f" {hms_s} {hstp_s} {hloss_s}"
        print(
            f"{i:>4}  {e.get('docker_name', '?'):<20} {e.get('experiment', '?'):<12} "
            f"{e.get('val_bpb') or 0:>8.4f} {e.get('val_bpb_int8') or 0:>9.4f} {e.get('val_bpb_ttt') or 0:>8.4f} "
            f"{e.get('val_loss') or 0:>7.4f} {e.get('avg_ms_per_step') or 0:>8.1f} {e.get('total_steps') or 0:>6} "
            f"{e.get('compressed_size_mb') or 0:>8.2f} {ptag:>18}"
            f"{state_str}{extra_str}"
        )
    print(sep)
    if show_state:
        print("  S column: ' '=finished  '!'=crashed  '~'=running")
    if show_h100:
        print("  H100 columns: projected ms/step, steps in time budget, projected val_loss")
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
        "--from-wandb", action="store_true",
        help="Use W&B as primary source (includes crashed runs). "
             "JSONL (if given) is used only to supplement compression/int8/TTT fields.",
    )
    parser.add_argument(
        "--entity", default="citaman", metavar="ENTITY",
        help="W&B entity (default: citaman)",
    )
    parser.add_argument(
        "--project", default="Openai-challenge-parameter-golf", metavar="PROJECT",
        help="W&B project name",
    )
    parser.add_argument(
        "--target-steps", type=int, default=None, metavar="N",
        help="Compare all runs at this step using real W&B curves. "
             "With --from-wandb: default is min steps seen. "
             "With --wandb: default is min total_steps in JSONL.",
    )
    parser.add_argument(
        "--project-h100", action="store_true",
        help="Project all runs to 8×H100 performance. Shows estimated ms/step, "
             "steps achievable in the time budget, and projected val_loss.",
    )
    parser.add_argument(
        "--speed-ratio", type=float, default=12.6, metavar="FLOAT",
        help="Speed ratio between current hardware and 8×H100 (default: 12.6, "
             "derived from 4×A10G baseline 548.5ms vs 8×H100 baseline 43.54ms).",
    )
    parser.add_argument(
        "--h100-time", type=int, default=600, metavar="SECONDS",
        help="Time budget in seconds for H100 projection (default: 600 = 10 min).",
    )
    args = parser.parse_args()

    if args.sort:
        invalid = [f for f in args.sort if f not in NUMERIC_FIELDS]
        if invalid:
            print(f"ERROR: unknown sort field(s): {', '.join(invalid)}")
            print(f"Available: {', '.join(NUMERIC_FIELDS)}")
            sys.exit(1)

    # --- Primary data source ---
    if args.from_wandb:
        # Load JSONL as supplement if provided (optional)
        jsonl_entries = None
        if args.results_file:
            p = Path(args.results_file)
            jsonl_entries = load_results(p) if p.exists() else None

        entries = load_from_wandb(args.entity, args.project, jsonl_entries)

        if args.target_steps:
            print(f"  Comparing at step {args.target_steps} (interpolated from W&B curves)")
            apply_target_steps(entries, args.target_steps)
        else:
            print("  Using each run's last checkpoint (pass --target-steps N for a common horizon)")

        show_proj = args.normalize is not None
        if show_proj:
            all_steps = [e.get("total_steps") or 0 for e in entries]
            target = max(all_steps) if args.normalize == -1 else args.normalize
            add_projected_scores(entries, target_steps=target, alpha=args.alpha)

        show_h100 = args.project_h100
        if show_h100:
            print(f"  H100 projection: speed_ratio={args.speed_ratio}x, "
                  f"time_budget={args.h100_time}s")
            add_h100_projection(entries, args.speed_ratio, args.h100_time,
                                args.alpha)

        sort_by = args.sort or (["h100_loss"] if show_h100 else ["val_bpb"])
        print_table(entries, sort_by=sort_by, show_proj=show_proj,
                    show_state=True, show_h100=show_h100)

    else:
        # JSONL-primary path (original behavior)
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

        show_h100 = args.project_h100
        if show_h100:
            print(f"\n  H100 projection: speed_ratio={args.speed_ratio}x, "
                  f"time_budget={args.h100_time}s")
            add_h100_projection(entries, args.speed_ratio, args.h100_time,
                                args.alpha)

        sort_by = args.sort
        if sort_by is None:
            if show_h100:
                sort_by = ["h100_loss"]
            elif show_wb:
                sort_by = ["wb_bpb"]
            elif show_proj:
                sort_by = ["proj_bpb"]

        print_table(entries, sort_by=sort_by, show_proj=show_proj,
                    show_wb=show_wb, show_h100=show_h100)

    if args.chart:
        print("Generating charts...")
        make_charts(entries, Path(args.out))


if __name__ == "__main__":
    main()
