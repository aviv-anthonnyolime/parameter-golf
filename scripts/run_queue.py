#!/usr/bin/env python3
"""
Queue runner: reads a YAML config with multiple runs and executes them
sequentially. After each run, commits results to git and pushes.

Usage:
    python scripts/run_queue.py experiments/phase1_UT/queue.yaml
    python scripts/run_queue.py queue.yaml --dry-run          # show commands only
    python scripts/run_queue.py queue.yaml --no-push          # commit but don't push

Example queue.yaml:
    script: experiments/phase1_UT/train_gpt_ut.py
    nproc: 4
    defaults:
      DATA_PATH: ./data/datasets/fineweb10B_sp1024
      TOKENIZER_PATH: ./data/tokenizers/fineweb_1024_bpe.model
      VOCAB_SIZE: "1024"
      MAX_WALLCLOCK_SECONDS: "2000"
      WANDB_ENABLED: "1"

    runs:
      - RUN_ID: ut_10L_d512
        NUM_LAYERS: "10"
        MODEL_DIM: "512"

      - RUN_ID: ut_10L_d640
        NUM_LAYERS: "10"
        MODEL_DIM: "640"

      - RUN_ID: ut_12L_d512
        NUM_LAYERS: "12"
        MODEL_DIM: "512"
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. pip install pyyaml")
    sys.exit(1)


def parse_latest_result(experiment_dir: Path):
    """Read the last line of results.jsonl in the experiment dir."""
    results_file = experiment_dir / "results.jsonl"
    if not results_file.exists():
        return None
    with open(results_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return None
    return json.loads(lines[-1].strip())


def build_torchrun_cmd(script: str, nproc: int, env_vars: dict):
    """Build the torchrun command."""
    cmd = f"torchrun --standalone --nproc_per_node={nproc} {script}"
    return cmd, env_vars


def git_commit_and_push(result: dict, push: bool = True):
    """Commit results and optionally push."""
    docker_name = result.get("docker_name", "unknown")
    params_tag = result.get("params_tag", "?")
    experiment = result.get("experiment", "?")
    val_loss = result.get("val_loss", 0)
    val_bpb = result.get("val_bpb_ttt", result.get("val_bpb", 0))

    commit_msg = (
        f"[{docker_name}] - [{params_tag}] - [{experiment}] - "
        f"(loss: {val_loss:.4f}) (val_bpb: {val_bpb:.4f})"
    )

    # Stage results and logs
    files_to_add = [
        "results/",
        "experiments/*/results.jsonl",
        "logs/",
    ]
    for pattern in files_to_add:
        subprocess.run(["git", "add", pattern], capture_output=True)

    # Check if there's anything to commit
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if not status.stdout.strip():
        print("  [git] Nothing to commit")
        return

    subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
    print(f"  [git] Committed: {commit_msg}")

    if push:
        push_result = subprocess.run(["git", "push"], capture_output=True, text=True)
        if push_result.returncode == 0:
            print("  [git] Pushed to remote")
        else:
            print(f"  [git] Push failed: {push_result.stderr.strip()}")


def run_single(script: str, nproc: int, env_vars: dict, run_index: int, total: int, dry_run: bool):
    """Execute a single training run."""
    run_id = env_vars.get("RUN_ID", f"run_{run_index}")
    cmd, env = build_torchrun_cmd(script, nproc, env_vars)

    print(f"\n{'='*60}")
    print(f"  Run {run_index}/{total}: {run_id}")
    print(f"{'='*60}")

    # Show env vars
    for k, v in sorted(env_vars.items()):
        print(f"  {k}={v}")
    print(f"  CMD: {cmd}")

    if dry_run:
        print("  [DRY RUN] Skipping execution")
        return None

    # Merge env
    full_env = os.environ.copy()
    full_env.update({k: str(v) for k, v in env_vars.items()})

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        shell=True,
        env=full_env,
        cwd=Path(script).resolve().parent.parent.parent,  # repo root
    )
    elapsed = time.time() - t0

    print(f"\n  Finished in {elapsed:.0f}s (exit code: {proc.returncode})")

    if proc.returncode != 0:
        print(f"  WARNING: Run {run_id} failed with exit code {proc.returncode}")
        return None

    return run_id


def main():
    parser = argparse.ArgumentParser(description="Run a queue of training experiments")
    parser.add_argument("queue_file", help="Path to YAML queue file")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    parser.add_argument("--no-push", action="store_true", help="Commit but don't push to remote")
    args = parser.parse_args()

    queue_path = Path(args.queue_file)
    if not queue_path.exists():
        print(f"Queue file not found: {queue_path}")
        sys.exit(1)

    def load_queue():
        with open(queue_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    config = load_queue()
    script = config.get("script", "experiments/phase1_UT/train_gpt_ut.py")
    total_runs = len(config.get("runs", []))

    if total_runs == 0:
        print("No runs defined in queue file.")
        sys.exit(1)

    print(f"Queue: {total_runs} runs (re-reads YAML before each run)")
    print(f"Script: {script}")
    print(f"GPUs: {config.get('nproc', 4)}")

    # Resolve experiment dir from script path
    experiment_dir = Path(script).resolve().parent

    i = 0
    completed = 0
    while True:
        # Re-read YAML before each run to pick up live edits
        config = load_queue()
        script = config.get("script", "experiments/phase1_UT/train_gpt_ut.py")
        nproc = config.get("nproc", 4)
        defaults = config.get("defaults", {})
        runs = config.get("runs", [])

        if i >= len(runs):
            break

        run_overrides = runs[i]
        env_vars = {**defaults}
        env_vars.update(run_overrides)

        print(f"\n  [YAML reloaded — {len(runs)} runs total, starting run {i+1}]")

        run_id = run_single(script, nproc, env_vars, i + 1, len(runs), args.dry_run)

        if run_id is not None and not args.dry_run:
            result = parse_latest_result(experiment_dir)
            if result:
                git_commit_and_push(result, push=not args.no_push)
            else:
                print("  WARNING: Could not find result entry for git commit")

        i += 1
        completed += 1

    print(f"\n{'='*60}")
    print(f"  Queue complete! {completed} runs finished.")
    print(f"{'='*60}\n")

    # Show final ranking
    results_file = experiment_dir / "results.jsonl"
    if results_file.exists() and not args.dry_run:
        print("Final ranking:")
        subprocess.run([sys.executable, "scripts/ranking.py", str(results_file)])


if __name__ == "__main__":
    main()
