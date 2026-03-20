#!/usr/bin/env python3
"""
Promote best local experiment runs to a RunPod-ready YAML queue.

Reads results.jsonl, picks the top N runs by val_bpb, and generates
a queue.yaml ready for 8xH100 execution.

Usage:
    python scripts/promote.py --top 3
    python scripts/promote.py --top 5 --from experiments/phase1_UT/results.jsonl
    python scripts/promote.py --top 3 --nproc 8 --wallclock 600
    python scripts/promote.py --top 3 --out runpod_queue.yaml
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. pip install pyyaml")
    sys.exit(1)


def load_results(path: Path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Promote best runs to RunPod queue")
    parser.add_argument("--top", type=int, default=3, help="Number of top runs to promote")
    parser.add_argument("--from", dest="results_file", default=None, help="Path to results.jsonl")
    parser.add_argument("--nproc", type=int, default=8, help="Number of GPUs for RunPod (default: 8)")
    parser.add_argument("--wallclock", type=int, default=600, help="Max wallclock seconds (default: 600)")
    parser.add_argument("--out", default=None, help="Output YAML path")
    parser.add_argument("--batch-tokens", type=int, default=524288, help="Train batch tokens for RunPod")
    args = parser.parse_args()

    # Find results file
    if args.results_file:
        results_path = Path(args.results_file)
    else:
        results_path = Path("results/all_runs.jsonl")

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        sys.exit(1)

    entries = load_results(results_path)
    if not entries:
        print("No results found.")
        sys.exit(1)

    # Sort by val_bpb_ttt (best first)
    entries.sort(key=lambda e: e.get("val_bpb_ttt", e.get("val_bpb", 99)))
    top_entries = entries[:args.top]

    print(f"\nTop {len(top_entries)} runs to promote:")
    for i, e in enumerate(top_entries, 1):
        print(
            f"  {i}. {e.get('docker_name', '?')} - {e.get('params_tag', '?')} "
            f"- val_bpb_ttt: {e.get('val_bpb_ttt', '?')}"
        )

    # Determine script from first entry
    first_exp = top_entries[0].get("experiment", "phase1_UT")
    # Try to find the script
    script_candidates = [
        f"experiments/{first_exp}/train_gpt_ut.py",
        f"experiments/{first_exp}/train_gpt.py",
    ]
    script = next((s for s in script_candidates if Path(s).exists()), script_candidates[0])

    # Build YAML queue
    queue = {
        "script": script,
        "nproc": args.nproc,
        "defaults": {
            "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
            "VOCAB_SIZE": "1024",
            "MAX_WALLCLOCK_SECONDS": str(args.wallclock),
            "TRAIN_BATCH_TOKENS": str(args.batch_tokens),
            "WANDB_ENABLED": "1",
        },
        "runs": [],
    }

    # Key hyperparams to extract per run
    hp_keys = [
        "NUM_LAYERS", "NUM_UNIQUE_LAYERS", "MODEL_DIM", "NUM_HEADS",
        "NUM_KV_HEADS", "MLP_MULT", "EMBED_LR", "HEAD_LR", "TIED_EMBED_LR",
        "MATRIX_LR", "SCALAR_LR", "MUON_MOMENTUM", "WARMUP_STEPS",
        "WARMDOWN_ITERS", "TRAIN_SEQ_LEN", "QK_GAIN_INIT", "LOGIT_SOFTCAP",
        "TTT_LORA_RANK", "TTT_LORA_LR",
    ]

    for e in top_entries:
        hp = e.get("hyperparams", {})
        run_env = {
            "RUN_ID": f"runpod_{e.get('docker_name', 'unknown')}",
        }
        # Copy relevant hyperparams (env vars are uppercase, hp dict may have lowercase)
        for key in hp_keys:
            lower_key = key.lower()
            if lower_key in hp:
                val = hp[lower_key]
                # Skip if same as what's probably default
                run_env[key] = str(val)

        queue["runs"].append(run_env)

    # Output
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(f"experiments/{first_exp}/queue_runpod.yaml")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(queue, f, default_flow_style=False, sort_keys=False)

    print(f"\nGenerated RunPod queue: {out_path}")
    print(f"  {len(queue['runs'])} runs, {args.nproc} GPUs, {args.wallclock}s wallclock")

    # Also print ready-to-paste commands
    print(f"\nReady-to-run commands:")
    print(f"  # On RunPod (8xH100):")
    print(f"  cd /workspace/parameter-golf")
    print(f"  python scripts/run_queue.py {out_path}")
    print()

    # Individual commands for manual use
    print("  # Or run individually:")
    for run_env in queue["runs"]:
        env_str = " ".join(f"{k}={v}" for k, v in run_env.items())
        defaults_str = " ".join(f"{k}={v}" for k, v in queue["defaults"].items())
        print(f"  {defaults_str} {env_str} torchrun --standalone --nproc_per_node={args.nproc} {script}")
    print()


if __name__ == "__main__":
    main()
