#!/usr/bin/env bash
# ============================================================
# Baseline run — 50-minute wallclock, 4×A10G
# Log every 20 steps | Val every 250 steps
# Expected: ~5,200 steps at ~548ms/step
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/train_gpt.py" ]]; then
    REPO_ROOT="$SCRIPT_DIR"
elif [[ -f "$SCRIPT_DIR/../train_gpt.py" ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    echo "ERROR: Cannot find train_gpt.py from $SCRIPT_DIR" && exit 1
fi
cd "$REPO_ROOT"

# ---------- logging ----------
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="$REPO_ROOT/logs"
LOG_FILE="$LOG_DIR/baseline_50min_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "  Baseline 50-min run"
echo "  Start : $(date '+%H:%M:%S')"
echo "  ETA   : $(date -d '+50 minutes' '+%H:%M:%S' 2>/dev/null || date -v+50M '+%H:%M:%S')"
echo "  Log   : $LOG_FILE"
echo "=================================================="

# ---------- hyperparameters ----------
export MAX_WALLCLOCK_SECONDS=2900     # 48min 20s — buffer for final eval + save
export ITERATIONS=20000               # wallclock will cap before this
export WARMDOWN_ITERS=500             # ~274s warmdown at 548ms/step (last ~9%)
export WARMUP_STEPS=20

export TRAIN_LOG_EVERY=20             # print train loss every 20 steps
export VAL_LOSS_EVERY=250             # validate every 250 steps

# Baseline model shape (unchanged from repo defaults)
export NUM_LAYERS=9
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=2
export VOCAB_SIZE=1024

# Optimizer defaults
export EMBED_LR=0.6
export HEAD_LR=0.008
export TIED_EMBED_LR=0.05
export MATRIX_LR=0.04
export SCALAR_LR=0.04
export MUON_MOMENTUM=0.95

export SEED=1337
export RUN_ID="baseline_50min_${TIMESTAMP}"

# ---------- run ----------
echo "" | tee -a "$LOG_FILE"
echo "[ENV] MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS" | tee -a "$LOG_FILE"
echo "[ENV] WARMDOWN_ITERS=$WARMDOWN_ITERS" | tee -a "$LOG_FILE"
echo "[ENV] TRAIN_LOG_EVERY=$TRAIN_LOG_EVERY" | tee -a "$LOG_FILE"
echo "[ENV] VAL_LOSS_EVERY=$VAL_LOSS_EVERY" | tee -a "$LOG_FILE"
echo "[ENV] SEED=$SEED" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_gpt.py \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=================================================="
echo "  Done : $(date '+%H:%M:%S')"
echo "  Log  : $LOG_FILE"
echo "=================================================="
