#!/usr/bin/env bash
# Kill all training processes and free GPU resources.
# Usage: bash scripts/kill_training.sh
#
# Safe to run multiple times. Won't error if nothing is running.

set -euo pipefail

echo "=== Killing training processes ==="

# 1. Kill the queue runner first (parent), then torchrun, then stragglers
for pattern in "run_queue.py" "torchrun" "torch.distributed" "train_gpt"; do
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  Killing $pattern (PIDs: $(echo $pids | tr '\n' ' '))"
        kill $pids 2>/dev/null || true
    fi
done

# 2. Wait briefly for graceful shutdown
sleep 2

# 3. Force-kill anything still alive
for pattern in "run_queue.py" "torchrun" "torch.distributed" "train_gpt"; do
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  Force-killing $pattern (PIDs: $(echo $pids | tr '\n' ' '))"
        kill -9 $pids 2>/dev/null || true
    fi
done

# 4. Clean up any orphaned NCCL shared memory
rm -f /dev/shm/nccl-* 2>/dev/null || true

# 5. Wait for GPUs to release
sleep 2

# 6. Show GPU state
echo ""
echo "=== GPU status ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null || true
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    if [ "$gpu_procs" -gt 0 ]; then
        echo "  WARNING: $gpu_procs process(es) still on GPU. Run again or reboot."
    else
        echo "  All GPUs free."
    fi
else
    echo "  nvidia-smi not found (no GPU?)"
fi

echo ""
echo "=== Done ==="
