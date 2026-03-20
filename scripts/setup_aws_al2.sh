#!/usr/bin/env bash
# Setup script for Amazon Linux 2 instances (e.g. SageMaker g5dn.12xlarge).
#
# Problems solved:
#   - Default GCC 7 is too old for NumPy 2.x  → use gcc10-gcc / gcc10-g++
#   - Default CMake 2.8 is too old for sentencepiece/pyarrow  → install cmake via pip
#   - SageMaker volume (/home/.../SageMaker) is only 4.8GB, too small for
#     PyTorch + CUDA libs (~2GB+) and training data (~20GB+)
#     → venv and data both live on root / (135GB), symlinked into the repo
#
# Usage:
#   First run:      bash scripts/setup_aws_al2.sh
#   After restart:  source /opt/pg-venv/bin/activate
#                   bash scripts/setup_aws_al2.sh --skip-install
#
# Re-run after every instance restart (exports don't persist across sessions).

set -e

VENV_DIR="${VENV_DIR:-/opt/pg-venv}"
DATA_STORE="${DATA_STORE:-/opt/pg-data}"
SKIP_INSTALL="${1:-}"

# ── 1. GCC 10 ────────────────────────────────────────────────────────────────
echo "[1/5] Setting GCC 10 as the active compiler..."
export CC=gcc10-gcc
export CXX=gcc10-g++

if ! command -v gcc10-gcc &>/dev/null; then
    echo "ERROR: gcc10-gcc not found. Install it with:"
    echo "  sudo yum install -y gcc10 gcc10-c++"
    exit 1
fi
echo "  CC  = $(gcc10-gcc --version | head -1)"
echo "  CXX = $(gcc10-g++ --version | head -1)"

# ── 2. Virtual environment on root filesystem ─────────────────────────────────
echo "[2/5] Setting up venv at $VENV_DIR..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  Created new venv at $VENV_DIR"
else
    echo "  Venv already exists at $VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "  Active Python: $(python --version)"

# ── 3. Modern CMake via pip ───────────────────────────────────────────────────
echo "[3/5] Installing a modern CMake via pip (system cmake 2.8 is too old)..."
pip install --quiet --cache-dir /tmp/pip-cache cmake
export PATH="$(python -c 'import cmake; import os; print(os.path.dirname(cmake.__file__))')/data/bin:$PATH"
echo "  cmake = $(cmake --version | head -1)"

# ── 4. Python dependencies ────────────────────────────────────────────────────
if [ "$SKIP_INSTALL" != "--skip-install" ]; then
    echo "[4/5] Installing Python dependencies (cache → /tmp/pip-cache)..."
    pip install --cache-dir /tmp/pip-cache -r requirements.txt
else
    echo "[4/5] Skipping pip install (--skip-install passed)."
fi

# ── 5. Data symlinks → root filesystem ───────────────────────────────────────
# The download script and train_gpt.py both default to ./data/datasets and
# ./data/tokenizers. We redirect those to /opt/pg-data via symlinks so the
# large binary files never touch the small SageMaker volume. No code changes.
echo "[5/5] Setting up data symlinks (actual data stored at $DATA_STORE)..."
mkdir -p "$DATA_STORE/datasets" "$DATA_STORE/tokenizers"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    REPO_ROOT="$SCRIPT_DIR"            # script is at repo root
elif [ -f "$SCRIPT_DIR/../requirements.txt" ]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"  # script is inside scripts/
else
    echo "ERROR: Cannot find repo root. Run from the repo root or scripts/ directory."
    exit 1
fi
REPO_DATA_DIR="$REPO_ROOT/data"

for subdir in datasets tokenizers; do
    target="$REPO_DATA_DIR/$subdir"
    store="$DATA_STORE/$subdir"
    if [ -L "$target" ]; then
        echo "  Symlink already exists: $target → $(readlink "$target")"
    elif [ -d "$target" ]; then
        echo "  Moving existing $subdir to $store and creating symlink..."
        cp -a "$target/." "$store/"
        rm -rf "$target"
        ln -s "$store" "$target"
        echo "  $target → $store"
    else
        ln -s "$store" "$target"
        echo "  $target → $store"
    fi
done

echo ""
echo "Done! Environment is ready."
echo ""
echo "Download data with:"
echo "  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10"
echo ""
echo "To activate this environment in a new terminal:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "NOTE: CC, CXX, and PATH exports are only active in this shell session."
echo "If you open a new terminal, re-run:  bash scripts/setup_aws_al2.sh --skip-install"
