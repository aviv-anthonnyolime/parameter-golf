#!/usr/bin/env bash
# Setup script for Amazon Linux 2 instances (e.g. SageMaker g5dn.12xlarge).
#
# Problems solved:
#   - Default GCC 7 is too old for NumPy 2.x  → use gcc10-gcc / gcc10-g++
#   - Default CMake 2.8 is too old for sentencepiece/pyarrow  → install cmake via pip
#   - SageMaker volume (/home/.../SageMaker) is only 4.8GB, too small for
#     PyTorch + CUDA libs (~2GB+) and training data (~20GB+)
#     → venv and data both live on root / (135GB), symlinked into the repo
#   - wandb >= 0.18 requires Go (wandb-core) and Rust/Cargo (gpu_stats)
#   - AL2's binutils 2.29 is too old for Rust's ring crate (needs >= 2.30)
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
GO_VERSION="${GO_VERSION:-1.26.1}"
BINUTILS_VERSION="${BINUTILS_VERSION:-2.42}"
SKIP_INSTALL="${1:-}"

# ── 1. GCC 10 ────────────────────────────────────────────────────────────────
echo "[1/8] Setting GCC 10 as the active compiler..."
export CC=gcc10-gcc
export CXX=gcc10-g++

if ! command -v gcc10-gcc &>/dev/null; then
    echo "ERROR: gcc10-gcc not found. Install it with:"
    echo "  sudo yum install -y gcc10 gcc10-c++"
    exit 1
fi
echo "  CC  = $(gcc10-gcc --version | head -1)"
echo "  CXX = $(gcc10-g++ --version | head -1)"

# ── 2. Go (required by wandb-core) ──────────────────────────────────────────
echo "[2/8] Checking Go >= $GO_VERSION..."
NEED_GO=true
if command -v go &>/dev/null; then
    CURRENT_GO="$(go version | grep -oP 'go\K[0-9]+\.[0-9]+\.[0-9]+')"
    if printf '%s\n%s\n' "$GO_VERSION" "$CURRENT_GO" | sort -V -C; then
        echo "  Go $CURRENT_GO already installed (>= $GO_VERSION), skipping."
        NEED_GO=false
    else
        echo "  Go $CURRENT_GO is too old, upgrading..."
    fi
fi
if $NEED_GO; then
    echo "  Installing Go $GO_VERSION..."
    curl -sOL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"
    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf "go${GO_VERSION}.linux-amd64.tar.gz"
    rm -f "go${GO_VERSION}.linux-amd64.tar.gz"
    echo "  Go $GO_VERSION installed."
fi
export PATH=/usr/local/go/bin:$PATH
echo "  go = $(go version)"

# ── 3. Rust / Cargo (required by wandb gpu_stats) ───────────────────────────
echo "[3/8] Checking Rust / Cargo..."
if command -v cargo &>/dev/null; then
    echo "  cargo = $(cargo --version)"
else
    echo "  Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
    echo "  cargo = $(cargo --version)"
fi
export PATH="$HOME/.cargo/bin:$PATH"

# ── 4. Newer binutils (AL2 ships 2.29, ring crate needs >= 2.30) ────────────
echo "[4/8] Checking binutils (assembler)..."
NEED_BINUTILS=true
if command -v /usr/local/bin/as &>/dev/null; then
    LOCAL_AS_VER="$(/usr/local/bin/as --version | head -1 | grep -oP '[0-9]+\.[0-9]+')"
    if printf '%s\n2.30\n' "$LOCAL_AS_VER" | sort -V -C 2>/dev/null; then
        echo "  binutils $LOCAL_AS_VER already installed (>= 2.30), skipping."
        NEED_BINUTILS=false
    fi
fi
if $NEED_BINUTILS; then
    echo "  Building binutils $BINUTILS_VERSION from source (this takes a few minutes)..."
    sudo yum install -y texinfo &>/dev/null || true
    pushd /tmp >/dev/null
    curl -sOL "https://ftp.gnu.org/gnu/binutils/binutils-${BINUTILS_VERSION}.tar.gz"
    tar xzf "binutils-${BINUTILS_VERSION}.tar.gz"
    cd "binutils-${BINUTILS_VERSION}"
    ./configure --prefix=/usr/local --quiet
    make -j"$(nproc)" --quiet
    sudo make install --quiet
    cd /tmp && rm -rf "binutils-${BINUTILS_VERSION}" "binutils-${BINUTILS_VERSION}.tar.gz"
    popd >/dev/null
    echo "  binutils $BINUTILS_VERSION installed."
fi
export PATH=/usr/local/bin:$PATH
echo "  as = $(as --version | head -1)"

# ── 5. Virtual environment on root filesystem ─────────────────────────────────
echo "[5/8] Setting up venv at $VENV_DIR..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  Created new venv at $VENV_DIR"
else
    echo "  Venv already exists at $VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "  Active Python: $(python --version)"

# ── 6. Upgrade pip itself first to avoid old pip not understanding modern package metadata ─────────────────────────────────
echo "[6/8] Upgrading pip itself..."
pip install --upgrade pip

# ── 7. Modern CMake via pip ───────────────────────────────────────────────────
echo "[7/8] Installing a modern CMake via pip (system cmake 2.8 is too old)..."
pip install --quiet --cache-dir /tmp/pip-cache cmake
export PATH="$(python -c 'import cmake; import os; print(os.path.dirname(cmake.__file__))')/data/bin:$PATH"
echo "  cmake = $(cmake --version | head -1)"

# ── 8. Python dependencies ────────────────────────────────────────────────────
if [ "$SKIP_INSTALL" != "--skip-install" ]; then
    echo "[8/8] Installing Python dependencies (cache → /tmp/pip-cache)..."
    pip install --cache-dir /tmp/pip-cache -r requirements.txt
else
    echo "[8/8] Skipping pip install (--skip-install passed)."
fi

# ── 9. Data symlinks → root filesystem ───────────────────────────────────────
# The download script and train_gpt.py both default to ./data/datasets and
# ./data/tokenizers. We redirect those to /opt/pg-data via symlinks so the
# large binary files never touch the small SageMaker volume. No code changes.
echo "[9/9] Setting up data symlinks (actual data stored at $DATA_STORE)..."
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
