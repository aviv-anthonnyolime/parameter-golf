#!/usr/bin/env bash
# Set up SSH-based git push on a remote training machine (RunPod / AWS).
# Usage: bash scripts/setup_git_ssh.sh
#
# What it does:
#   1. Generates an SSH key (if none exists)
#   2. Prints the public key for you to add to GitHub
#   3. Switches the origin remote from HTTPS → SSH
#   4. Tests the connection
#
# After running this script, add the printed public key to:
#   https://github.com/settings/ssh/new

set -euo pipefail

echo "=== Git SSH Setup ==="

# 1. Generate SSH key if needed
KEY="$HOME/.ssh/id_ed25519"
if [ -f "$KEY" ]; then
    echo "  SSH key already exists: $KEY"
else
    echo "  Generating SSH key..."
    mkdir -p "$HOME/.ssh"
    ssh-keygen -t ed25519 -N "" -f "$KEY" -q
    echo "  Created: $KEY"
fi

# 2. Print public key
echo ""
echo "=== Add this public key to GitHub ==="
echo "  → https://github.com/settings/ssh/new"
echo ""
cat "${KEY}.pub"
echo ""

# 3. Switch origin to SSH (if currently HTTPS)
CURRENT_URL=$(git remote get-url origin 2>/dev/null || echo "")
if [[ "$CURRENT_URL" == https://github.com/* ]]; then
    # Extract owner/repo from https://github.com/owner/repo.git
    SSH_URL=$(echo "$CURRENT_URL" | sed 's|https://github.com/|git@github.com:|')
    echo "  Switching origin: $CURRENT_URL → $SSH_URL"
    git remote set-url origin "$SSH_URL"
elif [[ "$CURRENT_URL" == git@github.com:* ]]; then
    echo "  Origin already uses SSH: $CURRENT_URL"
else
    echo "  WARNING: Unexpected remote URL: $CURRENT_URL"
    echo "  Please set it manually: git remote set-url origin git@github.com:<owner>/<repo>.git"
fi

# 4. Test connection
echo ""
echo "=== Testing GitHub SSH connection ==="
echo "  (If this is the first connection, type 'yes' to accept the host key)"
echo ""
ssh -T git@github.com 2>&1 || true

echo ""
echo "=== Done ==="
echo "  If you see 'Hi <user>! You've successfully authenticated', git push will work."
echo "  If not, make sure you added the public key above to https://github.com/settings/ssh/new"
