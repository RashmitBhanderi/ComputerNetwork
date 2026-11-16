#!/bin/bash
# One-shot setup on the Mac: installs dependencies, tests every connection, then
# installs the background service. Run from anywhere:  bash scripts/setup_mac.sh
set -euo pipefail
cd "$(dirname "$0")/.."
command -v claude >/dev/null || { echo "Claude Code CLI missing. Install: curl -fsSL https://claude.ai/install.sh | bash   then run: claude   and log in."; exit 1; }
nc -z 127.0.0.1 11111 2>/dev/null || { echo "OpenD is not reachable on port 11111. Open moomoo OpenD, log in, click Unlock Trade, then rerun."; exit 1; }
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
echo "--- connection check (no orders) ---"
.venv/bin/python -m bot check
echo
read -r -p "Everything above look right? Type LIVE to install the service and start trading real money: " ans
[ "$ans" = "LIVE" ] || { echo "Stopped. Nothing installed."; exit 0; }
bash scripts/install_mac.sh
