#!/bin/bash
# Installs the bot as a macOS background service that starts at login, restarts
# if it crashes, and keeps the Mac awake while it runs. Run from the optionbot folder:
#   bash scripts/install_mac.sh            (install / reinstall)
#   bash scripts/install_mac.sh uninstall
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.optionbot"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
if [ "${1:-}" = "uninstall" ]; then rm -f "$PLIST"; echo "optionbot service removed"; exit 0; fi

[ -x "$DIR/.venv/bin/python" ] || { echo "Run the setup steps in README.md first (.venv missing)"; exit 1; }
CLAUDE_DIR="$(dirname "$(command -v claude || echo /usr/local/bin/claude)")"
mkdir -p "$DIR/data" "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/caffeinate</string><string>-i</string>
    <string>$DIR/.venv/bin/python</string><string>-m</string><string>bot</string><string>run</string>
  </array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>$CLAUDE_DIR:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    <key>HOME</key><string>$HOME</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$DIR/data/service.log</string>
  <key>StandardErrorPath</key><string>$DIR/data/service.log</string>
</dict></plist>
PL
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "optionbot service installed and running. Log: $DIR/data/service.log"
