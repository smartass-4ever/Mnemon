#!/usr/bin/env bash
# setup.sh — install Mnemon persistent memory for Claude Code
# Run once from the examples/claude_code_memory directory

set -e

CLAUDE_DIR="$HOME/.claude"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing Mnemon Claude Code memory integration..."

# Copy session init script
cp "$SCRIPT_DIR/mnemon_session_init.py" "$CLAUDE_DIR/mnemon_session_init.py"
echo "  Copied mnemon_session_init.py -> $CLAUDE_DIR/"

# Merge settings.json hook into existing settings (or create new)
SETTINGS="$CLAUDE_DIR/settings.json"
if [ -f "$SETTINGS" ]; then
    echo "  $SETTINGS already exists — merge the PreToolUse hook manually:"
    echo ""
    cat "$SCRIPT_DIR/settings.json"
    echo ""
    echo "  Add the above PreToolUse block to your existing settings.json."
else
    cp "$SCRIPT_DIR/settings.json" "$SETTINGS"
    echo "  Created $SETTINGS"
fi

echo ""
echo "Done. Start a new Claude Code session to verify:"
echo "  [Mnemon] Session context loaded:"
echo "    project : ..."
echo "    prefs   : ..."
echo "    last    : ..."
