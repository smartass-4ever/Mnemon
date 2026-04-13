"""
mnemon_session_init.py — runs once per Claude session via PreToolUse hook.

Uses a temp file keyed by parent PID to ensure it only fires once,
not on every tool call.
"""

import asyncio
import os
import sys
import json
import tempfile

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

MNEMON_SCRIPT = r"C:\Users\MAHIKA\.claude\mnemon_claude.py"
GUARD_DIR     = tempfile.gettempdir()


def guard_path():
    # Parent PID = the Claude Code process — unique per session
    ppid = os.getppid()
    return os.path.join(GUARD_DIR, f"mnemon_session_{ppid}.lock")


def already_ran():
    return os.path.exists(guard_path())


def mark_ran():
    with open(guard_path(), "w") as f:
        f.write(str(os.getpid()))


if __name__ == "__main__":
    if already_ran():
        sys.exit(0)   # silent — already loaded this session

    mark_ran()

    cwd = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

    # Run session-start via subprocess to keep import path clean
    import subprocess
    result = subprocess.run(
        [sys.executable, MNEMON_SCRIPT, "session-start", cwd],
        capture_output=True, text=True, encoding="utf-8"
    )

    if result.returncode == 0 and result.stdout.strip():
        try:
            data = json.loads(result.stdout)
            print("\n[Mnemon] Session context loaded:")
            if data.get("project_facts"):
                print(f"  project : {data['project_facts'][:100]}")
            if data.get("user_preferences"):
                print(f"  prefs   : {data['user_preferences']}")
            if data.get("recent_session"):
                print(f"  last    : {data['recent_session'][:100]}")
        except Exception:
            print(result.stdout)
