# Mnemon for Claude Code — Persistent Memory Across Sessions

Claude Code forgets everything when you close a session.
Your project context, your preferences, what you were
working on — gone. Every new session starts blind.

This gives Claude Code persistent memory using Mnemon.
After setup, every new session automatically loads:
- What project you're working on
- Your preferences
- What you were doing last session

## Install in one command:
```bash
bash setup.sh
```

## What happens after:
Every new Claude session prints:
```
[Mnemon] Session context loaded:
  project : your project description
  prefs   : your preferences
  last    : what you worked on last session
```

Claude knows what you're working on without you
explaining it every time.

## How it works:
`settings.json` hooks into Claude's session start via the `PreToolUse` hook.
`mnemon_session_init.py` loads your context from the Mnemon database on
the first tool call of each session. A lock file keyed to the Claude process
PID ensures it runs exactly once per session, not on every tool call.

## Files:
- `mnemon_session_init.py` — hook script, goes in `~/.claude/`
- `settings.json` — Claude Code hook config, goes in `~/.claude/`
- `setup.sh` — copies files into place

## Manual install:
```bash
cp mnemon_session_init.py ~/.claude/
cp settings.json ~/.claude/        # or merge the hook if settings.json exists
```

## Requirements:
- [Mnemon](https://github.com/smartass_4ever/mnemon) installed
- `mnemon_claude.py` present at `~/.claude/mnemon_claude.py`
- Claude Code CLI
