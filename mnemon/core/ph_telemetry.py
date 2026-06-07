"""
Anonymous opt-out telemetry — PostHog.
Disable by setting MNEMON_NO_TELEMETRY=1 in the environment.

What is sent:  framework, cache_level, tokens_saved, latency_ms, mnemon_version, python_version.
What is never sent: prompts, goals, content, API keys, file paths, user identity.

Runs in a daemon thread — never blocks, never raises.
"""

import hashlib
import json
import os
import sys
import threading
import urllib.request
from typing import List

_KEY = os.environ.get(
    "MNEMON_POSTHOG_KEY",
    "phc_kgVmx3ixuEj5qQ2zDKniQrWxeRj4dfurm79LrDLce8eQ",
)
_URL = "https://us.i.posthog.com/capture/"
_ENABLED = os.environ.get("MNEMON_NO_TELEMETRY", "").strip() != "1"


def _anon_id() -> str:
    """Stable anonymous ID — hostname hash. Not reversible to user identity."""
    try:
        import socket
        seed = socket.gethostname()
    except Exception:
        seed = "unknown"
    return "mnemon-" + hashlib.sha256(seed.encode()).hexdigest()[:16]


def _fire(event: str, props: dict, cli: bool = False) -> None:
    if not _ENABLED:
        return
    def _post():
        try:
            from mnemon.core.models import MNEMON_VERSION
            payload = json.dumps({
                "api_key": _KEY,
                "event": event,
                "distinct_id": _anon_id(),
                "properties": {
                    **props,
                    "mnemon_version": MNEMON_VERSION,
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                    "$lib": "mnemon",
                },
            }).encode()
            req = urllib.request.Request(
                _URL, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            pass
    threading.Thread(target=_post, daemon=True).start()


def track_hit(framework: str, cache_level: str, tokens_saved: int, latency_ms: float) -> None:
    _fire("cache_hit", {
        "framework": framework,
        "cache_level": cache_level,
        "tokens_saved": tokens_saved,
        "latency_ms": round(latency_ms, 1),
    })


def track_miss(framework: str) -> None:
    _fire("cache_miss", {"framework": framework})


def track_init(frameworks: List[str]) -> None:
    _fire("init", {"frameworks": frameworks})


def track_session_end(
    session_hits: int,
    session_runs: int,
    tokens_saved: int,
    lifetime_hits: int,
    lifetime_runs: int,
    frameworks: List[str],
) -> None:
    hit_rate = round(session_hits / session_runs, 3) if session_runs > 0 else 0.0
    _fire("session_end", {
        "session_hits":   session_hits,
        "session_runs":   session_runs,
        "session_hit_rate": hit_rate,
        "tokens_saved":   tokens_saved,
        "lifetime_hits":  lifetime_hits,
        "lifetime_runs":  lifetime_runs,
        "had_any_hit":    session_hits > 0,
        "frameworks":     frameworks,
    })
