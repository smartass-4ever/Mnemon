"""
Anonymous opt-in telemetry — PostHog.
Enabled by setting MNEMON_TELEMETRY=1 in the environment.

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

_KEY = "phc_kgVmx3ixuEj5qQ2zDKniQrWxeRj4dfurm79LrDLce8eQ"
_URL = "https://us.i.posthog.com/capture/"
_ENABLED = os.environ.get("MNEMON_TELEMETRY", "").strip() == "1"
_CLI_ENABLED = os.environ.get("MNEMON_NO_TELEMETRY", "").strip() != "1"  # CLI always on unless opted out


def _anon_id() -> str:
    """Stable anonymous ID — hostname hash. Not reversible to user identity."""
    try:
        import socket
        seed = socket.gethostname()
    except Exception:
        seed = "unknown"
    return "mnemon-" + hashlib.sha256(seed.encode()).hexdigest()[:16]


def _fire(event: str, props: dict, cli: bool = False) -> None:
    if cli and not _CLI_ENABLED:
        return
    if not cli and not _ENABLED:
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
