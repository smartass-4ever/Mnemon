"""
Shared utilities for all moth integrations.
All functions are fail-safe — they never raise.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mnemon import MnemonSync

logger = logging.getLogger(__name__)


def prompt_hash(messages: List[Dict], system: Optional[str], model: str) -> str:
    """Stable hash for an LLM call — used for System 1 client-level cache."""
    try:
        key = json.dumps(
            {"m": messages, "s": system or "", "model": model},
            sort_keys=True,
            default=str,
        )
        return hashlib.md5(key.encode()).hexdigest()
    except Exception:
        return hashlib.md5(str(messages).encode()).hexdigest()


def extract_query(messages: List[Dict], system: Optional[str] = None) -> str:
    """
    Extract the intent signal for EME semantic matching.
    Uses system prompt + last user message together — the system prompt
    describes what the agent is trying to do, the last user message is the task.
    Returns "" if no meaningful signal can be extracted.
    """
    parts: List[str] = []

    if system and isinstance(system, str):
        parts.append(system[:200])

    try:
        for msg in reversed(messages):
            if isinstance(msg, dict):
                role = msg.get("role", "")
                if role in ("user", "human"):
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        parts.append(content[:200])
                        break
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "").strip()
                                if text:
                                    parts.append(text[:200])
                                    break
                        break
    except Exception:
        pass

    return " ".join(parts).strip()[:350]


def track_cache_hit(
    m: Any,
    source: str,
    tokens: Optional[int] = None,
    model: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> None:
    """Record a cache hit to MothStats and Bus. Never raises."""
    try:
        if hasattr(m, "_stats") and m._stats is not None:
            m._stats.record_hit(source, tokens, model, input_tokens, output_tokens)
    except Exception:
        pass
    # Notify Bus so it has data for pattern detection even on MOTH-intercepted calls
    try:
        inner = getattr(m, "_m", None)
        if inner and getattr(inner, "_bus", None):
            task_id = f"moth:{source}:{time.time():.0f}"
            m._run(inner._bus.record_outcome(
                task_id=task_id, task_type=source,
                outcome="success", latency_ms=0.0,
            ))
    except Exception:
        pass
