"""
Shared utilities for all moth integrations.
All functions are fail-safe — they never raise.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mnemon import MnemonSync

logger = logging.getLogger(__name__)


def recall_as_context(m: "MnemonSync", query: str, top_k: int = 5, source: str = "") -> str:
    """
    Recall relevant memories and return as a formatted string ready to
    inject into any system prompt. Returns "" if nothing relevant found.
    Tracks injection/gate events in m._stats when source is provided.
    """
    try:
        result = m.recall(query, top_k=top_k)
        return _format_recall_result(result, m, query, source)
    except Exception as e:
        logger.debug(f"Mnemon recall_as_context failed: {e}")
        return ""


async def recall_as_context_async(m: Any, query: str, top_k: int = 5, source: str = "") -> str:
    """
    Async variant — awaits m._m.recall() directly so it never calls
    run_until_complete() inside an already-running event loop.
    Drop-in replacement for recall_as_context in all async patch paths.
    """
    try:
        result = await m._m.recall(query, top_k=top_k)
        return _format_recall_result(result, m, query, source)
    except Exception as e:
        logger.debug(f"Mnemon recall_as_context_async failed: {e}")
        return ""


def _format_recall_result(result: Dict, m: Any, query: str, source: str) -> str:
    memories: List[Dict] = result.get("memories", [])
    facts: Dict = result.get("facts", {})

    lines: List[str] = []

    if memories:
        lines.append("[Mnemon — context from previous sessions]")
        for mem in memories:
            # recall() returns {text: ...} at top level; compressed_context uses {content: {text: ...}}
            text = (
                mem.get("text")
                or mem.get("content", {}).get("text")
                or mem.get("content", {}).get("value")
            )
            if text:
                lines.append(f"- {text.strip()}")

    if facts:
        if not lines:
            lines.append("[Mnemon — known facts]")
        for key, val in facts.items():
            lines.append(f"- {key}: {val}")

    context = "\n".join(lines)

    if source and hasattr(m, "_stats") and m._stats is not None:
        if context:
            m._stats.record_injection(source, query, context)
        elif query:
            m._stats.record_gate(source, query)

    return context


def record_outcome(m: "MnemonSync", goal: str, outcome: str, importance: float = 0.65) -> None:
    """Record a task outcome to Mnemon. Never raises."""
    try:
        text = f"{goal[:120]} -> {outcome[:400]}" if goal else outcome[:500]
        m.remember(text, importance=importance)
    except Exception as e:
        logger.debug(f"Mnemon record_outcome failed: {e}")
    try:
        if goal and hasattr(m, "_stats") and m._stats is not None:
            m._stats.record_query(goal)
    except Exception:
        pass


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
    Extract the intent signal for protein bond recall.

    Uses system prompt + last user message together — the system prompt
    describes what the agent is trying to do (the real intent), while the
    last user message is the specific task. Together they give protein bond
    the full picture so only genuinely relevant memories bond.

    Returns "" if no meaningful signal can be extracted — which means
    recall returns nothing and nothing gets injected. No context rot.
    """
    parts: List[str] = []

    # System prompt carries the agent's intent — highest signal
    if system and isinstance(system, str):
        parts.append(system[:200])

    # Last user message is the specific task
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


def track_cache_hit(m: Any, source: str, tokens: Optional[int] = None) -> None:
    """Record a cache hit to MothStats. Never raises."""
    try:
        if hasattr(m, "_stats") and m._stats is not None:
            m._stats.record_hit(source, tokens)
    except Exception:
        pass


def inject_into_system(existing_system: Optional[str], context: str) -> str:
    """Prepend Mnemon context to a system prompt."""
    if not context:
        return existing_system or ""
    if existing_system:
        return f"{context}\n\n{existing_system}"
    return context


def inject_into_openai_messages(messages: List[Dict], context: str) -> List[Dict]:
    """Inject context into OpenAI-style messages list as a system message."""
    if not context:
        return messages
    messages = list(messages)
    if messages and messages[0].get("role") == "system":
        messages[0] = {
            **messages[0],
            "content": f"{context}\n\n{messages[0]['content']}",
        }
    else:
        messages.insert(0, {"role": "system", "content": context})
    return messages
