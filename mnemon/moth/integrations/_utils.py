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


def recall_as_context(m: "MnemonSync", query: str, top_k: int = 5) -> str:
    """
    Recall relevant memories and return as a formatted string ready to
    inject into any system prompt. Returns "" if nothing relevant found.
    """
    try:
        result = m.recall(query, top_k=top_k)
        memories: List[Dict] = result.get("memories", [])
        facts: Dict = result.get("facts", {})

        lines: List[str] = []

        if memories:
            lines.append("[Mnemon — context from previous sessions]")
            for mem in memories:
                content = mem.get("content", {})
                text = (
                    content.get("text")
                    or content.get("value")
                    or (str(content) if content and content != {} else None)
                )
                if text and text != "{}":
                    lines.append(f"- {text.strip()}")

        if facts:
            if not lines:
                lines.append("[Mnemon — known facts]")
            for key, val in facts.items():
                lines.append(f"- {key}: {val}")

        return "\n".join(lines)
    except Exception as e:
        logger.debug(f"Mnemon recall_as_context failed: {e}")
        return ""


def record_outcome(m: "MnemonSync", goal: str, outcome: str, importance: float = 0.65) -> None:
    """Record a task outcome to Mnemon. Never raises."""
    try:
        text = f"{goal[:120]} → {outcome[:400]}" if goal else outcome[:500]
        m.remember(text, importance=importance)
    except Exception as e:
        logger.debug(f"Mnemon record_outcome failed: {e}")


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


def extract_query(messages: List[Dict]) -> str:
    """Extract the most recent human/user message as the recall query."""
    try:
        for msg in reversed(messages):
            if isinstance(msg, dict):
                role = msg.get("role", "")
                if role in ("user", "human"):
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content[:300]
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                return block.get("text", "")[:300]
    except Exception:
        pass
    return ""


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
