"""
Shared utilities for all moth integrations.
All functions are fail-safe — they never raise.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from mnemon import MnemonSync

logger = logging.getLogger(__name__)

# ── Heuristic importance signals ─────────────────────────────────────────────
# Ordered by priority. First match wins for importance score.
_HEURISTIC_RULES: List[Tuple[re.Pattern, float]] = [
    # Explicit preferences / standing instructions — highest priority
    (re.compile(
        r"\b(prefer|always|never|must\s+not|must|should\s+not|should|require[sd]?|"
        r"expect[sd]?|want[sd]?\s+to|like[sd]?\s+to|insist[sd]?)\b", re.I), 0.92),
    # Avoidance / constraints
    (re.compile(
        r"\b(avoid|don'?t|do\s+not|cannot|can'?t|won'?t|will\s+not|"
        r"not\s+allowed|prohibited|forbidden|blocked)\b", re.I), 0.90),
    # Decisions and confirmed choices
    (re.compile(
        r"\b(decided|chose|selected|confirmed|agreed|will\s+use|going\s+to\s+use|"
        r"adopted|switched\s+to|migrated\s+to|settled\s+on)\b", re.I), 0.87),
    # Named entities with factual verbs — specific facts worth keeping
    (re.compile(
        r"\b(uses?|using|runs?\s+on|is\s+built\s+on|is\s+hosted|"
        r"is\s+called|is\s+named|is\s+located|works?\s+at|reports?\s+to)\b", re.I), 0.80),
    # General factual statements
    (re.compile(r"\b(is|are|was|were|has|have|had|contains?|includes?|provides?)\b", re.I), 0.74),
]

_MIN_SENTENCE_CHARS = 20   # skip fragments too short to be meaningful
_HEURISTIC_THRESHOLD = 0.74  # only store sentences that match at least a general fact


def _heuristic_extract(goal: str, outcome: str) -> List[Tuple[str, float]]:
    """
    Split outcome into sentences and score each for long-term importance.
    Returns [(memory_text, importance)] — always includes a fallback raw episode.
    Never raises.
    """
    results: List[Tuple[str, float]] = []
    seen: set = set()

    try:
        # Split on sentence boundaries (handles ". ", "! ", "? ", newlines)
        sentences = re.split(r'(?<=[.!?])\s+|\n+', outcome)
        goal_prefix = goal[:80].strip() if goal else ""

        for sent in sentences:
            sent = sent.strip().rstrip(".!?,;")
            if len(sent) < _MIN_SENTENCE_CHARS:
                continue

            best_imp = 0.0
            for pattern, imp in _HEURISTIC_RULES:
                if pattern.search(sent):
                    best_imp = max(best_imp, imp)
                    if best_imp >= 0.90:
                        break  # can't do better — short-circuit

            if best_imp >= _HEURISTIC_THRESHOLD:
                text = f"{goal_prefix}: {sent}" if goal_prefix else sent
                text = text[:500]
                key = text.lower()[:80]
                if key not in seen:
                    seen.add(key)
                    results.append((text, best_imp))

    except Exception:
        pass

    # Always store a raw episodic record at lower importance.
    # Individual extracted sentences supersede this for retrieval,
    # but the episode gives temporal/contextual grounding.
    try:
        raw = f"{goal[:120]} → {outcome[:350]}" if goal else outcome[:450]
        raw_key = raw.lower()[:80]
        if raw_key not in seen:
            results.append((raw, 0.55))
    except Exception:
        pass

    return results


# ── Background LLM extraction ─────────────────────────────────────────────────

_EXTRACT_PROMPT = """\
You are a memory extraction assistant. Given an AI interaction, extract facts, \
preferences, decisions, and standing rules worth remembering across future sessions.

Rules:
- Only extract things that would be useful in a future, unrelated session
- Skip generic task outputs, step-by-step plans, and boilerplate
- Each memory must be a complete, standalone sentence (no pronouns like "it" or "they")
- importance: 0.92 for preferences/rules, 0.87 for decisions, 0.78 for specific facts

Goal: {goal}
Response: {outcome}

Return a JSON array only — no other text:
[{{"text": "...", "importance": 0.0}}]
If nothing is worth remembering, return [].
"""

_LLM_EXTRACT_MIN_CHARS = 80   # don't bother extracting from very short responses
_LLM_EXTRACT_MAX_MEMORIES = 8  # cap per interaction


async def _llm_extract_and_store(m: Any, goal: str, outcome: str) -> None:
    """
    Background LLM extraction. Fire-and-forget — never awaited by callers.
    Uses the cheapest configured model. Completely fail-safe.
    """
    try:
        llm = getattr(getattr(m, "_m", None), "_llm", None)
        if llm is None:
            return
        if len(outcome) < _LLM_EXTRACT_MIN_CHARS:
            return

        prompt = _EXTRACT_PROMPT.format(
            goal=goal[:300],
            outcome=outcome[:800],
        )

        # Use the cheapest model — this is a background auxiliary call
        raw = await llm.complete(prompt, model=None, max_tokens=500, temperature=0.0)

        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        items = json.loads(clean)
        if not isinstance(items, list):
            return

        for item in items[:_LLM_EXTRACT_MAX_MEMORIES]:
            text = str(item.get("text", "")).strip()
            imp = float(item.get("importance", 0.78))
            if text and len(text) >= _MIN_SENTENCE_CHARS:
                await m._m.remember(text, importance=min(imp, 0.95))

    except Exception as e:
        logger.debug(f"Mnemon LLM extraction failed (non-critical): {e}")


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
    """
    Extract and store what matters from an LLM interaction.

    Two tiers:
      1. Heuristic extraction (always, zero latency) — sentence-level scoring
         by importance signals (preferences > decisions > facts > episodes).
      2. LLM extraction (background, async contexts only) — structured JSON
         extraction using the cheapest configured model. Fire-and-forget.
    """
    # Tier 1: heuristic extraction — always runs, no LLM, no latency
    try:
        extracted = _heuristic_extract(goal, outcome)
        for text, imp in extracted:
            try:
                m.remember(text, importance=imp)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Mnemon heuristic extraction failed: {e}")

    # Tier 2: background LLM extraction — only fires when inside a running
    # async event loop (i.e. async moth patches). Silently skipped in sync paths.
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_llm_extract_and_store(m, goal, outcome))
    except RuntimeError:
        pass  # no running loop — sync context, heuristic covers it
    except Exception:
        pass

    # Waste report tracking
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


def track_cache_hit(
    m: Any,
    source: str,
    tokens: Optional[int] = None,
    model: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> None:
    """Record a cache hit to MothStats. Never raises."""
    try:
        if hasattr(m, "_stats") and m._stats is not None:
            m._stats.record_hit(source, tokens, model, input_tokens, output_tokens)
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
