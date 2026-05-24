"""
Shared utilities for all moth integrations.
All functions are fail-safe — they never raise.

FeedbackExtractor: framework-aware signal extractor. Builds EvidenceRecords from:
  - Hard crashes (exceptions, with framework-specific error type mapping)
  - Tool errors (is_error blocks in Anthropic/OpenAI responses)
  - Validation failures (OutputParserException, PydanticValidationError, etc.)
  - Wrong plans (max_iter hits, graph routing errors, schema mismatches)
  - Soft failure language in LLM output text
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from mnemon import MnemonSync

logger = logging.getLogger(__name__)


def prompt_hash(messages: List[Dict], system: Optional[str], model: str) -> str:
    """Stable hash for an LLM call — used for System 1 client-level cache."""
    try:
        key = json.dumps(
            {"m": messages, "s": system or "", "model": model},
            sort_keys=True, default=str,
        )
        return hashlib.md5(key.encode()).hexdigest()
    except Exception:
        return hashlib.md5(str(messages).encode()).hexdigest()


def extract_query(messages: List[Dict], system: Optional[str] = None) -> str:
    """
    Extract the intent signal for EME semantic matching.
    Uses system prompt + last user message — system describes what the agent
    is doing, last user message is the task.
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


# ─────────────────────────────────────────────
# FEEDBACK EXTRACTOR
# ─────────────────────────────────────────────

class FeedbackExtractor:
    """
    Framework-aware feedback signal extractor.
    Builds EvidenceRecords from framework exceptions and response scanning.
    All methods are classmethods — no instance state, no bleed between calls.
    """

    # Soft failure language in LLM output text
    _SOFT_FAIL_RE = re.compile(
        r"i (was unable|couldn'?t|cannot|failed) to"
        r"|error (occurred|happened|encountered)"
        r"|the tool (returned|produced) an? error"
        r"|unable to (complete|execute|perform|process)"
        r"|(unfortunately|sorry)[,\s]",
        re.IGNORECASE,
    )

    # Framework exception class names → failure_class
    _ERROR_MAP: Dict[str, str] = {
        "OutputParserException":     "validation",
        "OutputParserError":         "validation",
        "ValidationError":           "schema",
        "PydanticValidationError":   "schema",
        "GraphRecursionError":       "max_iter",
        "NodeInterrupt":             "wrong_plan",
        "InvalidUpdateError":        "schema",
        "TaskRepeatedAttemptError":  "max_iter",
        "ToolException":             "tool_error",
        "ToolExecutionError":        "tool_error",
        "TimeoutError":              "exception",
        "asyncio.TimeoutError":      "exception",
    }

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        framework: str,
        task_id: str,
        template_id: Optional[str],
        fragment_ids: List[str],
        goal_hash: Optional[str],
        goal_type: Optional[str],
        tenant_id: str,
        latency_ms: float,
        failed_step: Optional[int] = None,
    ) -> Any:
        """Build EvidenceRecord from a caught exception."""
        try:
            from mnemon.core.models import EvidenceRecord
        except ImportError:
            return None
        error_type    = type(exc).__name__
        failure_class = cls._ERROR_MAP.get(error_type, "exception")
        if failed_step is None:
            failed_step = cls._step_from_exc(exc)
        return EvidenceRecord(
            task_id=task_id,
            tenant_id=tenant_id,
            template_id=template_id,
            fragment_ids_used=fragment_ids,
            framework=framework,
            outcome="failure",
            failure_class=failure_class,
            error_type=error_type,
            error_message=str(exc)[:500],
            failed_step=failed_step,
            cascade_root=None,
            tool_name=cls._tool_from_exc(exc),
            goal_hash=goal_hash,
            goal_type=goal_type,
            latency_ms=latency_ms,
            timestamp=time.time(),
        )

    @classmethod
    def from_response(
        cls,
        response: Any,
        framework: str,
        task_id: str,
        template_id: Optional[str],
        fragment_ids: List[str],
        goal_hash: Optional[str],
        goal_type: Optional[str],
        tenant_id: str,
        latency_ms: float,
    ) -> Any:
        """
        Scan a successful response for embedded failure signals.
        Returns EvidenceRecord with outcome="success" if clean,
        "wrong_plan" if framework signals found, "failure" if hard error in response.
        """
        try:
            from mnemon.core.models import EvidenceRecord
        except ImportError:
            return None
        failure_class, error_type, error_msg, tool_name, failed_step = (
            cls._scan_response(response, framework)
        )
        outcome = "success"
        if failure_class:
            outcome = "wrong_plan" if failure_class not in ("exception",) else "failure"
        return EvidenceRecord(
            task_id=task_id,
            tenant_id=tenant_id,
            template_id=template_id,
            fragment_ids_used=fragment_ids,
            framework=framework,
            outcome=outcome,
            failure_class=failure_class,
            error_type=error_type,
            error_message=error_msg,
            failed_step=failed_step,
            cascade_root=None,
            tool_name=tool_name,
            goal_hash=goal_hash,
            goal_type=goal_type,
            latency_ms=latency_ms,
            timestamp=time.time(),
        )

    @classmethod
    def _scan_response(
        cls, response: Any, framework: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[int]]:
        """
        Scan response for embedded failure signals.
        Returns (failure_class, error_type, error_msg, tool_name, failed_step).
        All None if response is clean.
        """
        try:
            # Anthropic — scan content blocks for tool_result is_error
            if hasattr(response, "content") and isinstance(response.content, list):
                for i, block in enumerate(response.content):
                    if hasattr(block, "type") and block.type == "tool_result":
                        if getattr(block, "is_error", False):
                            return (
                                "tool_error", "tool_result_error",
                                str(getattr(block, "content", ""))[:300],
                                getattr(block, "tool_use_id", None),
                                i,
                            )

            # OpenAI dict response
            if isinstance(response, dict):
                for i, tc in enumerate(response.get("tool_calls") or []):
                    if isinstance(tc, dict):
                        result = tc.get("result") or tc.get("output", "")
                        if isinstance(result, dict) and result.get("error"):
                            return (
                                "tool_error", "tool_call_error",
                                str(result.get("error", ""))[:300],
                                tc.get("function", {}).get("name"),
                                i,
                            )
                finish = (
                    response.get("finish_reason")
                    or (response.get("choices") or [{}])[0].get("finish_reason")
                )
                if finish in ("content_filter", "function_call_error"):
                    return ("wrong_plan", finish, f"finish_reason={finish}", None, None)

            # Soft failure language in text
            text = cls._extract_text(response)
            if text and cls._SOFT_FAIL_RE.search(text):
                return ("wrong_plan", "soft_failure_language", text[:200], None, None)

        except Exception:
            pass
        return (None, None, None, None, None)

    @classmethod
    def _extract_text(cls, response: Any) -> str:
        try:
            if isinstance(response, str):
                return response
            if hasattr(response, "content"):
                c = response.content
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    return " ".join(
                        getattr(b, "text", "") or (b.get("text", "") if isinstance(b, dict) else "")
                        for b in c
                    )
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
        except Exception:
            pass
        return ""

    @classmethod
    def _step_from_exc(cls, exc: Exception) -> Optional[int]:
        try:
            msg = str(exc)
            m = re.search(r"step[_\s]?(\d+)", msg, re.IGNORECASE)
            if m:
                return int(m.group(1))
            m = re.search(r"node[_\s]?(\d+)", msg, re.IGNORECASE)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return None

    @classmethod
    def _tool_from_exc(cls, exc: Exception) -> Optional[str]:
        try:
            m = re.search(r"tool[_\s]?['\"]?(\w+)['\"]?", str(exc), re.IGNORECASE)
            if m:
                return m.group(1)
        except Exception:
            pass
        return None


# ─────────────────────────────────────────────
# CACHE HIT / MISS TRACKING
# ─────────────────────────────────────────────

def track_cache_miss(m: Any, source: str) -> None:
    """Print first-run or new-input message on MOTH cache miss. Never raises."""
    try:
        import sys as _sys
        import os as _os
        inner = getattr(m, "_m", None)
        if inner is None:
            return
        silent = getattr(inner, "_silent", False) or getattr(m, "_kwargs", {}).get("silent", False)
        if silent:
            return
        db_dir    = getattr(inner, "_db_dir", ".")
        tenant_id = getattr(inner, "tenant_id", "default")
        flag = _os.path.join(db_dir, f".mnemon_welcomed_{tenant_id}")
        if not _os.path.exists(flag):
            try:
                open(flag, "w").close()
            except OSError:
                pass
            msg = (
                f"Mnemon [{source}]: first run — response cached, next call will be instant\n"
                f"  Thank you for installing Mnemon!"
                f" Drop a line at mahikajadhav22@gmail.com if caching isn't working.\n"
                f"  I can also personally integrate this into your workflow — same email."
            )
        else:
            msg = f"Mnemon [{source}]: new input — cached, next call will be instant"
        print(msg, file=_sys.stderr, flush=True)
    except Exception:
        pass
    try:
        from mnemon.core import ph_telemetry
        ph_telemetry.track_miss(framework=source)
    except Exception:
        pass


def track_cache_hit(
    m: Any,
    source: str,
    tokens: Optional[int] = None,
    model: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    latency_ms: float = 0.0,
) -> None:
    """Record a cache hit to MothStats and Bus. Never raises."""
    try:
        silent = getattr(m, "_kwargs", {}).get("silent", False)
        if not silent:
            import sys as _sys
            total_tokens = (input_tokens or 0) + (output_tokens or 0) if (input_tokens or output_tokens) else (tokens or 0)
            cost = total_tokens * 0.000003
            msg = f"Mnemon: cache hit [{source}]"
            if total_tokens:
                msg += f" · {total_tokens:,} tokens saved · ~${cost:.4f}"
            print(msg, file=_sys.stderr, flush=True)
    except Exception:
        pass
    try:
        if hasattr(m, "_stats") and m._stats is not None:
            m._stats.record_hit(source, tokens, model, input_tokens, output_tokens)
    except Exception:
        pass
    try:
        from mnemon.core import ph_telemetry
        total_tokens = (input_tokens or 0) + (output_tokens or 0) if (input_tokens or output_tokens) else (tokens or 0)
        ph_telemetry.track_hit(framework=source, cache_level="moth", tokens_saved=total_tokens, latency_ms=latency_ms)
    except Exception:
        pass
    try:
        inner = getattr(m, "_m", None)
        if inner and getattr(inner, "_bus", None):
            try:
                from mnemon.core.models import EvidenceRecord
            except ImportError:
                return
            task_id  = f"moth:{source}:{time.time():.0f}"
            evidence = EvidenceRecord(
                task_id=task_id,
                tenant_id=getattr(inner, "tenant_id", "unknown"),
                template_id=None,
                fragment_ids_used=[],
                framework=source,
                outcome="success",
                failure_class=None,
                error_type=None,
                error_message=None,
                failed_step=None,
                cascade_root=None,
                tool_name=None,
                goal_hash=None,
                goal_type=source,
                latency_ms=latency_ms,
                timestamp=time.time(),
            )
            import asyncio as _asyncio
            coro = inner._bus.record_evidence(evidence)
            try:
                loop = _asyncio.get_running_loop()
                loop.create_task(coro)
            except RuntimeError:
                m._run(coro)
    except Exception:
        pass


# ─────────────────────────────────────────────
# EVIDENCE ROUTING — shared by all integrations
# ─────────────────────────────────────────────

def route_evidence(m: Any, evidence: Any) -> None:
    """Fire a pre-built EvidenceRecord to the Bus. Never raises."""
    try:
        inner = getattr(m, "_m", None)
        if not inner:
            return
        bus = getattr(inner, "_bus", None)
        if not bus:
            return
        import asyncio as _asyncio
        coro = bus.record_evidence(evidence)
        try:
            loop = _asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            m._run(coro)
    except Exception:
        pass


def build_call_evidence(
    m: Any,
    framework: str,
    query: str,
    latency_ms: float,
    response: Any = None,
    exc: Exception = None,
) -> Any:
    """
    Build an EvidenceRecord for a single patched LLM call.
    Pass either response= (successful call) or exc= (exception).
    Returns None silently on any error.
    """
    try:
        from mnemon.core.models import EvidenceRecord
        inner     = getattr(m, "_m", None)
        tenant_id = getattr(inner, "tenant_id", "unknown") if inner else "unknown"
        task_id   = hashlib.md5(
            f"moth:{framework}:{query[:40]}:{time.time():.3f}".encode()
        ).hexdigest()[:12]
        goal_hash = hashlib.md5(query.encode()).hexdigest()[:16] if query else None

        if exc is not None:
            return FeedbackExtractor.from_exception(
                exc=exc, framework=framework, task_id=task_id,
                template_id=None, fragment_ids=[], goal_hash=goal_hash,
                goal_type="llm_call", tenant_id=tenant_id, latency_ms=latency_ms,
            )
        if response is not None:
            return FeedbackExtractor.from_response(
                response=response, framework=framework, task_id=task_id,
                template_id=None, fragment_ids=[], goal_hash=goal_hash,
                goal_type="llm_call", tenant_id=tenant_id, latency_ms=latency_ms,
            )
    except Exception:
        pass
    return None
