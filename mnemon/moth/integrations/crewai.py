"""
Mnemon moth — CrewAI integration.

Registers directly with crewai_event_bus (not via BaseEventListener) so
we hold exact function references and unpatch() reliably removes them.

  TaskCompletedEvent → record task cache key for future EME hits

System 2 EME:
  Patches Task.execute_sync to cache task outputs. On a cache hit the agent
  is not called. Falls back gracefully if Task.execute_sync changes between
  CrewAI versions.

Tested against crewai 0.80+ / 1.x.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import sys
import types
from typing import Any, Dict, Optional

from mnemon.moth import MnemonIntegration
from ._utils import prompt_hash, track_cache_hit, track_cache_miss
from ._eme_bridge import MothCache

logger = logging.getLogger(__name__)


class CrewAIIntegration(MnemonIntegration):
    """
    Instruments CrewAI via its event bus using direct handler registration.
    Storing exact handler references guarantees unpatch() removes them cleanly.
    """

    name = "crewai"

    def __init__(self) -> None:
        self._handlers: Dict[Any, Any] = {}
        self._original_execute_sync: Optional[Any] = None
        self._mnemon: Optional[Any] = None
        self._task_cache: Optional[MothCache] = None

    def is_available(self) -> bool:
        return "crewai" in sys.modules

    def patch(self, mnemon: Any) -> None:
        if self._handlers:
            return

        self._mnemon = mnemon
        m = mnemon
        self._task_cache = MothCache(m, "crewai")
        task_cache = self._task_cache

        # System 2 EME: patch Task.execute_sync
        try:
            from crewai.task import Task
            self._original_execute_sync = Task.execute_sync
            orig_exec = self._original_execute_sync

            def patched_execute_sync(
                _self: Any,
                agent: Any = None,
                context: Optional[str] = None,
                tools: Any = None,
            ) -> Any:
                description = getattr(_self, "description", "") or ""
                cache_key   = _task_cache_key(_self, agent, context)
                cached = task_cache.check(query=description, capabilities=[], hash_key=cache_key)
                if cached is not None:
                    track_cache_hit(m, "crewai")
                    logger.debug(f"Mnemon: CrewAI Task '{description[:40]}' cache hit")
                    if isinstance(cached, str):
                        return _crewai_from_text(cached)
                    return cached

                result = orig_exec(_self, agent=agent, context=context, tools=tools)
                text   = _crewai_text(result)
                task_cache.store(description, [], cache_key, result, text)
                track_cache_miss(m, "crewai")
                return result

            Task.execute_sync = patched_execute_sync
        except Exception as e:
            logger.debug(f"Mnemon: CrewAI Task.execute_sync patch failed — {e}")

    def unpatch(self) -> None:
        try:
            if self._original_execute_sync is not None:
                from crewai.task import Task
                Task.execute_sync = self._original_execute_sync
        except Exception as e:
            logger.debug(f"Mnemon: CrewAI execute_sync unpatch failed — {e}")
        finally:
            self._original_execute_sync = None
            self._handlers = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _task_cache_key(task: Any, agent: Any, context: Optional[str]) -> str:
    description = getattr(task, "description", "") or ""
    agent_role  = getattr(agent, "role", "") if agent else ""
    key = hashlib.md5(
        f"{description}|{agent_role}|{context or ''}".encode()
    ).hexdigest()
    return f"crewai_task:{key}"


def _crewai_text(result: Any) -> str:
    """Extract plain text from a CrewAI task result for DB persistence."""
    if isinstance(result, str):
        return result
    return (
        getattr(result, "raw", None)
        or getattr(result, "output", None)
        or str(result)
    )


def _crewai_from_text(text: str) -> Any:
    """Reconstruct a minimal CrewAI-compatible result from cached text."""
    return types.SimpleNamespace(raw=text, output=text, pydantic=None)
