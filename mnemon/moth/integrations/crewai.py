"""
Mnemon moth — CrewAI integration.

Registers directly with crewai_event_bus (not via BaseEventListener) so
we hold exact function references and unpatch() reliably removes them.

  AgentExecutionStartedEvent  → inject recalled memory into agent.backstory
  AgentExecutionCompletedEvent → record outcome + restore original backstory
  TaskCompletedEvent          → additional outcome recording at task granularity

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
from typing import Any, Dict, Optional

from mnemon.moth import MnemonIntegration
from ._utils import recall_as_context, record_outcome, prompt_hash, track_cache_hit
from ._cache import BoundedTTLCache

logger = logging.getLogger(__name__)

_task_cache = BoundedTTLCache(maxsize=500, ttl=3600)

# Temporary storage: agent id → original backstory (restored after execution)
_backstory_originals: Dict[int, str] = {}


class CrewAIIntegration(MnemonIntegration):
    """
    Instruments CrewAI via its event bus using direct handler registration.
    Storing exact handler references guarantees unpatch() removes them cleanly.
    """

    name = "crewai"

    def __init__(self) -> None:
        self._handlers: Dict[Any, Any] = {}   # event_type → handler fn
        self._original_execute_sync: Optional[Any] = None
        self._mnemon: Optional[Any] = None

    def is_available(self) -> bool:
        # Only activate if crewai is actually imported in the user's process.
        return "crewai" in sys.modules

    def patch(self, mnemon: Any) -> None:
        if self._handlers:
            return  # already patched — prevents duplicate listeners on re-init

        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.events.types.agent_events import (
                AgentExecutionStartedEvent,
                AgentExecutionCompletedEvent,
            )
            from crewai.events.types.task_events import TaskCompletedEvent
        except Exception as e:
            logger.debug(f"Mnemon: CrewAI event types unavailable — {e}")
            return

        self._mnemon = mnemon
        m = mnemon

        def on_agent_start(source: Any, event: Any) -> None:
            try:
                agent = event.agent
                task_text = getattr(event.task, "description", "") or getattr(event, "task_prompt", "")
                query = f"{getattr(agent, 'role', '')} {task_text}"[:300]
                context = recall_as_context(m, query, source="crewai")
                if context:
                    original = getattr(agent, "backstory", "") or ""
                    _backstory_originals[id(agent)] = original
                    try:
                        object.__setattr__(agent, "backstory", f"{context}\n\n{original}")
                    except (AttributeError, TypeError):
                        agent.__dict__["backstory"] = f"{context}\n\n{original}"
            except Exception as e:
                logger.debug(f"Mnemon: CrewAI on_agent_start failed — {e}")

        def on_agent_completed(source: Any, event: Any) -> None:
            try:
                agent = event.agent
                original = _backstory_originals.pop(id(agent), None)
                if original is not None:
                    try:
                        object.__setattr__(agent, "backstory", original)
                    except (AttributeError, TypeError):
                        agent.__dict__["backstory"] = original
                task_text = getattr(event.task, "description", "") or ""
                goal = f"{getattr(agent, 'role', '')} {task_text}"[:120]
                record_outcome(m, goal, str(event.output)[:400])
            except Exception as e:
                logger.debug(f"Mnemon: CrewAI on_agent_completed failed — {e}")

        def on_task_completed(source: Any, event: Any) -> None:
            try:
                task = event.task
                description = getattr(task, "description", "") if task else ""
                output_str = str(event.output.raw if hasattr(event.output, "raw") else event.output)
                record_outcome(m, description[:120], output_str[:400], importance=0.7)
            except Exception as e:
                logger.debug(f"Mnemon: CrewAI on_task_completed failed — {e}")

        # Register directly — store the exact objects the bus holds so off() works
        try:
            crewai_event_bus.on(AgentExecutionStartedEvent)(on_agent_start)
            crewai_event_bus.on(AgentExecutionCompletedEvent)(on_agent_completed)
            crewai_event_bus.on(TaskCompletedEvent)(on_task_completed)
            self._handlers = {
                AgentExecutionStartedEvent:   on_agent_start,
                AgentExecutionCompletedEvent: on_agent_completed,
                TaskCompletedEvent:           on_task_completed,
            }
        except Exception as e:
            logger.debug(f"Mnemon: CrewAI event registration failed — {e}")
            return

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
                cache_key = _task_cache_key(_self, agent, context)
                if cache_key in _task_cache:
                    track_cache_hit(mnemon, "crewai")
                    logger.debug(
                        f"Mnemon: CrewAI Task '{getattr(_self, 'description', '')[:40]}' "
                        f"System 2 cache hit"
                    )
                    return _task_cache[cache_key]

                result = orig_exec(_self, agent=agent, context=context, tools=tools)
                _task_cache[cache_key] = result
                return result

            Task.execute_sync = patched_execute_sync
        except Exception as e:
            logger.debug(f"Mnemon: CrewAI Task.execute_sync patch failed — {e}")

    def unpatch(self) -> None:
        try:
            from crewai.events.event_bus import crewai_event_bus
            for event_type, handler in list(self._handlers.items()):
                try:
                    crewai_event_bus.off(event_type, handler)
                except Exception as e:
                    logger.debug(f"Mnemon: CrewAI off({event_type.__name__}) failed — {e}")
        except Exception as e:
            logger.debug(f"Mnemon: CrewAI event bus import failed during unpatch — {e}")
        self._handlers = {}

        try:
            if self._original_execute_sync is not None:
                from crewai.task import Task
                Task.execute_sync = self._original_execute_sync
        except Exception as e:
            logger.debug(f"Mnemon: CrewAI execute_sync unpatch failed — {e}")
        finally:
            self._original_execute_sync = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _task_cache_key(task: Any, agent: Any, context: Optional[str]) -> str:
    description = getattr(task, "description", "") or ""
    agent_role = getattr(agent, "role", "") if agent else ""
    key = hashlib.md5(
        f"{description}|{agent_role}|{context or ''}".encode()
    ).hexdigest()
    return f"crewai_task:{key}"
