"""
Mnemon moth — CrewAI integration.

Hooks into CrewAI's native event bus (crewai_event_bus) using the
BaseEventListener pattern. No monkey-patching of internals required.

  AgentExecutionStartedEvent → inject recalled memory into agent.backstory
    before the agent calls the LLM. Restores original on completion.

  AgentExecutionCompletedEvent → record outcome to experience bus + restore
    agent.backstory to its original value.

  TaskCompletedEvent → additional outcome recording at task granularity.

System 2 EME:
  Patches Task.execute_sync to cache task outputs keyed by (task_description
  + agent_role + context_hash). On a cache hit the agent is not called.
  Falls back gracefully if Task.execute_sync changes between CrewAI versions.

Tested against crewai 0.80+ / 1.x. BaseEventListener pattern is stable
across versions; execute_sync patch falls back to no-op on failure.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
from typing import Any, Dict, Optional

from mnemon.moth import MnemonIntegration
from ._utils import recall_as_context, record_outcome, prompt_hash
from ._cache import BoundedTTLCache

logger = logging.getLogger(__name__)

_task_cache = BoundedTTLCache(maxsize=500, ttl=3600)

# Temporary storage: agent id → original backstory (restored after execution)
_backstory_originals: Dict[int, str] = {}


class CrewAIIntegration(MnemonIntegration):
    """
    Instruments CrewAI via its native event bus.
    No internal patching — uses BaseEventListener pattern exactly as
    CrewAI intends for observability tools.
    """

    name = "crewai"

    def __init__(self) -> None:
        self._listener: Optional[Any] = None
        self._original_execute_sync: Optional[Any] = None
        self._mnemon: Optional[Any] = None

    def is_available(self) -> bool:
        return importlib.util.find_spec("crewai") is not None

    def patch(self, mnemon: Any) -> None:
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.base_event_listener import BaseEventListener
        from crewai.events.types.agent_events import (
            AgentExecutionStartedEvent,
            AgentExecutionCompletedEvent,
        )
        from crewai.events.types.task_events import TaskCompletedEvent

        self._mnemon = mnemon
        m = mnemon

        class _MnemonListener(BaseEventListener):
            def setup_listeners(self, bus: Any) -> None:
                @bus.on(AgentExecutionStartedEvent)
                def on_agent_start(source: Any, event: AgentExecutionStartedEvent) -> None:
                    try:
                        agent = event.agent
                        task_text = getattr(event.task, "description", "") or event.task_prompt
                        query = f"{getattr(agent, 'role', '')} {task_text}"[:300]
                        context = recall_as_context(m, query)
                        if context:
                            original = getattr(agent, "backstory", "") or ""
                            _backstory_originals[id(agent)] = original
                            try:
                                object.__setattr__(agent, "backstory", f"{context}\n\n{original}")
                            except (AttributeError, TypeError):
                                agent.__dict__["backstory"] = f"{context}\n\n{original}"
                    except Exception as e:
                        logger.debug(f"Mnemon: CrewAI on_agent_start failed — {e}")

                @bus.on(AgentExecutionCompletedEvent)
                def on_agent_completed(source: Any, event: AgentExecutionCompletedEvent) -> None:
                    try:
                        agent = event.agent
                        # Restore original backstory
                        original = _backstory_originals.pop(id(agent), None)
                        if original is not None:
                            try:
                                object.__setattr__(agent, "backstory", original)
                            except (AttributeError, TypeError):
                                agent.__dict__["backstory"] = original

                        # Record to experience bus
                        task_text = getattr(event.task, "description", "") or ""
                        goal = f"{getattr(agent, 'role', '')} {task_text}"[:120]
                        record_outcome(m, goal, str(event.output)[:400])
                    except Exception as e:
                        logger.debug(f"Mnemon: CrewAI on_agent_completed failed — {e}")

                @bus.on(TaskCompletedEvent)
                def on_task_completed(source: Any, event: TaskCompletedEvent) -> None:
                    try:
                        task = event.task
                        description = getattr(task, "description", "") if task else ""
                        output_str = str(event.output.raw if hasattr(event.output, "raw") else event.output)
                        record_outcome(m, description[:120], output_str[:400], importance=0.7)
                    except Exception as e:
                        logger.debug(f"Mnemon: CrewAI on_task_completed failed — {e}")

                # Store handler refs on the listener for unpatch
                self._on_agent_start = on_agent_start
                self._on_agent_completed = on_agent_completed
                self._on_task_completed = on_task_completed

        self._listener = _MnemonListener()

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
            from crewai.events.types.agent_events import (
                AgentExecutionStartedEvent,
                AgentExecutionCompletedEvent,
            )
            from crewai.events.types.task_events import TaskCompletedEvent

            if self._listener is not None:
                listener = self._listener
                if hasattr(listener, "_on_agent_start"):
                    crewai_event_bus.off(AgentExecutionStartedEvent, listener._on_agent_start)
                if hasattr(listener, "_on_agent_completed"):
                    crewai_event_bus.off(AgentExecutionCompletedEvent, listener._on_agent_completed)
                if hasattr(listener, "_on_task_completed"):
                    crewai_event_bus.off(TaskCompletedEvent, listener._on_task_completed)
        except Exception as e:
            logger.debug(f"Mnemon: CrewAI event bus off failed — {e}")

        try:
            if self._original_execute_sync is not None:
                from crewai.task import Task
                Task.execute_sync = self._original_execute_sync
        except Exception as e:
            logger.debug(f"Mnemon: CrewAI execute_sync unpatch failed — {e}")
        finally:
            self._listener = None
            self._original_execute_sync = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _task_cache_key(task: Any, agent: Any, context: Optional[str]) -> str:
    description = getattr(task, "description", "") or ""
    agent_role = getattr(agent, "role", "") if agent else ""
    key = hashlib.md5(
        f"{description}|{agent_role}|{context or ''}".encode()
    ).hexdigest()
    return f"crewai_task:{key}"
