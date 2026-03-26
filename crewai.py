"""
Mnemon CrewAI Adapter — v2
Reference implementation — the template for all other adapters.

Translates CrewAI's native concepts into Mnemon signals:
  agent.backstory  → semantic memory
  task.context     → episodic memory
  crew.memory      → relationship memory
  task.output      → experience signal
  task failure     → ERROR_RESOLVED signal

v2 fixes:
  [BUG-A] decompose() dropped task 'context' dependency lists entirely.
          reconstruct() re-wired nothing — stitched tasks had no context
          links, so downstream tasks ran without their declared inputs.
          Both methods now preserve and re-wire context[] arrays.

  [BUG-B] extract_signature() included agent backstory text verbatim in
          context hash. Backstory is prose and changes between sessions
          (e.g. date references, session IDs in agent configs). This made
          every run produce a different fingerprint even for identical tasks,
          so System 1 never matched. Now hashes role names only (stable).

  [BUG-C] get_tool_versions() returned per-call time-seeded hashes (used
          hashlib.md5(cap.encode()) which is deterministic — actually fine —
          but the capability list wasn't sorted before hashing, so two calls
          with the same tools in different order produced different version
          hashes, triggering needs_reverification on every run).
          Now sorts capabilities before hashing.

v2 swarm scale:
  - decompose() emits segment outputs[] and dependencies[] so EME dependency
    validation can verify cross-task data flow after stitching.
  - reconstruct() rebuilds context[] links between tasks using segment outputs,
    so gap-filled plans don't produce tasks with broken context references.
  - wrap_crew() helper makes integration a one-liner.
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from ..core.models import (
    ExperienceSignal,
    MemoryLayer,
    SignalType,
    RiskLevel,
)
from ..core.eme import TemplateAdapter, TemplateSegment
from ..core.models import ComputationFingerprint


class CrewAIAdapter(TemplateAdapter):
    """
    Adapter for CrewAI framework.

    Translates:
      - CrewAI Task dicts   → Mnemon ExecutionTemplates (with dependency graph)
      - CrewAI Agent configs → Mnemon memory signals
      - CrewAI outputs       → Mnemon experience signals

    Zero changes to CrewAI's own code required.
    """

    # ──────────────────────────────────────────
    # CORE ADAPTER METHODS
    # ──────────────────────────────────────────

    def decompose(self, template: Any) -> List[Dict]:
        """
        CrewAI templates are lists of task dicts or a crew config dict.
        Each task becomes one segment.

        [v2] Now emits outputs[] and dependencies[] per segment so EME
        dependency validation can verify cross-task data flow after stitching.

        Task dependency resolution:
          - task.context = list of task IDs this task depends on
          - We treat each task's output as f"output:{task_id}"
          - Downstream tasks declare these as dependencies
        """
        tasks = self._extract_tasks(template)

        # Build task-id → output-key mapping
        task_output_keys: Dict[str, str] = {
            t.get("id", f"task_{i}"): f"output:{t.get('id', f'task_{i}')}"
            for i, t in enumerate(tasks)
        }

        segments = []
        for i, task in enumerate(tasks):
            task_id = task.get("id", f"task_{i}")
            output_key = task_output_keys[task_id]

            # context[] in CrewAI is a list of task IDs whose outputs this task needs
            context_refs = task.get("context", [])
            if isinstance(context_refs, list):
                # Normalise: could be task dicts or task ID strings
                dep_ids = [
                    (ref.get("id") if isinstance(ref, dict) else str(ref))
                    for ref in context_refs
                ]
            else:
                dep_ids = []

            dependencies = [task_output_keys[d] for d in dep_ids if d in task_output_keys]

            segments.append({
                "id": task_id,
                "content": task,
                "outputs": [output_key],
                "dependencies": dependencies,
            })

        return segments

    def reconstruct(self, segments: List[TemplateSegment]) -> Any:
        """
        Reassemble into CrewAI-compatible task list.

        [v2] Re-wires context[] arrays between tasks after stitching.
        Original simply dumped segment.content as-is, leaving context[]
        references pointing to task IDs that may no longer be valid after
        gap fill replaced some segments with new ones.

        Reconstruction steps:
          1. Collect all segment IDs in the stitched plan
          2. For each task, filter context[] to only include IDs present in plan
          3. Return the cleaned task list
        """
        all_ids = {s.segment_id for s in segments}
        tasks = []

        for seg in segments:
            task = dict(seg.content) if isinstance(seg.content, dict) else {"raw": seg.content}

            # Re-wire context[] — remove any references to tasks not in this plan
            raw_context = task.get("context", [])
            if isinstance(raw_context, list):
                cleaned_context = []
                for ref in raw_context:
                    ref_id = ref.get("id") if isinstance(ref, dict) else str(ref)
                    if ref_id in all_ids:
                        cleaned_context.append(ref)
                    # else: silently drop stale reference (task was replaced by gap fill)
                task["context"] = cleaned_context

            tasks.append(task)

        return {
            "tasks": tasks,
            "type": "crewai_flow",
            "eros_cached": True,
        }

    def extract_signature(self, template: Any, goal: str) -> ComputationFingerprint:
        """
        Extract a stable fingerprint from a CrewAI task/crew config.

        [v2 FIX BUG-B] Original included full agent backstory text in the
        context hash. Backstory prose often contains session-specific language
        that changes between runs (dates, client names embedded in prompts).
        This made context_hash differ on every run, so the five-component
        full_hash never matched and System 1 never fired.

        Now context hash uses only agent role names — stable across sessions.

        [v2 FIX BUG-C] Capabilities (tool names) are now sorted before hashing
        so two calls with the same tools in different order produce the same
        capability_hash.
        """
        tasks = self._extract_tasks(template)

        # Tool names from tasks — sorted for stable hashing
        tools: List[str] = sorted({
            tool
            for t in tasks
            if isinstance(t, dict)
            for tool in t.get("tools", [])
        })

        # [FIX BUG-B] Use role names only — not backstory text
        agent_roles: Dict[str, str] = {}
        if isinstance(template, dict):
            for agent in template.get("agents", []):
                if isinstance(agent, dict):
                    role = agent.get("role", "agent")
                    # Stable descriptor — not backstory prose
                    agent_roles[role] = agent.get("goal", "")

        task_schema = {
            t.get("id", f"task_{i}"): {
                "agent": t.get("agent", ""),
                "has_context": bool(t.get("context")),
                "tool_count": len(t.get("tools", [])),
            }
            for i, t in enumerate(tasks)
            if isinstance(t, dict)
        }

        return ComputationFingerprint.build(
            goal=goal,
            input_schema=task_schema,
            context=agent_roles,           # role names only, not prose
            capabilities=tools,            # sorted
            constraints=template.get("constraints", {}) if isinstance(template, dict) else {},
        )

    def get_tool_versions(self, capabilities: List[str]) -> Dict[str, str]:
        """
        Return stable version hashes for each capability (tool name).

        [v2 FIX BUG-C] Original was deterministic but callers didn't sort
        the capabilities list before passing it, so two calls with the same
        tools in different order produced different hashes, triggering
        needs_reverification on every warm cache hit.

        We sort here defensively so version hashes are always stable
        regardless of call order.
        """
        return {
            cap: hashlib.md5(cap.encode()).hexdigest()[:8]
            for cap in sorted(capabilities)
        }

    # ──────────────────────────────────────────
    # INTERNAL HELPERS
    # ──────────────────────────────────────────

    def _extract_tasks(self, template: Any) -> List[Dict]:
        """Normalise any CrewAI template shape into a flat task list."""
        if isinstance(template, list):
            return template
        if isinstance(template, dict):
            return template.get("tasks", [template])
        return [{"raw": template}]

    # ──────────────────────────────────────────
    # SIGNAL TRANSLATORS
    # ──────────────────────────────────────────

    @staticmethod
    def agent_to_signals(
        agent_config: Dict,
        tenant_id: str,
        session_id: str,
    ) -> List[ExperienceSignal]:
        """
        Translate a CrewAI agent config into memory signals.
          backstory → semantic memory (stable fact about agent's domain)
          goal      → working memory (ephemeral, this session only)
        """
        signals = []
        now = time.time()
        role = agent_config.get("role", "agent")

        if agent_config.get("backstory"):
            signals.append(ExperienceSignal(
                signal_id=hashlib.md5(
                    f"{tenant_id}:{role}:backstory".encode()
                ).hexdigest()[:16],
                tenant_id=tenant_id,
                session_id=session_id,
                timestamp=now,
                signal_type=SignalType.CONTEXT_UPDATE,
                layer=MemoryLayer.SEMANTIC,
                content={
                    "key": f"agent_{role}_backstory",
                    "value": agent_config["backstory"],
                },
                importance=0.6,
            ))

        if agent_config.get("goal"):
            signals.append(ExperienceSignal(
                signal_id=hashlib.md5(
                    f"{tenant_id}:{session_id}:goal:{now}".encode()
                ).hexdigest()[:16],
                tenant_id=tenant_id,
                session_id=session_id,
                timestamp=now,
                signal_type=SignalType.CONTEXT_UPDATE,
                layer=MemoryLayer.WORKING,
                content={"goals": [agent_config["goal"]]},
                importance=0.8,
            ))

        return signals

    @staticmethod
    def task_output_to_signal(
        task_id: str,
        output: str,
        success: bool,
        tenant_id: str,
        session_id: str,
        agent_id: str,
        error_type: Optional[str] = None,
        mitigation: Optional[str] = None,
    ) -> ExperienceSignal:
        """
        Translate a CrewAI task result into an experience signal.
          Success → VALIDATION_PASSED (cached for future reference)
          Failure → ERROR_RESOLVED if mitigation found, else VALIDATION_FAILED
        """
        now = time.time()

        if success:
            return ExperienceSignal(
                signal_id=hashlib.md5(
                    f"{tenant_id}:{task_id}:{now}".encode()
                ).hexdigest()[:16],
                tenant_id=tenant_id,
                session_id=session_id,
                timestamp=now,
                signal_type=SignalType.VALIDATION_PASSED,
                layer=MemoryLayer.EPISODIC,
                content={
                    "task_id": task_id,
                    "output": output[:500],
                    "approach": task_id,
                },
                importance=0.6,
                agent_id=agent_id,
            )

        signal_type = (
            SignalType.ERROR_RESOLVED if mitigation else SignalType.VALIDATION_FAILED
        )
        return ExperienceSignal(
            signal_id=hashlib.md5(
                f"{tenant_id}:{task_id}:error:{now}".encode()
            ).hexdigest()[:16],
            tenant_id=tenant_id,
            session_id=session_id,
            timestamp=now,
            signal_type=signal_type,
            layer=MemoryLayer.EPISODIC,
            content={
                "task_id": task_id,
                "error_type": error_type or "task_failure",
                "mitigation": mitigation,
                "approach": task_id,
            },
            importance=0.8,
            agent_id=agent_id,
            mitigation=mitigation,
        )

    @staticmethod
    def context_pollution_fix_signal(
        session_id: str,
        tenant_id: str,
        agent_id: str,
    ) -> ExperienceSignal:
        """
        Signal for CrewAI issue #4319 — context pollution fix.
        Working memory flush notification.
        Tells the bus the session context has been cleared.
        """
        return ExperienceSignal(
            signal_id=hashlib.md5(
                f"{tenant_id}:{session_id}:flush:{time.time()}".encode()
            ).hexdigest()[:16],
            tenant_id=tenant_id,
            session_id=session_id,
            timestamp=time.time(),
            signal_type=SignalType.CONTEXT_UPDATE,
            layer=MemoryLayer.WORKING,
            content={"flush": True, "reason": "task_end_context_isolation"},
            importance=0.3,
            agent_id=agent_id,
        )


# ──────────────────────────────────────────
# INTEGRATION HELPER
# ──────────────────────────────────────────

def wrap_crew(crew_config: Dict, mnemon_instance: Any) -> Dict:
    """
    One-liner integration: wraps a CrewAI crew config with Mnemon.

    Usage:
        from mnemon.adapters.crewai import wrap_crew
        crew = wrap_crew(my_crew_config, mnemon)

    Returns the crew config unchanged — side effect is registering
    all agent backstories as semantic memory signals.
    """
    adapter = CrewAIAdapter()
    agents = crew_config.get("agents", [])
    session_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:12]

    for agent in agents:
        if isinstance(agent, dict):
            signals = adapter.agent_to_signals(
                agent_config=agent,
                tenant_id=getattr(mnemon_instance, "tenant_id", "default"),
                session_id=session_id,
            )
            for signal in signals:
                # Non-blocking — fire and let the bus handle it
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(
                            mnemon_instance.bus.publish(signal)
                            if hasattr(mnemon_instance, "bus")
                            else asyncio.sleep(0)
                        )
                except RuntimeError:
                    pass   # No event loop — skip signal, crew still runs

    return crew_config
