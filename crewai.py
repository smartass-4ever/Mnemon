"""
Mnemon CrewAI Adapter
Reference implementation — the template for all other adapters.

Translates CrewAI's native concepts into Mnemon signals:
  agent.backstory    → semantic memory
  task.context       → episodic memory
  crew.memory        → relationship memory
  task.output        → experience signal
  task failure       → ERROR_RESOLVED signal

Usage:
    from .crewai import CrewAIAdapter, wrap_crew

    adapter = CrewAIAdapter()
    mnemon = Mnemon(tenant_id="my_company", adapter=adapter)
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from ..core.models import (
    ExperienceSignal, MemoryLayer, SignalType, RiskLevel
)
from ..core.eme import TemplateAdapter, ComputationFingerprint, TemplateSegment


class CrewAIAdapter(TemplateAdapter):
    """
    Adapter for CrewAI framework.

    Translates:
    - CrewAI Task dicts → Mnemon ExecutionTemplates
    - CrewAI Agent configs → Mnemon memory signals
    - CrewAI outputs → Mnemon experience signals

    ~50 lines of translation code.
    Zero changes to CrewAI's own code.
    """

    def decompose(self, template: Any) -> List[Dict]:
        """
        CrewAI templates are lists of task dicts or a crew config dict.
        Each task becomes one segment.
        """
        if isinstance(template, list):
            return [
                {
                    "id":      task.get("id", f"task_{i}"),
                    "content": task,
                }
                for i, task in enumerate(template)
            ]
        if isinstance(template, dict):
            tasks = template.get("tasks", [template])
            return [
                {
                    "id":      t.get("id", f"task_{i}") if isinstance(t, dict) else f"task_{i}",
                    "content": t,
                }
                for i, t in enumerate(tasks)
            ]
        return [{"id": "task_0", "content": template}]

    def reconstruct(self, segments: List[TemplateSegment]) -> Any:
        """Reassemble into CrewAI-compatible task list."""
        return {
            "tasks":       [s.content for s in segments],
            "type":        "crewai_flow",
            "eros_cached": True,
        }

    def extract_signature(self, template: Any, goal: str) -> ComputationFingerprint:
        """Extract fingerprint from a CrewAI task/crew config."""
        tasks = self.decompose(template)

        # Tool names from tasks
        tools = []
        for t in tasks:
            content = t.get("content", {})
            if isinstance(content, dict):
                tools.extend(content.get("tools", []))

        # Agent instructions / backstory
        agents_context = {}
        if isinstance(template, dict):
            for agent in template.get("agents", []):
                if isinstance(agent, dict):
                    agents_context[agent.get("role", "agent")] = agent.get("backstory", "")

        return ComputationFingerprint.build(
            goal=goal,
            input_schema={t["id"]: type(t["content"]).__name__ for t in tasks},
            context=agents_context,
            capabilities=tools,
            constraints=template.get("constraints", {}) if isinstance(template, dict) else {},
        )

    def get_tool_versions(self, capabilities: List[str]) -> Dict[str, str]:
        """
        In production: fetch actual tool version hashes from CrewAI tool registry.
        Here we use capability names as version proxies.
        """
        return {cap: hashlib.md5(cap.encode()).hexdigest()[:8] for cap in capabilities}

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
        backstory → semantic memory
        goal      → episodic context
        """
        signals = []
        now = time.time()

        # Backstory → semantic memory (stable facts about this agent's domain)
        if agent_config.get("backstory"):
            signals.append(ExperienceSignal(
                signal_id=hashlib.md5(
                    f"{tenant_id}:{agent_config.get('role', 'agent')}:backstory".encode()
                ).hexdigest()[:16],
                tenant_id=tenant_id,
                session_id=session_id,
                timestamp=now,
                signal_type=SignalType.CONTEXT_UPDATE,
                layer=MemoryLayer.SEMANTIC,
                content={
                    "key":   f"agent_{agent_config.get('role', 'agent')}_backstory",
                    "value": agent_config["backstory"],
                },
                importance=0.6,
            ))

        # Goal → working memory for this session
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
        Failure → ERROR_RESOLVED if fix found
        """
        now = time.time()

        if success:
            return ExperienceSignal(
                signal_id=hashlib.md5(f"{tenant_id}:{task_id}:{now}".encode()).hexdigest()[:16],
                tenant_id=tenant_id,
                session_id=session_id,
                timestamp=now,
                signal_type=SignalType.VALIDATION_PASSED,
                layer=MemoryLayer.EPISODIC,
                content={
                    "task_id": task_id,
                    "output":  output[:500],
                    "approach": task_id,
                },
                importance=0.6,
                agent_id=agent_id,
            )
        else:
            signal_type = SignalType.ERROR_RESOLVED if mitigation else SignalType.VALIDATION_FAILED
            return ExperienceSignal(
                signal_id=hashlib.md5(f"{tenant_id}:{task_id}:error:{now}".encode()).hexdigest()[:16],
                tenant_id=tenant_id,
                session_id=session_id,
                timestamp=now,
                signal_type=signal_type,
                layer=MemoryLayer.EPISODIC,
                content={
                    "task_id":    task_id,
                    "error_type": error_type or "task_failure",
                    "mitigation": mitigation,
                    "approach":   task_id,
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
            signal_id=hashlib.md5(f"{tenant_id}:{session_id}:flush:{time.time()}".encode()).hexdigest()[:16],
            tenant_id=tenant_id,
            session_id=session_id,
            timestamp=time.time(),
            signal_type=SignalType.CONTEXT_UPDATE,
            layer=MemoryLayer.WORKING,
            content={"flush": True, "reason": "task_end_context_isolation"},
            importance=0.3,
            agent_id=agent_id,
        )
