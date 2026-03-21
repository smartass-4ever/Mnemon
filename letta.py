"""
EROS Letta Adapter
Translates Letta/MemGPT concepts into EROS signals.

Directly addresses the RFC filed at Letta (triaged into official issue tracker):
- Heartbeat contention → StatelessSignalingBus pattern
- Sleep-time compute integration → async memory writes
- 1000+ concurrent conversations → Working memory isolation

Letta concepts translated:
  heartbeat event      → ExperienceSignal to experience bus
  core memory update   → semantic memory write
  archival memory      → episodic memory write
  recall memory query  → EROS retrieve()
  in-context memory    → working memory
  sleep-time compute   → async LLM router + tag verification

Usage:
    from eros.adapters.letta import LettaAdapter
    eros = EROS(tenant_id="x", adapter=LettaAdapter())
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from ..core.types import (
    ExperienceSignal, MemoryLayer, SignalType, RiskLevel
)
from ..core.eme import TemplateAdapter, ComputationFingerprint, TemplateSegment


class LettaAdapter(TemplateAdapter):
    """
    Adapter for Letta (MemGPT) framework.

    Translates:
    - Letta agent steps → EROS ExecutionTemplate segments
    - Letta core_memory → EROS semantic memory
    - Letta archival_memory → EROS episodic memory
    - Letta heartbeat → EROS experience signal + PAD health report
    """

    def decompose(self, template: Any) -> List[Dict]:
        """
        Letta templates are agent step sequences or function call chains.
        Each function call becomes one segment.
        """
        if isinstance(template, list):
            return [
                {
                    "id":      step.get("name", f"step_{i}") if isinstance(step, dict) else f"step_{i}",
                    "content": step,
                }
                for i, step in enumerate(template)
            ]
        if isinstance(template, dict):
            steps = template.get("steps", template.get("functions", [template]))
            return [
                {
                    "id":      s.get("name", f"step_{i}") if isinstance(s, dict) else f"step_{i}",
                    "content": s,
                }
                for i, s in enumerate(steps)
            ]
        return [{"id": "step_0", "content": template}]

    def reconstruct(self, segments: List[TemplateSegment]) -> Any:
        return {
            "steps":       [s.content for s in segments],
            "type":        "letta_agent_flow",
            "eros_cached": True,
        }

    def extract_signature(self, template: Any, goal: str) -> ComputationFingerprint:
        steps    = self.decompose(template)
        tools    = []
        persona  = ""
        human    = ""

        if isinstance(template, dict):
            persona = template.get("persona", "")
            human   = template.get("human", "")
            for s in steps:
                content = s.get("content", {})
                if isinstance(content, dict) and "name" in content:
                    tools.append(content["name"])

        return ComputationFingerprint.build(
            goal=goal,
            input_schema={s["id"]: "step" for s in steps},
            context={"persona": persona[:100], "human": human[:100]},
            capabilities=tools,
            constraints=template.get("constraints", {}) if isinstance(template, dict) else {},
        )

    # ──────────────────────────────────────────
    # SIGNAL TRANSLATORS
    # ──────────────────────────────────────────

    @staticmethod
    def heartbeat_to_signal(
        agent_id: str,
        heartbeat_data: Dict,
        tenant_id: str,
        session_id: str,
        goal: str = "",
    ) -> ExperienceSignal:
        """
        Translate a Letta heartbeat event into an EROS experience signal.

        Instead of the heartbeat directly editing memory (causing DB contention),
        it broadcasts a signal. The bus sequences it atomically.
        This is the core fix for the Letta RFC heartbeat contention issue.
        """
        now = time.time()
        content_type = heartbeat_data.get("type", "context_update")

        signal_type = SignalType.CONTEXT_UPDATE
        layer       = MemoryLayer.WORKING

        if content_type in ("tool_call_result", "function_result"):
            signal_type = SignalType.VALIDATION_PASSED
            layer       = MemoryLayer.EPISODIC
        elif content_type in ("error", "exception"):
            signal_type = SignalType.ERROR_RESOLVED
            layer       = MemoryLayer.EPISODIC
        elif content_type == "memory_edit":
            layer = MemoryLayer.SEMANTIC

        return ExperienceSignal(
            signal_id=hashlib.md5(
                f"{tenant_id}:{agent_id}:{now}:heartbeat".encode()
            ).hexdigest()[:16],
            tenant_id=tenant_id,
            session_id=session_id,
            timestamp=now,
            signal_type=signal_type,
            layer=layer,
            content={
                **heartbeat_data,
                "agent_id":     agent_id,
                "heartbeat_at": now,
            },
            importance=0.5,
            agent_id=agent_id,
        )

    @staticmethod
    def core_memory_to_signal(
        key: str,
        value: str,
        tenant_id: str,
        session_id: str,
        agent_id: str,
        confidence: float = 1.0,
    ) -> ExperienceSignal:
        """
        Translate a Letta core_memory_replace into EROS semantic memory.
        Core memory = stable facts → SemanticMemory layer.
        """
        now = time.time()
        return ExperienceSignal(
            signal_id=hashlib.md5(
                f"{tenant_id}:{key}:{now}".encode()
            ).hexdigest()[:16],
            tenant_id=tenant_id,
            session_id=session_id,
            timestamp=now,
            signal_type=SignalType.CONTEXT_UPDATE,
            layer=MemoryLayer.SEMANTIC,
            content={
                "key":        key,
                "value":      value,
                "confidence": confidence,
            },
            importance=0.8,
            agent_id=agent_id,
        )

    @staticmethod
    def archival_insert_to_signal(
        content: str,
        tenant_id: str,
        session_id: str,
        agent_id: str,
        importance: float = 0.6,
    ) -> ExperienceSignal:
        """
        Translate a Letta archival_memory_insert into EROS episodic memory.
        Archival memory = timestamped experiences → EpisodicMemory layer.
        """
        now = time.time()
        return ExperienceSignal(
            signal_id=hashlib.md5(
                f"{tenant_id}:{agent_id}:{now}:archival".encode()
            ).hexdigest()[:16],
            tenant_id=tenant_id,
            session_id=session_id,
            timestamp=now,
            signal_type=SignalType.CONTEXT_UPDATE,
            layer=MemoryLayer.EPISODIC,
            content={"text": content, "source": "archival_insert"},
            importance=importance,
            agent_id=agent_id,
        )

    @staticmethod
    def sleep_time_signals(
        session_id: str,
        tenant_id: str,
        agent_id: str,
        recent_messages: List[Dict],
    ) -> List[ExperienceSignal]:
        """
        Generate memory consolidation signals during Letta sleep-time compute.

        Letta's sleep-time compute runs when the agent is idle.
        EROS uses this window to write memory signals that were
        queued during active conversation — completing the async write pipeline.

        Returns signals for each significant moment in recent_messages.
        """
        signals = []
        now = time.time()

        for i, msg in enumerate(recent_messages[-10:]):  # process last 10
            role    = msg.get("role", "unknown")
            content = msg.get("content", "")

            if not content or role == "system":
                continue

            # Classify message content
            importance = 0.4
            layer      = MemoryLayer.EPISODIC

            if any(kw in content.lower() for kw in ["remember", "note", "important", "always"]):
                importance = 0.8
                layer      = MemoryLayer.SEMANTIC

            elif any(kw in content.lower() for kw in ["prefer", "like", "usually", "style"]):
                layer      = MemoryLayer.RELATIONSHIP
                importance = 0.7

            signals.append(ExperienceSignal(
                signal_id=hashlib.md5(
                    f"{tenant_id}:{session_id}:sleep:{i}:{now}".encode()
                ).hexdigest()[:16],
                tenant_id=tenant_id,
                session_id=session_id,
                timestamp=now - (len(recent_messages) - i),  # preserve order
                signal_type=SignalType.CONTEXT_UPDATE,
                layer=layer,
                content={"text": content[:500], "role": role, "source": "sleep_time"},
                importance=importance,
                agent_id=agent_id,
            ))

        return signals
