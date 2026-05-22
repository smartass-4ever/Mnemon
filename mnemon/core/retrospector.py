"""
Mnemon Retrospector — Failure Analyst (v2)

Receives full EvidenceRecords from the Bus (real data, not phantom signals).
Also receives DecisionTraces from EME via submit_trace().

Pipeline:
  _diagnose()          — precise attribution using cascade_root/failed_step/failure_class
                         instead of always blaming the last fragment
  _correlate()         — confirm pattern (≥2 failures, same fragment + reason)
  _update_reputation() — update fragment reputation + edge weights on EVERY evidence
                         (not just confirmed patterns — each failure is a data point)
  _prescribe()         — quarantine / framework-incompatibility / edge penalty
                         only after pattern confirmed, with confidence-gated quarantine

No LLM calls — _compress_finding() haiku call replaced with code classification table.
Every method fully guarded — Retrospector failure never affects EME or Bus.

Architecture by Mahika Jadhav (smartass-4ever).
"""

import asyncio
import hashlib
import logging
import time
from typing import Dict, List, Optional

from .models import (
    DecisionTrace,
    EvidenceRecord,
    ExperienceSignal,
    MemoryLayer,
    RiskLevel,
    SignalType,
)
from .system_db import SystemDatabase

logger = logging.getLogger(__name__)

PATTERN_CONFIRM_THRESHOLD = 2

# Quarantine TTL by failure class — severity determines duration
_QUARANTINE_TTL: Dict[str, int] = {
    "exception":   168,   # 7 days — hard crash
    "tool_error":   72,   # 3 days
    "validation":   48,   # 2 days
    "schema":       48,
    "max_iter":     24,   # 1 day — fragile, not broken
    "retry":        12,   # 12 hours
    "wrong_plan":   96,   # 4 days
    "manual":      168,
    "default":     168,
}

# Code-based signal classification — replaces haiku call
_FAILURE_CLASS_TO_SIGNAL: Dict[str, str] = {
    "exception":   SignalType.FAILURE.value,
    "tool_error":  SignalType.ANOMALY.value,
    "validation":  SignalType.PATTERN_FOUND.value,
    "schema":      SignalType.PATTERN_FOUND.value,
    "max_iter":    SignalType.DEGRADATION.value,
    "retry":       SignalType.DEGRADATION.value,
    "wrong_plan":  SignalType.PATTERN_FOUND.value,
    "manual":      SignalType.FAILURE.value,
}


class Retrospector:
    """
    Failure Analyst — diagnoses, correlates, prescribes, updates fragment reputation.

    Two entry points:
      submit_trace(trace)        — from EME after every run
      analyse_evidence(evidence) — from Bus after framework signal

    Both converge on: _diagnose → _correlate → _update_reputation → _prescribe.
    """

    def __init__(
        self,
        bus,
        eme,
        memory,
        system_db: SystemDatabase,
        llm_client=None,
        signal_db=None,
    ):
        self.bus       = bus
        self.eme       = eme
        self.memory    = memory
        self.system_db = system_db
        self.signal_db = signal_db
        self.llm       = llm_client

        self._running = False
        self._quarantine_loop_task: Optional[asyncio.Task] = None

    # ──────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────

    async def start(self):
        try:
            self._running = True
            self._quarantine_loop_task = asyncio.create_task(
                self._quarantine_check_loop()
            )
            logger.info("Retrospector started")
        except Exception as e:
            logger.error(f"Retrospector start failed: {e}")

    async def stop(self):
        try:
            self._running = False
            if self._quarantine_loop_task:
                self._quarantine_loop_task.cancel()
                try:
                    await self._quarantine_loop_task
                except asyncio.CancelledError:
                    pass
            logger.info("Retrospector stopped")
        except Exception as e:
            logger.error(f"Retrospector stop failed: {e}")

    # ──────────────────────────────────────────
    # ENTRY POINT 1 — from EME
    # ──────────────────────────────────────────

    async def submit_trace(self, trace: DecisionTrace):
        """
        Persist a decision trace. On failure, build a minimal EvidenceRecord
        and route through the full analysis pipeline.
        """
        try:
            await self.system_db.write_trace({
                "trace_id":             trace.trace_id,
                "tenant_id":            trace.tenant_id,
                "task_id":              trace.task_id,
                "goal_hash":            trace.goal_hash,
                "fragment_ids_used":    trace.fragment_ids_used,
                "memory_ids_retrieved": trace.memory_ids_retrieved,
                "segments_generated":   trace.segments_generated,
                "tools_called":         trace.tools_called,
                "step_outcomes":        trace.step_outcomes,
                "overall_outcome":      trace.overall_outcome,
                "latency_ms":           trace.latency_ms,
                "timestamp":            trace.timestamp,
            })
            if trace.overall_outcome == "failure":
                await self.analyse_evidence(EvidenceRecord(
                    task_id=trace.task_id,
                    tenant_id=trace.tenant_id,
                    template_id=None,
                    fragment_ids_used=trace.fragment_ids_used,
                    framework="unknown",
                    outcome="failure",
                    failure_class="manual",
                    error_type=None,
                    error_message=None,
                    failed_step=self._step_from_outcomes(trace.step_outcomes),
                    cascade_root=None,
                    tool_name=None,
                    goal_hash=trace.goal_hash,
                    goal_type=None,
                    latency_ms=trace.latency_ms,
                    timestamp=trace.timestamp,
                ))
        except Exception as e:
            logger.error(f"submit_trace failed (non-fatal): {e}")

    def _step_from_outcomes(self, step_outcomes: Dict) -> Optional[int]:
        try:
            for i, (_, outcome) in enumerate(step_outcomes.items()):
                if outcome in ("fail", "timeout"):
                    return i
        except Exception:
            pass
        return None

    # ──────────────────────────────────────────
    # ENTRY POINT 2 — from Bus
    # ──────────────────────────────────────────

    async def analyse_evidence(self, evidence: EvidenceRecord):
        """Full analysis pipeline. Must never raise."""
        try:
            diagnosis = await self._diagnose(evidence)
            if not diagnosis:
                return

            # Update reputation on every evidence — each data point matters
            await self._update_reputation(evidence, diagnosis["fragment_id"])

            pattern_confirmed = await self._correlate(
                diagnosis["fragment_id"], evidence.tenant_id, evidence.failure_class
            )
            if pattern_confirmed:
                await self._prescribe(evidence, diagnosis)

        except Exception as e:
            logger.error(f"analyse_evidence failed (non-fatal): {e}")

    # ──────────────────────────────────────────
    # BUS CALLBACK
    # ──────────────────────────────────────────

    async def _on_bus_signal(self, signal: ExperienceSignal):
        """
        Tier 1 bus pattern signals (DEGRADATION, PATTERN_FOUND, ANOMALY, RECOVERY).
        These are aggregate pattern alerts — individual failure routing happens via
        analyse_evidence(). Just log here; no fragment_id available to act on.
        """
        try:
            logger.info(
                f"Retrospector: bus signal [{signal.signal_type.value}] "
                f"task_type={signal.content.get('task_type')} "
                f"reason={signal.content.get('reason')} "
                f"framework={signal.content.get('framework', 'unknown')}"
            )
        except Exception as e:
            logger.error(f"_on_bus_signal failed (non-fatal): {e}")

    # ──────────────────────────────────────────
    # DIAGNOSIS
    # ──────────────────────────────────────────

    async def _diagnose(self, evidence: EvidenceRecord) -> Optional[Dict]:
        """
        Attribute failure to specific fragment with confidence score.

        Priority:
          1. cascade_root (Bus confirmed root cause) — confidence 0.95
          2. failed_step (framework told us position) — confidence 0.90
          3. failure_class inference (tool/schema signals) — confidence 0.60-0.65
          4. last fragment fallback — confidence 0.40, logged as uncertain
        """
        try:
            frags = evidence.fragment_ids_used
            if not frags:
                return None

            if evidence.cascade_root is not None:
                idx = min(evidence.cascade_root, len(frags) - 1)
                return {"fragment_id": frags[idx], "step": evidence.cascade_root,
                        "confidence": 0.95, "method": "cascade_root",
                        "reason": f"cascade root at step {evidence.cascade_root}"}

            if evidence.failed_step is not None:
                idx = min(evidence.failed_step, len(frags) - 1)
                return {"fragment_id": frags[idx], "step": evidence.failed_step,
                        "confidence": 0.90, "method": "failed_step",
                        "reason": f"{evidence.failure_class} at step {evidence.failed_step}"}

            if evidence.failure_class == "tool_error" and evidence.tool_name:
                return {"fragment_id": frags[-1], "step": len(frags) - 1,
                        "confidence": 0.60, "method": "tool_inference",
                        "reason": f"tool_error on {evidence.tool_name}"}

            if evidence.failure_class in ("validation", "schema"):
                return {"fragment_id": frags[-1], "step": len(frags) - 1,
                        "confidence": 0.65, "method": "schema_inference",
                        "reason": f"{evidence.failure_class} failure on output"}

            return {"fragment_id": frags[-1], "step": len(frags) - 1,
                    "confidence": 0.40, "method": "last_fragment_fallback",
                    "reason": "no step information available"}

        except Exception as e:
            logger.error(f"_diagnose failed (non-fatal): {e}")
            return None

    # ──────────────────────────────────────────
    # CORRELATION
    # ──────────────────────────────────────────

    async def _correlate(
        self, fragment_id: str, tenant_id: str, failure_class: Optional[str]
    ) -> bool:
        """Confirm pattern: ≥ PATTERN_CONFIRM_THRESHOLD failures for this fragment + tenant."""
        try:
            traces = await self.system_db.fetch_traces_by_fragment(fragment_id)
            failed = [
                t for t in traces
                if t.get("tenant_id") == tenant_id
                and t.get("overall_outcome") == "failure"
            ]
            confirmed = len(failed) >= PATTERN_CONFIRM_THRESHOLD
            if confirmed:
                logger.info(
                    f"Pattern confirmed: fragment={fragment_id} failures={len(failed)} "
                    f"tenant={tenant_id} failure_class={failure_class}"
                )
            return confirmed
        except Exception as e:
            logger.error(f"_correlate failed (non-fatal): {e}")
            return False

    # ──────────────────────────────────────────
    # REPUTATION UPDATE — every evidence, not just confirmed patterns
    # ──────────────────────────────────────────

    async def _update_reputation(self, evidence: EvidenceRecord, fragment_id: str):
        """
        Update fragment reputation and edge weights on every evidence record.
        On success: strengthen ALL edges in the plan (Hebbian).
        On failure: weaken the edge leading to the failed fragment.
        """
        if not self.signal_db:
            return
        try:
            fw        = evidence.framework or "unknown"
            gt        = evidence.goal_type  or "general"
            outcome   = evidence.outcome
            issue     = evidence.failure_class or ""

            await self.signal_db.update_fragment_reputation(
                fragment_id=fragment_id,
                framework=fw,
                goal_type=gt,
                outcome=outcome,
                issue=issue if outcome != "success" else None,
            )

            frags = evidence.fragment_ids_used
            if outcome == "success":
                # Strengthen all edges in the successful plan
                for i in range(len(frags) - 1):
                    asyncio.create_task(
                        self.signal_db.update_edge_strength(
                            from_id=frags[i], to_id=frags[i + 1],
                            framework=fw, success=True, issue=None,
                        )
                    )
            else:
                # Weaken the edge leading into the failed fragment
                if fragment_id in frags:
                    idx = frags.index(fragment_id)
                    if idx > 0:
                        asyncio.create_task(
                            self.signal_db.update_edge_strength(
                                from_id=frags[idx - 1], to_id=fragment_id,
                                framework=fw, success=False, issue=issue,
                            )
                        )
        except Exception as e:
            logger.error(f"_update_reputation failed (non-fatal): {e}")

    # ──────────────────────────────────────────
    # PRESCRIPTION — only on confirmed patterns
    # ──────────────────────────────────────────

    async def _prescribe(self, evidence: EvidenceRecord, diagnosis: Dict):
        """
        After pattern confirmed:
          confidence ≥ 0.70 → full quarantine with failure-class TTL
          confidence < 0.70 → soft-flag only (finding written, no quarantine)
        Broadcasts corrective signal with fragment_id so downstream components
        (EME, future Moth checks) can act on it.
        """
        try:
            fragment_id   = diagnosis["fragment_id"]
            confidence    = diagnosis["confidence"]
            failure_class = evidence.failure_class or "default"
            framework     = evidence.framework or "unknown"
            ttl           = _QUARANTINE_TTL.get(failure_class, _QUARANTINE_TTL["default"])

            if confidence >= 0.70:
                await self.system_db.quarantine(
                    item_type="fragment",
                    item_id=fragment_id,
                    tenant_id=evidence.tenant_id,
                    reason=(
                        f"{failure_class} confirmed via {diagnosis['method']} "
                        f"(conf={confidence:.2f}): {diagnosis['reason']}"
                    ),
                    confidence=confidence,
                    ttl_hours=ttl,
                )
                logger.info(
                    f"Fragment quarantined: {fragment_id} reason={failure_class} "
                    f"framework={framework} ttl={ttl}h conf={confidence:.2f}"
                )
            else:
                logger.info(
                    f"Fragment soft-flagged (conf={confidence:.2f} < 0.70): "
                    f"{fragment_id} reason={failure_class}"
                )

            signal_type_str = _FAILURE_CLASS_TO_SIGNAL.get(failure_class, SignalType.ANOMALY.value)
            try:
                signal_type = SignalType(signal_type_str)
            except ValueError:
                signal_type = SignalType.ANOMALY

            finding_id = hashlib.md5(
                f"{evidence.tenant_id}:{fragment_id}:{time.time()}".encode()
            ).hexdigest()[:16]

            await self.system_db.write_finding({
                "finding_id":  finding_id,
                "tenant_id":   evidence.tenant_id,
                "signal_type": signal_type.value,
                "affected_id": fragment_id,
                "summary": (
                    f"Fragment {fragment_id} linked to {failure_class} failures "
                    f"framework={framework} tenant={evidence.tenant_id}. "
                    f"Diagnosis: {diagnosis['reason']} (conf={confidence:.2f})"
                ),
                "created_at": time.time(),
            })

            if self.bus:
                await self.bus.broadcast_signal(ExperienceSignal(
                    signal_id=finding_id,
                    tenant_id=evidence.tenant_id,
                    session_id="retrospector",
                    timestamp=time.time(),
                    signal_type=signal_type,
                    layer=MemoryLayer.EPISODIC,
                    content={
                        "fragment_id":   fragment_id,
                        "failure_class": failure_class,
                        "framework":     framework,
                        "confidence":    confidence,
                        "method":        diagnosis["method"],
                        "summary":       diagnosis["reason"],
                        "source":        "retrospector",
                    },
                    importance=0.9,
                    risk_level=RiskLevel.HIGH,
                ))
        except Exception as e:
            logger.error(f"_prescribe failed (non-fatal): {e}")

    # ──────────────────────────────────────────
    # BACKGROUND LOOP
    # ──────────────────────────────────────────

    async def _quarantine_check_loop(self):
        while self._running:
            try:
                await asyncio.sleep(3600)
                await self.system_db.expire_quarantines()
                logger.debug("Quarantine expiry check complete")
            except asyncio.CancelledError:
                raise  # re-raise so asyncio can properly transition task to cancelled
            except Exception as e:
                logger.error(f"_quarantine_check_loop error (non-fatal): {e}")
