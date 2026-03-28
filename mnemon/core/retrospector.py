"""
Mnemon Retrospector
Post-hoc decision analysis: traces EME decisions, detects failure patterns,
quarantines bad fragments, and emits corrective signals via the bus.

Never imports from or writes to EROSDatabase directly.
Every method has try/except — Retrospector failure must never affect EME or bus.

Architecture by Mahika Jadhav (smartass-4ever).
"""

import asyncio
import hashlib
import logging
import time
from typing import Optional

from .models import (
    DecisionTrace,
    ExperienceSignal,
    MemoryLayer,
    RiskLevel,
    SignalType,
)
from .system_db import SystemDatabase

logger = logging.getLogger(__name__)

# Number of failed traces referencing the same fragment to confirm a pattern
PATTERN_CONFIRM_THRESHOLD = 2


class Retrospector:
    """
    Post-hoc decision auditor.

    Lifecycle:
        await retrospector.start()   # starts hourly quarantine-expiry loop
        await retrospector.stop()    # cancels background task

    Primary entry point:
        await retrospector.submit_trace(trace)   # called by EME via create_task

    All public methods are fully guarded — any exception is caught and logged.
    """

    def __init__(
        self,
        bus,
        eme,
        memory,
        system_db: SystemDatabase,
        llm_client=None,
    ):
        self.bus       = bus
        self.eme       = eme
        self.memory    = memory
        self.system_db = system_db
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
    # TRACE SUBMISSION
    # ──────────────────────────────────────────

    async def submit_trace(self, trace: DecisionTrace):
        """
        Persist a decision trace and trigger retrospective analysis when the
        overall outcome indicates a failure. Called by EME via create_task —
        must never raise.
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

            if trace.overall_outcome in ("fail", "failure", "error"):
                await self._analyse_failure(trace)
        except Exception as e:
            logger.error(f"submit_trace failed (non-fatal): {e}")

    # ──────────────────────────────────────────
    # BUS CALLBACK
    # ──────────────────────────────────────────

    async def _on_bus_signal(self, signal: ExperienceSignal):
        """
        Tier 1 bus callback registered via register_retrospector().
        Currently logs SUCCESS signals for positive correlation tracking.
        """
        try:
            if signal.signal_type != SignalType.SUCCESS:
                return
            logger.debug(
                f"Retrospector: SUCCESS signal {signal.signal_id} "
                f"tenant={signal.tenant_id}"
            )
        except Exception as e:
            logger.error(f"_on_bus_signal failed (non-fatal): {e}")

    # ──────────────────────────────────────────
    # FAILURE ANALYSIS PIPELINE
    # ──────────────────────────────────────────

    async def _analyse_failure(self, trace: DecisionTrace):
        """Full retrospective pipeline for a failed trace."""
        try:
            cause = await self._locate_cause(trace)
            suspect_frag_id = cause.get("suspect_fragment_id")
            if not suspect_frag_id:
                return

            pattern_confirmed = await self._correlate_patterns(
                suspect_frag_id, trace.tenant_id
            )
            if not pattern_confirmed:
                return

            pattern_row = {
                "fragment_id": suspect_frag_id,
                "tenant_id":   trace.tenant_id,
                "failed_step": cause.get("failed_step"),
                "outcome":     trace.overall_outcome,
            }
            signal_type_str = await self._compress_finding(pattern_row)

            try:
                signal_type = SignalType(signal_type_str)
            except ValueError:
                signal_type = SignalType.ANOMALY

            summary = (
                f"Fragment {suspect_frag_id} linked to repeated failures "
                f"in tenant {trace.tenant_id}. "
                f"First bad step: {cause.get('failed_step')}."
            )
            await self._apply_signal(
                signal_type, suspect_frag_id, trace.tenant_id, summary
            )
        except Exception as e:
            logger.error(f"_analyse_failure failed (non-fatal): {e}")

    async def _locate_cause(self, trace: DecisionTrace) -> dict:
        """
        Pure math — scan step_outcomes for the first failed/timeout step and
        map it to the fragment that was in use at that position.

        Returns dict with keys 'failed_step' and 'suspect_fragment_id'.
        """
        try:
            failed_step = None
            for step_id, outcome in trace.step_outcomes.items():
                if outcome in ("fail", "timeout"):
                    failed_step = step_id
                    break

            if not trace.fragment_ids_used:
                return {"failed_step": failed_step, "suspect_fragment_id": None}

            if failed_step is None:
                # No individual step failed but overall outcome was failure.
                # Point at the last fragment used as the most likely culprit.
                suspect = trace.fragment_ids_used[-1]
                return {"failed_step": None, "suspect_fragment_id": suspect}

            # Map failed step index → fragment at same position in the used list.
            step_ids = list(trace.step_outcomes.keys())
            step_idx = step_ids.index(failed_step)
            idx = min(step_idx, len(trace.fragment_ids_used) - 1)
            suspect = trace.fragment_ids_used[idx]

            return {"failed_step": failed_step, "suspect_fragment_id": suspect}
        except Exception as e:
            logger.error(f"_locate_cause failed (non-fatal): {e}")
            return {}

    async def _correlate_patterns(
        self, suspect_fragment_id: str, tenant_id: str
    ) -> bool:
        """
        Fetch traces that used this fragment over the last 7 days.
        Returns True when >= PATTERN_CONFIRM_THRESHOLD of them are failures
        for this tenant, confirming a repeatable pattern.
        """
        try:
            traces = await self.system_db.fetch_traces_by_fragment(suspect_fragment_id)
            failed = [
                t for t in traces
                if t.get("tenant_id") == tenant_id
                and t.get("overall_outcome") in ("fail", "failure", "error")
            ]
            confirmed = len(failed) >= PATTERN_CONFIRM_THRESHOLD
            if confirmed:
                logger.info(
                    f"Pattern confirmed: fragment={suspect_fragment_id} "
                    f"appeared in {len(failed)} failures for tenant={tenant_id}"
                )
            return confirmed
        except Exception as e:
            logger.error(f"_correlate_patterns failed (non-fatal): {e}")
            return False

    async def _compress_finding(self, pattern_row: dict) -> str:
        """
        Single claude-haiku-4-5 call to classify the corrective signal type.
        Returns a SignalType value string. Falls back to 'anomaly' on any error.
        max_tokens=100 — this must stay cheap.
        """
        try:
            if not self.llm:
                return SignalType.ANOMALY.value

            prompt = (
                "An AI execution-cache fragment caused repeated task failures.\n"
                f"Fragment ID : {pattern_row.get('fragment_id')}\n"
                f"Failed step : {pattern_row.get('failed_step')}\n"
                f"Outcome     : {pattern_row.get('outcome')}\n\n"
                "Classify this event as exactly one of:\n"
                "  anomaly | degradation | failure | pattern_found\n"
                "Reply with only the single classification word."
            )
            response = await self.llm.complete(
                prompt=prompt,
                model="claude-haiku-4-5-20251001",
                max_tokens=100,
            )
            result = response.strip().lower()
            valid = {st.value for st in SignalType}
            return result if result in valid else SignalType.ANOMALY.value
        except Exception as e:
            logger.error(f"_compress_finding failed (non-fatal): {e}")
            return SignalType.ANOMALY.value

    async def _apply_signal(
        self,
        signal_type: SignalType,
        affected_id: str,
        tenant_id: str,
        summary: str,
    ):
        """
        Quarantine the suspect fragment, write a finding record, and emit a
        corrective signal on the bus so the rest of the system learns.
        """
        try:
            await self.system_db.quarantine(
                item_type="fragment",
                item_id=affected_id,
                tenant_id=tenant_id,
                reason=summary,
                confidence=0.85,
                ttl_hours=168,
            )

            finding_id = hashlib.md5(
                f"{tenant_id}:{affected_id}:{time.time()}".encode()
            ).hexdigest()[:16]

            await self.system_db.write_finding({
                "finding_id":  finding_id,
                "tenant_id":   tenant_id,
                "signal_type": signal_type.value,
                "affected_id": affected_id,
                "summary":     summary,
                "created_at":  time.time(),
            })

            if self.bus:
                signal = ExperienceSignal(
                    signal_id=finding_id,
                    tenant_id=tenant_id,
                    session_id="retrospector",
                    timestamp=time.time(),
                    signal_type=signal_type,
                    layer=MemoryLayer.EPISODIC,
                    content={
                        "fragment_id": affected_id,
                        "summary":     summary,
                        "source":      "retrospector",
                    },
                    importance=0.9,
                    risk_level=RiskLevel.HIGH,
                )
                await self.bus.broadcast_signal(signal)

            logger.info(
                f"Applied {signal_type.value} signal: fragment={affected_id} "
                f"tenant={tenant_id}"
            )
        except Exception as e:
            logger.error(f"_apply_signal failed (non-fatal): {e}")

    # ──────────────────────────────────────────
    # BACKGROUND LOOP
    # ──────────────────────────────────────────

    async def _quarantine_check_loop(self):
        """Runs every hour, removing quarantine entries that have expired."""
        while self._running:
            try:
                await asyncio.sleep(3600)
                await self.system_db.expire_quarantines()
                logger.debug("Quarantine expiry check complete")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"_quarantine_check_loop error (non-fatal): {e}")
