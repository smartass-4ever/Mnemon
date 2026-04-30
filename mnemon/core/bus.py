"""
Mnemon Experience Bus — Evidence Collector (v2)

Replaced phantom success/failure binary with rich EvidenceRecords assembled
by FeedbackExtractor from real framework signals.

Sources:
  m.run()            — plan-level outcomes with fragment_ids and real latency
  FeedbackExtractor  — framework-level signals: exceptions, tool errors,
                       validation failures, wrong plans, near misses
  m.mark_failure()   — explicit user signal

Detects:
  DEGRADATION    — latency spike vs rolling baseline (real latency, not 0ms)
  PATTERN_FOUND  — task_type failure rate > 30%
  ANOMALY        — sudden failure after string of successes
  RECOVERY       — success after failure streak
  CASCADE        — root cause identified when multiple consecutive steps fail

Routes full EvidenceRecord to Retrospector — dead pathway replaced.
record_outcome() kept for backward compatibility.

Architecture by Mahika Jadhav (smartass-4ever).
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import (
    EvidenceRecord,
    ExperienceSignal,
    MemoryLayer,
    SignalType,
    MNEMON_VERSION,
)
from .persistence import EROSDatabase

logger = logging.getLogger(__name__)

_RETRY_WINDOW_S   = 30.0   # seconds — same goal_hash within this window = near miss
_CASCADE_MIN_TAIL = 2       # min fragments after failed_step to call it a cascade


@dataclass
class Tier1Observation:
    """Lightweight internal record for rolling baseline tracking."""
    tenant_id:     str
    task_id:       str
    task_type:     str
    outcome:       str           # "success"|"failure"|"wrong_plan"|"near_miss"
    latency_ms:    float
    token_cost:    int
    timestamp:     float
    failure_class: Optional[str] = None
    framework:     str = "unknown"


class Tier1Observer:
    """
    Rolling pattern detector — baselines per task_type, signal emission.
    Now tracks framework + failure_class breakdown for richer pattern detection.
    wrong_plan counts as failure for failure-rate computation.
    """

    def __init__(self, tenant_id: str):
        self.tenant_id       = tenant_id
        self._history:       deque = deque(maxlen=1000)
        self._baselines:     Dict[str, Dict] = {}
        self._failure_rates: Dict[str, float] = {}
        self._callbacks:     List[Callable] = []
        self._lock           = asyncio.Lock()

    def register_callback(self, cb: Callable) -> None:
        self._callbacks.append(cb)

    async def record(self, obs: Tier1Observation) -> None:
        async with self._lock:
            self._history.append(obs)
            await self._update_baseline(obs)
            signals = await self._detect_patterns(obs)
        for sig in signals:
            for cb in self._callbacks:
                try:
                    await cb(sig)
                except Exception as e:
                    logger.warning(f"Tier1 callback error: {e}")

    async def _update_baseline(self, obs: Tier1Observation) -> None:
        t = obs.task_type
        if t not in self._baselines:
            self._baselines[t] = {"latency": obs.latency_ms, "tokens": obs.token_cost, "n": 1}
        else:
            b = self._baselines[t]
            n = b["n"]
            b["latency"] = (b["latency"] * n + obs.latency_ms) / (n + 1)
            b["tokens"]  = (b["tokens"]  * n + obs.token_cost)  / (n + 1)
            b["n"]       = n + 1
        recent = [h for h in self._history if h.task_type == t][-20:]
        if recent:
            failures = sum(1 for h in recent if h.outcome in ("failure", "wrong_plan"))
            self._failure_rates[t] = failures / len(recent)

    async def _detect_patterns(self, obs: Tier1Observation) -> List[ExperienceSignal]:
        signals = []
        t        = obs.task_type
        baseline = self._baselines.get(t, {})

        # Latency spike — only when latency > 0 (Moth cache hits send 0.0)
        if baseline.get("latency") and obs.latency_ms > 0 and obs.latency_ms > baseline["latency"] * 1.5:
            signals.append(self._make_signal(
                SignalType.DEGRADATION, obs,
                {"reason": "latency_spike", "baseline_ms": baseline["latency"],
                 "current_ms": obs.latency_ms},
            ))

        # High failure rate
        failure_rate = self._failure_rates.get(t, 0.0)
        if failure_rate > 0.30:
            signals.append(self._make_signal(
                SignalType.PATTERN_FOUND, obs,
                {"reason": "high_failure_rate", "rate": failure_rate, "task_type": t},
            ))

        # Sudden failure after run of successes
        recent = [h for h in self._history if h.task_type == t][-5:]
        if len(recent) >= 4:
            prev_ok   = all(h.outcome == "success" for h in recent[:-1])
            curr_fail = obs.outcome in ("failure", "wrong_plan")
            if prev_ok and curr_fail:
                signals.append(self._make_signal(
                    SignalType.ANOMALY, obs,
                    {"reason": "sudden_failure_after_successes", "task_type": t,
                     "failure_class": obs.failure_class, "framework": obs.framework},
                ))

        # Recovery
        if obs.outcome == "success" and failure_rate > 0.0:
            recent_outcomes = [h.outcome for h in list(self._history)[-5:]]
            if any(o in ("failure", "wrong_plan") for o in recent_outcomes):
                signals.append(self._make_signal(
                    SignalType.RECOVERY, obs,
                    {"reason": "recovery_after_failures", "task_type": t},
                ))

        return signals

    def _make_signal(
        self, signal_type: SignalType, obs: Tier1Observation, content: Dict
    ) -> ExperienceSignal:
        return ExperienceSignal(
            signal_id=hashlib.md5(
                f"{self.tenant_id}:{obs.task_id}:{signal_type.value}:{time.time()}".encode()
            ).hexdigest()[:16],
            tenant_id=self.tenant_id,
            session_id=obs.task_id,
            timestamp=time.time(),
            signal_type=signal_type,
            layer=MemoryLayer.EPISODIC,
            content={**content, "task_type": obs.task_type, "task_id": obs.task_id},
            importance=0.7,
        )

    def get_stats(self) -> Dict:
        return {
            "observations":  len(self._history),
            "task_types":    len(self._baselines),
            "failure_rates": dict(self._failure_rates),
        }


class ExperienceBus:
    """
    Evidence Collector — v2.

    record_evidence(evidence) is the primary entry point.
    record_outcome() is kept for backward compatibility.
    """

    def __init__(self, tenant_id: str, db: EROSDatabase, **_ignored):
        self.tenant_id      = tenant_id
        self.db             = db
        self.tier1          = Tier1Observer(tenant_id)
        self.tier1_signals  = 0
        self._retrospector  = None
        self._running       = False
        # goal_hash → (task_id, timestamp) for near-miss detection
        self._recent_hashes: Dict[str, Tuple[str, float]] = {}

        self.tier1.register_callback(self._on_tier1_signal)

    def register_retrospector(self, retrospector) -> None:
        self._retrospector = retrospector
        self.tier1.register_callback(retrospector._on_bus_signal)

    # ──────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────

    def _sidecar_path(self) -> Optional[str]:
        db_dir = getattr(self.db, "db_dir", None)
        if not db_dir or db_dir == ":memory:":
            return None
        return os.path.join(db_dir, f"mnemon_bus_{self.tenant_id}.json")

    async def start(self) -> None:
        self._running = True
        self._load_sidecar()
        logger.info(f"Experience bus started for tenant {self.tenant_id}")

    def _load_sidecar(self) -> None:
        path = self._sidecar_path()
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.tier1._baselines.update(data.get("baselines", {}))
            self.tier1._failure_rates.update(data.get("failure_rates", {}))
            for entry in data.get("history", [])[-200:]:
                self.tier1._history.append(Tier1Observation(**entry))
            logger.debug(f"Bus sidecar loaded: {path}")
        except Exception as e:
            logger.debug(f"Bus sidecar load failed (non-critical): {e}")

    async def stop(self) -> None:
        self._running = False
        self._save_sidecar()
        logger.info(f"Experience bus stopped for tenant {self.tenant_id}")

    def _save_sidecar(self) -> None:
        path = self._sidecar_path()
        if not path:
            return
        try:
            history_snapshot = [
                {
                    "tenant_id":    o.tenant_id,
                    "task_id":      o.task_id,
                    "task_type":    o.task_type,
                    "outcome":      o.outcome,
                    "latency_ms":   o.latency_ms,
                    "token_cost":   o.token_cost,
                    "timestamp":    o.timestamp,
                    "failure_class": o.failure_class,
                    "framework":    o.framework,
                }
                for o in list(self.tier1._history)[-200:]
            ]
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "baselines":     self.tier1._baselines,
                    "failure_rates": self.tier1._failure_rates,
                    "history":       history_snapshot,
                }, f)
            logger.debug(f"Bus sidecar saved: {path}")
        except Exception as e:
            logger.debug(f"Bus sidecar save failed (non-critical): {e}")

    # ──────────────────────────────────────────
    # PRIMARY ENTRY POINT
    # ──────────────────────────────────────────

    async def record_evidence(self, evidence: EvidenceRecord) -> None:
        """
        Accept a rich EvidenceRecord and:
          1. Run cascade detection when fragment_ids + failed_step are present
          2. Detect near-miss (retry pattern via goal_hash registry)
          3. Update Tier1Observer rolling baseline
          4. Route full EvidenceRecord to Retrospector on failure/wrong_plan/near_miss
        """
        try:
            evidence = self._detect_cascade(evidence)
            evidence = self._check_near_miss(evidence)

            obs = Tier1Observation(
                tenant_id=evidence.tenant_id,
                task_id=evidence.task_id,
                task_type=evidence.goal_type or "general",
                outcome=evidence.outcome,
                latency_ms=evidence.latency_ms,
                token_cost=0,
                timestamp=evidence.timestamp,
                failure_class=evidence.failure_class,
                framework=evidence.framework,
            )
            await self.tier1.record(obs)
            self.tier1_signals += 1

            # Update near-miss registry
            if evidence.goal_hash:
                self._recent_hashes[evidence.goal_hash] = (
                    evidence.task_id, evidence.timestamp
                )
                cutoff = time.time() - _RETRY_WINDOW_S * 10
                self._recent_hashes = {
                    h: v for h, v in self._recent_hashes.items() if v[1] > cutoff
                }

            # Route to Retrospector with full context — every failure/wrong_plan/near_miss
            if (self._retrospector
                    and evidence.outcome in ("failure", "wrong_plan", "near_miss")):
                try:
                    asyncio.create_task(
                        self._retrospector.analyse_evidence(evidence)
                    )
                except Exception as e:
                    logger.debug(f"Bus→Retrospector route failed (non-critical): {e}")

        except Exception as e:
            logger.warning(f"record_evidence failed (non-critical): {e}")

    # ──────────────────────────────────────────
    # SIGNAL DETECTION
    # ──────────────────────────────────────────

    def _detect_cascade(self, evidence: EvidenceRecord) -> EvidenceRecord:
        """
        If failed_step N exists and there are ≥ _CASCADE_MIN_TAIL more fragments
        after N, set cascade_root = N. Everything after N is collateral damage —
        Retrospector should blame N, not the last fragment.
        """
        try:
            if (evidence.failed_step is not None
                    and evidence.cascade_root is None
                    and len(evidence.fragment_ids_used) > evidence.failed_step + _CASCADE_MIN_TAIL):
                return EvidenceRecord(
                    task_id=evidence.task_id,
                    tenant_id=evidence.tenant_id,
                    template_id=evidence.template_id,
                    fragment_ids_used=evidence.fragment_ids_used,
                    framework=evidence.framework,
                    outcome=evidence.outcome,
                    failure_class=evidence.failure_class,
                    error_type=evidence.error_type,
                    error_message=evidence.error_message,
                    failed_step=evidence.failed_step,
                    cascade_root=evidence.failed_step,
                    tool_name=evidence.tool_name,
                    framework_context=evidence.framework_context,
                    goal_hash=evidence.goal_hash,
                    goal_type=evidence.goal_type,
                    latency_ms=evidence.latency_ms,
                    timestamp=evidence.timestamp,
                    is_retry=evidence.is_retry,
                    retry_diff=evidence.retry_diff,
                )
        except Exception:
            pass
        return evidence

    def _check_near_miss(self, evidence: EvidenceRecord) -> EvidenceRecord:
        """
        If this success arrives within _RETRY_WINDOW_S of a prior attempt with
        the same goal_hash, classify it as near_miss (retry-then-succeed).
        """
        try:
            if (evidence.outcome == "success"
                    and not evidence.is_retry
                    and evidence.goal_hash):
                prev = self._recent_hashes.get(evidence.goal_hash)
                if prev and time.time() - prev[1] < _RETRY_WINDOW_S:
                    return EvidenceRecord(
                        task_id=evidence.task_id,
                        tenant_id=evidence.tenant_id,
                        template_id=evidence.template_id,
                        fragment_ids_used=evidence.fragment_ids_used,
                        framework=evidence.framework,
                        outcome="near_miss",
                        failure_class="retry",
                        error_type=None,
                        error_message=None,
                        failed_step=None,
                        cascade_root=None,
                        tool_name=None,
                        framework_context=evidence.framework_context,
                        goal_hash=evidence.goal_hash,
                        goal_type=evidence.goal_type,
                        latency_ms=evidence.latency_ms,
                        timestamp=evidence.timestamp,
                        is_retry=True,
                        retry_diff=evidence.retry_diff,
                    )
        except Exception:
            pass
        return evidence

    # ──────────────────────────────────────────
    # BACKWARD COMPATIBILITY
    # ──────────────────────────────────────────

    async def record_outcome(
        self,
        task_id:    str,
        task_type:  str,
        outcome:    str,
        latency_ms: float,
        token_cost: int = 0,
    ) -> None:
        """Backward-compatible wrapper — builds minimal EvidenceRecord."""
        await self.record_evidence(EvidenceRecord(
            task_id=task_id,
            tenant_id=self.tenant_id,
            template_id=None,
            fragment_ids_used=[],
            framework="unknown",
            outcome=outcome,
            failure_class="manual" if outcome == "failure" else None,
            error_type=None,
            error_message=None,
            failed_step=None,
            cascade_root=None,
            tool_name=None,
            goal_hash=None,
            goal_type=task_type,
            latency_ms=latency_ms,
            timestamp=time.time(),
        ))

    async def _on_tier1_signal(self, signal: ExperienceSignal) -> None:
        logger.debug(
            f"Bus signal [{signal.signal_type.value}] "
            f"task_type={signal.content.get('task_type')} "
            f"reason={signal.content.get('reason')}"
        )

    async def broadcast_signal(self, signal: ExperienceSignal) -> None:
        for cb in self.tier1._callbacks:
            try:
                await cb(signal)
            except Exception as e:
                logger.warning(f"Bus broadcast_signal callback error: {e}")

    def get_stats(self) -> Dict:
        return {
            "tenant_id":     self.tenant_id,
            "tier1":         self.tier1.get_stats(),
            "tier1_signals": self.tier1_signals,
        }
