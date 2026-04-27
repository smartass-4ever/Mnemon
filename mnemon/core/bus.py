"""
Mnemon Experience Bus — Tier 1 (System Learning Loop)

Passive observer. Records every computation outcome, detects patterns,
feeds EME with success/failure intelligence.

No agents needed. Always on. Gets smarter with every run.

Signals emitted:
  DEGRADATION  — latency spike vs rolling baseline
  PATTERN_FOUND — task type failure rate > 30%
  ANOMALY      — sudden failure after string of successes
  RECOVERY     — success after failure streak

Tier 2 (PAD monitor, belief registry, agent propagation) → see _future/bus_tier2.py

Architecture by Mahika Jadhav (smartass-4ever).
"""

import asyncio
import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .models import ExperienceSignal, MemoryLayer, SignalType, MNEMON_VERSION
from .persistence import EROSDatabase

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TIER 1 — OBSERVATION
# ─────────────────────────────────────────────

@dataclass
class Tier1Observation:
    """One recorded computation outcome."""
    tenant_id:    str
    task_id:      str
    task_type:    str
    outcome:      str        # "success" | "failure"
    latency_ms:   float
    token_cost:   int
    timestamp:    float
    context_hash: str = ""


class Tier1Observer:
    """
    Passive observer — records everything, detects patterns.
    Always on. No agents needed.

    Feeds EME fragment library with success patterns.
    Detects degradation before it becomes a user-facing problem.
    """

    def __init__(self, tenant_id: str):
        self.tenant_id       = tenant_id
        self._history:       deque = deque(maxlen=1000)
        self._baselines:     Dict[str, Dict] = {}      # task_type → {avg_latency, avg_tokens, n}
        self._failure_rates: Dict[str, float] = {}     # task_type → failure rate
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
            failures = sum(1 for h in recent if h.outcome == "failure")
            self._failure_rates[t] = failures / len(recent)

    async def _detect_patterns(self, obs: Tier1Observation) -> List[ExperienceSignal]:
        signals = []
        t        = obs.task_type
        baseline = self._baselines.get(t, {})

        # Latency spike vs rolling baseline
        if baseline.get("latency") and obs.latency_ms > baseline["latency"] * 1.5:
            signals.append(self._make_signal(
                SignalType.DEGRADATION, obs,
                {"reason": "latency_spike", "baseline_ms": baseline["latency"], "current_ms": obs.latency_ms}
            ))

        # High failure rate
        failure_rate = self._failure_rates.get(t, 0.0)
        if failure_rate > 0.30:
            signals.append(self._make_signal(
                SignalType.PATTERN_FOUND, obs,
                {"reason": "high_failure_rate", "rate": failure_rate, "task_type": t}
            ))

        # Sudden failure after string of successes
        recent = [h for h in self._history if h.task_type == t][-5:]
        if len(recent) >= 4:
            if all(h.outcome == "success" for h in recent[:-1]) and obs.outcome == "failure":
                signals.append(self._make_signal(
                    SignalType.ANOMALY, obs,
                    {"reason": "sudden_failure_after_successes", "task_type": t}
                ))

        # Recovery after failures
        if obs.outcome == "success" and failure_rate > 0.0:
            recent_outcomes = [h.outcome for h in list(self._history)[-5:]]
            if "failure" in recent_outcomes:
                signals.append(self._make_signal(
                    SignalType.RECOVERY, obs,
                    {"reason": "recovery_after_failures", "task_type": t}
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


# ─────────────────────────────────────────────
# EXPERIENCE BUS
# ─────────────────────────────────────────────

class ExperienceBus:
    """
    Tier 1 experience bus — system learning loop.

    Intercepts every computation outcome. Builds baselines per task type.
    Detects degradation, anomalies, recovery. Feeds the Retrospector.
    Gets smarter with every run — zero configuration required.
    """

    def __init__(self, tenant_id: str, db: EROSDatabase, **_ignored):
        self.tenant_id     = tenant_id
        self.db            = db
        self.tier1         = Tier1Observer(tenant_id)
        self.tier1_signals = 0
        self._retrospector = None
        self._running      = False

        self.tier1.register_callback(self._on_tier1_signal)

    def register_retrospector(self, retrospector) -> None:
        """Attach Retrospector to receive Tier 1 signals."""
        self._retrospector = retrospector
        self.tier1.register_callback(retrospector._on_bus_signal)

    async def start(self) -> None:
        self._running = True
        logger.info(f"Experience bus started for tenant {self.tenant_id}")

    async def stop(self) -> None:
        self._running = False
        logger.info(f"Experience bus stopped for tenant {self.tenant_id}")

    async def record_outcome(
        self,
        task_id:    str,
        task_type:  str,
        outcome:    str,
        latency_ms: float,
        token_cost: int = 0,
    ) -> None:
        """Record one computation outcome. Universal entry point — call after every run."""
        obs = Tier1Observation(
            tenant_id=self.tenant_id,
            task_id=task_id,
            task_type=task_type,
            outcome=outcome,
            latency_ms=latency_ms,
            token_cost=token_cost,
            timestamp=time.time(),
        )
        await self.tier1.record(obs)
        self.tier1_signals += 1

    async def _on_tier1_signal(self, signal: ExperienceSignal) -> None:
        logger.debug(
            f"Bus signal [{signal.signal_type.value}] "
            f"task_type={signal.content.get('task_type')} "
            f"reason={signal.content.get('reason')}"
        )

    def get_stats(self) -> Dict:
        return {
            "tenant_id":     self.tenant_id,
            "tier1":         self.tier1.get_stats(),
            "tier1_signals": self.tier1_signals,
        }
