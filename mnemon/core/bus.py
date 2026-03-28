"""
Mnemon Experience Bus
Two-tier collective learning and coordination system.

Tier 1 — System Learning Loop:
    Universal. No agents needed. Passive observer records outcomes,
    detects patterns, feeds EME and memory.
    Signals: SUCCESS, FAILURE, DEGRADATION, RECOVERY, PATTERN, ANOMALY

Tier 2 — Agent Intelligence Layer:
    Activates when agents are present.
    PAD health monitor, knowledge propagation, atomic belief registry.
    Single sequential processor — no race conditions ever.

Architecture by Mahika Jadhav (smartass-4ever).
Extended with: two-tier design, PAD semantic similarity,
relevance filtering, belief registry with optimistic locking,
bootstrap on join, retry queue, Tier 1/2 cross-feeding.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .models import (
    ExperienceSignal, PADVector, PADSeverity, SignalType,
    MemoryLayer, MNEMON_VERSION
)
from .persistence import EROSDatabase
from .memory import SimpleEmbedder, CognitiveMemorySystem

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PAD THRESHOLDS
# ─────────────────────────────────────────────

PAD_PLEASURE_DIVERGE = 0.20
PAD_AROUSAL_LOOP     = 0.80
PAD_DOMINANCE_FREEZE = 0.30
PAD_TREND_WINDOW     = 5     # readings for trend check
PAD_FLATLINE_WINDOW  = 10    # readings for flatline check
PAD_FLATLINE_EPSILON = 0.03  # max variance for flatline detection

# Queue overflow
QUEUE_MAX_SIZE = 10_000
BOOTSTRAP_BATCH = 50         # max historical signals per bootstrap


# ─────────────────────────────────────────────
# TIER 1 — SYSTEM LEARNING LOOP
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

    Feeds:
    - EME fragment library (success patterns → promoted fragments)
    - Memory system (anomalies → episodic signals)
    - PAD thresholds (per task-type failure rates)
    - Warnings upstream before failure
    """

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self._history:   deque = deque(maxlen=1000)
        self._baselines: Dict[str, Dict] = {}  # task_type → {avg_latency, avg_tokens}
        self._failure_rates: Dict[str, float] = {}  # task_type → failure rate
        self._callbacks: List = []  # registered listeners (EME, memory, etc.)
        self._lock = asyncio.Lock()

    def register_callback(self, cb):
        """Register a listener for Tier 1 signals."""
        self._callbacks.append(cb)

    async def record(self, obs: Tier1Observation):
        """Record a computation outcome and detect patterns."""
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

    async def _update_baseline(self, obs: Tier1Observation):
        t = obs.task_type
        if t not in self._baselines:
            self._baselines[t] = {"latency": obs.latency_ms, "tokens": obs.token_cost, "n": 1}
        else:
            b = self._baselines[t]
            n = b["n"]
            b["latency"] = (b["latency"] * n + obs.latency_ms) / (n + 1)
            b["tokens"]  = (b["tokens"]  * n + obs.token_cost)  / (n + 1)
            b["n"]       = n + 1

        # Update failure rate
        recent = [h for h in self._history if h.task_type == t][-20:]
        if recent:
            failures = sum(1 for h in recent if h.outcome == "failure")
            self._failure_rates[t] = failures / len(recent)

    async def _detect_patterns(self, obs: Tier1Observation) -> List[ExperienceSignal]:
        signals = []
        t = obs.task_type
        baseline = self._baselines.get(t, {})

        # Degradation detection
        if baseline.get("latency") and obs.latency_ms > baseline["latency"] * 1.5:
            signals.append(self._make_signal(
                SignalType.DEGRADATION, obs,
                {"reason": "latency_spike", "baseline_ms": baseline["latency"], "current_ms": obs.latency_ms}
            ))

        # High failure rate warning
        failure_rate = self._failure_rates.get(t, 0.0)
        if failure_rate > 0.30:
            signals.append(self._make_signal(
                SignalType.PATTERN_FOUND, obs,
                {"reason": "high_failure_rate", "rate": failure_rate, "task_type": t}
            ))

        # Anomaly — sudden failure after string of successes
        recent = [h for h in self._history if h.task_type == t][-5:]
        if len(recent) >= 4:
            prev_successes = all(h.outcome == "success" for h in recent[:-1])
            if prev_successes and obs.outcome == "failure":
                signals.append(self._make_signal(
                    SignalType.ANOMALY, obs,
                    {"reason": "sudden_failure_after_successes", "task_type": t}
                ))

        # Recovery
        if obs.outcome == "success" and failure_rate > 0.0:
            recent_outcomes = [h.outcome for h in list(self._history)[-5:]]
            if "failure" in recent_outcomes and obs.outcome == "success":
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

    def get_pad_adjustment(self, task_type: str) -> float:
        """
        Tier 1 → Tier 2 feedback: adjust PAD pleasure threshold
        based on known failure rates for this task type.
        High historical failure rate → lower the pleasure threshold
        so PAD is more sensitive.
        """
        failure_rate = self._failure_rates.get(task_type, 0.0)
        # If 30% failure rate → raise pleasure threshold by 0.1
        return failure_rate * 0.33

    def get_stats(self) -> Dict:
        return {
            "observations":   len(self._history),
            "task_types":     len(self._baselines),
            "failure_rates":  self._failure_rates,
        }


# ─────────────────────────────────────────────
# TIER 2 — PAD MONITOR
# ─────────────────────────────────────────────

class PADMonitor:
    """
    Continuous PAD health monitoring per agent.
    Pleasure uses semantic similarity (embedding-based).
    Arousal and dominance use heuristics.
    """

    def __init__(self, embedder: Optional[SimpleEmbedder] = None):
        self.embedder = embedder or SimpleEmbedder()
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=PAD_FLATLINE_WINDOW))

    def compute_pad(
        self, agent_id: str, recent_output: str,
        goal: str, tenant_id: str, task_type: str = "",
        tier1: Optional[Tier1Observer] = None,
    ) -> PADVector:
        """
        Compute PAD vector from agent output.
        Uses embedding similarity for pleasure (more accurate than keywords).
        """
        # Pleasure — semantic similarity between output and goal
        if goal:
            out_embed  = self.embedder.embed(recent_output[:500])
            goal_embed = self.embedder.embed(goal[:200])
            pleasure   = SimpleEmbedder.cosine_similarity(out_embed, goal_embed)
        else:
            pleasure = 0.5

        # Tier 1 adjustment — be more sensitive for historically failing task types
        if tier1 and task_type:
            adjustment = tier1.get_pad_adjustment(task_type)
            pleasure = max(0.0, pleasure - adjustment)

        # Arousal — error signals and repetition heuristics
        text = recent_output.lower()
        error_signals = sum(
            text.count(kw)
            for kw in ["error", "failed", "exception", "timeout", "retry", "traceback", "invalid"]
        )
        repetition = self._detect_repetition(recent_output)
        arousal = min(1.0, error_signals * 0.15 + repetition * 0.4)

        # Dominance — confident vs uncertain language
        confident_words = sum(
            text.count(kw)
            for kw in ["will", "done", "complete", "success", "implementing", "executing", "confirmed"]
        )
        uncertain_words = sum(
            text.count(kw)
            for kw in ["maybe", "perhaps", "not sure", "unclear", "might", "possibly", "i think"]
        )
        dominance = max(0.0, min(1.0, (confident_words - uncertain_words * 1.5) / 5.0 + 0.5))

        pad = PADVector(
            pleasure=round(pleasure, 3),
            arousal=round(arousal, 3),
            dominance=round(dominance, 3),
            agent_id=agent_id,
            tenant_id=tenant_id,
            timestamp=time.time(),
        )
        self._history[agent_id].append(pad)
        return pad

    def _detect_repetition(self, text: str) -> float:
        """Detect if agent is repeating itself (sign of loop)."""
        sentences = text.split(".")
        if len(sentences) < 4:
            return 0.0
        unique = len(set(s.strip().lower() for s in sentences if len(s.strip()) > 10))
        total  = len([s for s in sentences if len(s.strip()) > 10])
        if total == 0:
            return 0.0
        repetition_rate = 1.0 - (unique / total)
        return min(1.0, repetition_rate * 1.5)

    def check_trend(self, agent_id: str) -> Optional[PADSeverity]:
        """
        Trend check over last N readings.
        Returns severity if trend is concerning.
        """
        history = list(self._history[agent_id])
        if len(history) < PAD_TREND_WINDOW:
            return None

        recent = history[-PAD_TREND_WINDOW:]

        # Pleasure declining steadily
        pleasure_trend = recent[-1].pleasure - recent[0].pleasure
        # Arousal climbing steadily
        arousal_trend = recent[-1].arousal - recent[0].arousal

        if pleasure_trend < -0.15 and arousal_trend > 0.15:
            return PADSeverity.CRITICAL
        if pleasure_trend < -0.10 or arousal_trend > 0.10:
            return PADSeverity.WARNING
        return None

    def check_flatline(self, agent_id: str) -> bool:
        """
        Flatline check — PAD completely static = agent stuck.
        """
        history = list(self._history[agent_id])
        if len(history) < PAD_FLATLINE_WINDOW:
            return False

        recent = history[-PAD_FLATLINE_WINDOW:]
        pleasures  = [p.pleasure for p in recent]
        arousals   = [p.arousal  for p in recent]
        dominances = [p.dominance for p in recent]

        def variance(vals):
            mean = sum(vals) / len(vals)
            return sum((v - mean) ** 2 for v in vals) / len(vals)

        return (
            variance(pleasures)  < PAD_FLATLINE_EPSILON and
            variance(arousals)   < PAD_FLATLINE_EPSILON and
            variance(dominances) < PAD_FLATLINE_EPSILON
        )

    def get_severity(self, agent_id: str, pad: PADVector) -> PADSeverity:
        """Combine instant reading, trend, and flatline for final severity."""
        if self.check_flatline(agent_id):
            return PADSeverity.FLATLINE
        trend = self.check_trend(agent_id)
        instant = pad.severity()
        if trend and trend.value > instant.value:
            return trend
        return instant


# ─────────────────────────────────────────────
# TIER 2 — BELIEF REGISTRY
# ─────────────────────────────────────────────

class BeliefRegistry:
    """
    Atomic shared source of truth for agent swarms.
    Versioned writes with optimistic locking.
    Agents read before every turn.
    Only orchestrator writes — via this class.
    Backed by EROSDatabase for persistence.
    """

    def __init__(self, tenant_id: str, db: EROSDatabase):
        self.tenant_id = tenant_id
        self.db        = db
        self._cache:   Dict[str, Any] = {}  # in-memory read cache
        self._lock     = asyncio.Lock()

    async def sync(self) -> Dict[str, Any]:
        """Read current beliefs — call before every agent turn."""
        beliefs = await self.db.get_all_beliefs(self.tenant_id)
        async with self._lock:
            self._cache = {k: v["value"] for k, v in beliefs.items()}
        return self._cache

    async def set(self, key: str, value: Any, expected_version: int) -> bool:
        """
        Optimistic locking write.
        Returns False if version mismatch — caller must retry.
        Conflict is explicit, never silent.
        """
        success = await self.db.set_belief(
            self.tenant_id, key, value, expected_version
        )
        if success:
            async with self._lock:
                self._cache[key] = value
            logger.debug(f"Belief updated: {key} = {value}")
        else:
            logger.warning(f"Belief conflict on key '{key}' — expected version {expected_version}")
        return success

    async def get(self, key: str) -> Optional[Any]:
        result = await self.db.get_belief(self.tenant_id, key)
        if result:
            return result["value"]
        return None

    async def get_version(self) -> int:
        return await self.db.get_belief_version(self.tenant_id)


# ─────────────────────────────────────────────
# EXPERIENCE BUS (FULL)
# ─────────────────────────────────────────────

class ExperienceBus:
    """
    Two-tier experience bus.

    Tier 1 always runs — system learning loop for any AI system.
    Tier 2 activates when agents register — collective immunity,
    PAD monitoring, belief registry.

    Single sequential processor ensures no race conditions.
    Bounded queue with DROP_OLDEST overflow policy.
    All signals auto-written to memory system.

    Cross-feeding:
    - Tier 1 → Tier 2: PAD threshold adjustments, belief registry updates
    - Tier 2 → Tier 1: agent PAD kills become Tier 1 FAILURE events
    """

    def __init__(
        self,
        tenant_id: str,
        db: EROSDatabase,
        memory: Optional[CognitiveMemorySystem] = None,
        embedder: Optional[SimpleEmbedder] = None,
        llm_client=None,
    ):
        self.tenant_id = tenant_id
        self.db        = db
        self.memory    = memory
        self.embedder  = embedder or SimpleEmbedder()
        self.llm       = llm_client

        # ── Tier 1 ──────────────────────────────
        self.tier1 = Tier1Observer(tenant_id)
        self.tier1.register_callback(self._on_tier1_signal)

        # ── Tier 2 ──────────────────────────────
        self.pad_monitor    = PADMonitor(self.embedder)
        self.belief_registry = BeliefRegistry(tenant_id, db)

        # Agent registry
        self._agents:        Dict[str, Dict] = {}   # agent_id → context
        self._agent_goals:   Dict[str, str]  = {}   # agent_id → current goal
        self._agent_task_types: Dict[str, str] = {} # agent_id → task_type

        # Signal infrastructure
        self._queue:   asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
        self._signals: Dict[str, ExperienceSignal] = {}

        # Signal indices for fast lookup
        self._error_solutions: Dict[str, List[str]] = defaultdict(list)
        self._workarounds:     Dict[str, List[str]] = defaultdict(list)
        self._validations:     Dict[str, bool]      = {}

        # Retry queue for failed propagations
        self._retry_queue:   List[Dict] = []

        # Statistics
        self.broadcasts_sent = 0
        self.immunizations   = 0
        self.tier1_signals   = 0
        self.pad_interventions = 0

        self._running = False
        self._processor_task = None
        self._retrospector = None

    def register_retrospector(self, retrospector) -> None:
        """
        Attach a Retrospector instance and subscribe it to Tier 1 signals.
        Call after bus.start() if you want the retrospector to receive signals.
        """
        self._retrospector = retrospector
        self.tier1.register_callback(retrospector._on_bus_signal)

    # ──────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────

    async def start(self):
        self._running = True
        self._processor_task = asyncio.create_task(self._process_loop())
        logger.info(f"Experience bus started for tenant {self.tenant_id}")

    async def stop(self):
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Experience bus stopped for tenant {self.tenant_id}")

    # ──────────────────────────────────────────
    # TIER 1 — RECORD OUTCOME
    # ──────────────────────────────────────────

    async def record_outcome(
        self,
        task_id: str,
        task_type: str,
        outcome: str,
        latency_ms: float,
        token_cost: int = 0,
    ):
        """
        Called after any computation completes.
        No agents needed. Universal entry point.
        """
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

    async def _on_tier1_signal(self, signal: ExperienceSignal):
        """Tier 1 pattern signals → write to memory, alert if critical."""
        if self.memory:
            try:
                await self.memory.write(signal)
            except Exception as e:
                logger.debug(f"Tier 1 memory write failed: {e}")

        # Tier 1 ANOMALY → update belief registry
        if signal.signal_type == SignalType.ANOMALY:
            try:
                version = await self.belief_registry.get_version()
                await self.belief_registry.set(
                    f"anomaly_{signal.content.get('task_type', 'unknown')}",
                    signal.content,
                    version
                )
            except Exception:
                pass

    # ──────────────────────────────────────────
    # TIER 2 — AGENT REGISTRATION
    # ──────────────────────────────────────────

    async def register_agent(
        self, agent_id: str, context: Dict, goal: str = "", task_type: str = ""
    ):
        """
        Register agent. Immediately bootstrap with historical learnings.
        """
        self._agents[agent_id]           = context
        self._agent_goals[agent_id]      = goal
        self._agent_task_types[agent_id] = task_type

        await self._bootstrap_agent(agent_id)
        logger.debug(f"Agent {agent_id} registered")

    async def unregister_agent(self, agent_id: str):
        for d in [self._agents, self._agent_goals, self._agent_task_types]:
            d.pop(agent_id, None)

    async def _bootstrap_agent(self, agent_id: str):
        """Give new agent all historical learnings relevant to its task."""
        context  = self._agents.get(agent_id, {})
        injected = 0

        for signal in list(self._signals.values())[-BOOTSTRAP_BATCH:]:
            if signal.agent_id == agent_id:
                continue
            if self._is_relevant(signal, agent_id):
                self._update_agent_context(agent_id, signal)
                injected += 1

        logger.debug(f"Agent {agent_id} bootstrapped with {injected} signals")

    # ──────────────────────────────────────────
    # TIER 2 — PAD MONITORING
    # ──────────────────────────────────────────

    async def report_pad(
        self,
        agent_id: str,
        recent_output: str,
        goal: str = "",
        task_type: str = "",
    ) -> PADSeverity:
        """
        Agent (or monitor) reports recent output for PAD analysis.
        Returns severity — caller handles SIG_KILL if FLATLINE.
        """
        goal = goal or self._agent_goals.get(agent_id, "")
        task_type = task_type or self._agent_task_types.get(agent_id, "")

        pad = self.pad_monitor.compute_pad(
            agent_id, recent_output, goal,
            self.tenant_id, task_type, self.tier1
        )
        severity = self.pad_monitor.get_severity(agent_id, pad)

        if severity == PADSeverity.WARNING:
            nudge = ExperienceSignal(
                signal_id=f"nudge_{agent_id}_{time.time():.0f}",
                tenant_id=self.tenant_id,
                session_id=agent_id,
                timestamp=time.time(),
                signal_type=SignalType.PAD_ALERT,
                layer=MemoryLayer.WORKING,
                content={
                    "type":    "nudge",
                    "message": f"You may be drifting from goal. Refocus on: {goal[:100]}",
                    "pad":     {"pleasure": pad.pleasure, "arousal": pad.arousal},
                },
                agent_id=agent_id,
            )
            await self.broadcast_signal(nudge)

        elif severity in (PADSeverity.CRITICAL, PADSeverity.FLATLINE):
            alert = ExperienceSignal(
                signal_id=f"pad_alert_{agent_id}_{time.time():.0f}",
                tenant_id=self.tenant_id,
                session_id=agent_id,
                timestamp=time.time(),
                signal_type=SignalType.PAD_ALERT,
                layer=MemoryLayer.EPISODIC,
                content={
                    "type":     severity.value,
                    "agent_id": agent_id,
                    "pad":      {"pleasure": pad.pleasure, "arousal": pad.arousal, "dominance": pad.dominance},
                    "action":   "SIG_KILL" if severity == PADSeverity.FLATLINE else "alert_orchestrator",
                },
                importance=0.9,
                agent_id=agent_id,
            )
            await self.broadcast_signal(alert)
            self.pad_interventions += 1

            # Tier 2 → Tier 1 feedback: kills become failure events
            if severity == PADSeverity.FLATLINE:
                await self.record_outcome(
                    task_id=agent_id,
                    task_type=self._agent_task_types.get(agent_id, "unknown"),
                    outcome="failure",
                    latency_ms=0,
                )

        return severity

    # ──────────────────────────────────────────
    # TIER 2 — KNOWLEDGE PROPAGATION
    # ──────────────────────────────────────────

    async def broadcast_signal(self, signal: ExperienceSignal):
        """
        Non-blocking broadcast. Agent continues immediately.
        Queue overflow: DROP_OLDEST (PAD_ALERT signals never dropped).
        """
        if self._queue.full():
            if signal.signal_type == SignalType.PAD_ALERT:
                # Never drop PAD alerts — remove oldest non-alert
                try:
                    items = []
                    while not self._queue.empty():
                        items.append(self._queue.get_nowait())
                    # Remove first non-PAD item
                    dropped = False
                    for i, item in enumerate(items):
                        if item.signal_type != SignalType.PAD_ALERT:
                            items.pop(i)
                            dropped = True
                            break
                    for item in items:
                        await self._queue.put(item)
                    if not dropped:
                        return  # queue full of PAD alerts, skip
                except Exception:
                    return
            else:
                logger.debug("Signal queue full — dropping oldest")
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass

        await self._queue.put(signal)
        self.broadcasts_sent += 1

    async def _process_loop(self):
        """
        Main sequential processor. One signal at a time.
        No concurrent writes. No race conditions. Ever.
        """
        while self._running:
            try:
                signal = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
                await self._handle_signal(signal)
            except asyncio.TimeoutError:
                await self._process_retries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Signal processor error: {e}")
                # One bad signal never stops the bus

    async def _handle_signal(self, signal: ExperienceSignal):
        """
        Process one signal:
        1. Store
        2. Index
        3. Write to memory
        4. Propagate to relevant agents
        """
        self._signals[signal.signal_id] = signal

        # Index by type
        if signal.signal_type == SignalType.ERROR_RESOLVED:
            error_key = signal.content.get("error_type", "unknown")
            self._error_solutions[error_key].append(signal.signal_id)

        elif signal.signal_type == SignalType.WORKAROUND_FOUND:
            api = signal.content.get("api", "unknown")
            self._workarounds[api].append(signal.signal_id)

        elif signal.signal_type in (SignalType.VALIDATION_PASSED, SignalType.VALIDATION_FAILED):
            approach = signal.content.get("approach", "")
            if approach:
                self._validations[approach] = (signal.signal_type == SignalType.VALIDATION_PASSED)

        # Write to memory system (permanent record)
        if self.memory and signal.layer:
            try:
                await self.memory.write(signal)
            except Exception as e:
                logger.debug(f"Memory write from bus failed: {e}")

        # Propagate to relevant agents (Tier 2 only)
        if self._agents:
            await self._propagate(signal)

    async def _propagate(self, signal: ExperienceSignal):
        """
        Surgical propagation — two-of-three relevance check.
        Does not broadcast to all agents.
        """
        propagated = 0
        for agent_id in list(self._agents.keys()):
            if agent_id == signal.agent_id:
                continue
            if self._is_relevant(signal, agent_id):
                success = self._update_agent_context(agent_id, signal)
                if success:
                    signal.propagated_to.add(agent_id)
                    propagated += 1
                else:
                    # Add to retry queue
                    self._retry_queue.append({
                        "signal_id": signal.signal_id,
                        "agent_id":  agent_id,
                        "attempts":  0,
                        "next_try":  time.time() + 5,
                    })

        if propagated > 0:
            self.immunizations += propagated
            logger.debug(f"Immunized {propagated} agents with {signal.signal_type.value}")

    def _is_relevant(self, signal: ExperienceSignal, agent_id: str) -> bool:
        """
        Two-of-three relevance check.
        Domain match, tool/API overlap, entity overlap.
        Signal propagates only when at least 2 of 3 match.
        """
        agent_ctx = self._agents.get(agent_id, {})
        matches = 0

        # Error solutions always relevant
        if signal.signal_type == SignalType.ERROR_RESOLVED:
            return True

        # Domain match
        signal_domain = signal.context.get("domain", "")
        agent_domain  = agent_ctx.get("domain", "")
        if signal_domain and agent_domain and signal_domain == agent_domain:
            matches += 1

        # Tool / API overlap
        signal_api  = signal.content.get("api", "")
        agent_apis  = agent_ctx.get("apis_used", [])
        if signal_api and signal_api in agent_apis:
            matches += 1

        # Entity overlap (client, system, project)
        signal_entities = set(signal.context.get("entities", []))
        agent_entities  = set(agent_ctx.get("entities", []))
        if signal_entities & agent_entities:
            matches += 1

        return matches >= 2

    def _update_agent_context(self, agent_id: str, signal: ExperienceSignal) -> bool:
        """Update agent context with new learning. Returns success."""
        if agent_id not in self._agents:
            return False
        ctx = self._agents[agent_id]
        if "learned_solutions" not in ctx:
            ctx["learned_solutions"] = []
        ctx["learned_solutions"].append({
            "signal_id":  signal.signal_id,
            "type":       signal.signal_type.value,
            "content":    signal.content,
            "learned_at": time.time(),
        })
        return True

    async def _process_retries(self):
        """Process retry queue with exponential backoff."""
        now = time.time()
        still_pending = []

        for item in self._retry_queue:
            if item["next_try"] > now:
                still_pending.append(item)
                continue
            if item["attempts"] >= 3:
                # Move to bootstrap queue — agent gets it on next sync
                logger.debug(f"Retry exhausted for signal {item['signal_id']} → agent {item['agent_id']}")
                continue

            signal = self._signals.get(item["signal_id"])
            if signal and item["agent_id"] in self._agents:
                self._update_agent_context(item["agent_id"], signal)
            else:
                item["attempts"] += 1
                item["next_try"] = now + (5 * (2 ** item["attempts"]))
                still_pending.append(item)

        self._retry_queue = still_pending

    # ──────────────────────────────────────────
    # QUERY INTERFACE
    # ──────────────────────────────────────────

    async def query(
        self,
        signal_type: SignalType,
        context: Dict,
    ) -> List[ExperienceSignal]:
        """Proactive query — check for known solutions before attempting work."""
        results = []

        if signal_type == SignalType.ERROR_RESOLVED:
            error_type = context.get("error_type")
            ids = self._error_solutions.get(error_type, [])
            results = [self._signals[sid] for sid in ids if sid in self._signals]

        elif signal_type == SignalType.WORKAROUND_FOUND:
            api  = context.get("api")
            ids  = self._workarounds.get(api, [])
            results = [self._signals[sid] for sid in ids if sid in self._signals]

        return results

    # ──────────────────────────────────────────
    # STATS
    # ──────────────────────────────────────────

    def get_stats(self) -> Dict:
        return {
            "tenant_id":          self.tenant_id,
            "tier1":              self.tier1.get_stats(),
            "tier2": {
                "active_agents":     len(self._agents),
                "broadcasts_sent":   self.broadcasts_sent,
                "immunizations":     self.immunizations,
                "pad_interventions": self.pad_interventions,
                "signals_stored":    len(self._signals),
                "error_solutions":   len(self._error_solutions),
                "workarounds":       len(self._workarounds),
                "queue_size":        self._queue.qsize(),
                "retry_pending":     len(self._retry_queue),
            },
        }
