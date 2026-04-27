"""
Mnemon Drift Detector
Tracks agent health across sessions and detects silent degradation.

The problem: your agent was answering well in week 1. By week 3 it's slower,
less accurate, and you have no idea why. Nobody noticed — it degraded silently.

What this does:
  - Records a health snapshot at the end of each session
  - Detects when cache hit rate or call efficiency drops over a rolling window
  - Identifies which memories were written during the degradation period
  - Provides auto-correction: marks those memories for reverification

DriftReport exposes:
  - severity: "healthy" | "warning" | "degraded" | "critical"
  - drift_since_ts: when the decline started
  - cache_hit_rate_trend: rolling comparison (positive = improving)
  - affected_sessions: how many sessions show degradation
  - recommendation: plain English explanation

Auto-correction (auto_correct()):
  - Fetches memories written after drift_since_ts
  - Runs conflict detection across them
  - Supersedes the weaker side of each conflict
  - Returns count of corrections made
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Rolling window sizes
_BASELINE_WINDOW  = 5   # sessions used to establish "healthy" baseline
_DRIFT_WINDOW     = 3   # sessions used to detect current degradation
_MIN_SESSIONS     = 4   # need at least this many sessions before detecting drift
_WARN_DROP        = 0.15  # 15% drop in cache_hit_rate → warning
_DEGRADE_DROP     = 0.30  # 30% drop → degraded
_CRITICAL_DROP    = 0.50  # 50% drop → critical


@dataclass
class DriftReport:
    severity:            str        # "healthy" | "warning" | "degraded" | "critical"
    drift_since_ts:      Optional[float]   # unix timestamp when decline started, None if healthy
    baseline_hit_rate:   float      # rolling baseline (healthy period)
    current_hit_rate:    float      # rolling current
    hit_rate_delta:      float      # current - baseline (negative = worse)
    affected_sessions:   int        # consecutive degraded sessions
    total_sessions:      int        # total sessions tracked
    recommendation:      str
    raw_history:         List[Dict] = field(default_factory=list)

    def is_healthy(self) -> bool:
        return self.severity == "healthy"

    def __str__(self) -> str:
        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Mnemon Drift Report  [{self.severity.upper()}]",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Sessions tracked:    {self.total_sessions}",
            f"  Baseline hit rate:   {self.baseline_hit_rate:.1%}",
            f"  Current hit rate:    {self.current_hit_rate:.1%}",
            f"  Change:              {self.hit_rate_delta:+.1%}",
        ]
        if self.drift_since_ts:
            import datetime
            since = datetime.datetime.fromtimestamp(self.drift_since_ts).strftime("%Y-%m-%d %H:%M")
            lines.append(f"  Drift since:         {since}")
            lines.append(f"  Affected sessions:   {self.affected_sessions}")
        lines += [
            "",
            f"  {self.recommendation}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        return "\n".join(lines)


class DriftDetector:
    """
    Detects silent performance degradation across sessions.
    One instance per Mnemon tenant, lives on the Mnemon object.
    """

    def __init__(self, tenant_id: str, db: Any):
        self.tenant_id = tenant_id
        self._db = db
        self._session_calls   = 0
        self._session_hits    = 0
        self._session_latency: List[float] = []
        self._session_id      = f"session_{time.time():.0f}"

    def record_call(self, cache_hit: bool, latency_ms: float = 0.0):
        """Call on every LLM invocation (hit or miss) to track session stats."""
        self._session_calls += 1
        if cache_hit:
            self._session_hits += 1
        if latency_ms > 0:
            self._session_latency.append(latency_ms)

    async def flush_session(self, session_id: Optional[str] = None, notes: str = ""):
        """Persist current session health. Call at session end or on stop()."""
        if self._session_calls == 0:
            return
        try:
            hit_rate   = self._session_hits / self._session_calls
            avg_lat    = sum(self._session_latency) / len(self._session_latency) if self._session_latency else 0.0
            sid        = session_id or self._session_id
            await self._db.write_session_health(
                tenant_id=self.tenant_id,
                session_id=sid,
                cache_hit_rate=hit_rate,
                memory_writes=0,
                total_calls=self._session_calls,
                avg_latency_ms=avg_lat,
                notes=notes,
            )
            self._reset_session()
        except Exception as e:
            logger.debug(f"DriftDetector.flush_session failed: {e}")

    def _reset_session(self):
        self._session_calls   = 0
        self._session_hits    = 0
        self._session_latency = []
        self._session_id      = f"session_{time.time():.0f}"

    async def detect(self) -> DriftReport:
        """Analyse session history and return a DriftReport."""
        try:
            history = await self._db.fetch_session_health(self.tenant_id, limit=30)
        except Exception as e:
            logger.debug(f"DriftDetector.detect failed to load history: {e}")
            return self._healthy_report(0)

        # History is DESC — reverse to chronological for analysis
        history = list(reversed(history))
        n = len(history)

        if n < _MIN_SESSIONS:
            return self._healthy_report(n, note="not enough sessions yet for drift analysis")

        # Compute rolling windows
        baseline_sessions = history[:_BASELINE_WINDOW]
        recent_sessions   = history[-_DRIFT_WINDOW:]

        baseline_rate = sum(s["cache_hit_rate"] for s in baseline_sessions) / len(baseline_sessions)
        current_rate  = sum(s["cache_hit_rate"] for s in recent_sessions)   / len(recent_sessions)
        delta         = current_rate - baseline_rate

        severity, recommendation = self._classify(delta, baseline_rate, current_rate, n)

        # Find drift_since_ts: earliest session in the degraded run
        drift_since_ts    = None
        affected_sessions = 0
        if severity != "healthy":
            for s in reversed(history):
                if s["cache_hit_rate"] < baseline_rate - _WARN_DROP:
                    drift_since_ts    = s["timestamp"]
                    affected_sessions += 1
                else:
                    break

        return DriftReport(
            severity=severity,
            drift_since_ts=drift_since_ts,
            baseline_hit_rate=baseline_rate,
            current_hit_rate=current_rate,
            hit_rate_delta=delta,
            affected_sessions=affected_sessions,
            total_sessions=n,
            recommendation=recommendation,
            raw_history=history[-10:],
        )

    def _classify(
        self, delta: float, baseline: float, current: float, n: int
    ) -> Tuple[str, str]:
        if delta >= -_WARN_DROP:
            return "healthy", (
                "Cache hit rate is stable. Mnemon is learning your agent's patterns well."
            )
        if delta >= -_DEGRADE_DROP:
            return "warning", (
                f"Cache hit rate dropped {abs(delta):.1%} from baseline ({baseline:.1%} → {current:.1%}). "
                f"Your agent may be asking questions in new contexts Mnemon hasn't seen yet. "
                f"Run auto_correct_drift() to reverify recent memories."
            )
        if delta >= -_CRITICAL_DROP:
            return "degraded", (
                f"Significant degradation: cache hit rate fell {abs(delta):.1%} "
                f"({baseline:.1%} → {current:.1%}). "
                f"Recent memories likely contain stale or conflicting information. "
                f"Run auto_correct_drift() immediately."
            )
        return "critical", (
            f"Critical drift: cache hit rate collapsed {abs(delta):.1%} "
            f"({baseline:.1%} → {current:.1%}). "
            f"Something fundamentally changed — new domain, stale facts, or conflicting memories. "
            f"Run auto_correct_drift() and review recent memory writes."
        )

    def _healthy_report(self, n: int, note: str = "") -> DriftReport:
        return DriftReport(
            severity="healthy",
            drift_since_ts=None,
            baseline_hit_rate=0.0,
            current_hit_rate=0.0,
            hit_rate_delta=0.0,
            affected_sessions=0,
            total_sessions=n,
            recommendation=note or "System is healthy.",
        )

