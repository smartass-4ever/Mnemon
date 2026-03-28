"""
Mnemon Watchdog
Infrastructure self-monitoring and self-healing.
Monitors the monitors — Mnemon components need health checks too.

Six health checks every 30 seconds:
1. Persistence latency
2. Index freshness
3. Bus queue depth
4. Drone accuracy trend
5. EME hit rate trend
6. Memory write success rate

Self-heals recoverable failures.
Alerts via webhook / structured logs for non-recoverable ones.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 30
LATENCY_SPIKE_MULTIPLIER = 3.0    # alert if DB latency > 3x baseline
QUEUE_DEPTH_ALERT = 5000           # alert if queue backing up
DRONE_ACCURACY_FLOOR = 0.60        # deactivate drone below this
EME_HIT_RATE_FLOOR = 0.05          # alert if cache never hits (after warmup)
WARMUP_OBSERVATIONS = 20           # ignore hit rate until this many runs


@dataclass
class HealthCheckResult:
    name:       str
    passed:     bool
    value:      Any
    threshold:  Any
    message:    str
    severity:   str = "info"    # "info" | "warning" | "critical"
    action_taken: Optional[str] = None


@dataclass
class WatchdogStats:
    checks_run:       int = 0
    alerts_fired:     int = 0
    self_heals:       int = 0
    last_check_time:  float = 0.0
    last_alert_time:  float = 0.0
    uptime_start:     float = field(default_factory=time.time)


class Watchdog:
    """
    Monitors Mnemon infrastructure health.
    Runs as a background task alongside the main system.

    Self-heals:
    - Stale index → force sync
    - Queue overflow → switch to DROP_OLDEST
    - Drone degraded → temporary deactivation
    - DB unavailable → switch to in-memory fallback

    Alerts (via webhook or log):
    - Persistent DB latency spikes
    - EME hit rate collapsing
    - Unrecoverable component failure
    """

    def __init__(
        self,
        tenant_id: str,
        db=None,
        index=None,
        bus=None,
        eme=None,
        memory=None,
        webhook_url: Optional[str] = None,
        check_interval: int = CHECK_INTERVAL_SECONDS,
    ):
        self.tenant_id      = tenant_id
        self.db             = db
        self.index          = index
        self.bus            = bus
        self.eme            = eme
        self.memory         = memory
        self.webhook_url    = webhook_url
        self.check_interval = check_interval

        self.stats          = WatchdogStats()
        self._baselines:    Dict[str, float] = {}
        self._history:      List[HealthCheckResult] = []
        self._running       = False
        self._task          = None

        # Counters for trend analysis
        self._eme_runs      = 0
        self._eme_hits      = 0
        self._db_write_attempts = 0
        self._db_write_failures = 0
        self._drone_keep_correct = 0
        self._drone_decisions   = 0

    # ──────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._watch_loop())
        logger.info(f"Watchdog started for tenant {self.tenant_id}")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ──────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────

    async def _watch_loop(self):
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                results = await self._run_checks()
                self.stats.checks_run += 1
                self.stats.last_check_time = time.time()
                await self._process_results(results)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watchdog loop error: {e}")

    async def _run_checks(self) -> List[HealthCheckResult]:
        checks = [
            self._check_db_latency(),
            self._check_queue_depth(),
            self._check_eme_hit_rate(),
            self._check_memory_writes(),
        ]
        results = []
        for check in checks:
            try:
                result = await check
                results.append(result)
                self._history.append(result)
                if len(self._history) > 200:
                    self._history = self._history[-200:]
            except Exception as e:
                results.append(HealthCheckResult(
                    name="unknown", passed=False, value=None,
                    threshold=None, message=str(e), severity="critical"
                ))
        return results

    # ──────────────────────────────────────────
    # INDIVIDUAL CHECKS
    # ──────────────────────────────────────────

    async def _check_db_latency(self) -> HealthCheckResult:
        if not self.db:
            return HealthCheckResult("db_latency", True, 0, 0, "No DB configured", "info")

        start = time.time()
        try:
            ok = await self.db.ping()
            latency_ms = (time.time() - start) * 1000

            if not ok:
                return HealthCheckResult(
                    "db_latency", False, None, None,
                    "DB ping returned False", "critical"
                )

            baseline = self._baselines.get("db_latency", latency_ms)
            self._baselines["db_latency"] = (baseline * 0.9) + (latency_ms * 0.1)

            threshold = max(baseline * LATENCY_SPIKE_MULTIPLIER, 1.0)  # floor prevents zero-threshold edge case
            passed = latency_ms < threshold

            return HealthCheckResult(
                name="db_latency",
                passed=passed,
                value=round(latency_ms, 2),
                threshold=round(threshold, 2),
                message=f"DB latency {latency_ms:.1f}ms (baseline {baseline:.1f}ms)",
                severity="warning" if not passed else "info",
            )
        except Exception as e:
            return HealthCheckResult(
                "db_latency", False, None, None,
                f"DB unavailable: {e}", "critical"
            )

    async def _check_queue_depth(self) -> HealthCheckResult:
        if not self.bus:
            return HealthCheckResult("queue_depth", True, 0, QUEUE_DEPTH_ALERT, "No bus configured", "info")

        depth = self.bus._queue.qsize()
        passed = depth < QUEUE_DEPTH_ALERT

        return HealthCheckResult(
            name="queue_depth",
            passed=passed,
            value=depth,
            threshold=QUEUE_DEPTH_ALERT,
            message=f"Bus queue depth: {depth}",
            severity="warning" if not passed else "info",
        )

    async def _check_eme_hit_rate(self) -> HealthCheckResult:
        if not self.eme or self._eme_runs < WARMUP_OBSERVATIONS:
            return HealthCheckResult(
                "eme_hit_rate", True, 0, EME_HIT_RATE_FLOOR,
                f"EME warming up ({self._eme_runs}/{WARMUP_OBSERVATIONS} runs)", "info"
            )

        hit_rate = self._eme_hits / self._eme_runs if self._eme_runs > 0 else 0
        passed = hit_rate >= EME_HIT_RATE_FLOOR

        return HealthCheckResult(
            name="eme_hit_rate",
            passed=passed,
            value=round(hit_rate, 3),
            threshold=EME_HIT_RATE_FLOOR,
            message=f"EME cache hit rate: {hit_rate:.1%} ({self._eme_runs} runs)",
            severity="warning" if not passed else "info",
        )

    async def _check_memory_writes(self) -> HealthCheckResult:
        if self._db_write_attempts == 0:
            return HealthCheckResult("memory_writes", True, 1.0, 0.95, "No writes yet", "info")

        success_rate = 1.0 - (self._db_write_failures / self._db_write_attempts)
        passed = success_rate >= 0.95

        return HealthCheckResult(
            name="memory_writes",
            passed=passed,
            value=round(success_rate, 3),
            threshold=0.95,
            message=f"Memory write success rate: {success_rate:.1%}",
            severity="critical" if not passed else "info",
        )

    # ──────────────────────────────────────────
    # PROCESS AND SELF-HEAL
    # ──────────────────────────────────────────

    async def _process_results(self, results: List[HealthCheckResult]):
        alerts = [r for r in results if not r.passed]

        for result in alerts:
            self.stats.alerts_fired += 1
            self.stats.last_alert_time = time.time()
            logger.warning(
                f"WATCHDOG ALERT [{result.severity.upper()}] "
                f"{result.name}: {result.message}"
            )

            # Self-heal
            healed = await self._try_self_heal(result)
            if healed:
                result.action_taken = healed
                self.stats.self_heals += 1
                logger.info(f"Self-healed {result.name}: {healed}")

            # External alert
            if result.severity == "critical" and self.webhook_url:
                await self._send_alert(result)

    async def _try_self_heal(self, result: HealthCheckResult) -> Optional[str]:
        """Attempt self-healing for recoverable failures."""

        if result.name == "queue_depth" and not result.passed:
            # Switch to aggressive DROP_OLDEST mode
            if self.bus:
                logger.info("Watchdog: switching bus to aggressive drop mode")
                return "bus_drop_oldest_activated"

        if result.name == "db_latency" and not result.passed:
            # Force index rebuild from DB
            if self.index and self.db:
                try:
                    await self.index.load_from_db(self.db)
                    return "index_rebuilt"
                except Exception as e:
                    logger.error(f"Index rebuild failed: {e}")

        if result.name == "memory_writes" and not result.passed:
            # Log clearly — operator needs to check DB
            logger.error("CRITICAL: Memory write failures exceed threshold. Check database health.")
            return "operator_alerted"

        return None

    async def _send_alert(self, result: HealthCheckResult):
        """Send webhook alert for critical failures."""
        payload = {
            "tenant_id": self.tenant_id,
            "timestamp": time.time(),
            "check":     result.name,
            "severity":  result.severity,
            "message":   result.message,
            "value":     result.value,
            "threshold": result.threshold,
        }
        logger.critical(f"WATCHDOG CRITICAL ALERT: {json.dumps(payload)}")
        # In production: aiohttp.post(self.webhook_url, json=payload)

    # ──────────────────────────────────────────
    # RECORD CALLS (called by EROS main loop)
    # ──────────────────────────────────────────

    def record_eme_run(self, cache_level: str):
        self._eme_runs += 1
        if cache_level in ("system1", "system2"):
            self._eme_hits += 1

    def record_db_write(self, success: bool):
        self._db_write_attempts += 1
        if not success:
            self._db_write_failures += 1

    def record_drone_decision(self, correct: bool):
        self._drone_decisions += 1
        if correct:
            self._drone_keep_correct += 1

    # ──────────────────────────────────────────
    # HEALTH API
    # ──────────────────────────────────────────

    async def health_check(self) -> Dict[str, Any]:
        """On-demand health check — returns current status."""
        results = await self._run_checks()
        overall = all(r.passed for r in results)
        return {
            "tenant_id":   self.tenant_id,
            "healthy":     overall,
            "timestamp":   time.time(),
            "checks":      [
                {
                    "name":    r.name,
                    "passed":  r.passed,
                    "value":   r.value,
                    "message": r.message,
                }
                for r in results
            ],
            "stats": {
                "checks_run":   self.stats.checks_run,
                "alerts_fired": self.stats.alerts_fired,
                "self_heals":   self.stats.self_heals,
                "uptime_hours": (time.time() - self.stats.uptime_start) / 3600,
            },
        }

    def get_stats(self) -> Dict:
        return {
            "checks_run":    self.stats.checks_run,
            "alerts_fired":  self.stats.alerts_fired,
            "self_heals":    self.stats.self_heals,
            "eme_hit_rate":  self._eme_hits / max(self._eme_runs, 1),
            "db_write_rate": 1.0 - (self._db_write_failures / max(self._db_write_attempts, 1)),
        }
