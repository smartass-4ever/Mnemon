"""
Mnemon Telemetry
Structured metrics and observability.
Collects what every component already produces and puts it in one place.

Metrics tracked:
- EME: hit rates by level, tokens saved, latency saved
- Memory: retrieval latency, drone accuracy, pool size by layer
- Bus: propagation latency, immunization rate, PAD intervention rate
- System: LLM call costs, DB latency, overall health score

Emits to: structured logs, optional metrics endpoint.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    name:      str
    value:     float
    tags:      Dict[str, str]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "metric":    self.name,
            "value":     self.value,
            "tags":      self.tags,
            "timestamp": self.timestamp,
        }


class Telemetry:
    """
    Lightweight telemetry collector.
    Every Mnemon component calls record() with metric points.
    Aggregates and exposes via get_report().

    No external dependencies — structured JSON to logs.
    In production: swap emit() for Datadog/CloudWatch/Prometheus client.
    """

    def __init__(self, tenant_id: str, agent_id: str = "default"):
        self.tenant_id = tenant_id
        self.agent_id  = agent_id
        self._points:  List[MetricPoint] = []
        self._counters: Dict[str, float] = {}
        self._gauges:   Dict[str, float] = {}
        self._start     = time.time()

    # ──────────────────────────────────────────
    # RECORDING
    # ──────────────────────────────────────────

    def record(self, name: str, value: float, **tags):
        point = MetricPoint(
            name=name, value=value,
            tags={"tenant": self.tenant_id, "agent": self.agent_id, **tags}
        )
        self._points.append(point)
        if len(self._points) > 10000:
            self._points = self._points[-5000:]

    def increment(self, name: str, by: float = 1.0):
        self._counters[name] = self._counters.get(name, 0) + by

    def gauge(self, name: str, value: float):
        self._gauges[name] = value

    # ──────────────────────────────────────────
    # EME METRICS
    # ──────────────────────────────────────────

    def eme_run(self, cache_level: str, tokens_saved: int, latency_ms: float, segments_reused: int):
        self.record("eme.run", 1, cache_level=cache_level)
        self.record("eme.tokens_saved", tokens_saved, cache_level=cache_level)
        self.record("eme.latency_ms", latency_ms)
        self.record("eme.segments_reused", segments_reused)
        self.increment(f"eme.hits.{cache_level}")
        self.increment("eme.runs.total")

    # ──────────────────────────────────────────
    # MEMORY METRICS
    # ──────────────────────────────────────────

    def memory_write(self, layer: str, latency_ms: float, success: bool):
        self.record("memory.write", 1, layer=layer, success=str(success))
        self.record("memory.write_latency_ms", latency_ms, layer=layer)
        self.increment(f"memory.writes.{layer}")

    def memory_retrieval(self, latency_ms: float, memories_returned: int, drone_used: bool, pool_size: int):
        self.record("memory.retrieval_latency_ms", latency_ms)
        self.record("memory.memories_returned", memories_returned)
        self.record("memory.drone_used", int(drone_used))
        self.gauge("memory.pool_size", pool_size)

    def drone_decision(self, kept: int, dropped: int, timeout: bool):
        self.record("memory.drone.kept", kept)
        self.record("memory.drone.dropped", dropped)
        self.record("memory.drone.timeout", int(timeout))
        total = kept + dropped
        if total > 0:
            self.record("memory.drone.precision", kept / total)

    # ──────────────────────────────────────────
    # BUS METRICS
    # ──────────────────────────────────────────

    def bus_signal(self, signal_type: str, propagated_to: int):
        self.record("bus.signal", 1, signal_type=signal_type)
        self.record("bus.immunizations", propagated_to)
        self.increment(f"bus.signals.{signal_type}")

    def pad_intervention(self, severity: str, agent_id: str):
        self.record("bus.pad_intervention", 1, severity=severity, agent=agent_id)
        self.increment(f"bus.pad.{severity}")

    def tier1_pattern(self, pattern_type: str, task_type: str):
        self.record("bus.tier1.pattern", 1, pattern=pattern_type, task_type=task_type)

    # ──────────────────────────────────────────
    # COST METRICS
    # ──────────────────────────────────────────

    def llm_call(self, component: str, model: str, tokens_in: int, tokens_out: int, cost_usd: float):
        self.record("llm.tokens_in",  tokens_in,  component=component, model=model)
        self.record("llm.tokens_out", tokens_out, component=component, model=model)
        self.record("llm.cost_usd",   cost_usd,   component=component, model=model)
        self.increment(f"llm.calls.{component}")
        self.increment("llm.total_cost_usd", cost_usd)

    # ──────────────────────────────────────────
    # REPORT
    # ──────────────────────────────────────────

    def get_report(self) -> Dict[str, Any]:
        """Structured observability report."""
        uptime = time.time() - self._start
        eme_runs  = self._counters.get("eme.runs.total", 0)
        eme_s1    = self._counters.get("eme.hits.system1", 0)
        eme_s2    = self._counters.get("eme.hits.system2", 0)

        # Average metrics from recent points
        def avg(metric: str, n: int = 100) -> float:
            vals = [p.value for p in self._points[-n:] if p.name == metric]
            return sum(vals) / len(vals) if vals else 0.0

        return {
            "tenant_id":    self.tenant_id,
            "agent_id":     self.agent_id,
            "uptime_hours": round(uptime / 3600, 2),
            "eme": {
                "total_runs":        int(eme_runs),
                "system1_hits":      int(eme_s1),
                "system2_hits":      int(eme_s2),
                "miss_rate":         round(1 - ((eme_s1 + eme_s2) / max(eme_runs, 1)), 3),
                "avg_tokens_saved":  round(avg("eme.tokens_saved"), 0),
                "avg_latency_ms":    round(avg("eme.latency_ms"), 1),
            },
            "memory": {
                "pool_size":         int(self._gauges.get("memory.pool_size", 0)),
                "avg_retrieval_ms":  round(avg("memory.retrieval_latency_ms"), 1),
                "avg_returned":      round(avg("memory.memories_returned"), 1),
                "drone_precision":   round(avg("memory.drone.precision"), 3),
            },
            "bus": {
                "total_signals":     int(sum(v for k, v in self._counters.items() if k.startswith("bus.signals."))),
                "total_immunizations": int(sum(p.value for p in self._points if p.name == "bus.immunizations")),
                "pad_warnings":      int(self._counters.get("bus.pad.warning", 0)),
                "pad_criticals":     int(self._counters.get("bus.pad.critical", 0)),
                "pad_flatlines":     int(self._counters.get("bus.pad.flatline", 0)),
            },
            "cost": {
                "total_llm_calls":  int(sum(v for k, v in self._counters.items() if k.startswith("llm.calls."))),
                "total_cost_usd":   round(self._counters.get("llm.total_cost_usd", 0.0), 4),
                "calls_by_component": {
                    k.replace("llm.calls.", ""): int(v)
                    for k, v in self._counters.items()
                    if k.startswith("llm.calls.")
                },
            },
        }

    def emit_log(self):
        """Emit current report to structured log."""
        report = self.get_report()
        logger.info(f"EROS_TELEMETRY: {json.dumps(report)}")
        return report
