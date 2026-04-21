"""
Mnemon — The intelligence layer between your agents and oblivion
Unified entry point combining EME, Memory, and Experience Bus.

Usage:
    from mnemon import Mnemon

    mnemon = Mnemon(tenant_id="my_company", agent_id="agent_01")
    await mnemon.start()

    result = await mnemon.run(
        goal="weekly security audit for Acme Corp",
        inputs={"client": "Acme Corp", "week": "March 17-21"},
        generation_fn=my_expensive_function,
    )

    await mnemon.stop()

Architecture by Mahika Jadhav (smartass-4ever).
"""

import asyncio
import atexit
import hashlib
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from mnemon.core.models import (
    ExperienceSignal, MemoryLayer, RiskLevel, SignalType, MNEMON_VERSION
)
from mnemon.core.persistence import EROSDatabase, InvertedIndex
from mnemon.core.memory import CognitiveMemorySystem, SimpleEmbedder
from mnemon.core.eme import ExecutionMemoryEngine, TemplateAdapter, CostBudget, EMEResult
from mnemon.core.bus import ExperienceBus
from mnemon.security.manager import SecurityManager, TenantSecurityConfig
from mnemon.observability.watchdog import Watchdog
from mnemon.observability.telemetry import Telemetry
from mnemon.llm.client import LLMClient, auto_client
from mnemon.core.drift import DriftDetector

logger = logging.getLogger(__name__)


class Mnemon:
    def __init__(
        self,
        tenant_id: str,
        agent_id: str = "default",
        db_dir: str = ".",
        memory_enabled: bool = True,
        eme_enabled:    bool = True,
        bus_enabled:    bool = True,
        enabled_layers: Optional[List[MemoryLayer]] = None,
        similarity_threshold: float = 0.70,
        resonance_floor: float = 0.70,
        adapter: Optional[TemplateAdapter] = None,
        cost_budget: Optional[CostBudget] = None,
        llm_client=None,
        router_model:   str = "claude-haiku-4-5-20251001",
        drone_model:    str = "claude-haiku-4-5-20251001",
        gap_fill_model: str = "claude-sonnet-4-6",
        data_region: str = "default",
        # Phase 2 — operational layer
        security_config: Optional[TenantSecurityConfig] = None,
        blocked_categories: Optional[List[str]] = None,
        enable_watchdog: bool = False,
        watchdog_webhook: Optional[str] = None,
        enable_telemetry: bool = True,
        prewarm_fragments: bool = True,
    ):
        self.tenant_id = tenant_id
        self.agent_id  = agent_id
        self.memory_enabled = memory_enabled
        self.eme_enabled    = eme_enabled
        self.bus_enabled    = bus_enabled

        self._embedder = SimpleEmbedder()
        self._db       = EROSDatabase(tenant_id=tenant_id, db_dir=db_dir)
        self._index    = InvertedIndex()
        # LLM client — explicit > auto-detect from env > None (rule-based)
        if llm_client is not None:
            resolved_llm = llm_client
        else:
            resolved_llm = auto_client()

        self._llm = resolved_llm

        # Security
        if security_config:
            self._security = SecurityManager(security_config)
        elif blocked_categories:
            cfg = TenantSecurityConfig(tenant_id=tenant_id, blocked_categories=blocked_categories)
            self._security = SecurityManager(cfg)
        else:
            self._security = SecurityManager(None)

        # Telemetry
        self._telemetry = Telemetry(tenant_id, agent_id) if enable_telemetry else None

        self._memory: Optional[CognitiveMemorySystem] = None
        self._eme:    Optional[ExecutionMemoryEngine]  = None
        self._bus:    Optional[ExperienceBus]          = None
        self._watchdog: Optional[Watchdog]             = None
        self._prewarm_fragments = prewarm_fragments

        if memory_enabled:
            self._memory = CognitiveMemorySystem(
                tenant_id=tenant_id, db=self._db, index=self._index,
                embedder=self._embedder, llm_client=resolved_llm,
                enabled_layers=enabled_layers, drone_model=drone_model,
                router_model=router_model, resonance_floor=resonance_floor,
            )
        if eme_enabled:
            self._eme = ExecutionMemoryEngine(
                tenant_id=tenant_id, db=self._db, embedder=self._embedder,
                llm_client=resolved_llm, adapter=adapter,
                similarity_threshold=similarity_threshold,
                gap_fill_model=gap_fill_model, cost_budget=cost_budget,
            )
        if bus_enabled:
            self._bus = ExperienceBus(
                tenant_id=tenant_id, db=self._db, memory=self._memory,
                embedder=self._embedder, llm_client=resolved_llm,
            )
        if enable_watchdog:
            self._watchdog = Watchdog(
                tenant_id=tenant_id,
                db=self._db, index=self._index,
                bus=self._bus, eme=self._eme, memory=self._memory,
                webhook_url=watchdog_webhook,
            )
        self._drift = DriftDetector(tenant_id=tenant_id, db=self._db)
        self._started = False

    async def start(self):
        await self._db.connect()
        await self._index.load_from_db(self._db)
        # Detect embedder dimension mismatch between stored vectors and current backend.
        # Cosine similarity between 64-dim and 384-dim vectors is undefined — retrieval
        # degrades silently to near-random without this check.
        stored_dim = await self._db.fetch_sample_embedding_dim()
        if stored_dim is not None and stored_dim != self._embedder.dim:
            logger.critical(
                f"Mnemon embedder dimension mismatch: stored vectors are {stored_dim}-dim "
                f"but current embedder produces {self._embedder.dim}-dim. "
                f"Retrieval will be unreliable until embeddings are re-indexed. "
                f"Run: mnemon reindex --tenant {self.tenant_id}"
            )
        if self._eme:
            await self._eme.warm()
            # Load pre-warmed fragments and templates on cold start
            if self._prewarm_fragments:
                try:
                    from mnemon.fragments.library import load_fragments, load_templates
                    existing = await self._db.fetch_fragments(self.tenant_id)
                    if len(existing) == 0:
                        frags = load_fragments(self.tenant_id)
                        for frag in frags:
                            await self._db.write_fragment(frag)
                        self._eme._fragments = frags
                        logger.info(f"Pre-warmed {len(frags)} fragments loaded")
                except Exception as e:
                    logger.debug(f"Fragment pre-warm skipped: {e}")
                try:
                    from mnemon.fragments.library import load_templates
                    existing_tmpl = await self._db.fetch_prewarmed_templates(self.tenant_id)
                    if len(existing_tmpl) == 0:
                        tmpls = load_templates(self.tenant_id)
                        for tmpl in tmpls:
                            await self._db.write_template(tmpl)
                            if tmpl.embedding:
                                await self._eme._template_index.add(
                                    self.tenant_id, tmpl.template_id, tmpl.embedding
                                )
                        logger.info(f"Pre-warmed {len(tmpls)} templates loaded")
                except Exception as e:
                    logger.debug(f"Template pre-warm skipped: {e}")
        if self._bus:
            await self._bus.start()
        if self._watchdog:
            await self._watchdog.start()
        self._started = True
        logger.info(f"Mnemon {MNEMON_VERSION} started — tenant={self.tenant_id}")

    async def stop(self):
        if self._watchdog:
            await self._watchdog.stop()
        if self._bus:
            await self._bus.stop()
        try:
            await self._drift.flush_session()
        except Exception:
            pass
        await self._db.disconnect()
        if self._telemetry:
            self._telemetry.emit_log()
        self._started = False

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()

    async def run(
        self,
        goal: str,
        inputs: Dict,
        generation_fn: Callable,
        context: Optional[Dict] = None,
        capabilities: Optional[List[str]] = None,
        constraints: Optional[Dict] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        task_type: str = "general",
    ) -> Dict[str, Any]:
        session_id  = session_id or f"{self.agent_id}_{time.time():.0f}"
        task_id     = task_id    or hashlib.md5(f"{goal}:{time.time()}".encode()).hexdigest()[:12]
        context     = context    or {}
        caps        = capabilities or []
        constraints = constraints  or {}
        start_time  = time.time()

        async def _retrieve():
            if not self._memory:
                return {}
            try:
                return await self._memory.retrieve(
                    task_signal=goal, session_id=session_id, task_goal=goal,
                )
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")
                return {}

        async def _run_eme():
            if not self._eme:
                try:
                    template = await generation_fn(goal, inputs, context, caps, constraints)
                    return EMEResult(status="miss", template=template, template_id=None)
                except Exception as e:
                    logger.error(f"Generation failed (eme disabled): {e}")
                    return None
            try:
                # memory_context=None: EME handles this gracefully (optional hint only).
                # Memory retrieval runs concurrently and the result is included in the
                # response dict after both tasks complete.
                return await self._eme.run(
                    goal=goal, inputs=inputs, context=context,
                    capabilities=caps, constraints=constraints,
                    generation_fn=generation_fn, task_id=task_id,
                    memory_context=None,
                )
            except Exception as e:
                logger.warning(f"EME failed: {e} — direct generation")
                try:
                    template = await generation_fn(goal, inputs, context, caps, constraints)
                    return EMEResult(status="fallback", template=template, template_id=None)
                except Exception as gen_e:
                    logger.error(f"Generation failed: {gen_e}")
                    return None

        memory_context, eme_result = await asyncio.gather(_retrieve(), _run_eme())

        latency_ms = (time.time() - start_time) * 1000

        if self._bus:
            try:
                outcome = "success" if eme_result and eme_result.template else "failure"
                await self._bus.record_outcome(
                    task_id=task_id, task_type=task_type,
                    outcome=outcome, latency_ms=latency_ms,
                )
            except Exception as e:
                logger.debug(f"Bus record failed: {e}")

        # Drift tracking — record every call regardless of cache outcome
        try:
            cache_hit = bool(eme_result and eme_result.cache_level in ("system1", "system2"))
            self._drift.record_call(cache_hit=cache_hit, latency_ms=latency_ms)
        except Exception:
            pass

        # Telemetry
        if self._telemetry and eme_result:
            self._telemetry.eme_run(
                cache_level=eme_result.cache_level,
                tokens_saved=eme_result.tokens_saved,
                latency_ms=latency_ms,
                segments_reused=eme_result.segments_reused,
            )
        if self._watchdog and eme_result:
            self._watchdog.record_eme_run(eme_result.cache_level)

        return {
            "template":         eme_result.template if eme_result else None,
            "cache_level":      eme_result.cache_level if eme_result else "error",
            "segments_reused":  eme_result.segments_reused if eme_result else 0,
            "tokens_saved":     eme_result.tokens_saved if eme_result else 0,
            "latency_saved_ms": eme_result.latency_saved_ms if eme_result else 0,
            "latency_ms":       latency_ms,
            "memory_context":   memory_context,
            "task_id":          task_id,
            "session_id":       session_id,
        }

    async def remember(self, content: Any, layer: Optional[MemoryLayer] = None,
                       importance: float = 0.5, session_id: str = "") -> Optional[str]:
        if not self._memory:
            return None
        signal = ExperienceSignal(
            signal_id=hashlib.md5(f"{time.time()}:{str(content)[:50]}".encode()).hexdigest()[:16],
            tenant_id=self.tenant_id, session_id=session_id or self.agent_id,
            timestamp=time.time(), signal_type=SignalType.CONTEXT_UPDATE,
            layer=layer, content=content if isinstance(content, dict) else {"text": str(content)},
            importance=importance,
        )
        try:
            result = await self._memory.write(signal)
            self._drift.record_memory_write()
            return result
        except Exception as e:
            logger.warning(f"remember() failed: {e}")
            return None

    async def recall(self, query: str, session_id: str = "", top_k: int = 10) -> Dict[str, Any]:
        if not self._memory:
            return {"memories": [], "working": {}, "conflicts": []}
        try:
            return await self._memory.retrieve(
                task_signal=query, session_id=session_id or self.agent_id,
                task_goal=query, top_k=top_k,
            )
        except Exception as e:
            logger.warning(f"recall() failed: {e}")
            return {"memories": [], "working": {}, "conflicts": []}

    async def learn_fact(self, key: str, value: Any, confidence: float = 1.0):
        if self._memory:
            try:
                await self._memory.write_fact(key, value, self.agent_id, confidence)
            except Exception as e:
                logger.warning(f"learn_fact() failed: {e}")

    async def recall_fact(self, key: str) -> Optional[Any]:
        if not self._memory:
            return None
        try:
            return await self._memory.read_fact(key)
        except Exception as e:
            logger.warning(f"recall_fact() failed: {e}")
            return None

    async def broadcast(self, signal_type: SignalType, content: Dict,
                        importance: float = 0.5, session_id: str = ""):
        if not self._bus:
            return
        signal = ExperienceSignal(
            signal_id=hashlib.md5(f"{self.agent_id}:{time.time()}:{signal_type.value}".encode()).hexdigest()[:16],
            tenant_id=self.tenant_id, session_id=session_id or self.agent_id,
            timestamp=time.time(), signal_type=signal_type,
            layer=MemoryLayer.EPISODIC, content=content,
            importance=importance, agent_id=self.agent_id,
        )
        try:
            await self._bus.broadcast_signal(signal)
        except Exception as e:
            logger.warning(f"broadcast() failed: {e}")

    async def report_health(self, recent_output: str, goal: str = ""):
        if not self._bus:
            return None
        try:
            return await self._bus.report_pad(self.agent_id, recent_output, goal)
        except Exception as e:
            logger.debug(f"report_health() failed: {e}")
            return None

    async def query_solutions(self, error_type: str) -> list:
        if not self._bus:
            return []
        try:
            return await self._bus.query(SignalType.ERROR_RESOLVED, {"error_type": error_type})
        except Exception as e:
            logger.debug(f"query_solutions() failed: {e}")
            return []

    async def start_session(self, session_id: str):
        if self._memory:
            await self._memory.start_session(session_id)

    async def end_session(self, session_id: str):
        if self._memory:
            await self._memory.end_session(session_id)

    async def delete_all_data(self):
        await self._db.delete_tenant_all(self.tenant_id)
        logger.info(f"All data deleted for tenant {self.tenant_id}")

    @classmethod
    def from_config(cls, config_path: str = "./mnemon.config.json") -> "Mnemon":
        """Load Mnemon from a config file created by `mnemon init`."""
        import json
        with open(config_path) as f:
            config = json.load(f)
        kwargs: dict = dict(
            tenant_id=config.get("tenant_id", "default"),
            db_dir=config.get("db_dir", config.get("db_path", ".")),
            memory_enabled=config.get("memory_enabled", True),
            eme_enabled=config.get("eme_enabled", True),
            bus_enabled=config.get("bus_enabled", True),
            prewarm_fragments=config.get("prewarm_fragments", True),
            enable_watchdog=config.get("enable_watchdog", False),
            enable_telemetry=config.get("enable_telemetry", True),
            similarity_threshold=config.get("similarity_threshold", 0.70),
            router_model=config.get("router_model", "claude-haiku-4-5-20251001"),
            gap_fill_model=config.get("gap_fill_model", "claude-sonnet-4-6"),
            drone_model=config.get("drone_model", "claude-haiku-4-5-20251001"),
            data_region=config.get("data_region", "default"),
        )
        if "blocked_categories" in config:
            kwargs["blocked_categories"] = config["blocked_categories"]
        if "watchdog_webhook" in config:
            kwargs["watchdog_webhook"] = config["watchdog_webhook"]
        return cls(**kwargs)

    async def health_check(self) -> dict:
        """On-demand health check. Returns structured health report."""
        if self._watchdog:
            return await self._watchdog.health_check()
        return {
            "tenant_id": self.tenant_id,
            "healthy":   self._started,
            "message":   "Watchdog not enabled — enable_watchdog=True for detailed health",
        }

    async def drift_report(self) -> "DriftReport":
        """
        Analyse cross-session health and return a DriftReport.
        Detects silent degradation before it becomes a user-facing problem.
        """
        return await self._drift.detect()

    async def auto_correct_drift(self) -> Dict[str, Any]:
        """
        Detect drift and attempt automatic correction.

        Finds memories written during the degradation window, runs conflict
        detection across them, and supersedes the weaker side of each conflict.
        Returns a summary of what was corrected.
        """
        report = await self._drift.detect()
        if report.is_healthy() or report.drift_since_ts is None:
            return {"status": "healthy", "corrections": 0, "report": str(report)}

        memory_ids = await self._drift.get_correction_targets(report.drift_since_ts)
        if not memory_ids:
            return {
                "status": report.severity,
                "corrections": 0,
                "message": "No memories found in drift window to correct.",
                "report": str(report),
            }

        corrections = 0
        if self._memory:
            try:
                memories = await self._db.fetch_memories(self.tenant_id, memory_ids)
                conflicts = self._memory._detect_conflicts(memories)
                for conflict in conflicts:
                    if len(conflict) >= 2:
                        # Supersede the memory with the lower importance score
                        weaker = min(conflict, key=lambda m: m.importance)
                        stronger = max(conflict, key=lambda m: m.importance)
                        await self._db.supersede_memory(
                            self.tenant_id, weaker.memory_id, stronger.memory_id
                        )
                        corrections += 1
            except Exception as e:
                logger.warning(f"auto_correct_drift: correction step failed: {e}")

        logger.info(
            f"auto_correct_drift [{self.tenant_id}]: "
            f"{corrections} conflict(s) resolved across {len(memory_ids)} candidate memories"
        )
        return {
            "status": report.severity,
            "corrections": corrections,
            "candidates_reviewed": len(memory_ids),
            "drift_since": report.drift_since_ts,
            "report": str(report),
        }

    def telemetry_report(self) -> dict:
        """Get current telemetry report."""
        if self._telemetry:
            return self._telemetry.get_report()
        return {"message": "Telemetry not enabled"}

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"version": MNEMON_VERSION, "tenant_id": self.tenant_id}
        if self._memory:
            stats["memory"] = self._memory.get_stats()
        if self._eme:
            stats["eme"] = self._eme.get_stats()
        if self._bus:
            stats["bus"] = self._bus.get_stats()
        if self._watchdog:
            stats["watchdog"] = self._watchdog.get_stats()
        if self._security:
            stats["security"] = self._security.get_stats()
        if self._telemetry:
            stats["telemetry"] = self._telemetry.get_report()
        stats["db"] = self._db.get_stats()
        return stats


def _cancel_all_tasks(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel every pending task on loop, then drain them. Prevents segfault on close."""
    try:
        pending = asyncio.all_tasks(loop)
        if not pending:
            return
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    except Exception:
        pass


class MnemonSync:
    """
    Synchronous wrapper around Mnemon for quick experimentation.

    Usage:
        with MnemonSync(tenant_id="my_company") as m:
            m.remember("Acme Corp prefers PDF reports")
            result = m.run(goal="weekly report", inputs={...}, generation_fn=fn)
            print(result["tokens_saved"])
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._m: Optional[Mnemon] = None
        self._loop = asyncio.new_event_loop()
        self._moth = None
        from mnemon.moth.stats import MothStats
        _db_dir    = kwargs.get("db_dir", ".")
        _tenant    = kwargs.get("tenant_id", "default")
        _stats_path = os.path.join(_db_dir, f"mnemon_stats_{_tenant}.json")
        self._stats = MothStats(persist_path=_stats_path)

    def __enter__(self):
        self._m = Mnemon(**self._kwargs)
        self._loop.run_until_complete(self._m.start())
        # Activate moth — auto-instrument installed frameworks
        try:
            from mnemon.moth import Moth
            self._moth = Moth()
            activated = self._moth.activate(self)
            if activated:
                logger.info(f"Mnemon moth activated: {', '.join(activated)}")
        except Exception as e:
            logger.debug(f"Mnemon moth failed to start: {e}")
        return self

    def __exit__(self, *args):
        if self._moth is not None:
            try:
                self._moth.deactivate()
            except Exception:
                pass
        self._loop.run_until_complete(self._m.stop())
        _cancel_all_tasks(self._loop)
        self._loop.close()

    @property
    def active_integrations(self) -> List[str]:
        """Names of frameworks currently instrumented by the moth."""
        if self._moth is None:
            return []
        return self._moth.active

    @property
    def stats(self) -> dict:
        """Cache hits, tokens saved, memory injections, protein bond gates."""
        return self._stats.summary

    @property
    def last_recall(self):
        """Most recent RecallTrace — what was injected or gated and why."""
        return self._stats.last_recall

    @property
    def recall_history(self):
        """Last 20 RecallTrace entries."""
        return self._stats.recall_history

    def show_stats(self) -> None:
        """Print a formatted stats summary."""
        print(repr(self._stats))

    @property
    def waste_report(self) -> str:
        """
        Personal waste report — which queries your agent repeated across sessions
        and what they cost you. Print it directly: print(m.waste_report)
        """
        return self._stats.waste_report()

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def run(self, goal: str, inputs: dict, generation_fn, **kwargs) -> dict:
        return self._run(self._m.run(goal=goal, inputs=inputs, generation_fn=generation_fn, **kwargs))

    def remember(self, text: str, layer=None, importance: float = 0.80):
        from mnemon.core.models import MemoryLayer as ML
        return self._run(self._m.remember(text, layer=layer or ML.EPISODIC, importance=importance))

    def recall(self, query: str, top_k: int = 5) -> dict:
        return self._run(self._m.recall(query, top_k=top_k))

    def learn_fact(self, key: str, value: str):
        return self._run(self._m.learn_fact(key, value))

    def recall_fact(self, key: str) -> Optional[str]:
        return self._run(self._m.recall_fact(key))

    def get_stats(self) -> dict:
        return self._m.get_stats()

    def drift_report(self):
        """
        Cross-session health analysis. Detects silent performance degradation.
        Returns a DriftReport — print it directly: print(m.drift_report())
        """
        return self._run(self._m.drift_report())

    def auto_correct_drift(self) -> dict:
        """
        Detect drift and automatically resolve conflicting memories in the
        degradation window. Returns a summary of what was corrected.
        """
        return self._run(self._m.auto_correct_drift())

    def close(self):
        if self._moth is not None:
            try:
                self._moth.deactivate()
            except Exception:
                pass
            self._moth = None
        if self._m is not None:
            self._loop.run_until_complete(self._m.stop())
            _cancel_all_tasks(self._loop)
            self._loop.close()
            self._m = None


# --- Global instance (set by init()) ---
_instance: Optional[MnemonSync] = None


def _detect_tenant_id() -> str:
    env = os.environ.get("MNEMON_TENANT")
    if env:
        return env
    return os.path.basename(os.getcwd()) or "default"


def _detect_adapter() -> Optional[TemplateAdapter]:
    try:
        import crewai  # noqa: F401
        from mnemon.adapters.crewai import CrewAIAdapter
        return CrewAIAdapter()
    except ImportError:
        pass
    return None


_USE_FLAGS: Dict[str, tuple] = {
    "all":    ("memory_enabled", "eme_enabled", "bus_enabled"),
    "memory": ("memory_enabled",),
    "cache":  ("eme_enabled",),
    "bus":    ("bus_enabled",),
}


def _resolve_use(use) -> Dict[str, bool]:
    """Map the `use` shorthand to memory_enabled / eme_enabled / bus_enabled flags."""
    defaults = {"memory_enabled": False, "eme_enabled": False, "bus_enabled": False}
    if use == "all" or use is None:
        return {"memory_enabled": True, "eme_enabled": True, "bus_enabled": True}
    tokens = [use] if isinstance(use, str) else list(use)
    for token in tokens:
        for flag in _USE_FLAGS.get(token, []):
            defaults[flag] = True
    return defaults


def init(
    tenant_id: Optional[str] = None,
    *,
    use: Any = "all",
    llm_client=None,
    adapter: Optional[TemplateAdapter] = None,
    **kwargs,
) -> MnemonSync:
    """One-line setup. Auto-detects tenant and framework adapter.

    use — controls which subsystems are active:
        "all"              all three (default)
        "memory"           memory only — no EME caching, no bus
        "cache"            EME caching only — no memory, no bus
        "bus"              experience bus only
        ["memory","cache"] any combination

    Usage:
        import mnemon
        m = mnemon.init()                      # all systems
        m = mnemon.init(use="memory")          # memory only
        m = mnemon.init(use=["memory","cache"])# memory + EME
        m.remember("Acme Corp prefers PDF reports")
        context = m.recall("Acme Corp")
    """
    global _instance
    if _instance is not None:
        return _instance

    resolved_tenant = tenant_id or _detect_tenant_id()
    resolved_adapter = adapter or _detect_adapter()

    # use= flags take effect only when not explicitly overridden in kwargs
    use_flags = _resolve_use(use)
    for flag, val in use_flags.items():
        kwargs.setdefault(flag, val)

    m = MnemonSync(
        tenant_id=resolved_tenant,
        llm_client=llm_client,
        adapter=resolved_adapter,
        **kwargs,
    )
    m.__enter__()

    atexit.register(m.close)
    _instance = m
    return m


def get() -> Optional[MnemonSync]:
    """Return the global Mnemon instance created by init(), or None."""
    return _instance


__all__ = [
    "Mnemon", "MnemonSync", "MemoryLayer", "SignalType", "RiskLevel",
    "ExperienceSignal", "CostBudget", "TemplateAdapter",
    "TenantSecurityConfig", "MNEMON_VERSION",
    "init", "get",
    "moth",
]
