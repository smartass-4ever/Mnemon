"""
Mnemon — execution caching and system learning for AI agents.

Zero code changes. One import line.
Gets smarter with every run. Saves tokens on every cache hit.

Usage:
    import mnemon
    m = mnemon.init()          # auto-detects tenant, patches frameworks

    result = m.run(
        goal="weekly security audit for Acme Corp",
        inputs={"client": "Acme Corp", "week": "March 17-21"},
        generation_fn=my_expensive_function,
    )
    print(result["tokens_saved"])

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
    ExperienceSignal, SignalType, RiskLevel, MNEMON_VERSION
)
from mnemon.core.persistence import EROSDatabase
from mnemon.core.embedder import SimpleEmbedder
from mnemon.core.eme import ExecutionMemoryEngine, TemplateAdapter, CostBudget, EMEResult
from mnemon.core.bus import ExperienceBus
from mnemon.security.manager import SecurityManager, TenantSecurityConfig, scrub_injection
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
        eme_enabled: bool = True,
        bus_enabled: bool = True,
        similarity_threshold: float = 0.70,
        adapter: Optional[TemplateAdapter] = None,
        llm_client=None,
        data_region: str = "default",
        security_config: Optional[TenantSecurityConfig] = None,
        blocked_categories: Optional[List[str]] = None,
        enable_watchdog: bool = False,
        watchdog_webhook: Optional[str] = None,
        enable_telemetry: bool = True,
        prewarm_fragments: bool = True,
        silent: bool = False,
    ):
        self.tenant_id   = tenant_id
        self.agent_id    = agent_id
        self.eme_enabled = eme_enabled
        self.bus_enabled = bus_enabled

        self._embedder = SimpleEmbedder()
        self._db       = EROSDatabase(tenant_id=tenant_id, db_dir=db_dir)

        resolved_llm = llm_client if llm_client is not None else auto_client()
        self._llm = resolved_llm

        if security_config:
            self._security = SecurityManager(security_config)
        elif blocked_categories:
            cfg = TenantSecurityConfig(tenant_id=tenant_id, blocked_categories=blocked_categories)
            self._security = SecurityManager(cfg)
        else:
            self._security = SecurityManager(None)

        self._telemetry = Telemetry(tenant_id, agent_id) if enable_telemetry else None

        self._eme:      Optional[ExecutionMemoryEngine] = None
        self._bus:      Optional[ExperienceBus]         = None
        self._watchdog: Optional[Watchdog]              = None
        self._prewarm_fragments = prewarm_fragments

        if eme_enabled:
            self._eme = ExecutionMemoryEngine(
                tenant_id=tenant_id, db=self._db, embedder=self._embedder,
                adapter=adapter, similarity_threshold=similarity_threshold,
            )
        if bus_enabled:
            self._bus = ExperienceBus(tenant_id=tenant_id, db=self._db)
        if enable_watchdog:
            self._watchdog = Watchdog(
                tenant_id=tenant_id, db=self._db,
                bus=self._bus, eme=self._eme,
                webhook_url=watchdog_webhook,
            )
        self._drift   = DriftDetector(tenant_id=tenant_id, db=self._db)
        self._started = False
        self._silent  = silent

        self._session_tokens_saved:    int   = 0
        self._session_latency_saved_ms: float = 0.0
        self._session_calls:           int   = 0
        self._session_plans_cached:    int   = 0
        self._session_future_tokens:   int   = 0

    async def start(self):
        await self._db.connect()

        if self._eme:
            await self._eme.warm()
            if self._prewarm_fragments:
                try:
                    from mnemon.fragments.library import load_fragments, load_templates
                    existing = await self._db.fetch_fragments(self.tenant_id)
                    if len(existing) == 0:
                        frags = load_fragments(self.tenant_id)
                        for frag in frags:
                            await self._db.write_fragment(frag)
                            if frag.signature:
                                await self._eme._fragment_index.add(
                                    self.tenant_id, frag.segment_id, frag.signature
                                )
                                self._eme._fragment_map[frag.segment_id] = frag
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
        if not self._silent:
            import sys as _sys
            parts = []
            if self._session_tokens_saved > 0:
                cost_usd = self._session_tokens_saved * 0.000003
                secs_saved = self._session_latency_saved_ms / 1000
                parts.append(
                    f"{self._session_tokens_saved:,} tokens saved · ${cost_usd:.4f}"
                    + (f" · {secs_saved:.1f}s faster" if secs_saved > 0 else "")
                )
            if self._session_plans_cached > 0:
                future_cost = self._session_future_tokens * 0.000003
                parts.append(
                    f"{self._session_plans_cached} plan(s) cached → "
                    f"next run saves ~{self._session_future_tokens:,} tokens (~${future_cost:.4f})"
                )
            if parts:
                print("\nMnemon: " + " · ".join(parts) + "\n", file=_sys.stderr, flush=True)

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
        session_id  = session_id  or f"{self.agent_id}_{time.time():.0f}"
        task_id     = task_id     or hashlib.md5(f"{goal}:{time.time()}".encode()).hexdigest()[:12]
        context     = context     or {}
        caps        = capabilities or []
        constraints = constraints  or {}
        start_time  = time.time()

        if self._eme:
            try:
                eme_result = await self._eme.run(
                    goal=goal, inputs=inputs, context=context,
                    capabilities=caps, constraints=constraints,
                    generation_fn=generation_fn, task_id=task_id,
                    memory_context=None,
                )
            except Exception as e:
                logger.warning(f"EME failed: {e} — direct generation")
                try:
                    template = await generation_fn(goal, inputs, context, caps, constraints)
                    eme_result = EMEResult(status="fallback", template=template, template_id=None)
                except Exception as gen_e:
                    logger.error(f"Generation failed: {gen_e}")
                    eme_result = None
        else:
            try:
                template = await generation_fn(goal, inputs, context, caps, constraints)
                eme_result = EMEResult(status="miss", template=template, template_id=None)
            except Exception as e:
                logger.error(f"Generation failed (eme disabled): {e}")
                eme_result = None

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

        try:
            cache_hit = bool(eme_result and eme_result.cache_level in ("system1", "system2"))
            self._drift.record_call(cache_hit=cache_hit, latency_ms=latency_ms)
        except Exception:
            pass

        if self._telemetry and eme_result:
            self._telemetry.eme_run(
                cache_level=eme_result.cache_level,
                tokens_saved=eme_result.tokens_saved,
                latency_ms=latency_ms,
                segments_reused=eme_result.segments_reused,
            )
        if self._watchdog and eme_result:
            self._watchdog.record_eme_run(eme_result.cache_level)

        self._session_calls += 1
        if eme_result:
            self._session_tokens_saved      += eme_result.tokens_saved or 0
            self._session_latency_saved_ms  += eme_result.latency_saved_ms or 0.0
            if eme_result.cache_level in ("miss", "system2_guided"):
                self._session_plans_cached  += 1
                total_segs = (eme_result.segments_reused or 0) + (eme_result.segments_generated or 0)
                self._session_future_tokens += total_segs * 250

        return {
            "template":         eme_result.template if eme_result else None,
            "cache_level":      eme_result.cache_level if eme_result else "error",
            "segments_reused":  eme_result.segments_reused if eme_result else 0,
            "tokens_saved":     eme_result.tokens_saved if eme_result else 0,
            "latency_saved_ms": eme_result.latency_saved_ms if eme_result else 0,
            "latency_ms":       latency_ms,
            "task_id":          task_id,
            "session_id":       session_id,
        }

    async def delete_all_data(self):
        await self._db.delete_tenant_all(self.tenant_id)
        logger.info(f"All data deleted for tenant {self.tenant_id}")

    @classmethod
    def from_config(cls, config_path: str = "./mnemon.config.json") -> "Mnemon":
        import json
        with open(config_path) as f:
            config = json.load(f)
        kwargs: dict = dict(
            tenant_id=config.get("tenant_id", "default"),
            db_dir=config.get("db_dir", config.get("db_path", ".")),
            eme_enabled=config.get("eme_enabled", True),
            bus_enabled=config.get("bus_enabled", True),
            prewarm_fragments=config.get("prewarm_fragments", True),
            enable_watchdog=config.get("enable_watchdog", False),
            enable_telemetry=config.get("enable_telemetry", True),
            silent=config.get("silent", False),
            similarity_threshold=config.get("similarity_threshold", 0.70),
            data_region=config.get("data_region", "default"),
        )
        if "blocked_categories" in config:
            kwargs["blocked_categories"] = config["blocked_categories"]
        if "watchdog_webhook" in config:
            kwargs["watchdog_webhook"] = config["watchdog_webhook"]
        return cls(**kwargs)

    async def health_check(self) -> dict:
        if self._watchdog:
            return await self._watchdog.health_check()
        return {
            "tenant_id": self.tenant_id,
            "healthy":   self._started,
            "message":   "Watchdog not enabled — enable_watchdog=True for detailed health",
        }

    async def drift_report(self) -> "DriftReport":
        return await self._drift.detect()

    def telemetry_report(self) -> dict:
        if self._telemetry:
            return self._telemetry.get_report()
        return {"message": "Telemetry not enabled"}

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"version": MNEMON_VERSION, "tenant_id": self.tenant_id}
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
        self._m._silent = True
        self._loop.run_until_complete(self._m.stop())
        _cancel_all_tasks(self._loop)
        self._loop.close()
        silent = self._kwargs.get("silent", False)
        if not silent:
            import sys as _sys
            s = self._stats.summary
            moth_tokens  = s.get("tokens_saved_est", 0)
            run_tokens   = self._m._session_tokens_saved
            total_tokens = max(moth_tokens, run_tokens)
            plans_cached = self._m._session_plans_cached
            secs_saved   = self._m._session_latency_saved_ms / 1000
            parts = []
            if total_tokens > 0:
                real_cost    = s.get("cost_saved_usd")
                cost_is_real = s.get("cost_is_real", False)
                if real_cost is not None:
                    cost_str = f"${real_cost:.4f}" if cost_is_real else f"~${real_cost:.4f}"
                else:
                    cost_str = f"~${total_tokens * 0.000003:.4f}"
                parts.append(
                    f"~{total_tokens:,} tokens saved · {cost_str}"
                    + (f" · {secs_saved:.1f}s faster" if secs_saved > 0 else "")
                )
            if plans_cached > 0:
                future_tokens = self._m._session_future_tokens
                future_cost   = future_tokens * 0.000015
                parts.append(
                    f"{plans_cached} plan(s) cached → "
                    f"next run saves ~{future_tokens:,} tokens (~${future_cost:.4f})"
                )
            if parts:
                print("\nMnemon: " + " · ".join(parts) + "\n", file=_sys.stderr, flush=True)

    @property
    def active_integrations(self) -> List[str]:
        if self._moth is None:
            return []
        return self._moth.active

    @property
    def stats(self) -> dict:
        return self._stats.summary

    @property
    def last_recall(self):
        return self._stats.last_recall

    @property
    def recall_history(self):
        return self._stats.recall_history

    def show_stats(self) -> None:
        print(repr(self._stats))

    @property
    def waste_report(self) -> str:
        return self._stats.waste_report()

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def run(self, goal: str, inputs: dict, generation_fn, **kwargs) -> dict:
        return self._run(self._m.run(goal=goal, inputs=inputs, generation_fn=generation_fn, **kwargs))

    def get_stats(self) -> dict:
        return self._m.get_stats()

    def drift_report(self):
        return self._run(self._m.drift_report())

    def close(self):
        if self._moth is not None:
            try:
                self._moth.deactivate()
            except Exception:
                pass
            self._moth = None
        if self._m is not None:
            self._m._silent = True
            self._loop.run_until_complete(self._m.stop())
            _cancel_all_tasks(self._loop)
            silent = self._kwargs.get("silent", False)
            if not silent:
                import sys as _sys
                s = self._stats.summary
                moth_tokens  = s.get("tokens_saved_est", 0)
                run_tokens   = self._m._session_tokens_saved
                total_tokens = max(moth_tokens, run_tokens)
                plans_cached = self._m._session_plans_cached
                secs_saved   = self._m._session_latency_saved_ms / 1000
                parts = []
                if total_tokens > 0:
                    real_cost    = s.get("cost_saved_usd")
                    cost_is_real = s.get("cost_is_real", False)
                    if real_cost is not None:
                        cost_str = f"${real_cost:.4f}" if cost_is_real else f"~${real_cost:.4f}"
                    else:
                        cost_str = f"~${total_tokens * 0.000003:.4f}"
                    parts.append(
                        f"~{total_tokens:,} tokens saved · {cost_str}"
                        + (f" · {secs_saved:.1f}s faster" if secs_saved > 0 else "")
                    )
                if plans_cached > 0:
                    future_tokens = self._m._session_future_tokens
                    future_cost   = future_tokens * 0.000015
                    parts.append(
                        f"{plans_cached} plan(s) cached → "
                        f"next run saves ~{future_tokens:,} tokens (~${future_cost:.4f})"
                    )
                if parts:
                    print("\nMnemon: " + " · ".join(parts) + "\n", file=_sys.stderr, flush=True)
            self._loop.close()
            self._m = None


# ── Global instance ──────────────────────────────────────────────────────────

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


def init(
    tenant_id: Optional[str] = None,
    *,
    llm_client=None,
    adapter: Optional[TemplateAdapter] = None,
    **kwargs,
) -> MnemonSync:
    """One-line setup. Auto-detects tenant and framework adapter.

    Usage:
        import mnemon
        m = mnemon.init()
        result = m.run(goal="...", inputs={...}, generation_fn=fn)
    """
    global _instance
    if _instance is not None:
        return _instance

    resolved_tenant  = tenant_id or _detect_tenant_id()
    resolved_adapter = adapter or _detect_adapter()

    kwargs.setdefault("eme_enabled", True)
    kwargs.setdefault("bus_enabled", True)

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
    "Mnemon", "MnemonSync", "SignalType", "RiskLevel",
    "ExperienceSignal", "CostBudget", "TemplateAdapter",
    "TenantSecurityConfig", "MNEMON_VERSION",
    "init", "get",
    "moth",
]
