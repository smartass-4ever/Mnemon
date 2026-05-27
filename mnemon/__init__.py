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

from mnemon.core.models import MNEMON_VERSION as __version__
import time
from typing import Any, Callable, Dict, List, Optional

from mnemon.core.models import (
    ExperienceSignal, SignalType, RiskLevel, MNEMON_VERSION
)
from mnemon.core.persistence import EROSDatabase
from mnemon.core.embedder import SimpleEmbedder
from mnemon.core.eme import ExecutionMemoryEngine, TemplateAdapter, CostBudget, EMEResult
from mnemon.core.bus import ExperienceBus
from mnemon.core.retrospector import Retrospector
from mnemon.core.system_db import SystemDatabase
from mnemon.core.signal_db import SignalDatabase
from mnemon.security.manager import SecurityManager, TenantSecurityConfig, scrub_injection
from mnemon.observability.watchdog import Watchdog
from mnemon.observability.telemetry import Telemetry
from mnemon.llm.client import LLMClient, auto_client
from mnemon.core.drift import DriftDetector
from mnemon.billing.quota import QuotaEnforcer

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
        prewarm_templates: bool = True,
        silent: bool = False,
        license_key: Optional[str] = None,
    ):
        self.tenant_id   = tenant_id
        self.agent_id    = agent_id
        self.eme_enabled = eme_enabled
        self.bus_enabled = bus_enabled

        self._embedder = SimpleEmbedder()
        self._db_dir   = db_dir
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

        self._eme:          Optional[ExecutionMemoryEngine] = None
        self._bus:          Optional[ExperienceBus]         = None
        self._watchdog:     Optional[Watchdog]              = None
        self._retrospector: Optional[Retrospector]          = None
        self._system_db:    Optional[SystemDatabase]        = None
        self._signal_db:    Optional[SignalDatabase]        = None
        self._prewarm_fragments  = prewarm_fragments
        self._prewarm_templates  = prewarm_templates

        if eme_enabled:
            self._eme = ExecutionMemoryEngine(
                tenant_id=tenant_id, db=self._db, embedder=self._embedder,
                adapter=adapter, similarity_threshold=similarity_threshold,
            )
        if bus_enabled:
            self._bus = ExperienceBus(tenant_id=tenant_id, db=self._db)
        if eme_enabled and bus_enabled:
            import os as _os
            self._system_db = SystemDatabase(
                db_path=_os.path.join(db_dir, f"mnemon_system_{tenant_id}.db")
            )
            self._signal_db = SignalDatabase(
                db_path=_os.path.join(db_dir, f"mnemon_signal_{tenant_id}.db")
            )
            self._retrospector = Retrospector(
                bus=self._bus, eme=self._eme, memory=None,
                system_db=self._system_db, llm_client=resolved_llm,
                signal_db=self._signal_db,
            )
        if enable_watchdog:
            self._watchdog = Watchdog(
                tenant_id=tenant_id, db=self._db,
                bus=self._bus, eme=self._eme,
                webhook_url=watchdog_webhook,
            )
        self._drift   = DriftDetector(tenant_id=tenant_id, db=self._db)
        self._quota   = QuotaEnforcer(db=self._db, tenant_id=tenant_id, license_key=license_key)
        self._started = False
        self._silent  = silent

        self._session_tokens_saved:    int   = 0
        self._session_latency_saved_ms: float = 0.0
        self._session_calls:           int   = 0
        self._session_plans_cached:    int   = 0
        self._session_future_tokens:   int   = 0

    def _prewarm_library_thread(self) -> None:
        """Daemon thread: compute embeddings and store curated library on first run.

        Uses its own DB connection and asyncio loop — fully independent of the
        main event loop so the main loop closing does not interrupt it.
        """
        try:
            asyncio.run(self._prewarm_library_async())
        except Exception as e:
            logger.debug(f"Mnemon: prewarm library thread failed — {e}")

    async def _prewarm_library_async(self) -> None:
        """Async body of prewarm — runs in the daemon thread's own event loop."""
        # Open a separate DB connection (WAL + busy_timeout make this safe).
        db = EROSDatabase(tenant_id=self.tenant_id, db_dir=self._db_dir)
        try:
            await db.connect()
        except Exception as e:
            logger.debug(f"Mnemon: prewarm DB connect failed — {e}")
            return

        try:
            if self._prewarm_fragments:
                try:
                    from mnemon.fragments.library import load_fragments
                    existing = await db.fetch_fragments(self.tenant_id)
                    if len(existing) == 0:
                        frags = load_fragments(self.tenant_id)  # heavy CPU
                        for frag in frags:
                            await db.write_fragment(frag)
                        # Don't touch self._eme here — its asyncio.Lock objects
                        # are bound to the main event loop; using them from this
                        # daemon thread's own loop corrupts asyncio state.
                        # The in-memory index is rebuilt from DB on next startup.
                        logger.info(f"Pre-warmed {len(frags)} fragments stored to DB")
                except Exception as e:
                    logger.info(f"Fragment pre-warm skipped: {e}")

            if self._prewarm_templates:
                try:
                    from mnemon.fragments.library import load_templates
                    existing_tmpl = await db.fetch_prewarmed_templates(self.tenant_id)
                    if len(existing_tmpl) == 0:
                        tmpls = load_templates(self.tenant_id)  # heavy CPU
                        for tmpl in tmpls:
                            await db.write_template(tmpl)
                        logger.info(f"Pre-warmed {len(tmpls)} templates stored to DB")
                except Exception as e:
                    logger.info(f"Template pre-warm skipped: {e}")
        finally:
            try:
                await db.disconnect()
            except Exception:
                pass

    async def start(self):
        await self._db.connect()
        if self._system_db:
            await self._system_db.connect()
        if self._signal_db:
            await self._signal_db.connect()

        if self._eme:
            await self._eme.warm()
            # Prewarm runs in a daemon thread with its own event loop and DB
            # connection — embedding 100+ fragments triggers sentence-transformers
            # load (15s) and must not block init().
            if self._prewarm_fragments or self._prewarm_templates:
                import threading as _threading
                _threading.Thread(
                    target=self._prewarm_library_thread,
                    daemon=True,
                    name="mnemon-prewarm",
                ).start()

        if self._bus:
            await self._bus.start()
        if self._retrospector:
            await self._retrospector.start()
            if self._bus:
                self._bus.register_retrospector(self._retrospector)
            if self._eme:
                self._eme.set_retrospector(self._retrospector)
        if self._watchdog:
            await self._watchdog.start()
        await self._quota.start()
        self._started = True
        logger.info(f"Mnemon {MNEMON_VERSION} started — tenant={self.tenant_id}")

    async def stop(self):
        if not self._started:
            return
        self._started = False
        if self._watchdog:
            await self._watchdog.stop()
        if self._retrospector:
            await self._retrospector.stop()
        if self._bus:
            await self._bus.stop()
        try:
            await self._drift.flush_session()
        except Exception:
            pass
        if self._eme:
            try:
                await self._eme._write_behind.flush_now()
            except Exception:
                pass
        if self._signal_db:
            await self._signal_db.disconnect()
        if self._system_db:
            await self._system_db.disconnect()
        await self._db.disconnect()
        if self._telemetry:
            self._telemetry.emit_log()
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
                # Quota gate: if free tier is exhausted, bypass the cache hit
                if (
                    eme_result
                    and eme_result.cache_level in ("system1", "system2", "system2_guided")
                    and not await self._quota.can_serve_cache_hit()
                ):
                    if not self._silent:
                        import sys as _sys
                        remaining = self._quota.hits_remaining
                        print(
                            f"Mnemon: free tier limit reached ({remaining} hits remaining today) — "
                            f"upgrade to Pro for unlimited caching: https://mnemon.lemonsqueezy.com",
                            file=_sys.stderr, flush=True,
                        )
                    template = await generation_fn(goal, inputs, context, caps, constraints)
                    eme_result = EMEResult(status="miss", template=template, template_id=None)
                elif eme_result and eme_result.cache_level in ("system1", "system2", "system2_guided"):
                    await self._quota.record_hit()
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
                from mnemon.core.models import EvidenceRecord
                outcome = "success" if eme_result and eme_result.template else "failure"
                frag_ids = (eme_result.fragment_ids_used if eme_result else []) or []
                await self._bus.record_evidence(EvidenceRecord(
                    task_id=task_id,
                    tenant_id=self.tenant_id,
                    template_id=eme_result.template_id if eme_result else None,
                    fragment_ids_used=frag_ids,
                    framework=context.get("_mnemon_framework", "unknown") if context else "unknown",
                    outcome=outcome,
                    failure_class=None if outcome == "success" else "manual",
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
            except Exception as e:
                logger.warning(f"Bus record failed: {e}")

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
        if eme_result:
            try:
                from mnemon.core import ph_telemetry
                if eme_result.cache_level in ("system1", "system2", "system2_guided"):
                    ph_telemetry.track_hit(
                        framework=context.get("_mnemon_framework", "unknown") if context else "unknown",
                        cache_level=eme_result.cache_level,
                        tokens_saved=eme_result.tokens_saved or 0,
                        latency_ms=latency_ms,
                    )
                else:
                    ph_telemetry.track_miss(
                        framework=context.get("_mnemon_framework", "unknown") if context else "unknown",
                    )
            except Exception:
                pass
        if self._watchdog and eme_result:
            self._watchdog.record_eme_run(eme_result.cache_level)

        self._session_calls += 1
        if eme_result:
            self._session_tokens_saved      += eme_result.tokens_saved or 0
            self._session_latency_saved_ms  += eme_result.latency_saved_ms or 0.0
            if eme_result.cache_level in ("miss", "system2_guided"):
                self._session_plans_cached  += 1
                total_segs = (eme_result.segments_reused or 0) + (eme_result.segments_generated or 0)
                self._session_future_tokens += max(total_segs * 250, 500)

        output = eme_result.template if eme_result else None
        cache_level = eme_result.cache_level if eme_result else "error"
        tokens_saved = eme_result.tokens_saved if eme_result else 0
        latency_saved_ms = eme_result.latency_saved_ms if eme_result else 0

        if not self._silent:
            import sys as _sys
            if cache_level in ("system1", "system2", "system2_guided"):
                cost = tokens_saved * 0.000003
                secs = latency_saved_ms / 1000
                msg = f"Mnemon: cache hit · {tokens_saved:,} tokens saved · ~${cost:.4f}"
                if secs > 0:
                    msg += f" · {secs:.1f}s faster"
            elif cache_level == "miss":
                flag = os.path.join(self._db_dir, f".mnemon_welcomed_{self.tenant_id}")
                if not os.path.exists(flag):
                    try:
                        open(flag, "w").close()
                    except OSError:
                        pass
                    msg = (
                        "Mnemon: first run — plan cached, next run will be instant\n"
                        "  Thank you for installing Mnemon! If your agent is loop-locking or token costs\n"
                        "  aren't dropping, drop a line at mahikajadhav22@gmail.com — I can look at your logs directly.\n"
                        "  I can also personally integrate this into your workflow — same email."
                    )
                else:
                    msg = "Mnemon: new input — cached, next run will be instant"
            else:
                msg = "Mnemon: ran (no cache)"
            print(msg, file=_sys.stderr, flush=True)

        return {
            "output":           output,
            "template":         output,   # alias kept for backwards compat
            "cache_level":      "system2" if cache_level == "system2_guided" else cache_level,
            "segments_reused":  eme_result.segments_reused if eme_result else 0,
            "tokens_saved":     tokens_saved,
            "latency_saved_ms": latency_saved_ms,
            "latency_ms":       latency_ms,
            "task_id":          task_id,
            "session_id":       session_id,
        }

    async def mark_failure(self, task_id: str) -> None:
        """Signal that a previously returned template failed in production."""
        if self._eme:
            try:
                await self._eme.mark_failure(task_id)
            except Exception as e:
                logger.debug(f"mark_failure failed (non-critical): {e}")
        if self._bus:
            try:
                await self._bus.record_outcome(
                    task_id=task_id, task_type="manual_failure",
                    outcome="failure", latency_ms=0.0,
                )
            except Exception:
                pass

    async def delete_all_data(self):
        await self._db.delete_tenant_all(self.tenant_id)
        logger.info(f"All data deleted for tenant {self.tenant_id}")

    @classmethod
    def from_config(cls, config_path: str = "./mnemon.config.json") -> "Mnemon":
        import json
        try:
            with open(config_path) as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Mnemon config not found: {config_path}\n"
                f"Run 'mnemon init' to create one, or pass config_path= explicitly."
            ) from None
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Mnemon config is invalid JSON ({config_path}): {e}"
            ) from None
        kwargs: dict = dict(
            tenant_id=config.get("tenant_id", "default"),
            db_dir=config.get("db_dir", config.get("db_path", ".")),
            eme_enabled=config.get("eme_enabled", True),
            bus_enabled=config.get("bus_enabled", True),
            prewarm_fragments=config.get("prewarm_fragments", True),
            prewarm_templates=config.get("prewarm_templates", True),
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
        if "license_key" in config:
            kwargs["license_key"] = config["license_key"]
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
        # One flush cycle lets tasks that suppressed CancelledError settle fully
        # before we ask for the pending set — avoids "Task was destroyed" warnings.
        loop.run_until_complete(asyncio.sleep(0))
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
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread = None   # set when loop runs in a background thread
        self._moth = None
        self._patch_thread = None  # background thread that loads + applies SDK patches
        from mnemon.moth.stats import MothStats
        _db_dir    = kwargs.get("db_dir", ".")
        _tenant    = kwargs.get("tenant_id", "default")
        _stats_path = os.path.join(_db_dir, f"mnemon_stats_{_tenant}.json")
        self._stats = MothStats(persist_path=_stats_path)

    def _patch_frameworks(self) -> None:
        """Background thread: import framework SDKs and apply patches."""
        import threading
        try:
            from mnemon.moth import Moth
            moth = Moth()
            activated = moth.activate(self)
            self._moth = moth  # atomic assign in CPython
            if activated:
                logger.info(f"Mnemon moth activated: {', '.join(activated)}")
                try:
                    from mnemon.core import ph_telemetry
                    ph_telemetry.track_init(frameworks=activated)
                except Exception:
                    pass
                try:
                    inner = getattr(self, "_m", None)
                    silent = self._kwargs.get("silent", False) if hasattr(self, "_kwargs") else False
                    if not silent:
                        emb = getattr(inner, "_embedder", None)
                        if emb:
                            emb._load()
                            if not getattr(emb, "system2_active", False):
                                import sys as _sys
                                print(
                                    "Mnemon: System 2 (semantic matching) inactive — "
                                    "cache persists across restarts but won't match rephrased inputs.\n"
                                    "  Fix: pip install mnemon-ai[full]",
                                    file=_sys.stderr, flush=True,
                                )
                except Exception:
                    pass
            else:
                import sys as _sys
                print(
                    "Mnemon: no supported frameworks detected -- caching is inactive.\n"
                    "  Install one of: anthropic, openai, langchain, langgraph, crewai\n"
                    "  Or use m.run() directly for explicit caching.",
                    file=_sys.stderr, flush=True,
                )
        except Exception as e:
            logger.warning(f"Mnemon moth failed to start: {e} -- framework auto-patching disabled")

    def __enter__(self):
        if self._m is not None:
            # Already entered (e.g. mnemon.init() called __enter__ before the
            # user's `with` statement does). Skip re-initialization.
            return self
        import threading
        self._loop = asyncio.new_event_loop()
        self._m = Mnemon(**self._kwargs)
        try:
            asyncio.get_running_loop()
            # Already inside a running event loop (e.g. Jupyter, FastAPI startup,
            # asyncio.run()) — run our dedicated loop in a background thread so
            # run_until_complete / run_coroutine_threadsafe can still work.
            self._loop_thread = threading.Thread(
                target=self._loop.run_forever, daemon=True, name="mnemon-loop"
            )
            self._loop_thread.start()
            asyncio.run_coroutine_threadsafe(self._m.start(), self._loop).result(timeout=30)
        except RuntimeError:
            # No running loop — safe to drive directly.
            self._loop.run_until_complete(self._m.start())
        # Patch frameworks in a background thread so init() returns in ~200ms.
        # SDK imports (anthropic, openai, langchain) are the slow step; they
        # complete well before the user's first LLM call fires.
        self._patch_thread = threading.Thread(
            target=self._patch_frameworks, daemon=True, name="mnemon-patch"
        )
        self._patch_thread.start()
        return self

    def __exit__(self, *args):
        if self._patch_thread is not None and self._patch_thread.is_alive():
            self._patch_thread.join(timeout=30)
        if self._moth is not None:
            try:
                self._moth.deactivate()
            except Exception:
                pass
        self._m._silent = True
        if self._loop is not None:
            if self._loop_thread is not None and self._loop_thread.is_alive():
                asyncio.run_coroutine_threadsafe(self._m.stop(), self._loop).result(timeout=10)
                _cancel_all_tasks(self._loop)
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._loop_thread.join(timeout=5)
                self._loop_thread = None
            else:
                self._loop.run_until_complete(self._m.stop())
                _cancel_all_tasks(self._loop)
            self._loop.close()
            self._loop = None
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
                future_cost   = future_tokens * 0.000003
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
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return asyncio.run_coroutine_threadsafe(coro, self._loop).result()
        return self._loop.run_until_complete(coro)

    def run(self, goal: str, inputs: dict, generation_fn, **kwargs) -> dict:
        import inspect
        if not inspect.iscoroutinefunction(generation_fn):
            _sync_fn = generation_fn
            async def generation_fn(*args, **kw):
                return _sync_fn(*args, **kw)
        return self._run(self._m.run(goal=goal, inputs=inputs, generation_fn=generation_fn, **kwargs))

    def get_stats(self) -> dict:
        return self._m.get_stats()

    def drift_report(self):
        return self._run(self._m.drift_report())

    def close(self):
        if self._patch_thread is not None and self._patch_thread.is_alive():
            self._patch_thread.join(timeout=30)
        if self._moth is not None:
            try:
                self._moth.deactivate()
            except Exception:
                pass
            self._moth = None
        if self._m is not None and self._loop is not None:
            self._m._silent = True
            if self._loop_thread is not None and self._loop_thread.is_alive():
                asyncio.run_coroutine_threadsafe(self._m.stop(), self._loop).result(timeout=10)
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._loop_thread.join(timeout=5)
                self._loop_thread = None
            else:
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
                    future_cost   = future_tokens * 0.000003
                    parts.append(
                        f"{plans_cached} plan(s) cached → "
                        f"next run saves ~{future_tokens:,} tokens (~${future_cost:.4f})"
                    )
                if parts:
                    print("\nMnemon: " + " · ".join(parts) + "\n", file=_sys.stderr, flush=True)
            self._loop.close()
            self._loop = None
            self._m = None


# ── Global instance ──────────────────────────────────────────────────────────

_instance: Optional[MnemonSync] = None


def _detect_tenant_id() -> str:
    env = os.environ.get("MNEMON_TENANT")
    if env:
        return env
    derived = os.path.basename(os.getcwd()) or "default"
    logger.info(
        f"Mnemon tenant auto-detected as '{derived}' from cwd. "
        "Set MNEMON_TENANT env var to pin a specific tenant."
    )
    return derived


def _detect_adapter() -> Optional[TemplateAdapter]:
    import sys
    if "crewai" not in sys.modules:
        return None  # don't cold-import heavy SDKs — user must import first
    try:
        from mnemon._future.adapters.crewai import CrewAIAdapter
        return CrewAIAdapter()
    except Exception:
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
    kwargs.setdefault("prewarm_templates", False)

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


def get() -> MnemonSync:
    """Return the global Mnemon instance created by init().

    Raises RuntimeError if init() has not been called yet.
    """
    if _instance is None:
        raise RuntimeError(
            "mnemon.get() called before mnemon.init().\n"
            "Add 'import mnemon; mnemon.init()' before using mnemon.get()."
        )
    return _instance


__all__ = [
    "Mnemon", "MnemonSync", "SignalType", "RiskLevel",
    "ExperienceSignal", "CostBudget", "TemplateAdapter",
    "TenantSecurityConfig", "MNEMON_VERSION",
    "init", "get",
    "moth",
]
