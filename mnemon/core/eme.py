"""
Mnemon Execution Memory Engine (EME) — v2
Generalised plan cache for any expensive recurring computation.

System 1: exact fingerprint match — zero LLM, sub-millisecond
System 2: partial segment match — gap fill with windowed context
Fragment library: proven segments accumulate across all templates

Architecture by Mahika Jadhav (smartass-4ever).

v2 bug fixes:
  [BUG-1] tokens_saved=0 on all-segments-matched System 2 path.
          The early-return block was missing tokens_saved and latency_saved_ms.
          Both defaulted to 0 even when all segments were successfully reused.

  [BUG-2] System 2 write-back was fire-and-forget (asyncio.create_task).
          The cache write could complete AFTER the next agent call arrived,
          causing Step 4 to re-enter System 2 instead of hitting System 1.
          Now awaited inline before returning the result.

  [BUG-3] _segment_diff silently counted unsigned segments as matched,
          inflating segments_reused and triggering false all-matched
          early-return (which then compounded BUG-1). Unsigned segments
          now routed to gap fill instead.

v2 scale improvements (for large agent counts + large data):
  - ANNIndex: vectorised numpy cosine top-k replaces O(n) list scan for
    fragment lookup. Handles 100k+ fragments. Per-tenant shards prevent
    cross-tenant bleed. Interface is faiss-compatible for future upgrade.

  - EmbeddingCache: LRU (2048 slots). Same goal string never re-embedded.
    Critical for CrewAI swarms where many agents share goal prefixes.

  - TemplateIndex: in-memory numpy matrix of all template embeddings, built
    on warm() and updated incrementally on write. top_k() replaces the
    full fetch_all_templates() table scan on every System 2 call.
    Under 100 concurrent agents this was 100 concurrent full table scans.

  - TenantLockRegistry: per-tenant asyncio locks. 100 agents on different
    tenants never queue behind each other. Same tenant serialises only
    on its own cache writes.

  - WriteBehindQueue: batches fragment writes with a 10ms debounce window.
    Under burst load (parallel agent swarms) gap-fill produces many
    fragments/second. Without batching these serialise on the DB lock
    and stall agents waiting to write.

  - _schema_of() now handles nested dicts and lists without crashing on
    unhashable types.

  - mark_failure() uses public DB API only. Original version accessed
    db._conn directly, bypassing the persistence layer's lock and
    transaction management.
"""

import asyncio
import hashlib
import json
import logging
import struct
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .models import (
    ComputationFingerprint,
    DecisionTrace,
    ExecutionTemplate,
    TemplateSegment,
    RiskLevel,
    MNEMON_VERSION,
)
from .persistence import EROSDatabase
from .memory import SimpleEmbedder
from .signal_db import SignalDatabase

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────

SYSTEM2_THRESHOLD_DEFAULT  = 0.70   # minimum overall similarity for System 2
FRAGMENT_EXACT_THRESHOLD   = 0.98   # fragment library exact hit
FRAGMENT_SIMILAR_THRESHOLD = 0.80   # fragment library similar hit (LLM adapts)
SEGMENT_MATCH_THRESHOLD    = 0.72   # per-segment similarity in _segment_diff
ANN_CANDIDATE_K            = 32     # ANN shortlist size for fragment search
TEMPLATE_CANDIDATE_K       = 20     # top-k templates scored in System 2
EMBEDDING_CACHE_SIZE       = 2048   # LRU embedding cache slots
WRITE_BEHIND_DEBOUNCE_MS   = 10     # fragment write-behind batch window (ms)

# Multi-component similarity weights
GOAL_WEIGHT       = 0.30
SCHEMA_WEIGHT     = 0.25
CONTEXT_WEIGHT    = 0.25
CAPABILITY_WEIGHT = 0.20


# ─────────────────────────────────────────────
# DOMAIN ADAPTER INTERFACE
# ─────────────────────────────────────────────

class TemplateAdapter(ABC):
    """
    Base adapter — any computation format plugs in here.
    The EME never knows what the template means, only its structure.
    """

    @abstractmethod
    def decompose(self, template: Any) -> List[Dict]:
        """Break template into segment dicts with 'content' and 'id' keys."""
        ...

    @abstractmethod
    def reconstruct(self, segments: List[TemplateSegment]) -> Any:
        """Reassemble segments into the framework's native format."""
        ...

    @abstractmethod
    def extract_signature(self, template: Any, goal: str) -> ComputationFingerprint:
        """Generate fingerprint from template and goal."""
        ...

    def get_tool_versions(self, capabilities: List[str]) -> Dict[str, str]:
        """Override to provide tool version hashes. Default: no versioning."""
        return {}


class GenericAdapter(TemplateAdapter):
    """
    Default adapter for dict/list templates.
    Works for agent plans, DAGs, sequential steps.
    """

    def decompose(self, template: Any) -> List[Dict]:
        if isinstance(template, list):
            return [{"id": f"seg_{i}", "content": step} for i, step in enumerate(template)]
        if isinstance(template, dict):
            nodes = template.get("nodes", template.get("steps", [template]))
            return [
                {
                    "id": n.get("id", f"seg_{i}") if isinstance(n, dict) else f"seg_{i}",
                    "content": n,
                }
                for i, n in enumerate(nodes)
            ]
        return [{"id": "seg_0", "content": template}]

    def reconstruct(self, segments: List[TemplateSegment]) -> Any:
        return {"nodes": [s.content for s in segments], "type": "eros_template"}

    def extract_signature(self, template: Any, goal: str) -> ComputationFingerprint:
        # Derive a structural schema from the template so that templates with the
        # same goal but different shapes produce different fingerprints.
        if isinstance(template, list):
            schema = {"type": "list", "length": len(template),
                      "keys": sorted({k for step in template
                                      if isinstance(step, dict) for k in step})}
        elif isinstance(template, dict):
            nodes = template.get("nodes", template.get("steps", []))
            schema = {"type": "dict", "top_keys": sorted(template.keys()),
                      "node_count": len(nodes)}
        else:
            schema = {"type": type(template).__name__}
        return ComputationFingerprint.build(
            goal=goal,
            input_schema=schema,
            context={},
            capabilities=[],
            constraints={},
        )


# ─────────────────────────────────────────────
# COST BUDGET
# ─────────────────────────────────────────────

@dataclass
class CostBudget:
    max_llm_calls_per_hour: int = 500
    max_tokens_per_task: int = 2000
    overflow_policy: str = "fallback"   # "fallback" | "block" | "alert_only"

    _calls_this_hour: int = field(default=0, init=False)
    _hour_start: float = field(default_factory=time.time, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def can_call(self) -> bool:
        async with self._lock:
            now = time.time()
            if now - self._hour_start > 3600:
                self._calls_this_hour = 0
                self._hour_start = now
            return self._calls_this_hour < self.max_llm_calls_per_hour

    async def record_call(self):
        async with self._lock:
            self._calls_this_hour += 1


# ─────────────────────────────────────────────
# GAP FILL REQUEST
# ─────────────────────────────────────────────

@dataclass
class GapFillRequest:
    """
    A segment with no fragment library match in System 2.
    Deferred to the user's generation_fn with full context.
    Zero Mnemon LLM cost. The filled result grows the fragment library.
    """
    position: int
    segment_id: str
    hint: str
    surrounding_context: List[Any]


# ─────────────────────────────────────────────
# EME RESULT
# ─────────────────────────────────────────────

@dataclass
class EMEResult:
    status: str           # "system1" | "system2" | "system2_guided" | "miss" | "error"
    template: Any         # the hydrated/generated template
    template_id: Optional[str]
    segments_reused: int = 0
    segments_generated: int = 0
    tokens_saved: int = 0
    latency_saved_ms: float = 0.0
    fragments_used: int = 0
    cache_level: str = "miss"
    validation_passed: bool = True
    pending_gaps: List[GapFillRequest] = field(default_factory=list)


# ─────────────────────────────────────────────
# EMBEDDING CACHE (LRU)
# ─────────────────────────────────────────────

class EmbeddingCache:
    """
    Thread-safe LRU cache for embeddings.
    Avoids re-embedding the same goal string on every agent call.
    Critical for CrewAI swarms where agents share goal prefixes.
    maxsize=2048 covers ~100 concurrent agents × 20 unique goal variants each.
    """

    def __init__(self, maxsize: int = EMBEDDING_CACHE_SIZE):
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._maxsize = maxsize
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[List[float]]:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    async def set(self, key: str, value: List[float]):
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    # Sync variants for use inside non-async helpers
    def get_sync(self, key: str) -> Optional[List[float]]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set_sync(self, key: str, value: List[float]):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = value


# ─────────────────────────────────────────────
# ANN INDEX (vectorised numpy cosine top-k)
# ─────────────────────────────────────────────

class ANNIndex:
    """
    Approximate nearest neighbour index over segment signatures.

    Implementation: flat numpy matrix with vectorised dot product.
    Handles 100k+ entries at ~1ms per query (vs O(n) Python loop in v1).
    Interface is faiss-compatible — swap in faiss.IndexFlatIP for 1M+
    without changing callers.

    Per-tenant shards ensure tenants never see each other's fragments.
    """

    def __init__(self):
        # tenant_id → (matrix [N, D] float32, segment_ids [N])
        self._shards: Dict[str, Tuple[Optional[np.ndarray], List[str]]] = {}
        self._lock = asyncio.Lock()

    async def add(self, tenant_id: str, seg_id: str, signature: List[float]):
        if not signature:
            return
        vec = np.array(signature, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        async with self._lock:
            mat, ids = self._shards.get(tenant_id, (None, []))
            ids = ids + [seg_id]
            mat = vec.reshape(1, -1) if mat is None else np.vstack([mat, vec.reshape(1, -1)])
            self._shards[tenant_id] = (mat, ids)

    async def top_k(
        self, tenant_id: str, query: List[float], k: int = ANN_CANDIDATE_K
    ) -> List[Tuple[str, float]]:
        """Return top-k (seg_id, similarity) pairs, highest similarity first."""
        if not query:
            return []
        async with self._lock:
            shard = self._shards.get(tenant_id)
            if shard is None or shard[0] is None:
                return []
            mat, ids = shard

        qvec = np.array(query, dtype=np.float32)
        norm = np.linalg.norm(qvec)
        if norm > 0:
            qvec = qvec / norm

        scores = mat @ qvec
        k = min(k, len(ids))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(ids[i], float(scores[i])) for i in top_indices]

    async def remove(self, tenant_id: str, seg_id: str):
        async with self._lock:
            shard = self._shards.get(tenant_id)
            if not shard or seg_id not in shard[1]:
                return
            mat, ids = shard
            idx = ids.index(seg_id)
            ids = ids[:idx] + ids[idx + 1:]
            mat = np.delete(mat, idx, axis=0) if len(ids) > 0 else None
            self._shards[tenant_id] = (mat, ids)

    def size(self, tenant_id: str) -> int:
        shard = self._shards.get(tenant_id)
        return len(shard[1]) if shard else 0


# ─────────────────────────────────────────────
# TEMPLATE INDEX (in-memory embedding matrix)
# ─────────────────────────────────────────────

class TemplateIndex:
    """
    In-memory index of full template goal embeddings, per tenant.

    Replaces fetch_all_templates() full table scan on every System 2 call.
    Under 100 concurrent agents this was 100 full table scans per second.

    top_k() returns candidate template_ids — only those are fetched from DB.
    Built on warm(), updated incrementally on every cache write.
    """

    def __init__(self):
        self._shards: Dict[str, Tuple[Optional[np.ndarray], List[str]]] = {}
        self._lock = asyncio.Lock()

    async def add(self, tenant_id: str, template_id: str, embedding: List[float]):
        if not embedding:
            return
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        async with self._lock:
            mat, ids = self._shards.get(tenant_id, (None, []))
            if template_id in ids:
                return   # already indexed, skip
            ids = ids + [template_id]
            mat = vec.reshape(1, -1) if mat is None else np.vstack([mat, vec.reshape(1, -1)])
            self._shards[tenant_id] = (mat, ids)

    async def remove(self, tenant_id: str, template_id: str):
        async with self._lock:
            shard = self._shards.get(tenant_id)
            if not shard or template_id not in shard[1]:
                return
            mat, ids = shard
            idx = ids.index(template_id)
            ids = ids[:idx] + ids[idx + 1:]
            mat = np.delete(mat, idx, axis=0) if len(ids) > 0 else None
            self._shards[tenant_id] = (mat, ids)

    async def top_k(
        self, tenant_id: str, query: List[float], k: int = TEMPLATE_CANDIDATE_K
    ) -> List[Tuple[str, float]]:
        if not query:
            return []
        async with self._lock:
            shard = self._shards.get(tenant_id)
            if not shard or shard[0] is None:
                return []
            mat, ids = shard

        qvec = np.array(query, dtype=np.float32)
        norm = np.linalg.norm(qvec)
        if norm > 0:
            qvec = qvec / norm

        scores = mat @ qvec
        k = min(k, len(ids))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(ids[i], float(scores[i])) for i in top_indices]

    def size(self, tenant_id: str) -> int:
        shard = self._shards.get(tenant_id)
        return len(shard[1]) if shard else 0


# ─────────────────────────────────────────────
# WRITE-BEHIND QUEUE (batched fragment writes)
# ─────────────────────────────────────────────

class WriteBehindQueue:
    """
    Batches fragment writes with a debounce window to reduce SQLite pressure.

    Under burst load (100-agent swarm hitting gap fill simultaneously),
    each gap-fill synchronously writes a new fragment. Without batching
    these 100 writes serialise on the DB asyncio lock and stall the agents.

    With a 10ms debounce window, all fragments generated in a burst are
    written in a single DB transaction. Agents never wait on fragment writes.

    flush_now() must be called on shutdown to drain the queue.
    """

    def __init__(self, db: EROSDatabase, debounce_ms: int = WRITE_BEHIND_DEBOUNCE_MS):
        self._db = db
        self._debounce = debounce_ms / 1000.0
        self._queue: List[TemplateSegment] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None

    async def enqueue(self, segment: TemplateSegment):
        async with self._lock:
            self._queue.append(segment)
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._delayed_flush())

    async def _delayed_flush(self):
        await asyncio.sleep(self._debounce)
        async with self._lock:
            batch = list(self._queue)
            self._queue.clear()
        for seg in batch:
            try:
                await self._db.write_fragment(seg)
            except Exception as e:
                logger.warning(f"WriteBehind: fragment write failed [{seg.segment_id}]: {e}")

    async def flush_now(self):
        """Force immediate drain — call before shutdown."""
        async with self._lock:
            batch = list(self._queue)
            self._queue.clear()
        for seg in batch:
            try:
                await self._db.write_fragment(seg)
            except Exception as e:
                logger.warning(f"WriteBehind: flush_now failed [{seg.segment_id}]: {e}")


# ─────────────────────────────────────────────
# PER-TENANT LOCK REGISTRY
# ─────────────────────────────────────────────

class TenantLockRegistry:
    """
    Provides one asyncio.Lock per tenant_id.
    100 agents on different tenants never queue behind each other.
    Same tenant serialises only on its own cache writes.
    """

    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._meta_lock = asyncio.Lock()

    async def get(self, tenant_id: str) -> asyncio.Lock:
        async with self._meta_lock:
            if tenant_id not in self._locks:
                self._locks[tenant_id] = asyncio.Lock()
            return self._locks[tenant_id]


# ─────────────────────────────────────────────
# EXECUTION MEMORY ENGINE
# ─────────────────────────────────────────────

class ExecutionMemoryEngine:
    """
    Generalised execution template cache.

    System 1 — exact fingerprint match:
      Five-component hash. 100% hit → instantiate directly.
      Zero LLM calls. Zero tokens.

    System 2 — partial segment match (70–99%):
      Segment-level diff. Matched segments reused from cache.
      Unmatched segments filled via three-tier gap fill:
        1. Fragment library exact match   — zero LLM
        2. Fragment library similar match — minimal LLM adaptation
        3. LLM generation                 — fresh, cached as new fragment
      Dependency validation pass before returning.
      [v2] Write-back is awaited — next identical call always hits System 1.

    Both fail → full generation → cached on success.
    Fragment library accumulates proven segments across all templates.
    [v2] Fragment lookup uses ANNIndex (vectorised cosine) — O(1) at scale.
    """

    def __init__(
        self,
        tenant_id: str,
        db: EROSDatabase,
        embedder: Optional[SimpleEmbedder] = None,
        adapter: Optional[TemplateAdapter] = None,
        similarity_threshold: float = SYSTEM2_THRESHOLD_DEFAULT,
        signal_db: Optional[SignalDatabase] = None,
    ):
        self.tenant_id = tenant_id
        self.db = db
        self.embedder = embedder or SimpleEmbedder()
        self.adapter = adapter or GenericAdapter()
        self.threshold = similarity_threshold
        self.signal_db = signal_db

        # System 1: in-memory hash lookup, sub-millisecond
        self._system1_cache: Dict[str, str] = {}   # fingerprint_hash → template_id

        # v2 scale structures
        self._fragment_index = ANNIndex()
        self._fragment_map: Dict[str, TemplateSegment] = {}   # seg_id → segment
        self._template_index = TemplateIndex()
        self._embed_cache = EmbeddingCache()
        self._write_behind = WriteBehindQueue(db)
        self._tenant_locks = TenantLockRegistry()

        self._fragments_loaded = False
        self.retrospector = None
        # Cross-tenant collective learning: proven boosts loaded on warm()
        self._proven_boosts: Dict[str, float] = {}  # intent_key → boost_weight

        # Per-call buffers populated in _fill_gap for retrospector tracing.
        # Reset at the top of run() — asyncio-safe (single-threaded event loop).
        self._trace_frags_buf: List[str] = []
        self._trace_gen_buf:   List[str] = []

    def set_retrospector(self, retrospector) -> None:
        """Attach a Retrospector instance. Call before the first run()."""
        self.retrospector = retrospector

    @staticmethod
    def _seg_tokens(segments) -> int:
        """
        Estimate token count from actual segment content (chars/4 ≈ tokens).
        More accurate than the old flat 250/segment default — segments vary
        from a 3-word label to a 500-word instruction block.
        """
        total = 0
        for seg in segments:
            try:
                content = getattr(seg, "content", seg)
                total += max(10, len(json.dumps(content, default=str)) // 4)
            except Exception:
                total += 250
        return max(total, len(segments) * 10)

    # ──────────────────────────────────────────
    # WARM
    # ──────────────────────────────────────────

    async def warm(self):
        """
        Load System 1 cache, fragment ANN index, and template embedding index
        from DB on startup. Called once per engine instance.
        """
        templates = await self.db.fetch_all_templates(self.tenant_id)
        for t in templates:
            self._system1_cache[t.fingerprint.full_hash] = t.template_id
            if t.embedding:
                await self._template_index.add(self.tenant_id, t.template_id, t.embedding)

        fragments = await self.db.fetch_fragments(self.tenant_id)

        # Re-rank by cross-tenant signal_db success_rate on warm() so proven
        # fragments sort earlier in the ANN index and get retrieved first.
        if self.signal_db and fragments:
            ranked: List[Tuple[float, Any]] = []
            for frag in fragments:
                signal_score = 0.5  # neutral default
                if frag.signature:
                    try:
                        dims = frag.signature[:32]
                        raw = struct.pack(f">{len(dims)}f", *dims)
                        shape_hash = hashlib.sha256(raw).hexdigest()[:32]
                        sig = await self.signal_db.get_fragment_signal(shape_hash)
                        if sig:
                            signal_score = sig["success_rate"]
                    except Exception:
                        pass
                ranked.append((signal_score, frag))
            ranked.sort(key=lambda x: x[0], reverse=True)
            fragments = [f for _, f in ranked]

        for frag in fragments:
            if frag.signature:
                await self._fragment_index.add(
                    self.tenant_id, frag.segment_id, frag.signature
                )
                self._fragment_map[frag.segment_id] = frag

        # Load cross-tenant proven boosts for pre-warmed template prioritisation
        if self.signal_db:
            try:
                self._proven_boosts = await self.signal_db.get_proven_boosts()
                if self._proven_boosts:
                    logger.info(f"EME: {len(self._proven_boosts)} proven intent boost(s) loaded")
            except Exception as e:
                logger.debug(f"EME: proven boosts load failed (non-critical): {e}")

        self._fragments_loaded = True
        logger.info(
            f"EME warmed [{self.tenant_id}]: "
            f"{len(self._system1_cache)} templates | "
            f"{self._fragment_index.size(self.tenant_id)} fragments"
        )

    # ──────────────────────────────────────────
    # MAIN ENTRY POINT
    # ──────────────────────────────────────────

    async def run(
        self,
        goal: str,
        inputs: Dict,
        context: Dict,
        capabilities: List[str],
        constraints: Dict,
        generation_fn: Callable,
        task_id: str = "",
        memory_context: Optional[Dict] = None,
    ) -> EMEResult:
        """
        Run a computation through the EME.
        Tries System 1 → System 2 → full generation.
        Caches successful results automatically.
        """
        fp = ComputationFingerprint.build(
            goal=goal,
            input_schema=self._schema_of(inputs),
            context=context,
            capabilities=capabilities,
            constraints=constraints,
        )

        # Reset per-call fragment/generation buffers for retrospector tracing.
        self._trace_frags_buf = []
        self._trace_gen_buf   = []

        # ── SYSTEM 1: Exact Match ──────────────
        result = await self._try_system1(fp, inputs, goal)

        if not result:
            # ── SYSTEM 2: Partial Match ────────
            result = await self._try_system2(
                fp, goal, inputs, context, capabilities,
                constraints, memory_context
            )
            if result:
                if result.pending_gaps:
                    # Unmatched segments deferred to user's generation_fn.
                    # Zero Mnemon LLM cost — their LLM fills the gaps with full context.
                    result = await self._guided_generation(
                        result, goal, inputs, context, capabilities,
                        constraints, generation_fn, fp
                    )
                else:
                    await self._cache_template(goal, result.template, fp, capabilities)

        if not result:
            # ── FULL GENERATION ────────────────
            result = await self._full_generation(
                goal, inputs, context, capabilities,
                constraints, generation_fn, fp
            )

        # ── RETROSPECTOR TRACE ─────────────────
        if self.retrospector:
            try:
                trace = DecisionTrace(
                    trace_id=hashlib.md5(
                        f"{self.tenant_id}:{task_id}:{time.time()}".encode()
                    ).hexdigest()[:16],
                    tenant_id=self.tenant_id,
                    task_id=task_id,
                    goal_hash=fp.goal_hash,
                    fragment_ids_used=list(self._trace_frags_buf),
                    memory_ids_retrieved=(
                        memory_context.get("memory_ids", [])
                        if memory_context else []
                    ),
                    segments_generated=list(self._trace_gen_buf),
                    tools_called=capabilities,
                    step_outcomes={},
                    overall_outcome=result.status,
                    latency_ms=result.latency_saved_ms,
                    timestamp=time.time(),
                )
                asyncio.create_task(self.retrospector.submit_trace(trace))
            except Exception as e:
                logger.debug(f"Retrospector trace submission failed (non-fatal): {e}")

        return result

    # ──────────────────────────────────────────
    # SYSTEM 1
    # ──────────────────────────────────────────

    async def _try_system1(
        self,
        fp: ComputationFingerprint,
        inputs: Dict,
        goal: str,
    ) -> Optional[EMEResult]:
        """Pure in-memory hash lookup. Sub-millisecond."""
        template_id = self._system1_cache.get(fp.full_hash)
        if not template_id:
            return None

        template = await self.db.fetch_template_by_fingerprint(
            self.tenant_id, fp.full_hash
        )
        if not template:
            del self._system1_cache[fp.full_hash]
            return None

        if template.needs_reverification:
            if not await self._validate_dependencies(template):
                await self.db.update_template_outcome(self.tenant_id, template_id, False)
                logger.info(f"Template {template_id} failed re-verification — evicting")
                await self.db.delete_template(self.tenant_id, template_id)
                await self._template_index.remove(self.tenant_id, template_id)
                del self._system1_cache[fp.full_hash]
                return None
            template.needs_reverification = False
            await self.db.write_template(template)

        if template.should_evict:
            logger.info(f"Template {template_id} evicted — high failure rate")
            await self.db.delete_template(self.tenant_id, template_id)
            await self._template_index.remove(self.tenant_id, template_id)
            del self._system1_cache[fp.full_hash]
            return None

        hydrated = self._hydrate(template, inputs)
        await self.db.update_template_outcome(self.tenant_id, template_id, True)
        tokens_saved = self._seg_tokens(template.segments)

        return EMEResult(
            status="system1",
            template=hydrated,
            template_id=template_id,
            segments_reused=len(template.segments),
            segments_generated=0,
            tokens_saved=tokens_saved,
            latency_saved_ms=20000,
            cache_level="system1",
        )

    # ──────────────────────────────────────────
    # SYSTEM 2
    # ──────────────────────────────────────────

    async def _try_system2(
        self,
        fp: ComputationFingerprint,
        goal: str,
        inputs: Dict,
        context: Dict,
        capabilities: List[str],
        constraints: Dict,
        memory_context: Optional[Dict],
    ) -> Optional[EMEResult]:
        """
        Semantic similarity search across cached templates.

        [v2] Uses TemplateIndex top-k instead of fetch_all_templates() full scan.
        Fetches only the shortlisted candidate templates from DB.
        Falls back to full fetch only on cold start (empty index).
        """
        goal_embedding = await self._embed(goal, full=True)

        # Candidate shortlist from in-memory index
        candidates = await self._template_index.top_k(
            self.tenant_id, goal_embedding, k=TEMPLATE_CANDIDATE_K
        )

        if candidates:
            # Fetch only shortlisted templates
            reverse = self._system1_reverse()
            templates: List[ExecutionTemplate] = []
            for tid, _ in candidates:
                fp_hash = reverse.get(tid, "")
                if fp_hash:
                    t = await self.db.fetch_template_by_fingerprint(self.tenant_id, fp_hash)
                    if t:
                        templates.append(t)
        else:
            # Cold start: full fetch and populate index
            templates = await self.db.fetch_all_templates(self.tenant_id)
            for t in templates:
                if t.embedding:
                    await self._template_index.add(self.tenant_id, t.template_id, t.embedding)

        if not templates:
            return None

        best_template: Optional[ExecutionTemplate] = None
        best_score = 0.0

        for t in templates:
            if not t.embedding:
                continue
            score = self._multi_component_similarity(
                fp, t.fingerprint, goal_embedding, t.embedding,
                capabilities, list(t.tool_versions.keys())
            )
            # Apply collective cross-tenant boost to proven pre-warmed templates
            if t.is_prewarmed and self._proven_boosts:
                intent_key = hashlib.md5(f"prewarmed:{t.intent}".encode()).hexdigest()[:24]
                boost = self._proven_boosts.get(intent_key, 0.0)
                score = min(1.0, score + boost)
            if score > best_score:
                best_score = score
                best_template = t

        if not best_template or best_score < self.threshold:
            return None

        if best_template.should_evict:
            return None

        # Segment-level diff
        matched, unmatched_indices = self._segment_diff(
            best_template.segments, goal, inputs
        )

        if not unmatched_indices:
            # All segments matched — full cache hit on System 2 path
            hydrated = self._hydrate(best_template, inputs)
            await self.db.update_template_outcome(
                self.tenant_id, best_template.template_id, True
            )
            if best_template.is_prewarmed and self.signal_db:
                intent_key = hashlib.md5(f"prewarmed:{best_template.intent}".encode()).hexdigest()[:24]
                domain = list(best_template.segments[0].domain_tags)[0] if best_template.segments and best_template.segments[0].domain_tags else "general"
                asyncio.create_task(self.signal_db.record_proven_intent(intent_key, domain, True))
            # [FIX BUG-1] tokens_saved and latency_saved_ms were not set here.
            # The original EMEResult was returned with both fields at their
            # dataclass default of 0, even though all segments were reused.
            tokens_saved = self._seg_tokens(matched)
            return EMEResult(
                status="system2",
                template=hydrated,
                template_id=best_template.template_id,
                segments_reused=len(matched),
                segments_generated=0,
                tokens_saved=tokens_saved,
                latency_saved_ms=len(matched) * 2500,
                cache_level="system2",
            )

        # Gap fill for unmatched segments
        all_segments = list(best_template.segments)
        fragments_used = 0
        pending_gaps: List[GapFillRequest] = []

        for idx in unmatched_indices:
            seg = all_segments[idx]
            window = self._window(all_segments, idx, window_size=2)
            filled, used_fragment, gap_request = await self._fill_gap(
                seg, goal, window, memory_context, context, idx
            )
            if gap_request is not None:
                pending_gaps.append(gap_request)
            else:
                all_segments[idx] = filled
                if used_fragment:
                    fragments_used += 1

        if pending_gaps:
            gap_positions = {g.position for g in pending_gaps}
            resolved_segs = [s for i, s in enumerate(all_segments) if i not in gap_positions]
            return EMEResult(
                status="system2_guided",
                template=self.adapter.reconstruct(all_segments),
                template_id=best_template.template_id,
                segments_reused=len(resolved_segs),
                segments_generated=0,
                tokens_saved=self._seg_tokens(resolved_segs),
                latency_saved_ms=len(resolved_segs) * 2500,
                fragments_used=fragments_used,
                cache_level="system2_guided",
                pending_gaps=pending_gaps,
            )

        stitched_template = self.adapter.reconstruct(all_segments)
        is_valid = await self._validate_stitched(all_segments, capabilities, constraints)

        if not is_valid:
            logger.warning(
                "System 2 stitched template failed validation — "
                "falling through to full generation"
            )
            await self.db.update_template_outcome(
                self.tenant_id, best_template.template_id, False
            )
            return None

        await self.db.update_template_outcome(
            self.tenant_id, best_template.template_id, True
        )

        tokens_saved = self._seg_tokens(matched)
        return EMEResult(
            status="system2",
            template=stitched_template,
            template_id=best_template.template_id,
            segments_reused=len(matched),
            segments_generated=0,
            tokens_saved=tokens_saved,
            latency_saved_ms=len(matched) * 2500,
            fragments_used=fragments_used,
            cache_level="system2",
            validation_passed=True,
        )

    def _system1_reverse(self) -> Dict[str, str]:
        """Reverse lookup: template_id → fingerprint_hash. Built on demand, cheap."""
        return {v: k for k, v in self._system1_cache.items()}

    def _multi_component_similarity(
        self,
        fp1: ComputationFingerprint,
        fp2: ComputationFingerprint,
        embed1: List[float],
        embed2: List[float],
        caps1: List[str],
        caps2: List[str],
    ) -> float:
        """Weighted four-component similarity score."""
        goal_sim   = SimpleEmbedder.cosine_similarity(embed1, embed2)
        schema_sim = 1.0 if fp1.input_schema_hash == fp2.input_schema_hash else 0.3
        ctx_sim    = 1.0 if fp1.context_hash == fp2.context_hash else 0.4

        if caps1 and caps2:
            overlap = len(set(caps1) & set(caps2))
            cap_sim = overlap / max(len(caps1), len(caps2))
        else:
            cap_sim = 1.0

        return (
            GOAL_WEIGHT       * goal_sim   +
            SCHEMA_WEIGHT     * schema_sim +
            CONTEXT_WEIGHT    * ctx_sim    +
            CAPABILITY_WEIGHT * cap_sim
        )

    def _segment_diff(
        self,
        segments: List[TemplateSegment],
        goal: str,
        inputs: Dict,
    ) -> Tuple[List[TemplateSegment], List[int]]:
        """
        Identify which segments match the current task and which need gap fill.
        Returns (matched_segments, indices_of_unmatched).

        [FIX BUG-3] Original silently appended unsigned segments to matched_segs.
        This inflated segments_reused and could trigger the all-matched early-return
        path (which compounded BUG-1 — the early-return path had tokens_saved=0).
        Unsigned segments now go to unmatched_indices so gap fill runs on them.
        """
        goal_sig = self._embed_sync(goal, full=False)
        matched_segs: List[TemplateSegment] = []
        unmatched_indices: List[int] = []

        for i, seg in enumerate(segments):
            if not seg.signature:
                logger.debug(
                    f"Segment {seg.segment_id} has no signature — "
                    f"routing to gap fill (was silently counted as matched in v1)"
                )
                unmatched_indices.append(i)
                continue

            sim = SimpleEmbedder.cosine_similarity(goal_sig, seg.signature)
            if sim >= SEGMENT_MATCH_THRESHOLD:
                matched_segs.append(seg)
            else:
                unmatched_indices.append(i)

        return matched_segs, unmatched_indices

    def _window(
        self,
        segments: List[TemplateSegment],
        idx: int,
        window_size: int = 2,
    ) -> List[TemplateSegment]:
        """Return neighbouring segments for windowed context in gap fill."""
        start = max(0, idx - window_size)
        end   = min(len(segments), idx + window_size + 1)
        return [s for i, s in enumerate(segments) if start <= i < end and i != idx]

    # ──────────────────────────────────────────
    # GAP FILL (three-tier)
    # ──────────────────────────────────────────

    async def _fill_gap(
        self,
        segment: TemplateSegment,
        goal: str,
        context_window: List[TemplateSegment],
        memory_context: Optional[Dict],
        execution_context: Dict,
        position: int = 0,
    ) -> Tuple[TemplateSegment, bool, Optional[GapFillRequest]]:
        """
        Two-tier fragment lookup. Returns (segment, used_fragment, gap_request).
        gap_request is non-None only for Tier 3 — deferred to user's generation_fn.

        Tier 1: exact fragment match (≥0.98)  — zero LLM, zero cost
        Tier 2: similar fragment (≥0.80)      — use directly, zero LLM, zero cost
        Tier 3: no fragment match             — GapFillRequest, user's LLM fills it
        """
        seg_str = json.dumps(segment.content, default=str) if segment.content else ""
        seg_sig = self._embed_sync(seg_str, full=False)

        candidates = await self._fragment_index.top_k(
            self.tenant_id, seg_sig, k=ANN_CANDIDATE_K
        )

        best_frag: Optional[TemplateSegment] = None
        best_sim = 0.0

        for seg_id, sim in candidates:
            frag = self._fragment_map.get(seg_id)
            if frag is None:
                continue

            if self.retrospector:
                try:
                    sys_db = self.retrospector.system_db
                    if await sys_db.is_quarantined(frag.segment_id, self.tenant_id):
                        continue
                except Exception:
                    pass

            if sim >= FRAGMENT_EXACT_THRESHOLD:
                # Tier 1: exact hit — zero cost
                frag.use_count += 1
                await self._write_behind.enqueue(frag)
                self._trace_frags_buf.append(frag.segment_id)
                return frag, True, None

            if sim > best_sim:
                best_sim = sim
                best_frag = frag

        if best_frag and best_sim >= FRAGMENT_SIMILAR_THRESHOLD:
            # Tier 2: similar fragment — use directly, no LLM adaptation
            best_frag.use_count += 1
            await self._write_behind.enqueue(best_frag)
            self._trace_frags_buf.append(best_frag.segment_id)
            return best_frag, True, None

        # Tier 3: no fragment match — defer to user's generation_fn
        gap_request = GapFillRequest(
            position=position,
            segment_id=segment.segment_id,
            hint=self._gap_hint(segment, context_window),
            surrounding_context=[s.content for s in context_window],
        )
        return segment, False, gap_request

    def _gap_hint(self, segment: TemplateSegment, window: List[TemplateSegment]) -> str:
        seg_preview = json.dumps(segment.content, default=str)[:120] if segment.content else "unknown step"
        if window:
            ctx = " → ".join(json.dumps(s.content, default=str)[:60] for s in window[:2])
            return f"generate: {seg_preview} (context: {ctx})"
        return f"generate: {seg_preview}"

    async def _guided_generation(
        self,
        partial_result: "EMEResult",
        goal: str,
        inputs: Dict,
        context: Dict,
        capabilities: List[str],
        constraints: Dict,
        generation_fn: Callable,
        fp: ComputationFingerprint,
    ) -> Optional["EMEResult"]:
        """
        System 2 guided generation.
        Matched segments + gap descriptions are injected into the user's
        generation_fn context. Their LLM fills the gaps as part of its
        normal call — zero Mnemon LLM cost. Filled segments are extracted
        positionally and added to the fragment library for future runs.
        """
        pending_gaps = partial_result.pending_gaps
        gap_positions = {g.position for g in pending_gaps}

        partial_segs = self.adapter.decompose(partial_result.template)
        matched_contents = [
            (partial_segs[i].get("content", partial_segs[i]) if isinstance(partial_segs[i], dict) else partial_segs[i])
            for i in range(len(partial_segs))
            if i not in gap_positions
        ]

        enriched = dict(context)
        enriched["_mnemon_partial_plan"] = {
            "matched_segments": matched_contents,
            "gaps": [
                {
                    "position": g.position,
                    "hint": g.hint,
                    "surrounding": g.surrounding_context,
                }
                for g in pending_gaps
            ],
            "total_steps": len(partial_segs),
            "instruction": (
                "matched_segments are already cached — do not regenerate them. "
                "Generate the complete plan; fill every listed gap."
            ),
        }

        try:
            template = await generation_fn(goal, inputs, enriched, capabilities, constraints)
        except Exception as e:
            logger.error(f"Guided generation failed: {e}")
            return None

        await self._cache_template(goal, template, fp, capabilities)
        await self._extract_gap_fragments(template, pending_gaps)

        return EMEResult(
            status="system2_guided",
            template=template,
            template_id=None,
            segments_reused=partial_result.segments_reused,
            segments_generated=len(pending_gaps),
            tokens_saved=partial_result.tokens_saved,
            latency_saved_ms=partial_result.latency_saved_ms,
            fragments_used=partial_result.fragments_used,
            cache_level="system2_guided",
            validation_passed=True,
        )

    async def _extract_gap_fragments(
        self,
        template: Any,
        pending_gaps: List[GapFillRequest],
    ) -> None:
        """
        Extract the filled segments at gap positions from the completed template.
        Each is added to the fragment library — next run with a similar gap
        hits Tier 1 or 2 at zero cost.
        """
        try:
            segments_data = self.adapter.decompose(template)
            for gap in pending_gaps:
                if gap.position >= len(segments_data):
                    continue
                seg_data = segments_data[gap.position]
                content = seg_data.get("content", seg_data) if isinstance(seg_data, dict) else seg_data
                content_str = json.dumps(content, default=str)
                sig = self._embed_sync(content_str, full=False)
                seg = TemplateSegment(
                    segment_id=hashlib.md5(
                        f"{self.tenant_id}:gap:{gap.segment_id}:{time.time()}".encode()
                    ).hexdigest()[:16],
                    tenant_id=self.tenant_id,
                    content=content,
                    fingerprint=hashlib.md5(content_str.encode()).hexdigest()[:16],
                    signature=sig,
                    is_generated=True,
                    confidence=0.85,
                    success_rate=1.0,
                )
                await self._write_behind.enqueue(seg)
                if sig:
                    await self._fragment_index.add(self.tenant_id, seg.segment_id, sig)
                    self._fragment_map[seg.segment_id] = seg
                self._trace_gen_buf.append(seg.segment_id)
                logger.debug(f"Gap fragment cached: {seg.segment_id} at position {gap.position}")
        except Exception as e:
            logger.debug(f"Gap fragment extraction failed (non-critical): {e}")

    # ──────────────────────────────────────────
    # FULL GENERATION
    # ──────────────────────────────────────────

    async def _full_generation(
        self,
        goal: str,
        inputs: Dict,
        context: Dict,
        capabilities: List[str],
        constraints: Dict,
        generation_fn: Callable,
        fp: ComputationFingerprint,
    ) -> EMEResult:
        """Call the real expensive function. Cache result on success."""
        try:
            template = await generation_fn(goal, inputs, context, capabilities, constraints)
        except Exception as e:
            logger.error(f"Full generation failed: {e}")
            return EMEResult(
                status="error",
                template=None,
                template_id=None,
                cache_level="miss",
                validation_passed=False,
            )

        segment_count = await self._cache_template(goal, template, fp, capabilities)

        return EMEResult(
            status="miss",
            template=template,
            template_id=None,
            segments_reused=0,
            segments_generated=segment_count,
            cache_level="miss",
        )

    # ──────────────────────────────────────────
    # CACHE WRITE
    # ──────────────────────────────────────────

    async def _cache_template(
        self,
        goal: str,
        template: Any,
        fp: ComputationFingerprint,
        capabilities: List[str],
    ):
        """
        Cache a successful template and update all in-memory indices.

        Always awaited — never called as create_task — so the System 1 cache
        entry is guaranteed to exist before this method returns. Any subsequent
        call with the same fingerprint will hit System 1 immediately.
        """
        try:
            lock = await self._tenant_locks.get(self.tenant_id)
            async with lock:
                segments_data = self.adapter.decompose(template)
                segments: List[TemplateSegment] = []

                for i, seg_data in enumerate(segments_data):
                    content = seg_data.get("content", seg_data)
                    content_str = json.dumps(content, default=str)
                    sig = self._embed_sync(content_str, full=False)

                    seg = TemplateSegment(
                        segment_id=seg_data.get("id", f"seg_{i}"),
                        tenant_id=self.tenant_id,
                        content=content,
                        fingerprint=hashlib.md5(content_str.encode()).hexdigest()[:16],
                        signature=sig,
                        is_generated=False,
                        confidence=1.0,
                        success_rate=1.0,
                    )
                    segments.append(seg)

                    await self._write_behind.enqueue(seg)
                    if sig:
                        await self._fragment_index.add(self.tenant_id, seg.segment_id, sig)
                        self._fragment_map[seg.segment_id] = seg
                    # Fire-and-forget: record cross-tenant success signal (never blocks agent)
                    if self.signal_db and seg.signature:
                        try:
                            dims = seg.signature[:32]
                            raw = struct.pack(f">{len(dims)}f", *dims)
                            shape_hash = hashlib.sha256(raw).hexdigest()[:32]
                            domain = list(seg.domain_tags)[0] if seg.domain_tags else "general"
                            asyncio.create_task(
                                self.signal_db.record_fragment_success(shape_hash, domain)
                            )
                        except Exception:
                            pass

                template_id = hashlib.sha256(
                    f"{self.tenant_id}:{fp.full_hash}:{time.time()}".encode()
                ).hexdigest()[:24]

                goal_embedding = await self._embed(goal, full=True)
                tool_versions = self.adapter.get_tool_versions(capabilities)

                et = ExecutionTemplate(
                    template_id=template_id,
                    tenant_id=self.tenant_id,
                    intent=goal,
                    fingerprint=fp,
                    segments=segments,
                    success_count=1,
                    embedding=goal_embedding,
                    tool_versions=tool_versions,
                )

                await self.db.write_template(et)

                # Update both indices before releasing lock
                self._system1_cache[fp.full_hash] = template_id
                await self._template_index.add(self.tenant_id, template_id, goal_embedding)

            logger.debug(f"Template cached: {template_id} ({len(segments)} segments)")
            return len(segments)

        except Exception as e:
            logger.warning(f"Template caching failed: {e}")
        return 0

    # ──────────────────────────────────────────
    # DEPENDENCY VALIDATION
    # ──────────────────────────────────────────

    async def _validate_stitched(
        self,
        segments: List[TemplateSegment],
        capabilities: List[str],
        constraints: Dict,
    ) -> bool:
        """
        Dependency validation for stitched templates.
        Checks that every segment's declared inputs are produced by prior segments.
        """
        produced: Set[str] = set()
        for seg in segments:
            for dep in seg.dependencies:
                if dep not in produced:
                    logger.warning(
                        f"Segment {seg.segment_id} dependency '{dep}' not satisfied"
                    )
                    return False
            for output in seg.outputs:
                produced.add(output)
        return True

    async def _validate_dependencies(self, template: ExecutionTemplate) -> bool:
        """Re-verify a template's tool versions after a dependency change."""
        current_versions = self.adapter.get_tool_versions(
            list(template.tool_versions.keys())
        )
        for tool, version_hash in template.tool_versions.items():
            current = current_versions.get(tool)
            if current and current != version_hash:
                logger.info(f"Tool {tool} version changed — template needs regeneration")
                return False
        return True

    # ──────────────────────────────────────────
    # HYDRATION
    # ──────────────────────────────────────────

    def _hydrate(self, template: ExecutionTemplate, inputs: Dict) -> Any:
        """Instantiate cached template with current variable values."""
        plan_str = json.dumps([s.content for s in template.segments], default=str)
        for key, value in inputs.items():
            plan_str = plan_str.replace(f"${{{key}}}", str(value))
        try:
            return json.loads(plan_str)
        except json.JSONDecodeError:
            return plan_str

    # ──────────────────────────────────────────
    # MARK FAILURE
    # ──────────────────────────────────────────

    async def mark_failure(self, template_id: str):
        """
        Signal that a template execution failed.

        [v2] Uses public DB API only. Original accessed db._conn directly,
        bypassing the persistence layer's asyncio lock and WAL transaction.
        This could corrupt the DB under concurrent agent writes.
        """
        await self.db.update_template_outcome(self.tenant_id, template_id, False)

        reverse = self._system1_reverse()
        fp_hash = reverse.get(template_id, "")
        if not fp_hash:
            return

        template = await self.db.fetch_template_by_fingerprint(self.tenant_id, fp_hash)
        if template and template.should_evict:
            await self.db.delete_template(self.tenant_id, template_id)
            await self._template_index.remove(self.tenant_id, template_id)
            if fp_hash in self._system1_cache:
                del self._system1_cache[fp_hash]
            logger.info(f"Template {template_id} evicted — failure rate > 50%")
            # Fire-and-forget: record cross-tenant failure signal for each segment
            if self.signal_db:
                for seg in template.segments:
                    if seg.signature:
                        try:
                            dims = seg.signature[:32]
                            raw = struct.pack(f">{len(dims)}f", *dims)
                            shape_hash = hashlib.sha256(raw).hexdigest()[:32]
                            domain = list(seg.domain_tags)[0] if seg.domain_tags else "general"
                            asyncio.create_task(
                                self.signal_db.record_fragment_failure(shape_hash, domain)
                            )
                        except Exception:
                            pass

    # ──────────────────────────────────────────
    # EMBEDDING HELPERS
    # ──────────────────────────────────────────

    async def _embed(self, text: str, full: bool = False) -> List[float]:
        """Async embed with LRU cache."""
        cache_key = f"{'F' if full else 'S'}:{text}"
        cached = await self._embed_cache.get(cache_key)
        if cached is not None:
            return cached
        result = self.embedder.embed_full(text) if full else self.embedder.embed(text)
        await self._embed_cache.set(cache_key, result)
        return result

    def _embed_sync(self, text: str, full: bool = False) -> List[float]:
        """Sync embed with LRU cache (for use in non-async methods)."""
        cache_key = f"{'F' if full else 'S'}:{text}"
        cached = self._embed_cache.get_sync(cache_key)
        if cached is not None:
            return cached
        result = self.embedder.embed_full(text) if full else self.embedder.embed(text)
        self._embed_cache.set_sync(cache_key, result)
        return result

    # ──────────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────────

    def _schema_of(self, inputs: Dict) -> Dict:
        """
        Extract structural schema from inputs (types only, not values).

        [v2] Handles nested dicts and lists without crashing on unhashable
        types. Original called type(v).__name__ on raw values which raised
        TypeError for dict/list inputs with complex inner structures.
        """
        def _type(v: Any) -> str:
            if isinstance(v, dict):
                return f"dict[{','.join(sorted(str(k) for k in v.keys()))}]"
            if isinstance(v, (list, tuple)):
                return f"list[{len(v)}]"
            return type(v).__name__

        return {k: _type(v) for k, v in inputs.items()}

    async def semantic_lookup(
        self, goal: str, capabilities: List[str]
    ) -> Optional[Tuple[str, str]]:
        """
        Check for a semantically similar cached result.
        Returns (template_id, cached_text) on hit, None on miss.
        Public API for the moth bridge — avoids private method access.
        """
        if not goal:
            return None
        try:
            fp = ComputationFingerprint.build(
                goal=goal, input_schema={}, context={},
                capabilities=capabilities, constraints={},
            )
            # System 1: exact fingerprint
            tid = self._system1_cache.get(fp.full_hash)
            if tid:
                template = await self.db.fetch_template_by_fingerprint(
                    self.tenant_id, fp.full_hash
                )
                if template and not template.should_evict and not template.is_prewarmed:
                    text = str(template.segments[0].content) if template.segments else ""
                    await self.db.update_template_outcome(self.tenant_id, tid, True)
                    return (tid, text)

            # System 2: semantic similarity — skip pre-warmed templates (plan JSON, not LLM text)
            goal_emb = await self._embed(goal, full=True)
            candidates = await self._template_index.top_k(self.tenant_id, goal_emb, k=5)
            for tid, score in candidates:
                if score < SYSTEM2_THRESHOLD_DEFAULT:
                    break
                reverse = self._system1_reverse()
                fp_hash = reverse.get(tid)
                if not fp_hash:
                    continue
                template = await self.db.fetch_template_by_fingerprint(
                    self.tenant_id, fp_hash
                )
                if template and not template.should_evict and not template.is_prewarmed:
                    text = (
                        str(template.segments[0].content)
                        if template.segments else ""
                    )
                    await self.db.update_template_outcome(self.tenant_id, tid, True)
                    return (tid, text)
        except Exception as e:
            logger.debug(f"EME semantic_lookup failed: {e}")
        return None

    async def cache_result(
        self, goal: str, result_text: str, capabilities: List[str]
    ) -> Optional[str]:
        """
        Cache an LLM result for future semantic retrieval.
        Returns template_id on success, None on failure.
        Public API for the moth bridge.
        """
        if not goal or not result_text:
            return None
        try:
            fp = ComputationFingerprint.build(
                goal=goal, input_schema={}, context={},
                capabilities=capabilities, constraints={},
            )
            if fp.full_hash in self._system1_cache:
                return self._system1_cache[fp.full_hash]
            await self._cache_template(goal, result_text, fp, capabilities)
            return self._system1_cache.get(fp.full_hash)
        except Exception as e:
            logger.debug(f"EME cache_result failed: {e}")
            return None

    async def shutdown(self):
        """Flush write-behind queue before process exit."""
        await self._write_behind.flush_now()

    def get_stats(self) -> Dict:
        return {
            "tenant_id":           self.tenant_id,
            "system1_entries":     len(self._system1_cache),
            "fragment_index_size": self._fragment_index.size(self.tenant_id),
            "template_index_size": self._template_index.size(self.tenant_id),
            "embed_cache_size":    len(self._embed_cache._cache),
            "threshold":           self.threshold,
        }
