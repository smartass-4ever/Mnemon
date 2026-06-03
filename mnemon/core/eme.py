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
from .embedder import SimpleEmbedder
from .signal_db import SignalDatabase

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────

SYSTEM2_THRESHOLD_DEFAULT  = 0.70   # minimum overall similarity for System 2
MIN_GOAL_SIMILARITY        = 0.60   # goal embedding must be at least this similar for System 2
FRAGMENT_EXACT_THRESHOLD   = 0.98   # fragment library exact hit
FRAGMENT_SIMILAR_THRESHOLD = 0.80   # fragment library similar hit (LLM adapts)
SEGMENT_MATCH_THRESHOLD    = 0.72   # per-segment intent similarity — clear match
INTENT_AMBIGUOUS_LOW       = 0.35   # below this: clear miss, skip drone
SPREADING_DECAY            = 0.85   # activation decay per hop in dependency graph
ANN_CANDIDATE_K            = 32     # ANN shortlist size for fragment search
TEMPLATE_CANDIDATE_K       = 20     # top-k templates scored in System 2
EMBEDDING_CACHE_SIZE       = 2048   # LRU embedding cache slots
WRITE_BEHIND_DEBOUNCE_MS   = 10     # fragment write-behind batch window (ms)

# Multi-component similarity weights
GOAL_WEIGHT       = 0.30
SCHEMA_WEIGHT     = 0.25
CONTEXT_WEIGHT    = 0.25
CAPABILITY_WEIGHT = 0.20

# Fragment-overlap scoring (Approach A)
FRAGMENT_OVERLAP_WEIGHT    = 0.30   # additive bonus on top of multi-component score
FRAGMENT_OVERLAP_GATE      = 0.60   # min overlap to pass the relaxed goal-sim gate
MIN_GOAL_SIMILARITY_RELAXED = 0.40  # lower goal-sim gate when fragment overlap compensates

# Fragment assembly (Approach B — cheap-LLM decomposition path)
FRAGMENT_ASSEMBLY_THRESHOLD = 0.50  # min fraction of steps that must have fragment matches


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
        if len(segments) == 1:
            return segments[0].content
        return [s.content for s in segments]

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
    fragment_ids_used: List[str] = field(default_factory=list)


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
            # Embedder dimension changed (e.g. hash-projection → sentence-transformers).
            # Reset shard rather than mixing incompatible vectors.
            if mat is not None and mat.shape[1] != vec.shape[0]:
                mat, ids = None, []
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
            if mat is not None and mat.shape[1] != vec.shape[0]:
                mat, ids = None, []
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
        drone_fn: Optional[Callable] = None,
    ):
        self.tenant_id = tenant_id
        self.db = db
        self.embedder = embedder or SimpleEmbedder()
        self.adapter = adapter or GenericAdapter()
        self.threshold = similarity_threshold
        self.signal_db = signal_db
        self.drone_fn = drone_fn  # async (goal: str, step_intent: str) -> bool

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
        # Segment objects from _try_system2 guided path — read by _guided_generation.
        self._guided_segments_buf: List[TemplateSegment] = []

        # Per-run context for reputation-weighted assembly
        self._run_framework: str = "unknown"
        self._run_goal_type: str = "general"

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
        run_start = time.time()

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
        self._guided_segments_buf = []

        self._run_framework = context.get("_mnemon_framework", "unknown") if context else "unknown"
        self._run_goal_type = context.get("_mnemon_goal_type", "general") if context else "general"

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
            # ── FRAGMENT ASSEMBLY ──────────────
            # Approach B: decompose goal → fragment lookup → guided generation
            # for gaps. Fires when no template matches but the fragment library
            # covers enough of the expected steps.
            result = await self._try_fragment_assembly(
                fp, goal, inputs, context, capabilities,
                constraints, generation_fn, memory_context
            )

        if not result:
            # ── FULL GENERATION ────────────────
            result = await self._full_generation(
                goal, inputs, context, capabilities,
                constraints, generation_fn, fp
            )

        # ── RETROSPECTOR TRACE ─────────────────
        if self.retrospector:
            try:
                _outcome_map = {
                    "system1": "success", "system2": "success",
                    "system2_guided": "success", "miss": "miss", "error": "failure",
                }
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
                    overall_outcome=_outcome_map.get(result.status, "miss"),
                    latency_ms=(time.time() - run_start) * 1000,
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

        # Short-form goal embedding used for per-segment fragment overlap scoring.
        # Computed once here so we don't re-embed on every candidate.
        goal_sig_for_overlap = self._embed_sync(goal, full=False)

        for t in templates:
            if not t.embedding:
                continue
            goal_sim = SimpleEmbedder.cosine_similarity(goal_embedding, t.embedding)

            fragment_overlap = self._compute_fragment_overlap(t.segments, goal_sig_for_overlap)
            score = self._multi_component_similarity(
                fp, t.fingerprint, goal_embedding, t.embedding,
                capabilities, list(t.tool_versions.keys())
            )
            score = min(1.0, score + fragment_overlap * FRAGMENT_OVERLAP_WEIGHT)
            # Gate: skip only when raw goal similarity, fragment overlap, AND
            # intent-based score are all too low. Intent score can rescue a low
            # raw cosine — no funnel, both run together.
            if goal_sim < MIN_GOAL_SIMILARITY_RELAXED and fragment_overlap < FRAGMENT_OVERLAP_GATE and score < self.threshold:
                continue

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

        # Segment-level diff (now async — drone may fire on ambiguous segments)
        matched, unmatched_indices = await self._segment_diff(
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
                fragment_ids_used=list(self._trace_frags_buf),
            )

        # Gap fill for unmatched segments
        all_segments = list(best_template.segments)
        fragments_used = 0
        fragment_filled_segs: List[TemplateSegment] = []
        pending_gaps: List[GapFillRequest] = []

        for idx in unmatched_indices:
            seg = all_segments[idx]
            window = self._window(all_segments, idx, window_size=2)
            prev_frag_id = (
                all_segments[idx - 1].segment_id if idx > 0 else None
            )
            filled, used_fragment, gap_request = await self._fill_gap(
                seg, goal, window, memory_context, context, idx,
                prev_fragment_id=prev_frag_id,
            )
            if gap_request is not None:
                pending_gaps.append(gap_request)
            else:
                all_segments[idx] = filled
                if used_fragment:
                    fragments_used += 1
                    fragment_filled_segs.append(filled)

        if pending_gaps:
            gap_positions = {g.position for g in pending_gaps}
            resolved_segs = [s for i, s in enumerate(all_segments) if i not in gap_positions]
            # Stash full segment objects so _guided_generation can build the brief
            # and stitch without re-fetching from DB.
            self._guided_segments_buf = list(all_segments)
            return EMEResult(
                status="system2_guided",
                template=self.adapter.reconstruct(all_segments),
                template_id=best_template.template_id,
                segments_reused=len(resolved_segs),
                segments_generated=0,
                tokens_saved=0,
                latency_saved_ms=0.0,
                fragments_used=fragments_used,
                cache_level="system2_guided",
                pending_gaps=pending_gaps,
                fragment_ids_used=list(self._trace_frags_buf),
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

        # Fragment-filled segments are also served from cache — count them in savings
        cached_count = len(matched) + fragments_used
        tokens_saved = self._seg_tokens(matched) + self._seg_tokens(fragment_filled_segs)
        return EMEResult(
            status="system2",
            template=stitched_template,
            template_id=best_template.template_id,
            segments_reused=cached_count,
            segments_generated=0,
            tokens_saved=tokens_saved,
            latency_saved_ms=cached_count * 2500,
            fragments_used=fragments_used,
            cache_level="system2",
            validation_passed=True,
            fragment_ids_used=list(self._trace_frags_buf),
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
        # Exact value match → 1.0; same structure different values → 0.85; different structure → 0.3
        if fp1.input_schema_hash == fp2.input_schema_hash:
            schema_sim = 1.0
        elif (fp1.structural_schema_hash and fp2.structural_schema_hash
              and fp1.structural_schema_hash == fp2.structural_schema_hash):
            schema_sim = 0.85
        else:
            schema_sim = 0.3
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

    def _compute_fragment_overlap(
        self, segments: List["TemplateSegment"], goal_sig: List[float]
    ) -> float:
        """
        Fraction of a template's segments whose intent signature scores
        >= SEGMENT_MATCH_THRESHOLD against the incoming goal.

        Used as an additive score component in System 2 candidate selection
        so templates with high structural relevance can be selected even when
        goal-string similarity is moderate.
        """
        if not segments:
            return 0.0
        matched = sum(
            1 for seg in segments
            if seg.signature and
            SimpleEmbedder.cosine_similarity(goal_sig, seg.signature) >= SEGMENT_MATCH_THRESHOLD
        )
        return matched / len(segments)

    @staticmethod
    def _extract_intent(seg_data: dict) -> str:
        """
        Rule-based intent extraction — no LLM, no I/O.

        GenericAdapter.decompose() always wraps step data as
        {"id": "seg_N", "content": <actual data>}.

        Priority:
          1. Plain text content  — LLM text response IS the intent phrase
          2. Anthropic tool_use  — name + key input fields
          3. LangChain agent     — action + action_input
          4. Generic structured  — action / tool / outputs / description
          5. List content        — join items as a sequence description
          6. Deep scan           — recurse one level for nested dicts
          7. First long string   — anything > 20 chars that isn't an ID
        """
        import re

        def _clean_text(text: str) -> str:
            return re.sub(
                r"^(step\s*\d+\s*[:\-\.]\s*|\d+\.\s*|#{1,3}\s*)",
                "", text.strip(), flags=re.IGNORECASE,
            ).strip()

        def _extract_from_dict(src: dict) -> List[str]:
            parts: List[str] = []

            # Anthropic tool_use block
            if src.get("type") == "tool_use" and src.get("name"):
                parts.append(src["name"])
                inp = src.get("input", {})
                if isinstance(inp, dict):
                    for v in list(inp.values())[:2]:
                        if isinstance(v, str) and len(v) > 3:
                            parts.append(v[:100])
                            break
                return parts

            # LangChain agent action style
            action_input = src.get("action_input")
            if action_input is not None:
                action = src.get("action", "")
                if isinstance(action, str) and action.strip():
                    parts.append(action.strip())
                if isinstance(action_input, str) and action_input.strip():
                    parts.append(action_input.strip()[:150])
                elif isinstance(action_input, dict):
                    for v in list(action_input.values())[:1]:
                        if isinstance(v, str) and v.strip():
                            parts.append(v.strip()[:150])
                if parts:
                    return parts

            # Generic structured fields
            for key in ("action", "step", "name", "task", "operation"):
                val = src.get(key)
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
                    break

            for key in ("tool", "tool_call", "function", "api"):
                val = src.get(key)
                if isinstance(val, str) and val.strip():
                    parts.append(f"using {val.strip()}")
                    break

            outputs = src.get("outputs") or src.get("produces") or src.get("returns")
            if outputs:
                if isinstance(outputs, list):
                    parts.append(f"produces {', '.join(str(o) for o in outputs[:3])}")
                elif isinstance(outputs, str):
                    parts.append(f"produces {outputs}")

            for key in ("goal", "description", "objective", "intent", "query", "input"):
                val = src.get(key)
                if isinstance(val, str) and len(val) > 5 and val not in parts:
                    parts.append(val.strip()[:150])
                    break

            return parts

        inner = seg_data.get("content")

        # ── 1. Plain text ────────────────────────────────────────────────
        if isinstance(inner, str) and len(inner) > 10:
            cleaned = _clean_text(inner)
            return cleaned[:200] if cleaned else inner[:200]

        # ── 2–4. Structured dict ─────────────────────────────────────────
        sources: List[dict] = []
        if isinstance(inner, dict):
            sources.append(inner)
        sources.append(seg_data)

        for src in sources:
            parts = _extract_from_dict(src)
            if parts:
                return " | ".join(parts)

        # ── 5. List content — describe as a sequence ─────────────────────
        if isinstance(inner, list):
            items = []
            for item in inner[:4]:
                if isinstance(item, str) and len(item) > 3:
                    items.append(_clean_text(item)[:60])
                elif isinstance(item, dict):
                    p = _extract_from_dict(item)
                    if p:
                        items.append(p[0][:60])
            if items:
                return " then ".join(items)

        # ── 6. Deep scan — one level of nesting ──────────────────────────
        for v in (inner or seg_data).values() if isinstance((inner or seg_data), dict) else []:
            if isinstance(v, dict):
                parts = _extract_from_dict(v)
                if parts:
                    return " | ".join(parts)

        # ── 7. First long string value that isn't an ID ──────────────────
        for v in seg_data.values():
            if isinstance(v, str) and len(v) > 20:
                return v[:200]
        if isinstance(inner, dict):
            for v in inner.values():
                if isinstance(v, str) and len(v) > 20:
                    return v[:200]

        # Last resort — truncated JSON (still better than empty)
        return json.dumps(seg_data.get("content", seg_data), default=str)[:200]

    async def _drone_verify(self, goal: str, step_intent: str) -> bool:
        """
        Lightweight LLM drone for ambiguous segment matches.
        Only fires when a segment's activated intent score falls in
        [INTENT_AMBIGUOUS_LOW, SEGMENT_MATCH_THRESHOLD) — typically 3-8% of segments.
        If no drone_fn is configured, defaults to False (conservative: cache miss).
        """
        if self.drone_fn is None:
            return False
        try:
            prompt = (
                f"Does this cached step serve the current goal?\n\n"
                f"Goal: {goal}\n"
                f"Step: {step_intent}\n\n"
                f"Reply with only YES or NO."
            )
            result = await self.drone_fn(prompt)
            return str(result).strip().upper().startswith("Y")
        except Exception as e:
            logger.debug(f"Intent drone failed (treating as miss): {e}")
            return False

    async def _segment_diff(
        self,
        segments: List[TemplateSegment],
        goal: str,
        inputs: Dict,
    ) -> Tuple[List[TemplateSegment], List[int]]:
        """
        Intent-based segment matching with spreading activation over the dependency graph.

        1. Base score: cosine(goal_embedding, seg.signature)
           - New segments: sig = intent phrase embedding (same semantic space as goal)
           - Legacy segments (no intent field): sig = content JSON embedding (old behavior)
        2. Spreading activation via BFS through dependency edges (bidirectional, decay=0.85)
           — a confident match in one step boosts adjacent steps in the plan graph
        3. Classification:
           - score >= 0.72 (SEGMENT_MATCH_THRESHOLD): matched, reuse from cache
           - score in [0.35, 0.72): ambiguous → conditional drone verification
           - score < 0.35 (INTENT_AMBIGUOUS_LOW): clear miss → gap fill
        4. No drone_fn: ambiguous defaults to gap fill (conservative)
        """
        goal_sig = self._embed_sync(goal, full=False)

        # Build bidirectional adjacency from stored dependency edges
        seg_by_id: Dict[str, TemplateSegment] = {s.segment_id: s for s in segments}
        adj: Dict[str, Set[str]] = {s.segment_id: set() for s in segments}
        for seg in segments:
            for dep_id in (seg.dependencies or []):
                if dep_id in adj:
                    adj[seg.segment_id].add(dep_id)
                    adj[dep_id].add(seg.segment_id)

        # Step 1: base cosine scores against intent embeddings
        base_scores: Dict[str, float] = {}
        for seg in segments:
            if not seg.signature:
                base_scores[seg.segment_id] = 0.0
            else:
                base_scores[seg.segment_id] = SimpleEmbedder.cosine_similarity(
                    goal_sig, seg.signature
                )

        # Step 2: spreading activation — high-scoring segments boost their neighbors
        activated: Dict[str, float] = dict(base_scores)
        for source_id, score in base_scores.items():
            if score < INTENT_AMBIGUOUS_LOW:
                continue
            queue: List[Tuple[str, float]] = [(source_id, score)]
            visited: Set[str] = {source_id}
            while queue:
                current_id, current_score = queue.pop(0)
                for neighbor_id in adj.get(current_id, set()):
                    if neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)
                    spread = current_score * SPREADING_DECAY
                    if spread > activated.get(neighbor_id, 0.0):
                        activated[neighbor_id] = spread
                        queue.append((neighbor_id, spread))

        # Step 3: classify using final activated scores
        matched_segs: List[TemplateSegment] = []
        unmatched_indices: List[int] = []

        for i, seg in enumerate(segments):
            if not seg.signature:
                logger.debug(f"Segment {seg.segment_id}: no intent signature — gap fill")
                unmatched_indices.append(i)
                continue

            score = activated.get(seg.segment_id, 0.0)

            if score >= SEGMENT_MATCH_THRESHOLD:
                matched_segs.append(seg)
            elif score >= INTENT_AMBIGUOUS_LOW:
                step_intent = seg.intent or seg.segment_id
                if await self._drone_verify(goal, step_intent):
                    matched_segs.append(seg)
                else:
                    unmatched_indices.append(i)
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
        prev_fragment_id: Optional[str] = None,
    ) -> Tuple[TemplateSegment, bool, Optional[GapFillRequest]]:
        """
        Three-tier fragment lookup with reputation × edge-strength weighting.

        Scoring: adjusted_sim = cosine×0.60 + reputation×0.30 + edge_strength×0.10
        Reputation and edge weights are time-decayed — recent evidence dominates.
        Quarantined fragments are skipped before scoring.

        Tier 1: adjusted_sim ≥ FRAGMENT_EXACT_THRESHOLD  — zero cost
        Tier 2: adjusted_sim ≥ FRAGMENT_SIMILAR_THRESHOLD — zero cost
        Tier 3: no match — GapFillRequest, user's LLM fills it
        """
        seg_str = json.dumps(segment.content, default=str) if segment.content else ""
        seg_sig = self._embed_sync(seg_str, full=False)

        candidates = await self._fragment_index.top_k(
            self.tenant_id, seg_sig, k=ANN_CANDIDATE_K
        )

        # Batch-fetch reputations and edges for all candidates at once
        cand_ids = [seg_id for seg_id, _ in candidates]
        reputations: Dict[str, float] = {}
        if self.signal_db and cand_ids:
            try:
                reputations = await self.signal_db.get_reputation_batch(
                    cand_ids, self._run_framework, self._run_goal_type
                )
            except Exception:
                pass

        best_frag: Optional[TemplateSegment] = None
        best_adj  = 0.0

        for seg_id, sim in candidates:
            frag = self._fragment_map.get(seg_id)
            if frag is None:
                continue

            # Skip quarantined fragments
            if self.retrospector:
                try:
                    if await self.retrospector.system_db.is_quarantined(
                        frag.segment_id, self.tenant_id
                    ):
                        continue
                except Exception:
                    pass

            # Reputation weight (time-decayed, defaults to 0.5 neutral)
            rep = reputations.get(seg_id, 0.5)

            # Edge strength from previous fragment (Hebbian synapse)
            edge = 0.5
            if self.signal_db and prev_fragment_id:
                try:
                    edge = await self.signal_db.get_edge_strength(
                        prev_fragment_id, seg_id, self._run_framework
                    )
                except Exception:
                    pass

            # Combined score: cosine dominates, reputation and edge refine
            adjusted = sim * 0.60 + rep * 0.30 + edge * 0.10

            if adjusted >= FRAGMENT_EXACT_THRESHOLD:
                frag.use_count += 1
                await self._write_behind.enqueue(frag)
                self._trace_frags_buf.append(frag.segment_id)
                return frag, True, None

            if adjusted > best_adj:
                best_adj  = adjusted
                best_frag = frag

        if best_frag and best_adj >= FRAGMENT_SIMILAR_THRESHOLD:
            best_frag.use_count += 1
            await self._write_behind.enqueue(best_frag)
            self._trace_frags_buf.append(best_frag.segment_id)
            return best_frag, True, None

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

    def _build_guided_brief(
        self,
        all_segments: List[TemplateSegment],
        pending_gaps: List[GapFillRequest],
    ) -> Dict:
        """
        Build a structured generation directive from segment metadata.
        No LLM, no I/O — pure code using intent/outputs already stored on each segment.

        Matched segments are represented as capsules (intent + outputs only).
        Full content stays in Mnemon memory and is unfurled during stitching —
        never sent to the user's LLM, so their context window only grows by
        the size of the capsule summaries, not the cached content.
        """
        gap_positions = {g.position for g in pending_gaps}

        pre_filled = []
        for i, seg in enumerate(all_segments):
            if i not in gap_positions:
                pre_filled.append({
                    "position": i,
                    "capsule_id": seg.segment_id,
                    "intent": seg.intent or f"step_{i}",
                    "outputs": list(seg.outputs) if seg.outputs else [],
                })

        gaps_to_fill = []
        for gap in pending_gaps:
            receives = []
            prev_idx = gap.position - 1
            if prev_idx >= 0 and prev_idx < len(all_segments) and prev_idx not in gap_positions:
                prev_seg = all_segments[prev_idx]
                receives = list(prev_seg.outputs) if prev_seg.outputs else []
            gaps_to_fill.append({
                "position": gap.position,
                "receives": receives,
                "hint": gap.hint,
            })

        return {
            "pre_filled": pre_filled,
            "gaps_to_fill": gaps_to_fill,
            "total_steps": len(all_segments),
            "instruction": (
                "pre_filled steps are already cached — do NOT regenerate them. "
                "Generate ONLY the steps listed in gaps_to_fill. "
                "Return strictly as JSON: {\"<position>\": <step_content>} "
                "with one key per gap position and nothing else."
            ),
        }

    def _format_guided_goal(
        self,
        goal: str,
        brief: Dict,
    ) -> str:
        """
        Inject the guided brief directly into the goal string as natural language.

        This ensures the LLM receives the brief regardless of whether the user's
        generation_fn reads the context parameter. Every generation_fn passes goal
        to the LLM — this is the only reliable injection point.

        Cached steps are shown as intent summaries only (not full content) so the
        LLM's context window stays small. Full content is restored during stitching.
        """
        lines = [goal, "", "[Mnemon: partial cache hit — generate only the missing steps]", ""]

        if brief.get("pre_filled"):
            lines.append("Steps already solved (DO NOT regenerate):")
            for step in brief["pre_filled"]:
                intent = step.get("intent", f"step {step['position']}")
                outputs = step.get("outputs", [])
                line = f"  - Step {step['position']}: {intent}"
                if outputs:
                    line += f" → produces {', '.join(str(o) for o in outputs)}"
                lines.append(line)
            lines.append("")

        lines.append("Generate ONLY these missing steps as JSON {\"position\": <step_content>}:")
        for gap in brief.get("gaps_to_fill", []):
            receives = gap.get("receives", [])
            hint = gap.get("hint", "")
            # Strip the "generate: " prefix from hint — already implied
            hint_clean = hint.replace("generate: ", "").split(" (context:")[0].strip()
            line = f"  - Step {gap['position']}"
            if receives:
                line += f" (receives: {', '.join(str(r) for r in receives)})"
            if hint_clean:
                line += f": {hint_clean}"
            lines.append(line)

        return "\n".join(lines)

    def _parse_gap_fills(
        self,
        output: Any,
        pending_gaps: List[GapFillRequest],
        all_segments: List[TemplateSegment],
    ) -> Optional[Dict[int, Any]]:
        """
        Three-tier robust parser for gap-fill output.

        Tier 1: output is or parses as JSON {"position": content}
        Tier 2: find the first embedded JSON object in text that satisfies all gaps
        Tier 3: positional decomposition via adapter — take segments at gap positions
        Returns None on all-tier failure — caller falls back to full generation.
        Never raises; every tier is individually guarded.
        """
        import re

        def _extract_from_dict(d: dict) -> Optional[Dict[int, Any]]:
            result = {}
            for g in pending_gaps:
                if str(g.position) in d:
                    result[g.position] = d[str(g.position)]
                elif g.position in d:
                    result[g.position] = d[g.position]
            return result if len(result) == len(pending_gaps) else None

        # Tier 1a: output is already a dict
        if isinstance(output, dict):
            r = _extract_from_dict(output)
            if r is not None:
                return r

        # Serialise for text-based tiers
        output_str = json.dumps(output, default=str) if isinstance(output, (dict, list)) else str(output)

        # Tier 1b: strict JSON parse of the full string
        try:
            parsed = json.loads(output_str)
            if isinstance(parsed, dict):
                r = _extract_from_dict(parsed)
                if r is not None:
                    return r
        except (json.JSONDecodeError, ValueError):
            pass

        # Tier 2: scan for any embedded JSON object that satisfies all gaps
        try:
            for match in re.finditer(r'\{[^{}]+\}', output_str, re.DOTALL):
                try:
                    parsed = json.loads(match.group())
                    if isinstance(parsed, dict):
                        r = _extract_from_dict(parsed)
                        if r is not None:
                            return r
                except (json.JSONDecodeError, ValueError):
                    continue
        except Exception:
            pass

        # Tier 3: positional decomposition — decompose the full output and
        # lift content at each gap position
        try:
            segs_data = self.adapter.decompose(output)
            if len(segs_data) >= len(all_segments):
                result = {}
                for g in pending_gaps:
                    if g.position < len(segs_data):
                        sd = segs_data[g.position]
                        result[g.position] = sd.get("content", sd) if isinstance(sd, dict) else sd
                if len(result) == len(pending_gaps):
                    return result
        except Exception:
            pass

        return None

    def _stitch_plan(
        self,
        all_segments: List[TemplateSegment],
        gap_fills: Dict[int, Any],
        pending_gaps: List[GapFillRequest],
    ) -> List[TemplateSegment]:
        """
        Unfurl capsules + splice generated gap fills into the final segment list.

        Capsule positions: full cached TemplateSegment objects already in all_segments
        — nothing to do, they were never sent to the user's LLM.
        Gap positions: build fresh TemplateSegments from parsed fill content,
        embed their intent so future System 2 calls can match against them.
        """
        stitched = list(all_segments)
        for gap in pending_gaps:
            fill_content = gap_fills[gap.position]
            content_str = json.dumps(fill_content, default=str)
            intent = self._extract_intent(
                fill_content if isinstance(fill_content, dict) else {"content": fill_content}
            )
            sig = self._embed_sync(intent, full=False)
            stitched[gap.position] = TemplateSegment(
                segment_id=hashlib.md5(
                    f"{self.tenant_id}:gap:{gap.segment_id}:{time.time()}".encode()
                ).hexdigest()[:16],
                tenant_id=self.tenant_id,
                content=fill_content,
                fingerprint=hashlib.md5(content_str.encode()).hexdigest()[:16],
                signature=sig,
                intent=intent,
                is_generated=True,
                confidence=0.85,
                success_rate=1.0,
            )
        return stitched

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
        System 2 guided generation — structured brief + robust parse + deterministic stitch.

        1. _build_guided_brief: capsule refs for matched segs (intent+outputs only),
           explicit positional gap directive, JSON output contract — zero LLM.
        2. generation_fn call: user's LLM receives a clear, minimal brief; their
           context window grows only by capsule summaries, not full cached content.
        3. _parse_gap_fills: 3-tier parser (strict JSON → embedded JSON → positional).
           On all-tier failure, falls back to the raw output as a full miss —
           never returns a corrupt or partially-stitched plan.
        4. _stitch_plan: unfurls capsules (cached content restored from memory),
           splices generated fills, builds final TemplateSegments with intent embeddings.
        5. _validate_stitched: dependency check before caching.
        """
        pending_gaps = partial_result.pending_gaps
        all_segments = self._guided_segments_buf

        brief = self._build_guided_brief(all_segments, pending_gaps)
        enriched = dict(context)
        enriched["_mnemon_brief"] = brief

        # Inject brief into the goal string — the only reliable delivery path.
        # generation_fn always receives goal; context is often ignored by user code.
        guided_goal = self._format_guided_goal(goal, brief)

        raw_output = await generation_fn(guided_goal, inputs, enriched, capabilities, constraints)

        gap_fills = self._parse_gap_fills(raw_output, pending_gaps, all_segments)

        if gap_fills is None:
            # All three parse tiers exhausted — treat as a full miss.
            # Still cache the raw output so subsequent identical calls hit System 1.
            logger.warning("Guided generation: all parse tiers failed — caching raw output as full miss")
            await self._cache_template(goal, raw_output, fp, capabilities)
            return EMEResult(
                status="miss",
                template=raw_output,
                template_id=None,
                segments_reused=0,
                segments_generated=len(all_segments),
                cache_level="miss",
            )

        stitched_segments = self._stitch_plan(all_segments, gap_fills, pending_gaps)

        is_valid = await self._validate_stitched(stitched_segments, capabilities, constraints)
        if not is_valid:
            logger.warning("Guided generation: stitched plan failed dependency validation — caching raw output as full miss")
            await self._cache_template(goal, raw_output, fp, capabilities)
            return EMEResult(
                status="miss",
                template=raw_output,
                template_id=None,
                segments_reused=0,
                segments_generated=len(all_segments),
                cache_level="miss",
            )

        final_template = self.adapter.reconstruct(stitched_segments)
        await self._cache_template(goal, final_template, fp, capabilities)

        # Register generated gap segments in the fragment library so future
        # calls with similar gaps hit Tier 1/2 at zero cost.
        for gap in pending_gaps:
            if gap.position < len(stitched_segments):
                seg = stitched_segments[gap.position]
                if seg.signature:
                    await self._fragment_index.add(self.tenant_id, seg.segment_id, seg.signature)
                    self._fragment_map[seg.segment_id] = seg
                    await self._write_behind.enqueue(seg)
                    self._trace_gen_buf.append(seg.segment_id)

        gap_positions = {g.position for g in pending_gaps}
        cached_segs = [s for i, s in enumerate(stitched_segments) if i not in gap_positions]
        tokens_saved = self._seg_tokens(cached_segs)
        if tokens_saved == 0 and all_segments:
            # All segments went to guided generation, but the cached template
            # avoided the planning phase — the LLM received a structured brief
            # instead of generating the plan structure from scratch. Credit the
            # template's planning-phase token value as savings.
            tokens_saved = self._seg_tokens(all_segments)
        latency_saved = partial_result.segments_reused * 2500
        if latency_saved == 0 and all_segments:
            latency_saved = len(all_segments) * 1000
        return EMEResult(
            status="system2_guided",
            template=final_template,
            template_id=None,
            segments_reused=partial_result.segments_reused,
            segments_generated=len(pending_gaps),
            tokens_saved=tokens_saved,
            latency_saved_ms=latency_saved,
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
    # FRAGMENT ASSEMBLY (Approach B)
    # ──────────────────────────────────────────

    async def _decompose_goal(
        self, goal: str, capabilities: List[str]
    ) -> Optional[List[str]]:
        """
        Return a list of step-intent strings for this goal.

        If capabilities are provided by the caller they already ARE the steps
        (free — no LLM call needed).  Otherwise fall back to drone_fn (cheap
        LLM) to produce a decomposition.  Returns None when neither is
        available so the caller skips fragment assembly gracefully.
        """
        if capabilities:
            return list(capabilities)
        if self.drone_fn is None:
            return None
        try:
            prompt = (
                f"Break this goal into 3-8 implementation steps.\n"
                f"Output ONLY a JSON array of short step descriptions.\n"
                f"Example: [\"validate input\", \"query database\", \"return result\"]\n\n"
                f"Goal: {goal}\n\nJSON:"
            )
            raw = await self.drone_fn(prompt)
            import re as _re
            m = _re.search(r"\[.*?\]", str(raw), _re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                if isinstance(parsed, list) and parsed:
                    return [str(s).strip() for s in parsed if str(s).strip()]
        except Exception as e:
            logger.debug(f"Fragment assembly decomposition failed: {e}")
        return None

    async def _try_fragment_assembly(
        self,
        fp: "ComputationFingerprint",
        goal: str,
        inputs: Dict,
        context: Dict,
        capabilities: List[str],
        constraints: Dict,
        generation_fn: Callable,
        memory_context: Optional[Dict],
    ) -> Optional["EMEResult"]:
        """
        Approach B: fragment assembly without a template match.

        When System 2 finds no template close enough, this path:
          1. Decomposes the goal into step intents (free via capabilities,
             or cheap via drone_fn LLM).
          2. Queries the fragment library for each step.
          3. If coverage >= FRAGMENT_ASSEMBLY_THRESHOLD (50%): builds a
             guided brief — pre-solved steps listed as capsules, gaps
             sent to the user's generation_fn.
          4. Stitches fragments + generated gaps → caches as new template.

        Token savings = tokens for the covered steps (not generated).
        Cost = one cheap decomposition call (or zero if capabilities given).
        """
        step_intents = await self._decompose_goal(goal, capabilities)
        if not step_intents:
            return None

        # ── Match each step against the fragment library ──────────────────
        assembled: List[Optional[TemplateSegment]] = []
        for intent in step_intents:
            intent_sig = self._embed_sync(intent, full=False)
            candidates = await self._fragment_index.top_k(
                self.tenant_id, intent_sig, k=8
            )
            best_frag: Optional[TemplateSegment] = None
            best_sim = 0.0
            for seg_id, sim in candidates:
                if sim < FRAGMENT_SIMILAR_THRESHOLD:
                    break
                frag = self._fragment_map.get(seg_id)
                if frag and sim > best_sim:
                    best_sim = sim
                    best_frag = frag
            assembled.append(best_frag)

        covered = sum(1 for s in assembled if s is not None)
        coverage = covered / len(step_intents)

        if coverage < FRAGMENT_ASSEMBLY_THRESHOLD:
            return None

        gap_positions = [i for i, s in enumerate(assembled) if s is None]

        # ── All steps covered — assemble directly, no LLM needed ─────────
        if not gap_positions:
            segs = [s for s in assembled if s is not None]
            final = self.adapter.reconstruct(segs)
            await self._cache_template(goal, final, fp, capabilities)
            return EMEResult(
                status="system2",
                template=final,
                template_id=None,
                segments_reused=covered,
                segments_generated=0,
                tokens_saved=self._seg_tokens(segs),
                latency_saved_ms=covered * 2500,
                cache_level="system2",
            )

        # ── Build guided brief: capsules for covered, gaps for the rest ───
        pre_filled = [
            {
                "position": i,
                "capsule_id": assembled[i].segment_id,
                "intent": assembled[i].intent or step_intents[i],
                "outputs": list(assembled[i].outputs) if assembled[i].outputs else [],
            }
            for i in range(len(step_intents)) if assembled[i] is not None
        ]
        gaps_to_fill = []
        for pos in gap_positions:
            receives: List[str] = []
            if pos > 0 and assembled[pos - 1] is not None:
                receives = list(assembled[pos - 1].outputs or [])
            gaps_to_fill.append({
                "position": pos,
                "receives": receives,
                "hint": f"generate: {step_intents[pos]}",
            })

        brief = {
            "pre_filled": pre_filled,
            "gaps_to_fill": gaps_to_fill,
            "total_steps": len(step_intents),
            "instruction": (
                "pre_filled steps are already solved — do NOT regenerate them. "
                "Generate ONLY the steps in gaps_to_fill as JSON: "
                "{\"<position>\": <step_content>} with one key per gap."
            ),
        }
        enriched = dict(context)
        enriched["_mnemon_brief"] = brief
        guided_goal = self._format_guided_goal(goal, brief)

        raw_output = await generation_fn(
            guided_goal, inputs, enriched, capabilities, constraints
        )

        # Build GapFillRequest objects for the parser
        pending_gaps = [
            GapFillRequest(
                position=pos,
                segment_id=f"fa_gap_{pos}",
                hint=step_intents[pos],
                surrounding_context=[],
            )
            for pos in gap_positions
        ]

        # Placeholder segments for parse_gap_fills (gaps need a TemplateSegment slot)
        all_segs_with_placeholders: List[TemplateSegment] = []
        for i, seg in enumerate(assembled):
            if seg is not None:
                all_segs_with_placeholders.append(seg)
            else:
                sig = self._embed_sync(step_intents[i], full=False)
                all_segs_with_placeholders.append(TemplateSegment(
                    segment_id=f"fa_gap_{i}",
                    tenant_id=self.tenant_id,
                    content={"gap": True, "intent": step_intents[i]},
                    fingerprint=f"gap_{i}",
                    signature=sig,
                    intent=step_intents[i],
                    is_generated=False,
                    confidence=0.0,
                    success_rate=0.0,
                ))

        gap_fills = self._parse_gap_fills(raw_output, pending_gaps, all_segs_with_placeholders)
        if gap_fills is None:
            logger.warning("Fragment assembly: gap fill parse failed — caching raw output as miss")
            await self._cache_template(goal, raw_output, fp, capabilities)
            return EMEResult(
                status="miss", template=raw_output, template_id=None,
                cache_level="miss",
            )

        stitched = self._stitch_plan(all_segs_with_placeholders, gap_fills, pending_gaps)
        final_template = self.adapter.reconstruct(stitched)
        await self._cache_template(goal, final_template, fp, capabilities)

        # Register newly generated gap segments in the fragment library
        for gap in pending_gaps:
            if gap.position < len(stitched):
                seg = stitched[gap.position]
                if seg.signature:
                    await self._fragment_index.add(self.tenant_id, seg.segment_id, seg.signature)
                    self._fragment_map[seg.segment_id] = seg
                    await self._write_behind.enqueue(seg)
                    self._trace_gen_buf.append(seg.segment_id)

        covered_segs = [s for i, s in enumerate(stitched) if i not in gap_positions]
        return EMEResult(
            status="system2",
            template=final_template,
            template_id=None,
            segments_reused=covered,
            segments_generated=len(gap_positions),
            tokens_saved=self._seg_tokens(covered_segs),
            latency_saved_ms=covered * 2500,
            cache_level="system2",
            validation_passed=True,
        )

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
        template = await generation_fn(goal, inputs, context, capabilities, constraints)

        if template is None:
            return EMEResult(
                status="miss",
                template=None,
                template_id=None,
                segments_reused=0,
                segments_generated=0,
                cache_level="miss",
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

    def _decompose_to_bricks(self, template: Any, capabilities: List[str]) -> List[Dict]:
        """
        Break any template into the smallest independently reusable bricks.

        Lego model: every cached plan must dissolve into individual fragments
        so future plans — even completely different ones — can pull individual
        bricks by semantic similarity and only generate what they're missing.

        Rules:
        1. Structured output (list/dict with steps): use adapter.decompose() as-is.
           Each step is already a brick.
        2. String output + capabilities provided: split into one brick per
           capability, aligned by semantic similarity between output sections
           and capability labels.  Each brick gets a '_capability_intent' so
           the fragment library stores it under a meaningful name.
        3. String output, no capabilities: split on paragraph/section boundaries
           (double newlines, markdown headers, numbered lines). Each paragraph
           is a brick with its own embedding rather than one opaque blob.
        """
        # Rule 1: already structured — adapter handles it
        if not isinstance(template, str):
            return self.adapter.decompose(template)

        raw_segs = self.adapter.decompose(template)

        # Rule 1b: adapter already split it into multiple segments
        if len(raw_segs) > 1:
            return raw_segs

        # Single-segment string — needs splitting
        text = str(template).strip()

        # Split into natural sections: double-newline, markdown headers, numbered lines
        import re as _re
        parts = _re.split(r'\n{2,}|(?=^#{1,3} )|(?=^\d+\. )', text, flags=_re.MULTILINE)
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 15]

        if not parts:
            return raw_segs  # nothing meaningful to split

        # Rule 2: capabilities provided — assign each part to the nearest capability
        if capabilities and len(capabilities) >= 2:
            cap_embeddings = [self._embed_sync(c, full=False) for c in capabilities]
            bricks: List[Dict] = []
            assigned: List[set] = [set() for _ in capabilities]

            for j, part in enumerate(parts):
                part_sig = self._embed_sync(part[:200], full=False)
                best_cap_idx = max(
                    range(len(capabilities)),
                    key=lambda k: SimpleEmbedder.cosine_similarity(part_sig, cap_embeddings[k])
                )
                brick_id = f"brick_{j}_{hashlib.md5(part.encode()).hexdigest()[:8]}"
                bricks.append({
                    "id": brick_id,
                    "content": part,
                    "_capability_intent": capabilities[best_cap_idx],
                })
            return bricks if bricks else raw_segs

        # Rule 3: no capabilities — each paragraph is its own brick
        return [
            {
                "id": f"brick_{j}_{hashlib.md5(p.encode()).hexdigest()[:8]}",
                "content": p,
            }
            for j, p in enumerate(parts)
        ] if len(parts) > 1 else raw_segs

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
                    intent = self._extract_intent(seg_data)
                    # Embed the intent phrase, not raw JSON — same semantic space as the goal.
                    # This is the fix for _segment_diff: goal vs JSON was always a miss.
                    sig = self._embed_sync(intent, full=False)

                    seg = TemplateSegment(
                        segment_id=seg_data.get("id", f"seg_{i}"),
                        tenant_id=self.tenant_id,
                        content=content,
                        fingerprint=hashlib.md5(content_str.encode()).hexdigest()[:16],
                        signature=sig,
                        intent=intent,
                        dependencies=seg_data.get("depends_on", []),
                        outputs=seg_data.get("outputs", []),
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

            # Lego brick injection — separate from template segments.
            # Template segments drive System 2 retrieval (leave them intact).
            # Bricks are extra fine-grained fragments added to the library so
            # Fragment Assembly can pull individual pieces from any past run
            # into a completely different future plan.
            bricks = self._decompose_to_bricks(template, capabilities)
            if len(bricks) > len(segments_data):
                for j, b in enumerate(bricks):
                    b_content = b.get("content", b)
                    b_intent = b.get("_capability_intent") or self._extract_intent(b)
                    b_sig = self._embed_sync(b_intent, full=False)
                    if not b_sig:
                        continue
                    b_id = b.get("id", f"brick_{template_id}_{j}")
                    brick_seg = TemplateSegment(
                        segment_id=b_id,
                        tenant_id=self.tenant_id,
                        content=b_content,
                        fingerprint=hashlib.md5(
                            json.dumps(b_content, default=str).encode()
                        ).hexdigest()[:16],
                        signature=b_sig,
                        intent=b_intent,
                        is_generated=False,
                        confidence=1.0,
                        success_rate=1.0,
                    )
                    await self._write_behind.enqueue(brick_seg)
                    await self._fragment_index.add(self.tenant_id, b_id, b_sig)
                    self._fragment_map[b_id] = brick_seg

            logger.debug(f"Template cached: {template_id} ({len(segments)} segments, {len(bricks)} bricks)")
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
        segs = template.segments
        if len(segs) == 1:
            # Single-segment: return content directly so type is preserved.
            # Multi-segment wraps in a list (plans, DAGs, step sequences).
            content_str = json.dumps(segs[0].content, default=str)
            for key, value in inputs.items():
                content_str = content_str.replace(f"${{{key}}}", str(value))
            try:
                return json.loads(content_str)
            except json.JSONDecodeError:
                return content_str
        plan_str = json.dumps([s.content for s in segs], default=str)
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
        Extract fingerprint key from inputs — values for primitives, schema for
        complex types.

        Primitive values (str, int, float, bool) are included as-is so that
        {"region": "US"} and {"region": "EU"} produce different fingerprints
        and never collide in the System 1 cache.

        Complex types (dict, list) use structural schema because their internal
        values are typically too large to hash efficiently and the plan structure
        is what matters for caching purposes.

        [v2] Handles nested dicts and lists without crashing on unhashable types.
        """
        def _key(v: Any) -> str:
            if isinstance(v, (str, int, float, bool)):
                return f"{type(v).__name__}:{v}"
            if isinstance(v, dict):
                return f"dict[{','.join(sorted(str(k) for k in v.keys()))}]"
            if isinstance(v, (list, tuple)):
                return f"list[{len(v)}]"
            return type(v).__name__

        return {k: _key(v) for k, v in inputs.items()}

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
            goal_sig = self._embed_sync(goal, full=False)
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
                    # Verify the cached response intent actually matches this query.
                    # Query→query similarity alone is not enough — a billing query and
                    # a password-reset query share the same system prompt prefix, so
                    # their embeddings look similar even though the correct responses
                    # are completely different. The segment signature is the short
                    # embedding of the cached response's intent (first 200 chars of
                    # the response text). Checking it against the incoming query
                    # ensures we only serve a cached response when its content is
                    # actually relevant.
                    if template.segments and template.segments[0].signature:
                        response_intent_sim = SimpleEmbedder.cosine_similarity(
                            goal_sig, template.segments[0].signature
                        )
                        if response_intent_sim < INTENT_AMBIGUOUS_LOW:
                            continue
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
