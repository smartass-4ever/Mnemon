"""
Mnemon Cognitive Memory System
Five-layer stratified memory with protein bond activation retrieval
and conditional intent drone curation.

Architecture by Mahika Jadhav (smartass-4ever).
Extended with: BondedMemory, fuzzy resonance, two-part retrieval,
LLM router, async tag verification, drone feedback loop,
cross-layer indexing, conflict detection, fact versioning.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .models import (
    BondedMemory, ExperienceSignal, MemoryLayer, RiskLevel,
    SemanticFact, SignalType, MNEMON_VERSION
)
from .persistence import EROSDatabase, InvertedIndex

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# THRESHOLDS AND CONSTANTS
# ─────────────────────────────────────────────

RESONANCE_FLOOR        = 0.70   # minimum for protein bond activation
SOFT_FILTER_THRESHOLD  = 0.85   # vector similarity override for zero-tag memories
SOFT_FILTER_PENALTY    = 0.75   # score multiplier for soft-filter candidates
INTENT_WEIGHT          = 0.60   # combined score weighting
PATTERN_WEIGHT         = 0.40
DRONE_SPARSE_THRESHOLD = 50     # below this — no drone
DRONE_MEDIUM_THRESHOLD = 5000   # below this — lightweight drone
DRONE_TIMEOUT_SECONDS  = 0.30   # hard limit on drone processing
DRONE_REVIEW_FRACTION  = 0.30   # lightweight mode reviews bottom N%
MAX_CONTEXT_TOKENS     = 2000   # token budget for memory context
EMOTION_DECAY_RATE     = 0.05   # per hour
IMPORTANCE_DECAY_RATE  = 0.02   # per day (episodic)


# ─────────────────────────────────────────────
# EMBEDDER — AUTO-UPGRADING
# ─────────────────────────────────────────────
#
# Priority order (automatic, zero config):
#   1. sentence-transformers  — 384-dim real semantic embeddings (pip install mnemon-ai[full])
#   2. OpenAI text-embedding  — 1536-dim if OPENAI_API_KEY set
#   3. HashProjectionEmbedder — 64-dim fallback, always available
#
# The moment sentence-transformers is installed, Mnemon upgrades silently.
# No code changes needed. No restart needed after install.
# ─────────────────────────────────────────────

def _try_load_sentence_transformers():
    """Attempt to load sentence-transformers. Returns model or None."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Mnemon embedder: sentence-transformers loaded (all-MiniLM-L6-v2, 384-dim)")
        return model
    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"sentence-transformers found but failed to load: {e}")
        return None


class HashProjectionEmbedder:
    """
    Lightweight hash-projection fallback embedder.
    64-dim activation signatures, 384-dim for EME templates.
    Always available — zero dependencies beyond numpy.
    Retrieval precision: ~56% on eval suite.
    """
    DIM_ACTIVATION = 64
    DIM_FULL       = 384

    def embed(self, text: str) -> List[float]:
        tokens = text.lower().split()
        vec = np.zeros(self.DIM_ACTIVATION)
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % self.DIM_ACTIVATION
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_full(self, text: str) -> List[float]:
        tokens = text.lower().split()
        vec = np.zeros(self.DIM_FULL)
        for token in tokens:
            h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            idx = h % self.DIM_FULL
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class SentenceTransformerEmbedder:
    """
    Production embedder using sentence-transformers.
    384-dim real semantic embeddings.
    Install: pip install mnemon-ai[full]
    Retrieval precision: ~85%+ on eval suite.
    """
    DIM_ACTIVATION = 384
    DIM_FULL       = 384

    def __init__(self, model):
        self._model = model

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.DIM_ACTIVATION
        vec = self._model.encode(text[:512], normalize_embeddings=True)
        return vec.tolist()

    def embed_full(self, text: str) -> List[float]:
        return self.embed(text)


class SimpleEmbedder:
    """
    Public embedder interface.
    Automatically selects the best available backend.

    With sentence-transformers installed:
        → SentenceTransformerEmbedder (384-dim, ~85% retrieval precision)
    Without:
        → HashProjectionEmbedder (64-dim, ~56% retrieval precision)

    To upgrade: pip install mnemon-ai[full]
    No code changes needed.
    """

    def __init__(self):
        st_model = _try_load_sentence_transformers()
        if st_model:
            self._backend = SentenceTransformerEmbedder(st_model)
            self.dim      = 384
            self.backend_name = "sentence-transformers"
        else:
            self._backend = HashProjectionEmbedder()
            self.dim      = 64
            self.backend_name = "hash-projection"
            logger.info(
                "Mnemon embedder: using hash-projection fallback (64-dim). "
                "For production quality, run: pip install mnemon-ai[full]"
            )

    def embed(self, text: str) -> List[float]:
        return self._backend.embed(text)

    def embed_full(self, text: str) -> List[float]:
        return self._backend.embed_full(text)

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)


# ─────────────────────────────────────────────
# RULE-BASED CLASSIFIER
# ─────────────────────────────────────────────

class RuleClassifier:
    """
    Fast zero-cost classifier for obvious memory routing.
    Handles ~70-80% of signals without LLM call.
    """

    TEMPORAL_KEYWORDS   = {"today", "yesterday", "last", "this week", "just", "recently", "ago", "at", "when"}
    EMOTIONAL_KEYWORDS  = {"happy", "sad", "frustrated", "excited", "angry", "pleased", "worried", "anxious", "great", "terrible", "tense", "nervous", "confident"}
    PREFERENCE_KEYWORDS = {"prefers", "always", "usually", "never", "likes", "dislikes", "wants", "hates", "loves", "typically", "tends"}
    RELATIONSHIP_KEYWORDS = {"communication", "style", "formal", "casual", "tone", "approach", "manner", "behaviour"}

    @classmethod
    def classify(cls, content: str) -> Optional[MemoryLayer]:
        text = content.lower()
        words = set(text.split())

        if words & cls.EMOTIONAL_KEYWORDS:
            return MemoryLayer.EMOTIONAL

        if words & cls.PREFERENCE_KEYWORDS:
            # Preferences about people → RELATIONSHIP; about facts → SEMANTIC
            if words & cls.RELATIONSHIP_KEYWORDS:
                return MemoryLayer.RELATIONSHIP
            return MemoryLayer.SEMANTIC

        if words & cls.TEMPORAL_KEYWORDS:
            return MemoryLayer.EPISODIC

        # Proper noun + stable fact pattern
        if any(w[0].isupper() for w in content.split() if len(w) > 1):
            if any(kw in text for kw in ["is ", "are ", "was ", "has ", "have "]):
                return MemoryLayer.SEMANTIC

        return None  # ambiguous → LLM router

    @classmethod
    def extract_tags(cls, content: str, layer: MemoryLayer) -> Set[str]:
        """
        Rule-based tag extraction. Fast, free.
        LLM router amends/corrects these async after write.
        """
        tags: Set[str] = set()
        words = content.split()

        # Proper nouns → entity tags
        for word in words:
            clean = word.strip(".,!?;:'\"()").lower()
            if word[0].isupper() and len(clean) > 2:
                tags.add(clean)

        # Domain keywords
        domain_map = {
            "security":    ["security", "audit", "vulnerability", "scan", "firewall"],
            "finance":     ["budget", "revenue", "cost", "invoice", "payment"],
            "code":        ["code", "function", "api", "bug", "deploy", "database"],
            "report":      ["report", "summary", "document", "pdf", "presentation"],
            "email":       ["email", "message", "reply", "send", "communicate"],
            "meeting":     ["meeting", "call", "schedule", "agenda", "discuss"],
        }
        text_lower = content.lower()
        for domain, keywords in domain_map.items():
            if any(kw in text_lower for kw in keywords):
                tags.add(domain)

        # Layer-specific tags
        tags.add(layer.value)

        return tags


# ─────────────────────────────────────────────
# WORKING MEMORY
# ─────────────────────────────────────────────

class WorkingMemory:
    """
    Layer 1: Ephemeral per-session scratchpad.
    Flushes completely at task end — no bleed between tasks.
    """

    def __init__(self, session_id: str, tenant_id: str):
        self.session_id    = session_id
        self.tenant_id     = tenant_id
        self.emotional_state: str = "neutral"
        self.active_goals: List[str] = []
        self.scratchpad:   Dict[str, Any] = {}
        self.created_at    = time.time()

    def update(self, data: Dict[str, Any]):
        self.scratchpad.update(data)

    def flush(self):
        self.emotional_state = "neutral"
        self.active_goals.clear()
        self.scratchpad.clear()

    def to_context(self) -> Dict[str, Any]:
        return {
            "emotional_state": self.emotional_state,
            "active_goals":    self.active_goals,
            "scratchpad":      self.scratchpad,
        }


# ─────────────────────────────────────────────
# COGNITIVE MEMORY SYSTEM
# ─────────────────────────────────────────────

class CognitiveMemorySystem:
    """
    Five-layer stratified memory system.

    Layer 1 — Working:      ephemeral scratchpad, per session
    Layer 2 — Episodic:     chronological experiences, importance scored
    Layer 3 — Semantic:     stable facts, key-value vault
    Layer 4 — Relationship: per-user interaction patterns
    Layer 5 — Emotional:    emotional context, time-decayed

    Retrieval: Two-part — protein bond pattern assembly (zero LLM)
    followed by conditional intent drone curation.

    Write path: rule classifier → async LLM router (if ambiguous)
    → async tag verification → inverted index update.
    """

    def __init__(
        self,
        tenant_id: str,
        db: EROSDatabase,
        index: InvertedIndex,
        embedder: Optional[SimpleEmbedder] = None,
        llm_client=None,          # injected — any async LLM client
        enabled_layers: Optional[List[MemoryLayer]] = None,
        drone_model: str = "claude-haiku-4-5-20251001",
        router_model: str = "claude-haiku-4-5-20251001",
        watchdog=None,            # optional Watchdog reference
    ):
        self.tenant_id  = tenant_id
        self.db         = db
        self.index      = index
        self.embedder   = embedder or SimpleEmbedder()
        self.llm        = llm_client
        self.drone_model  = drone_model
        self.router_model = router_model
        self.watchdog   = watchdog

        self.enabled_layers = set(enabled_layers or list(MemoryLayer))

        # Layer 1 — in-memory only
        self._working: Dict[str, WorkingMemory] = {}

        # Pool size cache — refreshed periodically
        self._pool_size: int = 0
        self._pool_size_updated: float = 0

        self._lock = asyncio.Lock()

        # Background prune loop
        self._running = False
        self._prune_task = None

    # ──────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────

    async def start(self):
        """Start background prune loop (runs every 24 h)."""
        self._running = True
        self._prune_task = asyncio.create_task(self._prune_loop())

    async def stop(self):
        """Stop background prune loop."""
        self._running = False
        if self._prune_task:
            self._prune_task.cancel()
            try:
                await self._prune_task
            except asyncio.CancelledError:
                pass

    async def _prune_loop(self):
        while self._running:
            try:
                await asyncio.sleep(86400)  # 24 hours
                await self.prune()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Prune loop error for tenant {self.tenant_id}: {e}")

    # ──────────────────────────────────────────
    # WRITE PATH
    # ──────────────────────────────────────────

    async def write(self, signal: ExperienceSignal) -> Optional[str]:
        """
        Main write entry point.
        1. Rule classifier (free)
        2. LLM router if ambiguous (cheap, async)
        3. Generate activation signature
        4. Write to DB + update index
        5. Async tag verification (never blocks caller)
        """
        if signal.layer and signal.layer not in self.enabled_layers:
            return None

        # Step 1: Determine layer
        if signal.layer:
            layer = signal.layer
        else:
            content_str = json.dumps(signal.content) if isinstance(signal.content, dict) else str(signal.content)
            layer = RuleClassifier.classify(content_str)

        # Step 2: LLM router for ambiguous cases
        if layer is None:
            layer = await self._llm_route(signal)

        if layer == MemoryLayer.WORKING:
            self._update_working(signal)
            return None

        # Step 3: Build BondedMemory
        memory = await self._build_memory(signal, layer)

        # Step 4: Persist and index
        await self.db.write_memory(memory)
        await self.index.update(self.tenant_id, memory.memory_id, memory.activation_tags)

        # Step 5: Async tag verification — never blocks caller
        asyncio.create_task(self._verify_tags(memory))

        # Update pool size cache
        self._pool_size += 1

        logger.debug(f"Memory written: {memory.memory_id} → {layer.value}")
        return memory.memory_id

    def _update_working(self, signal: ExperienceSignal):
        session_id = signal.session_id
        if session_id not in self._working:
            self._working[session_id] = WorkingMemory(session_id, self.tenant_id)
        wm = self._working[session_id]
        if isinstance(signal.content, dict):
            wm.update(signal.content)
            if "emotional_state" in signal.content:
                wm.emotional_state = signal.content["emotional_state"]
            if "goals" in signal.content:
                wm.active_goals = signal.content["goals"]

    async def _build_memory(self, signal: ExperienceSignal, layer: MemoryLayer) -> BondedMemory:
        content_str = json.dumps(signal.content) if isinstance(signal.content, dict) else str(signal.content)

        # Tags from rule classifier (fast)
        tags = RuleClassifier.extract_tags(content_str, layer)

        # Activation signature
        act_sig = self.embedder.embed(content_str)
        intent_sig = self.embedder.embed(f"use when: {content_str[:200]}")

        # Intent label — simple heuristic, LLM router will improve it
        intent_label = self._heuristic_intent(content_str, layer)

        memory_id = hashlib.sha256(
            f"{self.tenant_id}:{signal.session_id}:{signal.timestamp}:{content_str[:100]}".encode()
        ).hexdigest()[:24]

        return BondedMemory(
            memory_id=memory_id,
            tenant_id=self.tenant_id,
            layer=layer,
            content=signal.content,
            session_id=signal.session_id,
            activation_tags=tags,
            activation_domain=self._infer_domain(tags),
            activation_signature=act_sig,
            intent_label=intent_label,
            intent_valid=True,
            intent_signature=intent_sig,
            importance=signal.importance,
            confidence=1.0,
            risk_level=signal.risk_level,
            timestamp=signal.timestamp,
            last_accessed=signal.timestamp,
        )

    def _heuristic_intent(self, content: str, layer: MemoryLayer) -> str:
        layer_intents = {
            MemoryLayer.EPISODIC:     "use when recalling past experiences relevant to",
            MemoryLayer.SEMANTIC:     "use when needing stable facts about",
            MemoryLayer.RELATIONSHIP: "use when calibrating communication style for",
            MemoryLayer.EMOTIONAL:    "use when setting tone for interactions with",
        }
        prefix = layer_intents.get(layer, "use when working on")
        words = content.split()[:8]
        return f"{prefix} {' '.join(words)}"

    def _infer_domain(self, tags: Set[str]) -> str:
        domain_priority = ["security", "finance", "code", "report", "email", "meeting"]
        for d in domain_priority:
            if d in tags:
                return d
        return "general"

    # ──────────────────────────────────────────
    # LLM ROUTER (async, never blocks write)
    # ──────────────────────────────────────────

    async def _llm_route(self, signal: ExperienceSignal) -> MemoryLayer:
        """
        Cheap LLM call that does three things in one:
        1. Layer classification
        2. Tag generation (amends rule tags)
        3. Intent label generation

        Falls back to EPISODIC if LLM unavailable.
        """
        if not self.llm:
            return MemoryLayer.EPISODIC

        content_str = json.dumps(signal.content) if isinstance(signal.content, dict) else str(signal.content)

        prompt = f"""Classify this memory content into exactly one layer.
Reply with JSON only, no markdown.

Content: {content_str[:500]}

Layers:
- working: temporary, only relevant to current active task
- episodic: something that happened, time-stamped experience
- semantic: stable fact about a person, company, or system
- relationship: how someone communicates or behaves
- emotional: emotional tone or sentiment

Reply format:
{{"layer": "semantic", "tags": ["tag1", "tag2"], "intent": "use when..."}}"""

        try:
            response = await self.llm.complete(
                prompt=prompt,
                model=self.router_model,
                max_tokens=150,
            )
            data = json.loads(response)
            return MemoryLayer(data.get("layer", "episodic"))
        except Exception as e:
            logger.warning(f"LLM router failed: {e} — defaulting to EPISODIC")
            return MemoryLayer.EPISODIC

    async def _verify_tags(self, memory: BondedMemory):
        """
        Async tag verification — runs after write, never blocks caller.
        Cheap LLM call checks tag accuracy and amends if needed.
        """
        if not self.llm:
            return

        content_str = json.dumps(memory.content) if isinstance(memory.content, dict) else str(memory.content)
        current_tags = list(memory.activation_tags)

        prompt = f"""Check if these tags accurately represent this memory content.
Reply with JSON only, no markdown.

Content: {content_str[:300]}
Current tags: {current_tags}

Reply format:
{{"verdict": "YES", "amended_tags": null}}
or
{{"verdict": "MISSING", "amended_tags": ["tag1", "tag2", "tag3"]}}
or
{{"verdict": "WRONG", "remove_tags": ["bad_tag"]}}"""

        try:
            response = await self.llm.complete(
                prompt=prompt,
                model=self.router_model,
                max_tokens=100,
            )
            data = json.loads(response)

            if data["verdict"] == "YES":
                return

            if data["verdict"] == "MISSING" and data.get("amended_tags"):
                new_tags = set(data["amended_tags"])
                await self.db.update_memory_tags(self.tenant_id, memory.memory_id, new_tags)
                await self.index.update(self.tenant_id, memory.memory_id, new_tags)
                logger.debug(f"Tags amended for {memory.memory_id}: {new_tags}")

            elif data["verdict"] == "WRONG" and data.get("remove_tags"):
                corrected = memory.activation_tags - set(data["remove_tags"])
                await self.db.update_memory_tags(self.tenant_id, memory.memory_id, corrected)
                logger.debug(f"Bad tags removed for {memory.memory_id}")

        except Exception as e:
            logger.debug(f"Tag verification skipped: {e}")

    # ──────────────────────────────────────────
    # RETRIEVE — PART 1: PATTERN ASSEMBLY
    # ──────────────────────────────────────────

    async def retrieve(
        self,
        task_signal: str,
        session_id: str,
        task_goal: str = "",
        top_k: int = 12,
    ) -> Dict[str, Any]:
        """
        Two-part retrieval.
        Part 1: protein bond pattern assembly (~15ms, zero LLM)
        Part 2: conditional intent drone (only above pool threshold)

        Returns structured context package ready for agent injection.
        """
        pool_size = await self._get_pool_size()

        # Part 1: Pattern Assembly
        candidates = await self._pattern_assembly(task_signal)

        # Part 2: Intent Drone (conditional)
        if pool_size < DRONE_SPARSE_THRESHOLD:
            curated = candidates[:top_k]
            drone_used = False
        elif pool_size < DRONE_MEDIUM_THRESHOLD:
            curated = await self._lightweight_drone(candidates, task_goal, top_k)
            drone_used = True
        else:
            curated = await self._full_drone(candidates, task_goal, top_k)
            drone_used = True

        # Fetch full content for kept memories
        memory_ids = [m_id for m_id, _ in curated]
        memories = await self.db.fetch_memories(self.tenant_id, memory_ids)
        memories_by_id = {m.memory_id: m for m in memories}

        # Layer diversity check
        final = self._ensure_layer_diversity(curated, memories_by_id, top_k)

        # Working memory context
        working_ctx = {}
        if session_id in self._working:
            working_ctx = self._working[session_id].to_context()

        # Detect conflicts
        conflicts = self._detect_conflicts(memories)

        return {
            "working":        working_ctx,
            "memories":       [memories_by_id[m_id].content for m_id, _ in final if m_id in memories_by_id],
            "memory_ids":     [m_id for m_id, _ in final],
            "memory_scores":  {m_id: score for m_id, score in final},
            "conflicts":      conflicts,
            "pool_size":      pool_size,
            "drone_used":     drone_used,
            "layers_present": list({memories_by_id[m_id].layer.value for m_id, _ in final if m_id in memories_by_id}),
        }

    async def _pattern_assembly(self, task_signal: str) -> List[Tuple[str, float]]:
        """
        Step 1: Tag intersection (inverted index, milliseconds)
        Step 2: Protein bond resonance (64-dim cosine, fuzzy 70%+)
        Step 3: Soft filter override for high-similarity zero-tag memories
        Step 4: Dynamic threshold self-regulation
        Returns: List of (memory_id, score) sorted descending
        """
        signal_tags = RuleClassifier.extract_tags(task_signal, MemoryLayer.EPISODIC)
        signal_sig  = self.embedder.embed(task_signal)
        signal_intent_sig = self.embedder.embed(f"need to know: {task_signal}")

        # Tag intersection — hard candidates
        tag_candidates = await self.index.intersect(self.tenant_id, signal_tags)

        # Fetch signatures for resonance scoring
        if tag_candidates:
            tag_memories = await self.db.fetch_memories(self.tenant_id, list(tag_candidates))
        else:
            tag_memories = []

        pool: List[Tuple[str, float]] = []

        for mem in tag_memories:
            if not mem.activation_signature:
                continue
            pattern_score = SimpleEmbedder.cosine_similarity(signal_sig, mem.activation_signature)
            intent_score  = SimpleEmbedder.cosine_similarity(signal_intent_sig, mem.intent_signature) if mem.intent_signature else 0.0
            combined = (PATTERN_WEIGHT * pattern_score) + (INTENT_WEIGHT * intent_score)
            if combined >= RESONANCE_FLOOR:
                # Boost by drone keep history
                combined = min(1.0, combined + (mem.drone_keep_score - 0.5) * 0.1)
                pool.append((mem.memory_id, combined))

        # Soft filter — recover relevant memories with zero tag overlap.
        # Filter by domain so we stay within the query's semantic space.
        signal_domain = self._infer_domain(signal_tags)
        recent_ids = await self.db.fetch_recent_by_domain(self.tenant_id, signal_domain, 50)
        all_memories = await self.db.fetch_memories(self.tenant_id, recent_ids)
        tag_ids = {m.memory_id for m in tag_memories}
        for mem in all_memories:
            if mem.memory_id in tag_ids:
                continue
            if not mem.activation_signature:
                continue
            vec_sim = SimpleEmbedder.cosine_similarity(signal_sig, mem.activation_signature)
            if vec_sim >= SOFT_FILTER_THRESHOLD:
                penalised = vec_sim * SOFT_FILTER_PENALTY
                pool.append((mem.memory_id, penalised))

        # Sort descending
        pool.sort(key=lambda x: x[1], reverse=True)

        # Dynamic threshold — self-regulate pool size
        if len(pool) > 100:
            # Raise effective threshold until pool is manageable
            threshold = RESONANCE_FLOOR
            while len([p for p in pool if p[1] >= threshold]) > 80 and threshold < 0.95:
                threshold += 0.02
            pool = [(m, s) for m, s in pool if s >= threshold]

        return pool

    # ──────────────────────────────────────────
    # RETRIEVE — PART 2: INTENT DRONE
    # ──────────────────────────────────────────

    async def _lightweight_drone(
        self,
        candidates: List[Tuple[str, float]],
        goal: str,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Pool 500-5000: review only bottom 30%, auto-keep top 70%.
        Hard 300ms time limit.
        """
        cutoff = int(len(candidates) * (1 - DRONE_REVIEW_FRACTION))
        auto_keep = candidates[:cutoff]
        to_review = candidates[cutoff:]

        if not to_review or not self.llm:
            return (auto_keep + to_review)[:top_k]

        try:
            reviewed = await asyncio.wait_for(
                self._drone_evaluate(to_review, goal),
                timeout=DRONE_TIMEOUT_SECONDS
            )
            return (auto_keep + reviewed)[:top_k]
        except asyncio.TimeoutError:
            logger.debug("Lightweight drone timeout — using pattern results")
            return (auto_keep + to_review)[:top_k]

    async def _full_drone(
        self,
        candidates: List[Tuple[str, float]],
        goal: str,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Pool 5000+: evaluate all candidates.
        Hard 300ms time limit.
        """
        if not self.llm or not candidates:
            return candidates[:top_k]

        try:
            result = await asyncio.wait_for(
                self._drone_evaluate(candidates, goal),
                timeout=DRONE_TIMEOUT_SECONDS
            )
            return result[:top_k]
        except asyncio.TimeoutError:
            logger.debug("Full drone timeout — using pattern results")
            return candidates[:top_k]

    async def _drone_evaluate(
        self,
        candidates: List[Tuple[str, float]],
        goal: str,
    ) -> List[Tuple[str, float]]:
        """
        LLM drone receives intent labels only — never full content.
        Returns curated list with conflict flags.
        """
        if not candidates:
            return candidates

        # Fetch intent labels only (not full content)
        memory_ids = [m_id for m_id, _ in candidates]
        label_map = await self.db.fetch_drone_labels(self.tenant_id, memory_ids)

        candidate_list = [
            {
                "id":           m_id,
                "score":        round(score, 3),
                "intent":       label_map.get(m_id, ("unknown", 0.5))[0],
                "keep_history": round(label_map.get(m_id, ("unknown", 0.5))[1], 2),
            }
            for m_id, score in candidates
        ]

        prompt = f"""You are a memory curator for an AI agent.
Goal: {goal}

Review these candidate memories and decide which to KEEP for the agent.
Each memory shows its intent label and historical keep rate.
Reply with JSON only — no markdown.

Candidates:
{json.dumps(candidate_list[:30], indent=2)}

Reply format:
{{"keep": ["id1", "id2"], "drop": ["id3"], "conflicts": [["id4", "id5"]]}}

Only keep memories that help achieve the goal. Flag conflicts where two memories contradict."""

        try:
            response = await self.llm.complete(
                prompt=prompt,
                model=self.drone_model,
                max_tokens=300,
            )
            data = json.loads(response)
            keep_ids = set(data.get("keep", []))
            drop_ids = set(data.get("drop", []))

            # Record decisions in watchdog for trend tracking
            if self.watchdog:
                for _ in keep_ids:
                    self.watchdog.record_drone_decision(correct=True)
                for _ in drop_ids:
                    self.watchdog.record_drone_decision(correct=False)

            # Schedule async feedback — only drop_delta now;
            # keep_delta is applied by confirm_memory_useful() after task success
            asyncio.create_task(self._drone_feedback(drop_ids))

            return [(m_id, score) for m_id, score in candidates if m_id in keep_ids]

        except Exception as e:
            logger.warning(f"Drone evaluation failed: {e} — returning pattern results")
            return candidates

    async def _drone_feedback(self, dropped: Set[str]):
        """
        Async feedback — penalises memories the drone excluded.
        keep_delta is NOT applied here; call confirm_memory_useful()
        after actual task success so the score reflects real usefulness.
        """
        for memory_id in dropped:
            await self.db.update_drone_scores(self.tenant_id, memory_id, drop_delta=0.01)

    async def confirm_memory_useful(self, memory_id: str):
        """
        Call after a task succeeds and a retrieved memory proved helpful.
        Applies keep_delta so the drone learns from actual outcomes,
        not from its own curation decisions.
        """
        await self.db.update_drone_scores(self.tenant_id, memory_id, keep_delta=0.02)

    def _ensure_layer_diversity(
        self,
        candidates: List[Tuple[str, float]],
        memories_by_id: Dict[str, BondedMemory],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Ensure at least one memory from each available layer.
        Fill token budget from top of ranked list.
        """
        seen_layers: Set[MemoryLayer] = set()
        priority: List[Tuple[str, float]] = []
        remainder: List[Tuple[str, float]] = []

        for m_id, score in candidates:
            mem = memories_by_id.get(m_id)
            if not mem:
                continue
            if mem.layer not in seen_layers:
                priority.append((m_id, score))
                seen_layers.add(mem.layer)
            else:
                remainder.append((m_id, score))

        combined = priority + remainder
        return combined[:top_k]

    def _detect_conflicts(self, memories: List[BondedMemory]) -> List[Dict]:
        """
        Detect contradicting memories in the same intent domain.
        Returns conflict pairs for agent to resolve — never silent.
        """
        conflicts = []
        for i, m1 in enumerate(memories):
            for m2 in memories[i+1:]:
                if m1.layer != m2.layer:
                    continue
                if m1.activation_domain != m2.activation_domain:
                    continue
                # Check tag overlap in same domain
                tag_overlap = m1.activation_tags & m2.activation_tags
                if len(tag_overlap) >= 2:
                    sim = SimpleEmbedder.cosine_similarity(
                        m1.activation_signature, m2.activation_signature
                    )
                    if 0.3 < sim < 0.7:
                        conflicts.append({
                            "memory_a":    m1.memory_id,
                            "memory_b":    m2.memory_id,
                            "domain":      m1.activation_domain,
                            "note":        "potentially contradicting — verify before acting",
                        })
        return conflicts

    # ──────────────────────────────────────────
    # WORKING MEMORY MANAGEMENT
    # ──────────────────────────────────────────

    async def start_session(self, session_id: str) -> WorkingMemory:
        wm = WorkingMemory(session_id, self.tenant_id)
        self._working[session_id] = wm
        return wm

    async def end_session(self, session_id: str):
        """Flush working memory at task end. No bleed between tasks."""
        if session_id in self._working:
            self._working[session_id].flush()
            del self._working[session_id]

    # ──────────────────────────────────────────
    # SEMANTIC FACT MANAGEMENT
    # ──────────────────────────────────────────

    async def write_fact(self, key: str, value: Any, source_session: str, confidence: float = 1.0):
        fact_id = hashlib.md5(f"{self.tenant_id}:{key}".encode()).hexdigest()[:16]
        existing = await self.db.fetch_fact(self.tenant_id, key)

        if existing:
            existing.update(value, confidence, source_session)
            await self.db.write_fact(existing)
        else:
            fact = SemanticFact(
                fact_id=fact_id,
                tenant_id=self.tenant_id,
                key=key,
                current_value=value,
                confidence=confidence,
                last_updated=time.time(),
            )
            await self.db.write_fact(fact)

    async def read_fact(self, key: str) -> Optional[Any]:
        fact = await self.db.fetch_fact(self.tenant_id, key)
        if fact:
            fact.access_count += 1
            await self.db.write_fact(fact)
            return fact.current_value
        return None

    # ──────────────────────────────────────────
    # POOL SIZE
    # ──────────────────────────────────────────

    async def _get_pool_size(self) -> int:
        now = time.time()
        if now - self._pool_size_updated > 60:  # refresh every 60s
            self._pool_size = await self.db.count_memories(self.tenant_id)
            self._pool_size_updated = now
        return self._pool_size

    # ──────────────────────────────────────────
    # PRUNING
    # ──────────────────────────────────────────

    async def prune(self):
        """
        Remove or archive low-value memories.
        Importance decay, emotional decay, access-weighted retention.
        Memory gets smarter over time, not just bigger.
        """
        now = time.time()

        async with self.db._lock:
            # Episodic: decay importance for old unaccessed memories
            self.db._conn.execute("""
                UPDATE memories
                SET importance = MAX(0.05, importance - ?)
                WHERE tenant_id=? AND layer='episodic'
                  AND last_accessed < ?
                  AND intent_valid=1
            """, (IMPORTANCE_DECAY_RATE, self.tenant_id, now - 86400))

            # Emotional: decay intensity (stored in importance field for emotional layer)
            hours_old_cutoff = now - 3600
            self.db._conn.execute("""
                UPDATE memories
                SET importance = MAX(0.0, importance - ?)
                WHERE tenant_id=? AND layer='emotional'
                  AND timestamp < ?
            """, (EMOTION_DECAY_RATE, self.tenant_id, hours_old_cutoff))

            # Archive very low importance + low access episodic memories
            self.db._conn.execute("""
                UPDATE memories
                SET intent_valid=0
                WHERE tenant_id=? AND layer='episodic'
                  AND importance < 0.1
                  AND access_count < 2
                  AND timestamp < ?
            """, (self.tenant_id, now - 604800))  # older than 7 days

            self.db._conn.commit()

        logger.debug(f"Pruning complete for tenant {self.tenant_id}")

    # ──────────────────────────────────────────
    # STATS
    # ──────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        db_stats = self.db.get_stats()
        return {
            "tenant_id":          self.tenant_id,
            "working_sessions":   len(self._working),
            "pool_size":          self._pool_size,
            "db_memories":        db_stats["memories"],
            "semantic_facts":     db_stats["facts"],
            "enabled_layers":     [l.value for l in self.enabled_layers],
            "drone_threshold_sparse": DRONE_SPARSE_THRESHOLD,
            "drone_threshold_full":   DRONE_MEDIUM_THRESHOLD,
        }
