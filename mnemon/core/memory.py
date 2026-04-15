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
import struct
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .models import (
    BondedMemory, ExperienceSignal, MemoryLayer, RiskLevel,
    SemanticFact, SignalType, MNEMON_VERSION
)
from .persistence import EROSDatabase, InvertedIndex
from .signal_db import SignalDatabase

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# THRESHOLDS AND CONSTANTS
# ─────────────────────────────────────────────

RESONANCE_FLOOR        = 0.25   # minimum for protein bond activation
INTENT_WEIGHT          = 0.60   # combined score weighting
PATTERN_WEIGHT         = 0.40
DRONE_SPARSE_THRESHOLD = 50     # below this — no drone
DRONE_MEDIUM_THRESHOLD = 5000   # below this — lightweight drone
DRONE_TIMEOUT_SECONDS  = 0.30   # hard limit on drone processing
DRONE_REVIEW_FRACTION  = 0.30   # lightweight mode reviews bottom N%
MAX_CONTEXT_TOKENS     = 2000   # token budget for memory context
EMOTION_DECAY_RATE     = 0.05   # per hour
IMPORTANCE_DECAY_RATE  = 0.02   # per day (episodic)
DRONE_CANDIDATE_CAP    = 80     # max candidates sent to drone LLM (was hard-coded 30)
UNION_FALLBACK_CAP     = 10_000 # max IDs from union fallback to prevent O(n) scan
SEMANTIC_FALLBACK_THRESHOLD = 0.20  # if tag candidates < 20% of pool, run full semantic scan
SEMANTIC_FALLBACK_CAP  = 5_000  # max memories to scan in semantic fallback
RECENCY_WEIGHT         = 0.05   # max recency bonus (decays exponentially by age)
RECENCY_HALFLIFE_DAYS  = 365    # recency bonus halves every year
DIVERSITY_SCORE_GAP    = 0.10   # only enforce layer diversity when gap < this


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

_ST_MODEL_CACHE = None  # module-level cache so model loads only once per process

def _try_load_sentence_transformers():
    """Attempt to load sentence-transformers. Returns cached model or None."""
    global _ST_MODEL_CACHE
    if _ST_MODEL_CACHE is not None:
        return _ST_MODEL_CACHE
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        _ST_MODEL_CACHE = model
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

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        truncated = [t[:512] if t and t.strip() else "" for t in texts]
        vecs = self._model.encode(truncated, normalize_embeddings=True, batch_size=256)
        return vecs.tolist()

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
        if va.shape != vb.shape:
            return 0.0
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)


# Module-level cache: vocab_list hash → (vocab_list, vocab_matrix)
# Avoids re-embedding the same META_VOCABULARY for every CognitiveMemorySystem.
_VOCAB_MATRIX_CACHE: dict = {}


# ─────────────────────────────────────────────
# SEMANTIC VOCABULARY TAGGER
# ─────────────────────────────────────────────

class SemanticVocabularyTagger:
    """
    Embedding-based tag assignment against a pre-built meta-vocabulary.

    No LLM on the write path for high-confidence content.
    Low-confidence content queued for async LLM tagging.

    Threshold: 0.72 cosine similarity.
    Above threshold → tags assigned immediately, no LLM.
    Below threshold → no tags assigned, memory added to async queue.
    """

    CONFIDENCE_THRESHOLD = 0.72
    TOP_K_TAGS = 5  # max tags to assign per memory

    def __init__(self, embedder: "SimpleEmbedder"):
        self.embedder = embedder
        self._vocab: List[str] = []
        self._vocab_matrix = None          # numpy array (N, dim)
        self._ready = False
        self._lock = asyncio.Lock()
        self._async_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task = None
        self._stopped = False              # flag for Python 3.11 wait_for cancellation bug

    async def start(self, vocabulary: List[str]):
        """
        Embed all vocabulary concepts into matrix at startup.
        Called once. After this, tag assignment is pure numpy.
        Caches the matrix so repeated CMS instances skip re-embedding.
        """
        self._vocab = vocabulary
        # Key on the underlying model object (cached globally) + vocab content
        backend = getattr(self.embedder, '_backend', self.embedder)
        model_obj = getattr(backend, '_model', backend)
        vocab_key = (id(model_obj), tuple(vocabulary))
        if vocab_key in _VOCAB_MATRIX_CACHE:
            self._vocab_matrix = _VOCAB_MATRIX_CACHE[vocab_key]
        else:
            if hasattr(self.embedder, 'embed_batch'):
                embeddings = self.embedder.embed_batch(vocabulary)
            else:
                embeddings = [self.embedder.embed(concept) for concept in vocabulary]
            self._vocab_matrix = np.array(embeddings)  # shape: (N, dim)
            _VOCAB_MATRIX_CACHE[vocab_key] = self._vocab_matrix
        # L2-normalise rows for fast cosine via dot product
        norms = np.linalg.norm(self._vocab_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self._vocab_matrix = self._vocab_matrix / norms
        self._ready = True
        self._worker_task = asyncio.create_task(self._llm_tag_worker())

    async def stop(self):
        self._stopped = True
        if self._worker_task:
            self._worker_task.cancel()
            try:
                # Use wait_for so Python 3.11.x wait_for/cancellation bugs can't hang us
                await asyncio.wait_for(self._worker_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    def assign_tags(self, content: str) -> Tuple[Set[str], bool]:
        """
        Assign tags synchronously using numpy matrix multiply.
        Returns (tags, high_confidence).

        high_confidence=False means caller should queue for LLM tagging.
        Never assigns low-confidence tags — returns empty set instead.
        """
        if not self._ready or self._vocab_matrix is None:
            # Not initialised — caller must fall back to RuleClassifier
            return set(), False

        content_emb = np.array(self.embedder.embed(content[:512]))
        norm = np.linalg.norm(content_emb)
        if norm > 0:
            content_emb = content_emb / norm

        # One matrix multiply gives all similarities simultaneously
        similarities = self._vocab_matrix @ content_emb  # shape: (N,)

        top_score = float(np.max(similarities))

        if top_score < self.CONFIDENCE_THRESHOLD:
            return set(), False

        # Get top-k concepts above threshold
        above_threshold = np.where(similarities >= self.CONFIDENCE_THRESHOLD)[0]
        top_indices = above_threshold[
            np.argsort(similarities[above_threshold])[::-1]
        ][:self.TOP_K_TAGS]

        tags: Set[str] = set()
        for idx in top_indices:
            concept = self._vocab[idx]
            # "security audit" → {"security", "audit", "security_audit"}
            words = concept.lower().split()
            tags.update(words)
            if len(words) > 1:
                tags.add("_".join(words))

        return tags, True

    async def queue_for_llm_tagging(
        self, memory_id: str, tenant_id: str, content: str
    ):
        """Queue low-confidence memory for async LLM tag assignment."""
        await self._async_queue.put({
            "memory_id": memory_id,
            "tenant_id": tenant_id,
            "content":   content[:500],
            "queued_at": time.time(),
        })

    async def _llm_tag_worker(self):
        """
        Background worker. Processes low-confidence memories.
        One Haiku call per item. Never blocks the write path.
        Runs until stop() is called.
        """
        while not self._stopped:
            try:
                item = await asyncio.wait_for(
                    self._async_queue.get(), timeout=1.0
                )
                await self._tag_with_llm(item)
            except asyncio.TimeoutError:
                continue  # re-check _stopped flag
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"LLM tag worker error: {e}")

    async def _tag_with_llm(self, item: dict):
        """Single LLM call to tag one low-confidence memory."""
        if not hasattr(self, '_llm') or not self._llm:
            return
        if not hasattr(self, '_db') or not self._db:
            return

        prompt = f"""Assign 3-5 tags to this content.
Reply with JSON only: {{"tags": ["tag1", "tag2", "tag3"]}}
Tags must be single words, lowercase, specific to the content domain.
Content: {item['content']}"""

        try:
            response = await self._llm.complete(
                prompt=prompt,
                model="claude-haiku-4-5-20251001",
                max_tokens=60,
            )
            data = json.loads(response)
            tags = set(data.get("tags", []))
            if tags:
                await self._db.update_memory_tags(
                    item["tenant_id"], item["memory_id"], tags
                )
                if hasattr(self, '_index') and self._index:
                    await self._index.update(
                        item["tenant_id"], item["memory_id"], tags
                    )
                logger.debug(f"LLM tagged {item['memory_id']}: {tags}")
        except Exception as e:
            logger.warning(f"LLM tagging failed for {item['memory_id']}: {e}")


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

        # Proper nouns → entity tags (≥2 chars so Q1, AI, UK are included)
        for word in words:
            clean = word.strip(".,!?;:'\"()").lower()
            if word[0].isupper() and len(clean) >= 2:
                tags.add(clean)

        # Domain keywords
        domain_map = {
            "security":    ["security", "audit", "vulnerability", "scan", "firewall",
                            "ssl", "tls", "cert", "encrypt", "decrypt", "vault",
                            "mfa", "cve", "oauth", "penetrat", "incident", "compliance",
                            "authentication", "authorisation", "authorization", "breach",
                            "soc", "owasp", "gdpr", "zero-trust", "intrusion"],
            "finance":     ["budget", "revenue", "cost", "invoice", "payment",
                            "ebitda", "arr", "fiscal", "payroll", "expense", "profit",
                            "margin", "quarterly", "forecast", "capex", "opex",
                            "netsuite", "concur", "churn", "licensing"],
            "code":        ["code", "function", "api", "bug", "deploy", "database",
                            "kubernetes", "k8s", "docker", "container", "terraform",
                            "pipeline", "ci/cd", "github", "staging", "microservice",
                            "postgresql", "postgres", "redis", "aws", "ec2", "s3",
                            "canary", "blue-green", "rollback", "test", "coverage"],
            "vendor":      ["vendor", "contract", "license", "renewal", "sla",
                            "datadog", "zendesk", "grafana", "snyk", "launchdarkly",
                            "bamboohr", "checkr", "pagerduty", "cloudflare", "jira"],
            "hr":          ["hire", "hired", "employee", "headcount", "pto", "leave",
                            "performance", "onboard", "compensation", "promotion",
                            "equity", "remote", "office", "review"],
            "report":      ["report", "summary", "document", "pdf", "presentation",
                            "dashboard", "confluence", "adr", "postmortem"],
            "email":       ["email", "message", "reply", "send", "communicate"],
            "meeting":     ["meeting", "call", "schedule", "agenda", "discuss",
                            "sprint", "standup", "retro", "okr", "dora"],
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
        signal_db: Optional[SignalDatabase] = None,
        resonance_floor: float = RESONANCE_FLOOR,
    ):
        self.tenant_id  = tenant_id
        self.db         = db
        self.index      = index
        self.embedder   = embedder or SimpleEmbedder()
        self.llm        = llm_client
        self.drone_model  = drone_model
        self.router_model = router_model
        self.watchdog   = watchdog
        self.signal_db  = signal_db
        self.resonance_floor = resonance_floor

        self.enabled_layers = set(enabled_layers or list(MemoryLayer))

        # Vocabulary tagger — embedding-based, no LLM on write path
        self.vocab_tagger = SemanticVocabularyTagger(self.embedder)
        self.vocab_tagger._llm   = self.llm
        self.vocab_tagger._db    = self.db
        self.vocab_tagger._index = self.index

        # Layer 1 — in-memory only
        self._working: Dict[str, WorkingMemory] = {}

        # Pool size cache — refreshed periodically
        self._pool_size: int = 0
        self._pool_size_updated: float = 0

        self._lock = asyncio.Lock()
        self._verify_sem = asyncio.Semaphore(10)
        self._causal_sem  = asyncio.Semaphore(3)   # max 3 concurrent causal-ref builds

        # Background prune loop
        self._running = False
        self._prune_task = None

    # ──────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────

    async def start(self):
        """Start background prune loop (runs every 24 h) and vocab tagger."""
        self._running = True
        self._prune_task = asyncio.create_task(self._prune_loop())
        from mnemon.fragments.library import META_VOCABULARY
        await self.vocab_tagger.start(META_VOCABULARY)

    async def stop(self):
        """Stop background prune loop and vocab tagger."""
        self._running = False
        if self._prune_task:
            self._prune_task.cancel()
            try:
                await self._prune_task
            except asyncio.CancelledError:
                pass
        await self.vocab_tagger.stop()

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
        memory, high_confidence = await self._build_memory(signal, layer)

        # Step 4: Persist and index
        await self.db.write_memory(memory)
        await self.index.update(self.tenant_id, memory.memory_id, memory.activation_tags)

        # Step 5: Queue low-confidence memories for async LLM tagging
        if not high_confidence:
            content_str = json.dumps(signal.content) if isinstance(signal.content, dict) else str(signal.content)
            await self.vocab_tagger.queue_for_llm_tagging(
                memory.memory_id, self.tenant_id, content_str
            )

        # Step 6: Async tag verification — never blocks caller
        asyncio.create_task(self._verify_tags(memory))

        # Step 7: Async causal ref graph — links this memory to related existing ones
        if self.llm and self._pool_size > 5:
            asyncio.create_task(self._build_causal_refs(memory))

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

    async def _build_memory(self, signal: ExperienceSignal, layer: MemoryLayer) -> Tuple["BondedMemory", bool]:
        if isinstance(signal.content, dict):
            content_str = signal.content.get("text", "") or signal.content.get("content", "") or json.dumps(signal.content)
        else:
            content_str = str(signal.content)

        # Tags: always union vocab-tagger tags with rule-based tags so the inverted
        # index contains both semantic concepts (for broad queries) and exact proper
        # nouns / domain keywords (for narrow queries).  Neither path replaces the other.
        rule_tags = RuleClassifier.extract_tags(content_str, layer)
        vocab_tags, high_confidence = self.vocab_tagger.assign_tags(content_str)
        if not self.vocab_tagger._ready:
            # Vocab tagger not yet initialized (start() not called) — rule tags only
            tags = rule_tags
            high_confidence = True
        elif not high_confidence:
            # Tagger below confidence — minimal vocab tags; still keep all rule tags
            domain = self._infer_domain(rule_tags)
            tags = {domain, layer.value} | rule_tags
        else:
            # High-confidence vocab tags + all rule tags → richest index entry
            tags = vocab_tags | rule_tags

        # Activation signature — embed the bare text, not a JSON wrapper
        act_sig = self.embedder.embed(content_str[:512])
        intent_sig = self.embedder.embed(f"need to know: {content_str[:200]}")

        # Intent label — simple heuristic, LLM router will improve it
        intent_label = self._heuristic_intent(content_str, layer)

        memory_id = hashlib.sha256(
            f"{self.tenant_id}:{signal.session_id}:{signal.timestamp}:{content_str[:100]}".encode()
        ).hexdigest()[:24]

        memory = BondedMemory(
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
        return memory, high_confidence

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

        async with self._verify_sem:
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

    async def _build_causal_refs(self, new_memory: "BondedMemory"):
        """
        Async — fires after write, never blocks caller.
        Fetches top-10 similar existing memories, asks LLM which ones
        have a CAUSAL or EXPLANATORY relationship (not just similarity).
        Writes bidirectional cross_layer_refs.

        Only runs when pool > 5 (checked by caller) and llm is present.
        Semaphore-capped at 3 concurrent builds to limit write-path cost.
        Uses router_model (Haiku) — cheap call, max 150 tokens.
        """
        async with self._causal_sem:
            try:
                content_str = (
                    json.dumps(new_memory.content)
                    if isinstance(new_memory.content, dict)
                    else str(new_memory.content)
                )

                # Candidate pool: tag intersection → top-10 by cosine
                signal_tags = RuleClassifier.extract_tags(content_str, new_memory.layer)
                candidate_ids = await self.index.intersect(self.tenant_id, signal_tags)
                candidate_ids.discard(new_memory.memory_id)
                if not candidate_ids:
                    return

                sig = self.embedder.embed(content_str)
                raw = await self.db.fetch_memories(self.tenant_id, list(candidate_ids)[:50])
                scored = sorted(
                    [(c, SimpleEmbedder.cosine_similarity(sig, c.activation_signature))
                     for c in raw if c.activation_signature],
                    key=lambda x: x[1], reverse=True
                )
                top = [c for c, _ in scored[:10]]
                if not top:
                    return

                candidates_list = [
                    {
                        "id":      c.memory_id,
                        "layer":   c.layer.value,
                        "content": str(c.content)[:200],
                    }
                    for c in top
                ]

                prompt = f"""You are analyzing a cognitive memory system.

New memory ({new_memory.layer.value}):
{content_str[:300]}

Candidates:
{json.dumps(candidates_list, indent=2)}

Which candidates have a CAUSAL or EXPLANATORY link to the new memory?
Examples of valid links: causes, explains, is caused by, contradicts, enables, is prerequisite for.
Similarity alone is NOT enough — we want structural meaning connections.

Reply with JSON only, no markdown:
{{"related_ids": ["id1", "id2"], "reason": "brief"}}
If none qualify: {{"related_ids": [], "reason": "no causal links"}}"""

                response = await self.llm.complete(
                    prompt=prompt,
                    model=self.router_model,
                    max_tokens=150,
                )
                data = json.loads(response)
                related_ids = [r for r in data.get("related_ids", []) if isinstance(r, str)]

                if not related_ids:
                    return

                # Write bidirectional refs
                await self.db.update_cross_layer_refs(
                    self.tenant_id, new_memory.memory_id, related_ids
                )
                for ref_id in related_ids:
                    await self.db.update_cross_layer_refs(
                        self.tenant_id, ref_id, [new_memory.memory_id]
                    )
                logger.debug(f"Causal refs built for {new_memory.memory_id}: {related_ids}")

            except Exception as e:
                logger.debug(f"Causal ref build failed (non-critical): {e}")

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

        # Part 1b: Spreading Activation — follow causal refs from seeds
        candidates = await self._spreading_activation(candidates)

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

        # Compressed context — pre-assembled picture for LLM injection
        compressed = self._compress_context(final, memories_by_id, conflicts)

        return {
            "working":            working_ctx,
            "memories":           [memories_by_id[m_id].content for m_id, _ in final if m_id in memories_by_id],
            "memory_ids":         [m_id for m_id, _ in final],
            "memory_scores":      {m_id: score for m_id, score in final},
            "conflicts":          conflicts,
            "pool_size":          pool_size,
            "drone_used":         drone_used,
            "layers_present":     list({memories_by_id[m_id].layer.value for m_id, _ in final if m_id in memories_by_id}),
            "compressed_context": compressed,
        }

    async def _pattern_assembly(self, task_signal: str) -> List[Tuple[str, float]]:
        """
        Step 1: Tag intersection (inverted index, milliseconds)
        Step 2: Protein bond resonance (64-dim cosine, fuzzy 70%+)
        Step 3: Score-gap-aware pool trimming (replaces blind count ratchet)
        Returns: List of (memory_id, score) sorted descending
        """
        signal_tags = RuleClassifier.extract_tags(task_signal, MemoryLayer.EPISODIC)
        # Strip the layer tag: on the read path we don't want to restrict to a single
        # layer.  Keeping "episodic" here causes intersection to exclude SEMANTIC
        # memories whenever episodic memories share another tag — union fallback never
        # fires and cross-layer recall collapses.
        signal_tags.discard(MemoryLayer.EPISODIC.value)
        signal_sig  = self.embedder.embed(task_signal)
        signal_intent_sig = self.embedder.embed(f"need to know: {task_signal}")

        # Tag intersection — hard candidates
        tag_candidates = await self.index.intersect(self.tenant_id, signal_tags)

        # Union fallback cap: prevent O(n) scan at 10M+ scale.
        # If union returned far more IDs than intersection, sort by ID hash to
        # get a stable, deterministic subsample rather than an arbitrary slice.
        if len(tag_candidates) > UNION_FALLBACK_CAP:
            import hashlib as _hl
            tag_candidates = set(
                sorted(tag_candidates, key=lambda x: _hl.md5(x.encode()).digest())
                [:UNION_FALLBACK_CAP]
            )

        # Fetch signatures for resonance scoring.
        # Semantic fallback: when tag candidates are sparse (< 20% of pool),
        # merge in a full embedding scan so narrow subdomain queries don't
        # miss memories that weren't tagged precisely enough.
        pool_size = await self._get_pool_size()
        sparse_threshold = max(10, int(pool_size * SEMANTIC_FALLBACK_THRESHOLD))
        if len(tag_candidates) < sparse_threshold and pool_size > 0:
            all_ids = await self.db.fetch_all_memory_ids(self.tenant_id, limit=SEMANTIC_FALLBACK_CAP)
            tag_candidates = tag_candidates | set(all_ids)

        if tag_candidates:
            tag_memories = await self.db.fetch_memories(self.tenant_id, list(tag_candidates))
        else:
            tag_memories = []

        now = time.time()
        pool: List[Tuple[str, float]] = []

        for mem in tag_memories:
            if not mem.activation_signature:
                continue
            try:
                pattern_score = SimpleEmbedder.cosine_similarity(signal_sig, mem.activation_signature)
                intent_score  = SimpleEmbedder.cosine_similarity(signal_intent_sig, mem.intent_signature) if mem.intent_signature else 0.0
            except Exception:
                continue
            combined = (PATTERN_WEIGHT * pattern_score) + (INTENT_WEIGHT * intent_score)
            if combined >= self.resonance_floor:
                # Boost by drone keep history
                combined = min(1.0, combined + (mem.drone_keep_score - 0.5) * 0.1)
                # Recency bonus: decays exponentially, max +RECENCY_WEIGHT for today's memory
                age_days = (now - mem.timestamp) / 86400.0
                recency_bonus = RECENCY_WEIGHT * (0.5 ** (age_days / RECENCY_HALFLIFE_DAYS))
                combined = min(1.0, combined + recency_bonus)
                pool.append((mem.memory_id, combined))

        # Sort descending
        pool.sort(key=lambda x: x[1], reverse=True)

        # Score-gap-aware trimming: find the natural score cliff and cut there.
        # Avoids the blind count ratchet that wiped correct moderate-score memories.
        if len(pool) > 100:
            scores = [s for _, s in pool]
            # Compute gaps between consecutive scores
            gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
            if gaps:
                # Find the largest gap in the top-200 range
                search_end = min(200, len(gaps))
                max_gap_idx = max(range(search_end), key=lambda i: gaps[i])
                gap_at_cut = gaps[max_gap_idx]
                # Only cut at this gap if it's a real cliff (gap >= 0.03)
                # and the cut keeps >= 10 memories (to avoid cutting too aggressively)
                if gap_at_cut >= 0.03 and (max_gap_idx + 1) >= 10:
                    pool = pool[:max_gap_idx + 1]
                else:
                    # No natural cliff: hard cap at 200 to bound scan cost
                    pool = pool[:200]

        return pool

    async def _spreading_activation(
        self,
        seeds: List[Tuple[str, float]],
        max_hops: int = 2,
        decay: float = 0.5,
        threshold: float = 0.15,
    ) -> List[Tuple[str, float]]:
        """
        BFS spreading activation across the causal ref graph.

        Starting from seed memories (protein bond output), follows
        cross_layer_refs with decaying activation per hop. Merges
        activated memories into the candidate list — taking max
        activation where a memory appears via multiple paths.

        If cross_layer_refs are empty (graph not yet built), this
        returns seeds unchanged. Zero-overhead when graph is sparse.

        max_hops=2, decay=0.5:
          hop 0 (seeds):   activation = cosine score       (e.g. 0.85)
          hop 1 (refs):    activation = 0.85 × 0.5 = 0.42
          hop 2 (ref-refs): activation = 0.42 × 0.5 = 0.21
          below threshold (0.15) → stops
        """
        if not seeds:
            return seeds

        # Start: seed activations
        activation: Dict[str, float] = {m_id: score for m_id, score in seeds}
        frontier = [m_id for m_id, _ in seeds]

        for _hop in range(max_hops):
            if not frontier:
                break

            # Fetch frontier memories to read their cross_layer_refs
            frontier_mems = await self.db.fetch_memories(self.tenant_id, frontier)

            # Check if any refs exist — if not, short-circuit
            has_refs = any(m.cross_layer_refs for m in frontier_mems)
            if not has_refs:
                break

            next_frontier: List[str] = []
            for mem in frontier_mems:
                current = activation.get(mem.memory_id, 0.0)
                spread = current * decay
                if spread < threshold:
                    continue
                for ref_id in mem.cross_layer_refs:
                    if activation.get(ref_id, 0.0) < spread:
                        activation[ref_id] = spread
                        next_frontier.append(ref_id)

            frontier = list(set(next_frontier))

        return sorted(activation.items(), key=lambda x: x[1], reverse=True)

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
{json.dumps(candidate_list[:DRONE_CANDIDATE_CAP], indent=2)}

Reply format:
{{"keep": ["id1", "id2"], "drop": ["id3"], "conflicts": [["id4", "id5"]]}}

Only keep memories that help achieve the goal. Flag conflicts where two memories contradict."""

        try:
            response = await self.llm.complete(
                prompt=prompt,
                model=self.drone_model,
                max_tokens=500,
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

            curated = [(m_id, score) for m_id, score in candidates if m_id in keep_ids]
            # Blackout guard: if drone dropped everything, fall back to top-k from
            # pattern assembly so the agent never receives an empty context.
            if not curated and candidates:
                logger.debug("Drone returned 0 keeps — blackout guard activated, using pattern top results")
                return candidates
            return curated

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
        # Fire-and-forget: record cross-tenant success signal (never blocks caller)
        if self.signal_db:
            asyncio.create_task(self._record_memory_signal(memory_id))

    async def _record_memory_signal(self, memory_id: str):
        """Background task: fetch memory and record its shape hash as a success signal."""
        try:
            memories = await self.db.fetch_memories(self.tenant_id, [memory_id])
            if memories:
                mem = memories[0]
                if mem.activation_signature:
                    dims = mem.activation_signature[:32]
                    raw = struct.pack(f">{len(dims)}f", *dims)
                    shape_hash = hashlib.sha256(raw).hexdigest()[:32]
                    domain = mem.activation_domain or "general"
                    await self.signal_db.record_fragment_success(shape_hash, domain)
        except Exception as e:
            logger.debug(f"_record_memory_signal failed (non-critical): {e}")

    def _ensure_layer_diversity(
        self,
        candidates: List[Tuple[str, float]],
        memories_by_id: Dict[str, BondedMemory],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Soft layer diversity: promote the first memory from each unseen layer
        ONLY when its score is within DIVERSITY_SCORE_GAP of the next
        same-layer memory it would displace. This prevents a low-relevance
        Emotional/Relationship memory (score 0.71) from pushing out a
        high-relevance Semantic memory (score 0.88) just to fill a layer slot.
        """
        # Track next-best score per layer for gap calculation
        layer_best: Dict[MemoryLayer, float] = {}
        for m_id, score in candidates:
            mem = memories_by_id.get(m_id)
            if mem and mem.layer not in layer_best:
                layer_best[mem.layer] = score

        seen_layers: Set[MemoryLayer] = set()
        priority: List[Tuple[str, float]] = []
        remainder: List[Tuple[str, float]] = []

        for m_id, score in candidates:
            mem = memories_by_id.get(m_id)
            if not mem:
                continue
            if mem.layer not in seen_layers:
                # Only force diversity if the candidate score is close to the
                # top of the overall list — i.e., not too far below the best score
                best_overall = candidates[0][1] if candidates else 1.0
                gap = best_overall - score
                if gap <= DIVERSITY_SCORE_GAP:
                    priority.append((m_id, score))
                    seen_layers.add(mem.layer)
                else:
                    # Gap too large — treat as ordinary candidate, don't force in
                    remainder.append((m_id, score))
                    seen_layers.add(mem.layer)  # still mark seen so we don't try again
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

    def _compress_context(
        self,
        final: List[Tuple[str, float]],
        memories_by_id: Dict[str, "BondedMemory"],
        conflicts: List[Dict],
    ) -> Dict[str, Any]:
        """
        Compress the activated memory set into a structured context package.
        Groups by layer, surfaces tensions, identifies anchor memories
        (those referenced by multiple others in this subgraph).

        The LLM receives this instead of a flat list — it gets a pre-assembled
        picture: what is fact, what happened, how things are done, and where
        there are tensions. No archaeology required.
        """
        import time as _time

        now = _time.time()

        # Group memories by layer, sorted by score descending
        by_layer: Dict[str, List[Dict]] = {}
        for m_id, score in final:
            mem = memories_by_id.get(m_id)
            if not mem:
                continue
            layer = mem.layer.value
            if layer not in by_layer:
                by_layer[layer] = []
            age_days = round((now - mem.timestamp) / 86400, 1)
            by_layer[layer].append({
                "id":         m_id,
                "content":    mem.content,
                "score":      round(score, 3),
                "confidence": round(mem.confidence, 2),
                "age_days":   age_days,
                "refs":       mem.cross_layer_refs,
            })
        for layer in by_layer:
            by_layer[layer].sort(key=lambda x: x["score"], reverse=True)

        # Anchor detection: memories referenced by 2+ others in this activated set
        # A high ref-count = load-bearing memory, even if not highest cosine scorer
        ref_counts: Dict[str, int] = {}
        for m_id, _ in final:
            mem = memories_by_id.get(m_id)
            if not mem:
                continue
            for ref_id in mem.cross_layer_refs:
                if ref_id in memories_by_id:
                    ref_counts[ref_id] = ref_counts.get(ref_id, 0) + 1

        anchors = [
            {
                "id":            mid,
                "referenced_by": count,
                "content":       memories_by_id[mid].content,
                "layer":         memories_by_id[mid].layer.value,
            }
            for mid, count in sorted(ref_counts.items(), key=lambda x: x[1], reverse=True)
            if count >= 2 and mid in memories_by_id
        ]

        # Enrich tensions with actual content so the LLM sees what contradicts what
        tensions = []
        for c in conflicts:
            ma = memories_by_id.get(c["memory_a"])
            mb = memories_by_id.get(c["memory_b"])
            if ma and mb:
                tensions.append({
                    "domain":   c["domain"],
                    "note":     c["note"],
                    "memory_a": {"id": c["memory_a"], "content": ma.content},
                    "memory_b": {"id": c["memory_b"], "content": mb.content},
                })

        return {
            "by_layer": by_layer,
            "anchors":  anchors,
            "tensions": tensions,
        }

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
