"""
Mnemon Persistence Layer
SQLite-backed storage with schema migration framework.
Inverted index lives in RAM, backed by SQLite for restarts.
Redis interface stub ready for distributed scale.
"""

import asyncio
import sqlite3
import json
import struct
import time
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import asdict
from pathlib import Path

from .models import (
    BondedMemory, SemanticFact, FactVersion, ExecutionTemplate,
    TemplateSegment, ComputationFingerprint, AuditEntry,
    LLMCallLog, MemoryLayer, RiskLevel, MNEMON_VERSION
)

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 4


class SchemaError(Exception):
    pass


class InvertedIndex:
    """
    In-RAM inverted index mapping tags → set of memory IDs.
    Partitioned by tenant_id so tenants never bleed into each other.
    Backed by SQLite for restarts.
    """

    def __init__(self):
        self._shards: Dict[str, Dict[str, Set[str]]] = {}
        self._lock = asyncio.Lock()

    async def update(self, tenant_id: str, memory_id: str, tags: Set[str]):
        async with self._lock:
            if tenant_id not in self._shards:
                self._shards[tenant_id] = {}
            shard = self._shards[tenant_id]
            for tag in tags:
                if tag not in shard:
                    shard[tag] = set()
                shard[tag].add(memory_id)

    # Max IDs returned from union fallback — prevents O(n) cosine scan at 10M scale.
    # Pattern assembly in memory.py applies an additional cap, but capping here too
    # avoids loading millions of rows from SQLite before the cosine stage.
    _UNION_CAP = 10_000

    async def intersect(self, tenant_id: str, tags: Set[str]) -> Set[str]:
        async with self._lock:
            shard = self._shards.get(tenant_id, {})
            if not shard or not tags:
                return set()
            sets = [shard.get(tag, set()) for tag in tags]
            if not sets:
                return set()
            # True intersection: only memories matching ALL queried tags.
            # Falls back to union when intersection is empty (sparse index or
            # single-tag queries) so recall never drops to zero.
            result = sets[0].copy()
            for s in sets[1:]:
                result &= s
            if not result:
                # Fallback: union across tags so recall stays non-zero.
                # Capped at _UNION_CAP to prevent O(n) cosine scan at 10M+ scale.
                # Stable deterministic subsample: sort by hash so the same memories
                # surface consistently across calls (not arbitrary slice order).
                import hashlib as _hl
                union: Set[str] = set()
                for s in sets:
                    union |= s
                if len(union) > self._UNION_CAP:
                    union = set(
                        sorted(union, key=lambda x: _hl.md5(x.encode()).digest())
                        [:self._UNION_CAP]
                    )
                result = union
            return result

    async def remove(self, tenant_id: str, memory_id: str, tags: Set[str]):
        async with self._lock:
            shard = self._shards.get(tenant_id, {})
            for tag in tags:
                if tag in shard:
                    shard[tag].discard(memory_id)

    async def load_from_db(self, db: "EROSDatabase"):
        """Rebuild RAM index from SQLite on startup."""
        rows = await db.fetch_all_memory_tags()
        for tenant_id, memory_id, tags_json in rows:
            tags = set(json.loads(tags_json))
            await self.update(tenant_id, memory_id, tags)
        logger.info(f"Inverted index rebuilt — {sum(len(s) for s in self._shards.values())} tag entries")

    def get_stats(self) -> Dict[str, Any]:
        total_tags = sum(
            sum(len(ids) for ids in shard.values())
            for shard in self._shards.values()
        )
        return {
            "tenants": len(self._shards),
            "total_tag_entries": total_tags,
        }


class TenantConnectionPool:
    """
    Manages multiple EROSDatabase instances, one per tenant.
    Uses per-tenant asyncio locks so concurrent requests for different tenants
    never block each other. LRU eviction when max_connections is reached.

    GDPR: delete_tenant() disconnects, deletes the DB file, and removes from pool.
    """

    def __init__(self, db_dir: str = ".", max_connections: int = 100):
        self._db_dir = db_dir
        self._max_connections = max_connections
        # OrderedDict as LRU: leftmost = least recently used
        self._pool: OrderedDict[str, "EROSDatabase"] = OrderedDict()
        # Per-tenant locks — never shared between tenants
        self._locks: Dict[str, asyncio.Lock] = {}
        # Brief global lock for dict mutations only (never held during I/O)
        self._meta_lock = asyncio.Lock()

    async def get(self, tenant_id: str) -> "EROSDatabase":
        """Return existing connection or create a new one. LRU eviction at capacity."""
        # Fast path: already connected — update LRU order
        async with self._meta_lock:
            if tenant_id in self._pool:
                self._pool.move_to_end(tenant_id)
                return self._pool[tenant_id]
            # Ensure per-tenant lock exists before releasing meta_lock
            if tenant_id not in self._locks:
                self._locks[tenant_id] = asyncio.Lock()
            tenant_lock = self._locks[tenant_id]

        # Slow path: create connection under per-tenant lock (no global blocking)
        async with tenant_lock:
            # Double-check: another coroutine may have connected while we waited
            async with self._meta_lock:
                if tenant_id in self._pool:
                    self._pool.move_to_end(tenant_id)
                    return self._pool[tenant_id]
                # Identify LRU candidate to evict if at capacity
                evict_id = None
                if len(self._pool) >= self._max_connections:
                    evict_id, _ = next(iter(self._pool.items()))

            # Evict outside meta_lock so disconnect I/O doesn't stall other tenants
            if evict_id is not None:
                logger.info(f"Evicting tenant {evict_id} from pool (LRU)")
                async with self._meta_lock:
                    evict_db = self._pool.pop(evict_id, None)
                if evict_db is not None:
                    try:
                        await evict_db.disconnect()
                    except Exception as e:
                        logger.warning(f"Error disconnecting evicted tenant {evict_id}: {e}")

            # Create and connect new per-tenant DB (I/O outside meta_lock)
            db = EROSDatabase(tenant_id=tenant_id, db_dir=self._db_dir)
            try:
                await db.connect()
            except Exception:
                logger.error(f"Failed to connect DB for tenant {tenant_id}")
                raise

            async with self._meta_lock:
                self._pool[tenant_id] = db
            return db

    async def release(self, tenant_id: str):
        """Mark connection as recently used (LRU update)."""
        async with self._meta_lock:
            if tenant_id in self._pool:
                self._pool.move_to_end(tenant_id)

    async def close_all(self):
        """Disconnect all open connections on shutdown."""
        async with self._meta_lock:
            pool_items = list(self._pool.items())
            self._pool.clear()
        for tenant_id, db in pool_items:
            try:
                await db.disconnect()
            except Exception as e:
                logger.warning(f"Error closing tenant {tenant_id} connection: {e}")
        logger.info(f"TenantConnectionPool closed {len(pool_items)} connections")

    async def delete_tenant(self, tenant_id: str):
        """
        GDPR Article 17 erasure: disconnect, delete the DB file, remove from pool.
        This is the right-to-erasure method — data is irrecoverably deleted.
        """
        async with self._meta_lock:
            db = self._pool.pop(tenant_id, None)
        if db is not None:
            try:
                await db.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting tenant {tenant_id} before deletion: {e}")
        if self._db_dir == ":memory:":
            logger.info(f"Tenant {tenant_id} in-memory DB dropped (GDPR erasure)")
            return
        db_path = Path(f"{self._db_dir}/mnemon_tenant_{tenant_id}.db")
        try:
            if db_path.exists():
                db_path.unlink()
                logger.info(f"Tenant {tenant_id} DB file deleted (GDPR erasure): {db_path}")
            else:
                logger.info(f"Tenant {tenant_id} DB file not found (already gone): {db_path}")
        except Exception as e:
            logger.error(f"Failed to delete tenant {tenant_id} DB file: {e}")


class EROSDatabase:
    """
    SQLite persistence layer with schema migration.
    Thread-safe via asyncio lock.
    All queries enforce tenant_id — no cross-tenant leakage possible.
    Each instance manages a single per-tenant DB file.
    """

    def __init__(self, tenant_id: str, db_dir: str = "."):
        self.tenant_id = tenant_id
        self.db_dir = db_dir
        if db_dir == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = f"{db_dir}/mnemon_tenant_{tenant_id}.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    async def connect(self):
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")       # safe with WAL; halves fsync cost
        self._conn.execute("PRAGMA cache_size=-65536")        # 64 MB page cache
        self._conn.execute("PRAGMA busy_timeout=5000")        # 5 s retry on SQLITE_BUSY
        self._conn.execute("PRAGMA wal_autocheckpoint=1000")  # checkpoint every 1000 WAL pages
        await self._migrate()
        logger.info(f"Database connected: {self.db_path}")

    async def disconnect(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ──────────────────────────────────────────
    # SCHEMA MIGRATION
    # ──────────────────────────────────────────

    async def _get_schema_version(self) -> int:
        try:
            row = self._conn.execute(
                "SELECT version FROM eros_schema_version LIMIT 1"
            ).fetchone()
            return row["version"] if row else 0
        except sqlite3.OperationalError:
            return 0

    async def _set_schema_version(self, version: int):
        self._conn.execute(
            "INSERT OR REPLACE INTO eros_schema_version(id, version) VALUES(1, ?)",
            (version,)
        )
        self._conn.commit()

    async def _migrate(self):
        current = await self._get_schema_version()
        if current == CURRENT_SCHEMA_VERSION:
            return

        logger.info(f"Migrating schema from v{current} to v{CURRENT_SCHEMA_VERSION}")

        migrations = {
            0: self._migration_v0_to_v1,
            1: self._migration_v1_to_v2,
            2: self._migration_v2_to_v3,
            3: self._migration_v3_to_v4,
        }

        for v in range(current, CURRENT_SCHEMA_VERSION):
            if v in migrations:
                await migrations[v]()
                await self._set_schema_version(v + 1)
                logger.info(f"Schema migrated to v{v + 1}")

    async def _migration_v0_to_v1(self):
        """Initial schema creation."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS eros_schema_version (
                id      INTEGER PRIMARY KEY,
                version INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memories (
                memory_id           TEXT NOT NULL,
                tenant_id           TEXT NOT NULL,
                schema_version      INTEGER DEFAULT 1,
                layer               TEXT NOT NULL,
                content             TEXT NOT NULL,
                session_id          TEXT DEFAULT '',
                activation_tags     TEXT DEFAULT '[]',
                activation_domain   TEXT DEFAULT 'general',
                activation_signature TEXT DEFAULT '[]',
                intent_label        TEXT DEFAULT '',
                intent_valid        INTEGER DEFAULT 1,
                intent_signature    TEXT DEFAULT '[]',
                superseded_at       REAL,
                superseded_by       TEXT,
                importance          REAL DEFAULT 0.5,
                confidence          REAL DEFAULT 1.0,
                risk_level          TEXT DEFAULT 'low',
                timestamp           REAL NOT NULL,
                last_accessed       REAL NOT NULL,
                access_count        INTEGER DEFAULT 0,
                cross_layer_refs    TEXT DEFAULT '[]',
                drone_keep_score    REAL DEFAULT 0.5,
                drone_drop_score    REAL DEFAULT 0.5,
                PRIMARY KEY (memory_id, tenant_id)
            );

            CREATE TABLE IF NOT EXISTS semantic_facts (
                fact_id         TEXT NOT NULL,
                tenant_id       TEXT NOT NULL,
                key             TEXT NOT NULL,
                current_value   TEXT NOT NULL,
                confidence      REAL DEFAULT 1.0,
                history         TEXT DEFAULT '[]',
                last_updated    REAL NOT NULL,
                access_count    INTEGER DEFAULT 0,
                PRIMARY KEY (fact_id, tenant_id)
            );

            CREATE TABLE IF NOT EXISTS execution_templates (
                template_id         TEXT NOT NULL,
                tenant_id           TEXT NOT NULL,
                intent              TEXT NOT NULL,
                fingerprint_hash    TEXT NOT NULL,
                fingerprint_data    TEXT NOT NULL,
                segments            TEXT NOT NULL,
                success_count       INTEGER DEFAULT 0,
                failure_count       INTEGER DEFAULT 0,
                created_at          REAL NOT NULL,
                last_used_at        REAL NOT NULL,
                embedding           TEXT DEFAULT '[]',
                tool_versions       TEXT DEFAULT '{}',
                api_schemas         TEXT DEFAULT '{}',
                needs_reverification INTEGER DEFAULT 0,
                PRIMARY KEY (template_id, tenant_id)
            );

            CREATE TABLE IF NOT EXISTS fragment_library (
                segment_id      TEXT NOT NULL,
                tenant_id       TEXT NOT NULL,
                content         TEXT NOT NULL,
                fingerprint     TEXT NOT NULL,
                signature       TEXT DEFAULT '[]',
                domain_tags     TEXT DEFAULT '[]',
                success_rate    REAL DEFAULT 1.0,
                use_count       INTEGER DEFAULT 0,
                confidence      REAL DEFAULT 1.0,
                PRIMARY KEY (segment_id, tenant_id)
            );

            CREATE TABLE IF NOT EXISTS belief_registry (
                tenant_id   TEXT NOT NULL,
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                version     INTEGER NOT NULL DEFAULT 0,
                updated_at  REAL NOT NULL,
                PRIMARY KEY (tenant_id, key)
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id       TEXT NOT NULL,
                task_id         TEXT NOT NULL,
                timestamp       REAL NOT NULL,
                action          TEXT NOT NULL,
                mnemon_version    TEXT NOT NULL,
                template_id     TEXT,
                memory_ids      TEXT DEFAULT '[]',
                risk_level      TEXT NOT NULL,
                human_approved  INTEGER,
                outcome         TEXT NOT NULL,
                component       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS llm_call_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id       TEXT NOT NULL,
                component       TEXT NOT NULL,
                model           TEXT NOT NULL,
                tokens_input    INTEGER NOT NULL,
                tokens_output   INTEGER NOT NULL,
                cost_usd        REAL NOT NULL,
                timestamp       REAL NOT NULL,
                task_id         TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memories_tenant
                ON memories(tenant_id, layer, intent_valid);
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp
                ON memories(tenant_id, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_domain
                ON memories(tenant_id, activation_domain, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_facts_tenant_key
                ON semantic_facts(tenant_id, key);
            CREATE INDEX IF NOT EXISTS idx_templates_fingerprint
                ON execution_templates(tenant_id, fingerprint_hash);
            CREATE INDEX IF NOT EXISTS idx_audit_tenant
                ON audit_log(tenant_id, timestamp DESC);
        """)
        self._conn.commit()

    async def _migration_v1_to_v2(self):
        """Add activation_domain index for fetch_recent_by_domain hot path."""
        self._conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_memories_domain
                ON memories(tenant_id, activation_domain, timestamp DESC);
        """)
        self._conn.commit()

    async def _migration_v3_to_v4(self):
        """Add session_health table for cross-session drift detection."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS session_health (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id       TEXT NOT NULL,
                session_id      TEXT NOT NULL,
                timestamp       REAL NOT NULL,
                cache_hit_rate  REAL DEFAULT 0.0,
                memory_writes   INTEGER DEFAULT 0,
                total_calls     INTEGER DEFAULT 0,
                avg_latency_ms  REAL DEFAULT 0.0,
                notes           TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_session_health_tenant
                ON session_health(tenant_id, timestamp DESC);
        """)
        self._conn.commit()

    async def _migration_v2_to_v3(self):
        """Add is_prewarmed column to execution_templates."""
        try:
            self._conn.execute(
                "ALTER TABLE execution_templates ADD COLUMN is_prewarmed INTEGER DEFAULT 0"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_templates_prewarmed "
                "ON execution_templates(tenant_id, is_prewarmed)"
            )
            self._conn.commit()
        except Exception as e:
            if "duplicate column" not in str(e).lower():
                raise

    # ──────────────────────────────────────────
    # MEMORY OPERATIONS
    # ──────────────────────────────────────────

    async def write_memory(self, memory: BondedMemory):
        async with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO memories VALUES (
                    :memory_id, :tenant_id, :schema_version, :layer,
                    :content, :session_id, :activation_tags, :activation_domain,
                    :activation_signature, :intent_label, :intent_valid,
                    :intent_signature, :superseded_at, :superseded_by,
                    :importance, :confidence, :risk_level,
                    :timestamp, :last_accessed, :access_count,
                    :cross_layer_refs, :drone_keep_score, :drone_drop_score
                )
            """, {
                "memory_id":           memory.memory_id,
                "tenant_id":           memory.tenant_id,
                "schema_version":      memory.schema_version,
                "layer":               memory.layer.value,
                "content":             json.dumps(memory.content),
                "session_id":          memory.session_id,
                "activation_tags":     json.dumps(list(memory.activation_tags)),
                "activation_domain":   memory.activation_domain,
                "activation_signature": json.dumps(memory.activation_signature),
                "intent_label":        memory.intent_label,
                "intent_valid":        int(memory.intent_valid),
                "intent_signature":    json.dumps(memory.intent_signature),
                "superseded_at":       memory.superseded_at,
                "superseded_by":       memory.superseded_by,
                "importance":          memory.importance,
                "confidence":          memory.confidence,
                "risk_level":          memory.risk_level.value,
                "timestamp":           memory.timestamp,
                "last_accessed":       memory.last_accessed,
                "access_count":        memory.access_count,
                "cross_layer_refs":    json.dumps(memory.cross_layer_refs),
                "drone_keep_score":    memory.drone_keep_score,
                "drone_drop_score":    memory.drone_drop_score,
            })
            self._conn.commit()

    async def fetch_memories(
        self,
        tenant_id: str,
        memory_ids: List[str],
        include_superseded: bool = False
    ) -> List[BondedMemory]:
        if not memory_ids:
            return []
        async with self._lock:
            placeholders = ",".join("?" * len(memory_ids))
            params = [tenant_id] + memory_ids
            if not include_superseded:
                params.append(1)
                rows = self._conn.execute(
                    f"SELECT * FROM memories WHERE tenant_id=? AND memory_id IN ({placeholders}) AND intent_valid=?",
                    params
                ).fetchall()
            else:
                rows = self._conn.execute(
                    f"SELECT * FROM memories WHERE tenant_id=? AND memory_id IN ({placeholders})",
                    params
                ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def fetch_all_memory_tags(self):
        async with self._lock:
            rows = self._conn.execute(
                "SELECT tenant_id, memory_id, activation_tags FROM memories WHERE tenant_id=?",
                (self.tenant_id,)
            ).fetchall()
        return [(r["tenant_id"], r["memory_id"], r["activation_tags"]) for r in rows]

    async def fetch_all_memory_ids(self, tenant_id: str, limit: int = 5000) -> List[str]:
        """Return up to `limit` active memory IDs for a tenant. Used for semantic fallback scan."""
        async with self._lock:
            rows = self._conn.execute(
                "SELECT memory_id FROM memories WHERE tenant_id=? AND intent_valid=1 LIMIT ?",
                (tenant_id, limit)
            ).fetchall()
        return [r["memory_id"] for r in rows]

    async def fetch_sample_embedding_dim(self) -> Optional[int]:
        """Return the vector length of the first stored activation_signature, or None."""
        async with self._lock:
            row = self._conn.execute(
                "SELECT activation_signature FROM memories WHERE tenant_id=? "
                "AND activation_signature != '[]' LIMIT 1",
                (self.tenant_id,)
            ).fetchone()
        if row is None:
            return None
        try:
            sig = json.loads(row["activation_signature"])
            return len(sig) if sig else None
        except Exception:
            return None

    async def update_memory_tags(self, tenant_id: str, memory_id: str, tags: Set[str]):
        async with self._lock:
            self._conn.execute(
                "UPDATE memories SET activation_tags=? WHERE tenant_id=? AND memory_id=?",
                (json.dumps(list(tags)), tenant_id, memory_id)
            )
            self._conn.commit()

    async def update_cross_layer_refs(self, tenant_id: str, memory_id: str, ref_ids: List[str]):
        """Append causal refs to a memory, deduplicating. Bidirectional writes caller's responsibility."""
        async with self._lock:
            row = self._conn.execute(
                "SELECT cross_layer_refs FROM memories WHERE tenant_id=? AND memory_id=?",
                (tenant_id, memory_id)
            ).fetchone()
            if not row:
                return
            existing = set(json.loads(row["cross_layer_refs"]))
            updated = list(existing | set(ref_ids))
            self._conn.execute(
                "UPDATE memories SET cross_layer_refs=? WHERE tenant_id=? AND memory_id=?",
                (json.dumps(updated), tenant_id, memory_id)
            )
            self._conn.commit()

    async def update_drone_scores(
        self, tenant_id: str, memory_id: str,
        keep_delta: float = 0.0, drop_delta: float = 0.0
    ):
        async with self._lock:
            self._conn.execute("""
                UPDATE memories
                SET drone_keep_score = MAX(0.0, MIN(1.0, drone_keep_score + ?)),
                    drone_drop_score = MAX(0.0, MIN(1.0, drone_drop_score + ?))
                WHERE tenant_id=? AND memory_id=?
            """, (keep_delta, drop_delta, tenant_id, memory_id))
            self._conn.commit()

    async def supersede_memory(self, tenant_id: str, memory_id: str, new_id: str):
        async with self._lock:
            self._conn.execute("""
                UPDATE memories SET intent_valid=0, superseded_at=?, superseded_by=?
                WHERE tenant_id=? AND memory_id=?
            """, (time.time(), new_id, tenant_id, memory_id))
            self._conn.commit()

    async def delete_tenant_memories(self, tenant_id: str):
        async with self._lock:
            self._conn.execute(
                "DELETE FROM memories WHERE tenant_id=?", (tenant_id,)
            )
            self._conn.commit()

    def _row_to_memory(self, row) -> BondedMemory:
        return BondedMemory(
            memory_id=row["memory_id"],
            tenant_id=row["tenant_id"],
            schema_version=row["schema_version"],
            layer=MemoryLayer(row["layer"]),
            content=json.loads(row["content"]),
            session_id=row["session_id"],
            activation_tags=set(json.loads(row["activation_tags"])),
            activation_domain=row["activation_domain"],
            activation_signature=json.loads(row["activation_signature"]),
            intent_label=row["intent_label"],
            intent_valid=bool(row["intent_valid"]),
            intent_signature=json.loads(row["intent_signature"]),
            superseded_at=row["superseded_at"],
            superseded_by=row["superseded_by"],
            importance=row["importance"],
            confidence=row["confidence"],
            risk_level=RiskLevel(row["risk_level"]),
            timestamp=row["timestamp"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            cross_layer_refs=json.loads(row["cross_layer_refs"]),
            drone_keep_score=row["drone_keep_score"],
            drone_drop_score=row["drone_drop_score"],
        )

    # ──────────────────────────────────────────
    # SEMANTIC FACTS
    # ──────────────────────────────────────────

    async def write_fact(self, fact: SemanticFact):
        async with self._lock:
            history_data = [
                {
                    "value": v.value,
                    "confidence": v.confidence,
                    "recorded_at": v.recorded_at,
                    "source_session": v.source_session,
                    "superseded_at": v.superseded_at,
                }
                for v in fact.history
            ]
            self._conn.execute("""
                INSERT OR REPLACE INTO semantic_facts VALUES (
                    :fact_id, :tenant_id, :key, :current_value,
                    :confidence, :history, :last_updated, :access_count
                )
            """, {
                "fact_id":       fact.fact_id,
                "tenant_id":     fact.tenant_id,
                "key":           fact.key,
                "current_value": json.dumps(fact.current_value),
                "confidence":    fact.confidence,
                "history":       json.dumps(history_data),
                "last_updated":  fact.last_updated,
                "access_count":  fact.access_count,
            })
            self._conn.commit()

    async def fetch_fact(self, tenant_id: str, key: str) -> Optional[SemanticFact]:
        async with self._lock:
            row = self._conn.execute(
                "SELECT * FROM semantic_facts WHERE tenant_id=? AND key=?",
                (tenant_id, key)
            ).fetchone()
        if not row:
            return None
        history = [
            FactVersion(
                value=v["value"],
                confidence=v["confidence"],
                recorded_at=v["recorded_at"],
                source_session=v["source_session"],
                superseded_at=v.get("superseded_at"),
            )
            for v in json.loads(row["history"])
        ]
        return SemanticFact(
            fact_id=row["fact_id"],
            tenant_id=row["tenant_id"],
            key=row["key"],
            current_value=json.loads(row["current_value"]),
            confidence=row["confidence"],
            history=history,
            last_updated=row["last_updated"],
            access_count=row["access_count"],
        )

    async def delete_tenant_facts(self, tenant_id: str):
        async with self._lock:
            self._conn.execute(
                "DELETE FROM semantic_facts WHERE tenant_id=?", (tenant_id,)
            )
            self._conn.commit()

    # ──────────────────────────────────────────
    # EXECUTION TEMPLATES (EME)
    # ──────────────────────────────────────────

    async def write_template(self, template: ExecutionTemplate):
        async with self._lock:
            segments_data = [
                {
                    "segment_id":   s.segment_id,
                    "content":      s.content,
                    "fingerprint":  s.fingerprint,
                    "signature":    s.signature,
                    "dependencies": s.dependencies,
                    "outputs":      s.outputs,
                    "domain_tags":  list(s.domain_tags),
                    "success_rate": s.success_rate,
                    "use_count":    s.use_count,
                    "confidence":   s.confidence,
                    "is_generated": s.is_generated,
                }
                for s in template.segments
            ]
            fp = template.fingerprint
            self._conn.execute("""
                INSERT OR REPLACE INTO execution_templates (
                    template_id, tenant_id, intent,
                    fingerprint_hash, fingerprint_data, segments,
                    success_count, failure_count,
                    created_at, last_used_at, embedding,
                    tool_versions, api_schemas, needs_reverification,
                    is_prewarmed
                ) VALUES (
                    :template_id, :tenant_id, :intent,
                    :fingerprint_hash, :fingerprint_data, :segments,
                    :success_count, :failure_count,
                    :created_at, :last_used_at, :embedding,
                    :tool_versions, :api_schemas, :needs_reverification,
                    :is_prewarmed
                )
            """, {
                "template_id":          template.template_id,
                "tenant_id":            template.tenant_id,
                "intent":               template.intent,
                "fingerprint_hash":     fp.full_hash,
                "fingerprint_data":     json.dumps({
                    "goal_hash":         fp.goal_hash,
                    "input_schema_hash": fp.input_schema_hash,
                    "context_hash":      fp.context_hash,
                    "capability_hash":   fp.capability_hash,
                    "constraint_hash":   fp.constraint_hash,
                }),
                "segments":             json.dumps(segments_data),
                "success_count":        template.success_count,
                "failure_count":        template.failure_count,
                "created_at":           template.created_at,
                "last_used_at":         template.last_used_at,
                "embedding":            json.dumps(template.embedding),
                "tool_versions":        json.dumps(template.tool_versions),
                "api_schemas":          json.dumps(template.api_schemas),
                "needs_reverification": int(template.needs_reverification),
                "is_prewarmed":         int(getattr(template, "is_prewarmed", False)),
            })
            self._conn.commit()

    async def fetch_template_by_fingerprint(
        self, tenant_id: str, fingerprint_hash: str
    ) -> Optional[ExecutionTemplate]:
        async with self._lock:
            row = self._conn.execute(
                "SELECT * FROM execution_templates WHERE tenant_id=? AND fingerprint_hash=?",
                (tenant_id, fingerprint_hash)
            ).fetchone()
        if not row:
            return None
        return self._row_to_template(row)

    async def fetch_all_templates(self, tenant_id: str) -> List[ExecutionTemplate]:
        async with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM execution_templates WHERE tenant_id=? ORDER BY last_used_at DESC",
                (tenant_id,)
            ).fetchall()
        return [self._row_to_template(r) for r in rows]

    async def fetch_prewarmed_templates(self, tenant_id: str) -> List[ExecutionTemplate]:
        """Return only pre-warmed (library) templates for a tenant."""
        async with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM execution_templates "
                "WHERE tenant_id=? AND is_prewarmed=1 ORDER BY last_used_at DESC",
                (tenant_id,)
            ).fetchall()
        return [self._row_to_template(r) for r in rows]

    async def update_template_outcome(
        self, tenant_id: str, template_id: str, success: bool
    ):
        async with self._lock:
            if success:
                self._conn.execute(
                    "UPDATE execution_templates SET success_count=success_count+1, last_used_at=? WHERE tenant_id=? AND template_id=?",
                    (time.time(), tenant_id, template_id)
                )
            else:
                self._conn.execute(
                    "UPDATE execution_templates SET failure_count=failure_count+1 WHERE tenant_id=? AND template_id=?",
                    (tenant_id, template_id)
                )
            self._conn.commit()

    async def delete_template(self, tenant_id: str, template_id: str):
        async with self._lock:
            self._conn.execute(
                "DELETE FROM execution_templates WHERE tenant_id=? AND template_id=?",
                (tenant_id, template_id)
            )
            self._conn.commit()

    async def delete_tenant_templates(self, tenant_id: str):
        async with self._lock:
            self._conn.execute(
                "DELETE FROM execution_templates WHERE tenant_id=?", (tenant_id,)
            )
            self._conn.commit()

    def _row_to_template(self, row) -> ExecutionTemplate:
        fp_data = json.loads(row["fingerprint_data"])
        fp = ComputationFingerprint(
            goal_hash=fp_data["goal_hash"],
            input_schema_hash=fp_data["input_schema_hash"],
            context_hash=fp_data["context_hash"],
            capability_hash=fp_data["capability_hash"],
            constraint_hash=fp_data["constraint_hash"],
        )
        segments_data = json.loads(row["segments"])
        segments = [
            TemplateSegment(
                segment_id=s["segment_id"],
                tenant_id=row["tenant_id"],
                content=s["content"],
                fingerprint=s["fingerprint"],
                signature=s.get("signature", []),
                dependencies=s.get("dependencies", []),
                outputs=s.get("outputs", []),
                domain_tags=set(s.get("domain_tags", [])),
                success_rate=s.get("success_rate", 1.0),
                use_count=s.get("use_count", 0),
                confidence=s.get("confidence", 1.0),
                is_generated=s.get("is_generated", False),
            )
            for s in segments_data
        ]
        return ExecutionTemplate(
            template_id=row["template_id"],
            tenant_id=row["tenant_id"],
            intent=row["intent"],
            fingerprint=fp,
            segments=segments,
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            created_at=row["created_at"],
            last_used_at=row["last_used_at"],
            embedding=json.loads(row["embedding"]),
            tool_versions=json.loads(row["tool_versions"]),
            api_schemas=json.loads(row["api_schemas"]),
            needs_reverification=bool(row["needs_reverification"]),
            is_prewarmed=bool(row["is_prewarmed"]) if "is_prewarmed" in row.keys() else False,
        )

    # ──────────────────────────────────────────
    # FRAGMENT LIBRARY
    # ──────────────────────────────────────────

    async def write_fragment(self, segment: TemplateSegment):
        async with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO fragment_library VALUES (
                    :segment_id, :tenant_id, :content, :fingerprint,
                    :signature, :domain_tags, :success_rate, :use_count, :confidence
                )
            """, {
                "segment_id":   segment.segment_id,
                "tenant_id":    segment.tenant_id,
                "content":      json.dumps(segment.content),
                "fingerprint":  segment.fingerprint,
                "signature":    json.dumps(segment.signature),
                "domain_tags":  json.dumps(list(segment.domain_tags)),
                "success_rate": segment.success_rate,
                "use_count":    segment.use_count,
                "confidence":   segment.confidence,
            })
            self._conn.commit()

    async def fetch_fragments(self, tenant_id: str) -> List[TemplateSegment]:
        async with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM fragment_library WHERE tenant_id=? ORDER BY success_rate DESC",
                (tenant_id,)
            ).fetchall()
        return [
            TemplateSegment(
                segment_id=r["segment_id"],
                tenant_id=tenant_id,
                content=json.loads(r["content"]),
                fingerprint=r["fingerprint"],
                signature=json.loads(r["signature"]),
                domain_tags=set(json.loads(r["domain_tags"])),
                success_rate=r["success_rate"],
                use_count=r["use_count"],
                confidence=r["confidence"],
            )
            for r in rows
        ]

    # ──────────────────────────────────────────
    # SESSION HEALTH (drift detection)
    # ──────────────────────────────────────────

    async def write_session_health(
        self,
        tenant_id: str,
        session_id: str,
        cache_hit_rate: float,
        memory_writes: int,
        total_calls: int,
        avg_latency_ms: float,
        notes: str = "",
    ):
        async with self._lock:
            self._conn.execute(
                "INSERT INTO session_health "
                "(tenant_id, session_id, timestamp, cache_hit_rate, memory_writes, "
                "total_calls, avg_latency_ms, notes) VALUES (?,?,?,?,?,?,?,?)",
                (tenant_id, session_id, time.time(), cache_hit_rate,
                 memory_writes, total_calls, avg_latency_ms, notes),
            )
            self._conn.commit()

    async def fetch_session_health(
        self, tenant_id: str, limit: int = 30
    ) -> List[Dict]:
        async with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM session_health WHERE tenant_id=? "
                "ORDER BY id DESC LIMIT ?",
                (tenant_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    async def fetch_memories_since(
        self, tenant_id: str, since_ts: float, limit: int = 200
    ) -> List[str]:
        """Return memory_ids written after since_ts — used by drift auto-correction."""
        async with self._lock:
            rows = self._conn.execute(
                "SELECT memory_id FROM memories WHERE tenant_id=? AND timestamp >= ? "
                "AND intent_valid=1 ORDER BY timestamp ASC LIMIT ?",
                (tenant_id, since_ts, limit),
            ).fetchall()
        return [r["memory_id"] for r in rows]

    # ──────────────────────────────────────────
    # BELIEF REGISTRY
    # ──────────────────────────────────────────

    async def get_belief(self, tenant_id: str, key: str) -> Optional[Dict]:
        async with self._lock:
            row = self._conn.execute(
                "SELECT * FROM belief_registry WHERE tenant_id=? AND key=?",
                (tenant_id, key)
            ).fetchone()
        if not row:
            return None
        return {
            "value":   json.loads(row["value"]),
            "version": row["version"],
        }

    async def get_belief_version(self, tenant_id: str) -> int:
        async with self._lock:
            row = self._conn.execute(
                "SELECT MAX(version) as v FROM belief_registry WHERE tenant_id=?",
                (tenant_id,)
            ).fetchone()
        return row["v"] or 0

    async def set_belief(
        self, tenant_id: str, key: str, value: Any,
        expected_version: int
    ) -> bool:
        """Optimistic locking — returns False if version mismatch."""
        async with self._lock:
            current_row = self._conn.execute(
                "SELECT version FROM belief_registry WHERE tenant_id=? AND key=?",
                (tenant_id, key)
            ).fetchone()

            current_version = current_row["version"] if current_row else 0
            if current_row and current_version != expected_version:
                return False

            new_version = current_version + 1
            self._conn.execute("""
                INSERT OR REPLACE INTO belief_registry VALUES (
                    :tenant_id, :key, :value, :version, :updated_at
                )
            """, {
                "tenant_id":  tenant_id,
                "key":        key,
                "value":      json.dumps(value),
                "version":    new_version,
                "updated_at": time.time(),
            })
            self._conn.commit()
            return True

    async def get_all_beliefs(self, tenant_id: str) -> Dict[str, Any]:
        async with self._lock:
            rows = self._conn.execute(
                "SELECT key, value, version FROM belief_registry WHERE tenant_id=?",
                (tenant_id,)
            ).fetchall()
        return {
            r["key"]: {
                "value":   json.loads(r["value"]),
                "version": r["version"],
            }
            for r in rows
        }

    # ──────────────────────────────────────────
    # AUDIT LOG
    # ──────────────────────────────────────────

    async def write_audit(self, entry: AuditEntry):
        async with self._lock:
            self._conn.execute("""
                INSERT INTO audit_log (
                    tenant_id, task_id, timestamp, action, mnemon_version,
                    template_id, memory_ids, risk_level, human_approved,
                    outcome, component
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                entry.tenant_id, entry.task_id, entry.timestamp,
                entry.action, entry.mnemon_version, entry.template_id,
                json.dumps(entry.memory_ids), entry.risk_level.value,
                entry.human_approved, entry.outcome, entry.component
            ))
            self._conn.commit()

    # ──────────────────────────────────────────
    # COST LOG
    # ──────────────────────────────────────────

    async def write_llm_call(self, log: LLMCallLog):
        async with self._lock:
            self._conn.execute("""
                INSERT INTO llm_call_log (
                    tenant_id, component, model, tokens_input, tokens_output,
                    cost_usd, timestamp, task_id
                ) VALUES (?,?,?,?,?,?,?,?)
            """, (
                log.tenant_id, log.component, log.model,
                log.tokens_input, log.tokens_output, log.cost_usd,
                log.timestamp, log.task_id
            ))
            self._conn.commit()

    async def get_cost_summary(self, tenant_id: str, since: float) -> Dict:
        async with self._lock:
            row = self._conn.execute("""
                SELECT
                    COUNT(*) as calls,
                    SUM(tokens_input + tokens_output) as total_tokens,
                    SUM(cost_usd) as total_cost
                FROM llm_call_log
                WHERE tenant_id=? AND timestamp >= ?
            """, (tenant_id, since)).fetchone()
        return {
            "calls":        row["calls"] or 0,
            "total_tokens": row["total_tokens"] or 0,
            "total_cost":   row["total_cost"] or 0.0,
        }

    # ──────────────────────────────────────────
    # TENANT MANAGEMENT
    # ──────────────────────────────────────────

    async def delete_tenant_all(self, tenant_id: str):
        """Full tenant data erasure — GDPR Article 17 compliant."""
        async with self._lock:
            tables = [
                "memories", "semantic_facts", "execution_templates",
                "fragment_library", "belief_registry"
            ]
            for table in tables:
                self._conn.execute(
                    f"DELETE FROM {table} WHERE tenant_id=?", (tenant_id,)
                )
            self._conn.commit()
        logger.info(f"Tenant {tenant_id} data fully erased")

    # ──────────────────────────────────────────
    # PUBLIC UTILITY METHODS (replacing direct _conn access)
    # ──────────────────────────────────────────

    async def ping(self) -> bool:
        """Health check — returns True if DB responds. Used by Watchdog."""
        try:
            async with self._lock:
                self._conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    async def fetch_recent_by_domain(
        self, tenant_id: str, domain: str, n: int
    ) -> List[str]:
        """
        Return up to n recent valid memory_ids whose activation_domain
        matches the given domain. Falls back to any domain if no match
        or domain is 'general'.
        """
        async with self._lock:
            if domain and domain != "general":
                rows = self._conn.execute(
                    "SELECT memory_id FROM memories "
                    "WHERE tenant_id=? AND intent_valid=1 AND activation_domain=? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (tenant_id, domain, n)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT memory_id FROM memories "
                    "WHERE tenant_id=? AND intent_valid=1 "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (tenant_id, n)
                ).fetchall()
        return [r["memory_id"] for r in rows]

    async def fetch_drone_labels(
        self, tenant_id: str, memory_ids: List[str]
    ) -> Dict[str, Tuple[str, float]]:
        """
        Return {memory_id: (intent_label, drone_keep_score)} for the
        given ids. Replaces direct _conn access in _drone_evaluate().
        """
        if not memory_ids:
            return {}
        async with self._lock:
            placeholders = ",".join("?" * len(memory_ids))
            rows = self._conn.execute(
                f"SELECT memory_id, intent_label, drone_keep_score FROM memories "
                f"WHERE tenant_id=? AND memory_id IN ({placeholders})",
                [tenant_id] + memory_ids
            ).fetchall()
        return {r["memory_id"]: (r["intent_label"], r["drone_keep_score"]) for r in rows}

    async def count_memories(self, tenant_id: str) -> int:
        """Count valid (non-superseded) memories for a tenant."""
        async with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) as c FROM memories WHERE tenant_id=? AND intent_valid=1",
                (tenant_id,)
            ).fetchone()
        return row["c"] if row else 0

    def get_stats(self) -> Dict[str, Any]:
        # sync, no await — safe in asyncio since no yield point means no write can interleave
        rows = self._conn.execute("""
            SELECT
                (SELECT COUNT(*) FROM memories WHERE tenant_id=?) as memories,
                (SELECT COUNT(*) FROM semantic_facts WHERE tenant_id=?) as facts,
                (SELECT COUNT(*) FROM execution_templates WHERE tenant_id=?) as templates,
                (SELECT COUNT(*) FROM fragment_library WHERE tenant_id=?) as fragments
        """, (self.tenant_id, self.tenant_id, self.tenant_id, self.tenant_id)).fetchone()
        return {
            "memories":  rows["memories"],
            "facts":     rows["facts"],
            "templates": rows["templates"],
            "fragments": rows["fragments"],
        }


# ──────────────────────────────────────────
# LEGACY MIGRATION HELPER
# ──────────────────────────────────────────

async def migrate_from_legacy(
    old_db_path: str,
    tenant_id: str,
    db_dir: str = ".",
) -> Dict[str, int]:
    """
    One-time migration helper for existing users.

    Reads all data for tenant_id from a legacy monolithic mnemon.db and writes
    it into the new per-tenant DB file (mnemon_tenant_{tenant_id}.db in db_dir).

    Returns {table_name: rows_migrated} so callers can verify the migration.
    Safe to run multiple times — uses INSERT OR REPLACE / INSERT OR IGNORE.

    Usage:
        import asyncio
        from mnemon.core.persistence import migrate_from_legacy
        result = asyncio.run(migrate_from_legacy("mnemon.db", "my_tenant", db_dir="/data"))
        print(result)
    """
    old_conn = sqlite3.connect(old_db_path)
    old_conn.row_factory = sqlite3.Row

    new_db = EROSDatabase(tenant_id=tenant_id, db_dir=db_dir)
    await new_db.connect()

    migrated: Dict[str, int] = {}

    # Tables whose PRIMARY KEY includes tenant_id
    pk_tables = [
        "memories",
        "semantic_facts",
        "execution_templates",
        "fragment_library",
        "belief_registry",
    ]
    for table in pk_tables:
        try:
            rows = old_conn.execute(
                f"SELECT * FROM {table} WHERE tenant_id=?", (tenant_id,)
            ).fetchall()
            for row in rows:
                cols = list(row.keys())
                placeholders = ",".join("?" * len(cols))
                # Direct _conn access is acceptable here — migration runs single-threaded
                new_db._conn.execute(
                    f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) "
                    f"VALUES ({placeholders})",
                    tuple(row),
                )
            new_db._conn.commit()
            migrated[table] = len(rows)
        except Exception as e:
            logger.warning(f"migrate_from_legacy: table '{table}' skipped — {e}")
            migrated[table] = 0

    # Tables with AUTOINCREMENT id — skip the id column to avoid PK conflicts
    auto_tables = ["audit_log", "llm_call_log"]
    for table in auto_tables:
        try:
            rows = old_conn.execute(
                f"SELECT * FROM {table} WHERE tenant_id=?", (tenant_id,)
            ).fetchall()
            for row in rows:
                cols = [k for k in row.keys() if k != "id"]
                placeholders = ",".join("?" * len(cols))
                new_db._conn.execute(
                    f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})",
                    tuple(row[c] for c in cols),
                )
            new_db._conn.commit()
            migrated[table] = len(rows)
        except Exception as e:
            logger.warning(f"migrate_from_legacy: table '{table}' skipped — {e}")
            migrated[table] = 0

    old_conn.close()
    await new_db.disconnect()

    logger.info(f"migrate_from_legacy complete [tenant={tenant_id}]: {migrated}")
    return migrated
