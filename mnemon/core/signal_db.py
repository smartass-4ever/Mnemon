"""
Mnemon Signal Database
Cross-tenant, privacy-preserving performance signals.
Stores only shape hashes of embedding vectors — never content, never tenant IDs.

Privacy guarantee (enforced in code):
  shape_hash = hashlib.sha256(struct.pack(">Nf", *embedding[:32])).hexdigest()[:32]
  NEVER store content, NEVER store tenant_id, NEVER store goal text.
  All methods work without knowing which tenant contributed.
"""

import asyncio
import logging
import sqlite3
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fragment_signals (
    shape_hash      TEXT PRIMARY KEY,
    domain          TEXT NOT NULL,
    success_count   INTEGER DEFAULT 0,
    failure_count   INTEGER DEFAULT 0,
    last_updated    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS approach_patterns (
    pattern_hash    TEXT PRIMARY KEY,
    domain          TEXT NOT NULL,
    success_rate    REAL DEFAULT 0.5,
    sample_count    INTEGER DEFAULT 0,
    last_updated    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS vocab_weights (
    concept         TEXT PRIMARY KEY,
    weight          REAL DEFAULT 0.5,
    tenant_count    INTEGER DEFAULT 0,
    last_updated    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS proven_intents (
    intent_key      TEXT PRIMARY KEY,
    domain          TEXT NOT NULL DEFAULT 'general',
    success_count   INTEGER DEFAULT 0,
    failure_count   INTEGER DEFAULT 0,
    boost_weight    REAL DEFAULT 0.0,
    last_updated    REAL NOT NULL
);
"""

# Minimum evidence before collective boost kicks in
_PROVEN_MIN_SUCCESSES = 3
_PROVEN_MIN_RATE      = 0.75
_PROVEN_MAX_BOOST     = 0.12   # max similarity score additive boost


class SignalDatabase:
    """
    Cross-tenant performance signal store.
    Schema: fragment_signals, approach_patterns, vocab_weights.

    Fully isolated from EROSDatabase — never imports from persistence.py.
    All methods have try/except so failures are logged but never propagate.
    All DB operations are fire-and-forget safe — callers use asyncio.create_task.
    """

    def __init__(self, db_path: str = "mnemon_signal.db"):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    # ──────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────

    async def connect(self):
        try:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            logger.info(f"SignalDatabase connected: {self._db_path}")
        except Exception as e:
            logger.error(f"SignalDatabase connect failed: {e}")

    async def disconnect(self):
        try:
            if self._conn:
                self._conn.close()
                self._conn = None
                logger.info("SignalDatabase disconnected")
        except Exception as e:
            logger.error(f"SignalDatabase disconnect error: {e}")

    # ──────────────────────────────────────────
    # FRAGMENT SIGNALS
    # shape_hash is sha256(struct.pack embedding dims)[:32]
    # No content or tenant info ever reaches this table.
    # ──────────────────────────────────────────

    async def record_fragment_success(self, shape_hash: str, domain: str):
        """Increment success_count, upsert row, update last_updated."""
        try:
            async with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO fragment_signals
                        (shape_hash, domain, success_count, failure_count, last_updated)
                    VALUES (?, ?, 1, 0, ?)
                    ON CONFLICT(shape_hash) DO UPDATE SET
                        success_count = success_count + 1,
                        last_updated  = excluded.last_updated
                    """,
                    (shape_hash, domain, time.time()),
                )
                self._conn.commit()
        except Exception as e:
            logger.debug(f"record_fragment_success failed (non-critical): {e}")

    async def record_fragment_failure(self, shape_hash: str, domain: str):
        """Increment failure_count, upsert row, update last_updated."""
        try:
            async with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO fragment_signals
                        (shape_hash, domain, success_count, failure_count, last_updated)
                    VALUES (?, ?, 0, 1, ?)
                    ON CONFLICT(shape_hash) DO UPDATE SET
                        failure_count = failure_count + 1,
                        last_updated  = excluded.last_updated
                    """,
                    (shape_hash, domain, time.time()),
                )
                self._conn.commit()
        except Exception as e:
            logger.debug(f"record_fragment_failure failed (non-critical): {e}")

    async def get_fragment_signal(self, shape_hash: str) -> Optional[dict]:
        """Returns {success_count, failure_count, success_rate, domain} or None if not found."""
        try:
            async with self._lock:
                row = self._conn.execute(
                    "SELECT * FROM fragment_signals WHERE shape_hash=?",
                    (shape_hash,),
                ).fetchone()
            if not row:
                return None
            total = row["success_count"] + row["failure_count"]
            success_rate = row["success_count"] / total if total > 0 else 0.5
            return {
                "success_count": row["success_count"],
                "failure_count": row["failure_count"],
                "success_rate":  success_rate,
                "domain":        row["domain"],
            }
        except Exception as e:
            logger.debug(f"get_fragment_signal failed (non-critical): {e}")
            return None

    async def get_top_fragments(self, domain: str, limit: int = 50) -> List[dict]:
        """
        Return fragments sorted by success_rate DESC for a domain.
        Used by EME on warm() to pre-rank the fragment library.
        """
        try:
            async with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT
                        shape_hash,
                        success_count,
                        failure_count,
                        CAST(success_count AS REAL) /
                            MAX(success_count + failure_count, 1) AS success_rate
                    FROM fragment_signals
                    WHERE domain = ?
                    ORDER BY success_rate DESC
                    LIMIT ?
                    """,
                    (domain, limit),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.debug(f"get_top_fragments failed (non-critical): {e}")
            return []

    # ──────────────────────────────────────────
    # APPROACH PATTERNS
    # pattern_hash is sha256(goal embedding cluster)[:32]
    # ──────────────────────────────────────────

    async def record_approach_success(self, pattern_hash: str, domain: str):
        """Incrementally update success_rate using online mean update."""
        try:
            async with self._lock:
                row = self._conn.execute(
                    "SELECT success_rate, sample_count FROM approach_patterns WHERE pattern_hash=?",
                    (pattern_hash,),
                ).fetchone()
                if row:
                    n = row["sample_count"] + 1
                    new_rate = row["success_rate"] + (1.0 - row["success_rate"]) / n
                    self._conn.execute(
                        "UPDATE approach_patterns SET success_rate=?, sample_count=?, last_updated=? "
                        "WHERE pattern_hash=?",
                        (new_rate, n, time.time(), pattern_hash),
                    )
                else:
                    self._conn.execute(
                        "INSERT INTO approach_patterns VALUES (?,?,?,?,?)",
                        (pattern_hash, domain, 1.0, 1, time.time()),
                    )
                self._conn.commit()
        except Exception as e:
            logger.debug(f"record_approach_success failed (non-critical): {e}")

    async def record_approach_failure(self, pattern_hash: str, domain: str):
        """Incrementally update success_rate using online mean update."""
        try:
            async with self._lock:
                row = self._conn.execute(
                    "SELECT success_rate, sample_count FROM approach_patterns WHERE pattern_hash=?",
                    (pattern_hash,),
                ).fetchone()
                if row:
                    n = row["sample_count"] + 1
                    new_rate = row["success_rate"] + (0.0 - row["success_rate"]) / n
                    self._conn.execute(
                        "UPDATE approach_patterns SET success_rate=?, sample_count=?, last_updated=? "
                        "WHERE pattern_hash=?",
                        (new_rate, n, time.time(), pattern_hash),
                    )
                else:
                    self._conn.execute(
                        "INSERT INTO approach_patterns VALUES (?,?,?,?,?)",
                        (pattern_hash, domain, 0.0, 1, time.time()),
                    )
                self._conn.commit()
        except Exception as e:
            logger.debug(f"record_approach_failure failed (non-critical): {e}")

    # ──────────────────────────────────────────
    # VOCAB WEIGHTS
    # Aggregated concept weights — no tenant info stored.
    # ──────────────────────────────────────────

    async def update_vocab_weight(self, concept: str, delta: float):
        """Update weight by delta, clamped to [0.1, 1.0]. Increments tenant_count."""
        try:
            async with self._lock:
                row = self._conn.execute(
                    "SELECT weight, tenant_count FROM vocab_weights WHERE concept=?",
                    (concept,),
                ).fetchone()
                if row:
                    new_weight = max(0.1, min(1.0, row["weight"] + delta))
                    self._conn.execute(
                        "UPDATE vocab_weights SET weight=?, tenant_count=?, last_updated=? WHERE concept=?",
                        (new_weight, row["tenant_count"] + 1, time.time(), concept),
                    )
                else:
                    new_weight = max(0.1, min(1.0, 0.5 + delta))
                    self._conn.execute(
                        "INSERT INTO vocab_weights VALUES (?,?,?,?)",
                        (concept, new_weight, 1, time.time()),
                    )
                self._conn.commit()
        except Exception as e:
            logger.debug(f"update_vocab_weight failed (non-critical): {e}")

    # ──────────────────────────────────────────
    # PROVEN INTENTS — collective learning
    # intent_key = md5(prewarmed_intent_string)[:24], privacy-safe
    # Tracks which pre-warmed execution strategies have been validated
    # cross-tenant so new tenants benefit from collective evidence.
    # ──────────────────────────────────────────

    async def record_proven_intent(self, intent_key: str, domain: str, success: bool):
        """Record a cross-tenant outcome for a pre-warmed intent strategy."""
        try:
            async with self._lock:
                row = self._conn.execute(
                    "SELECT success_count, failure_count FROM proven_intents WHERE intent_key=?",
                    (intent_key,),
                ).fetchone()
                now = time.time()
                if row:
                    sc = row["success_count"] + (1 if success else 0)
                    fc = row["failure_count"] + (0 if success else 1)
                    total = sc + fc
                    rate  = sc / total if total > 0 else 0.5
                    # Log-scaled boost: more evidence = more confidence, capped at max
                    import math
                    boost = min(_PROVEN_MAX_BOOST, rate * math.log1p(total) * 0.03)
                    self._conn.execute(
                        "UPDATE proven_intents SET success_count=?, failure_count=?, "
                        "boost_weight=?, last_updated=? WHERE intent_key=?",
                        (sc, fc, boost, now, intent_key),
                    )
                else:
                    self._conn.execute(
                        "INSERT INTO proven_intents VALUES (?,?,?,?,?,?)",
                        (intent_key, domain, 1 if success else 0, 0 if success else 1, 0.0, now),
                    )
                self._conn.commit()
        except Exception as e:
            logger.debug(f"record_proven_intent failed: {e}")

    async def get_proven_boosts(self) -> Dict[str, float]:
        """
        Return {intent_key: boost_weight} for all intents with sufficient evidence.
        Used by EME.warm() to boost pre-warmed templates validated cross-tenant.
        """
        try:
            async with self._lock:
                rows = self._conn.execute(
                    "SELECT intent_key, boost_weight FROM proven_intents "
                    "WHERE success_count >= ? AND "
                    "CAST(success_count AS REAL) / (success_count + failure_count + 1) >= ?",
                    (_PROVEN_MIN_SUCCESSES, _PROVEN_MIN_RATE),
                ).fetchall()
            return {r["intent_key"]: r["boost_weight"] for r in rows}
        except Exception as e:
            logger.debug(f"get_proven_boosts failed: {e}")
            return {}

    async def get_vocab_weights(self, concepts: List[str]) -> Dict[str, float]:
        """
        Returns {concept: weight} for requested concepts.
        Unknown concepts return 0.5 (neutral).
        """
        if not concepts:
            return {}
        result: Dict[str, float] = {c: 0.5 for c in concepts}
        try:
            async with self._lock:
                placeholders = ",".join("?" * len(concepts))
                rows = self._conn.execute(
                    f"SELECT concept, weight FROM vocab_weights WHERE concept IN ({placeholders})",
                    concepts,
                ).fetchall()
            for row in rows:
                result[row["concept"]] = row["weight"]
        except Exception as e:
            logger.debug(f"get_vocab_weights failed (non-critical): {e}")
        return result
