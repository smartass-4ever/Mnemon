"""
Mnemon System Database
Separate SQLite store for Retrospector analysis data.
Completely isolated from EROSDatabase — never imports from persistence.py.

Architecture by Mahika Jadhav (smartass-4ever).
"""

import asyncio
import json
import logging
import sqlite3
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decision_traces (
    trace_id             TEXT PRIMARY KEY,
    tenant_id            TEXT NOT NULL,
    task_id              TEXT NOT NULL,
    goal_hash            TEXT NOT NULL,
    fragment_ids_used    TEXT NOT NULL,
    memory_ids_retrieved TEXT NOT NULL,
    segments_generated   TEXT NOT NULL,
    tools_called         TEXT NOT NULL,
    step_outcomes        TEXT NOT NULL,
    overall_outcome      TEXT NOT NULL,
    latency_ms           REAL NOT NULL,
    timestamp            REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS failure_patterns (
    pattern_id    TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    fragment_id   TEXT NOT NULL,
    failure_count INTEGER NOT NULL DEFAULT 1,
    last_seen     REAL NOT NULL,
    confirmed     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS quarantine_list (
    item_type  TEXT NOT NULL,
    item_id    TEXT NOT NULL,
    tenant_id  TEXT NOT NULL,
    reason     TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    PRIMARY KEY (item_id, tenant_id)
);

CREATE TABLE IF NOT EXISTS findings (
    finding_id  TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    affected_id TEXT NOT NULL,
    summary     TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""


class SystemDatabase:
    """
    Standalone SQLite database for retrospector data.
    Schema: decision_traces, failure_patterns, quarantine_list, findings.

    Fully isolated from EROSDatabase — uses a different file and never imports
    from persistence.py. All methods have try/except so failures are logged but
    never propagate to callers.
    """

    def __init__(self, db_path: str = "mnemon_system.db"):
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
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            logger.info(f"SystemDatabase connected: {self._db_path}")
        except Exception as e:
            logger.error(f"SystemDatabase connect failed: {e}")

    async def disconnect(self):
        try:
            if self._conn:
                self._conn.close()
                self._conn = None
                logger.info("SystemDatabase disconnected")
        except Exception as e:
            logger.error(f"SystemDatabase disconnect error: {e}")

    # ──────────────────────────────────────────
    # DECISION TRACES
    # ──────────────────────────────────────────

    async def write_trace(self, trace: dict):
        """Persist a DecisionTrace dict to decision_traces table."""
        try:
            async with self._lock:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO decision_traces
                    (trace_id, tenant_id, task_id, goal_hash,
                     fragment_ids_used, memory_ids_retrieved, segments_generated,
                     tools_called, step_outcomes, overall_outcome, latency_ms, timestamp)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        trace["trace_id"],
                        trace["tenant_id"],
                        trace["task_id"],
                        trace["goal_hash"],
                        json.dumps(trace.get("fragment_ids_used", [])),
                        json.dumps(trace.get("memory_ids_retrieved", [])),
                        json.dumps(trace.get("segments_generated", [])),
                        json.dumps(trace.get("tools_called", [])),
                        json.dumps(trace.get("step_outcomes", {})),
                        trace["overall_outcome"],
                        float(trace.get("latency_ms", 0.0)),
                        float(trace.get("timestamp", time.time())),
                    ),
                )
                self._conn.commit()
        except Exception as e:
            logger.error(f"write_trace failed: {e}")

    async def fetch_traces_by_fragment(
        self, fragment_id: str, days: int = 7
    ) -> List[dict]:
        """
        Return all traces that used fragment_id within the last `days` days.
        Searches the JSON-encoded fragment_ids_used column via LIKE.
        """
        try:
            cutoff = time.time() - days * 86400
            async with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT * FROM decision_traces
                    WHERE fragment_ids_used LIKE ?
                      AND timestamp >= ?
                    ORDER BY timestamp DESC
                    """,
                    (f'%"{fragment_id}"%', cutoff),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"fetch_traces_by_fragment failed: {e}")
            return []

    # ──────────────────────────────────────────
    # QUARANTINE
    # ──────────────────────────────────────────

    async def quarantine(
        self,
        item_type: str,
        item_id: str,
        tenant_id: str,
        reason: str,
        confidence: float,
        ttl_hours: int = 168,
    ):
        """Add or refresh a quarantine entry. Default TTL: 7 days."""
        try:
            now = time.time()
            expires_at = now + ttl_hours * 3600
            async with self._lock:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO quarantine_list
                    (item_type, item_id, tenant_id, reason, confidence, created_at, expires_at)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (item_type, item_id, tenant_id, reason, confidence, now, expires_at),
                )
                self._conn.commit()
            logger.info(
                f"Quarantined {item_type} {item_id} "
                f"[tenant={tenant_id}, ttl={ttl_hours}h, confidence={confidence:.2f}]"
            )
        except Exception as e:
            logger.error(f"quarantine failed: {e}")

    async def is_quarantined(self, item_id: str, tenant_id: str) -> bool:
        """Return True if item_id is currently quarantined for this tenant."""
        try:
            now = time.time()
            async with self._lock:
                row = self._conn.execute(
                    """
                    SELECT 1 FROM quarantine_list
                    WHERE item_id = ? AND tenant_id = ? AND expires_at > ?
                    LIMIT 1
                    """,
                    (item_id, tenant_id, now),
                ).fetchone()
            return row is not None
        except Exception as e:
            logger.error(f"is_quarantined failed: {e}")
            return False

    async def expire_quarantines(self):
        """Delete all quarantine entries whose expires_at is in the past."""
        try:
            now = time.time()
            async with self._lock:
                cursor = self._conn.execute(
                    "DELETE FROM quarantine_list WHERE expires_at <= ?", (now,)
                )
                self._conn.commit()
            logger.debug(f"expire_quarantines: removed {cursor.rowcount} expired entries")
        except Exception as e:
            logger.error(f"expire_quarantines failed: {e}")

    # ──────────────────────────────────────────
    # FINDINGS
    # ──────────────────────────────────────────

    async def write_finding(self, finding: dict):
        """Persist a retrospector finding."""
        try:
            async with self._lock:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO findings
                    (finding_id, tenant_id, signal_type, affected_id, summary, created_at)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (
                        finding["finding_id"],
                        finding["tenant_id"],
                        finding["signal_type"],
                        finding["affected_id"],
                        finding["summary"],
                        float(finding.get("created_at", time.time())),
                    ),
                )
                self._conn.commit()
        except Exception as e:
            logger.error(f"write_finding failed: {e}")
