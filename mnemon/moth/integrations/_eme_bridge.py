"""
Bridge between moth integrations and the Mnemon EME.

MothCache provides a unified cache interface:
  1. BoundedTTLCache (hash-based, O(1), no I/O) — fastest path
  2. EME semantic index (embedding similarity) — catches same task, different phrasing

Sync paths use m._run() to call EME on Mnemon's private loop.
Async paths await EME directly — safe on Python 3.10+ where asyncio locks
are loop-agnostic.

When EME is unavailable (eme_enabled=False), falls back to hash cache only.
One MothCache instance per integration per Mnemon instance — caches are
tenant-isolated since the EME itself is per-tenant.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ._cache import BoundedTTLCache

logger = logging.getLogger(__name__)


class MothCache:
    """Unified hash + semantic cache for a single moth integration."""

    def __init__(self, m: Any, source: str, maxsize: int = 500, ttl: float = 3600.0) -> None:
        self._m = m
        self._source = source
        self._hash_cache = BoundedTTLCache(maxsize=maxsize, ttl=ttl)
        # EME template_id → original response/output object
        # (EME stores text; we need the real object to return to callers)
        self._obj_store: Dict[str, Any] = {}

    def _eme(self) -> Optional[Any]:
        try:
            if self._m._m and self._m._m._eme:
                return self._m._m._eme
        except Exception:
            pass
        return None

    # ── Sync ─────────────────────────────────────────────────────────────────

    def check(self, query: str, capabilities: List[str], hash_key: str) -> Optional[Any]:
        """Check hash cache then EME semantic. Returns stored object or None."""
        cached = self._hash_cache.get(hash_key)
        if cached is not None:
            return cached

        eme = self._eme()
        if eme and query:
            try:
                result = self._m._run(eme.semantic_lookup(query, capabilities))
                if result:
                    tid, text = result
                    obj = self._obj_store.get(tid)
                    if obj is not None:
                        logger.debug(f"Mnemon: {self._source} EME semantic hit")
                        return obj
                    if text:
                        # Cold start: process restarted, obj_store is empty but
                        # EME still has the text. Return text so caller can synthesize.
                        logger.debug(f"Mnemon: {self._source} EME cold-start text hit")
                        return text
            except Exception as e:
                logger.debug(f"Mnemon: {self._source} EME sync check failed — {e}")

        return None

    def store(
        self, query: str, capabilities: List[str], hash_key: str, obj: Any, text: str
    ) -> None:
        """Store in hash cache and EME."""
        self._hash_cache[hash_key] = obj

        eme = self._eme()
        if eme and query and text:
            try:
                tid = self._m._run(eme.cache_result(query, text, capabilities))
                if tid:
                    self._obj_store[tid] = obj
            except Exception as e:
                logger.debug(f"Mnemon: {self._source} EME sync store failed — {e}")

    # ── Async ─────────────────────────────────────────────────────────────────

    async def async_check(
        self, query: str, capabilities: List[str], hash_key: str
    ) -> Optional[Any]:
        """Async: check hash cache then EME semantic."""
        cached = self._hash_cache.get(hash_key)
        if cached is not None:
            return cached

        eme = self._eme()
        if eme and query:
            try:
                result = await eme.semantic_lookup(query, capabilities)
                if result:
                    tid, text = result
                    obj = self._obj_store.get(tid)
                    if obj is not None:
                        logger.debug(f"Mnemon: {self._source} EME async semantic hit")
                        return obj
                    if text:
                        logger.debug(f"Mnemon: {self._source} EME async cold-start text hit")
                        return text
            except Exception as e:
                logger.debug(f"Mnemon: {self._source} EME async check failed — {e}")

        return None

    async def async_store(
        self, query: str, capabilities: List[str], hash_key: str, obj: Any, text: str
    ) -> None:
        """Async: store in hash cache and EME."""
        self._hash_cache[hash_key] = obj

        eme = self._eme()
        if eme and query and text:
            try:
                tid = await eme.cache_result(query, text, capabilities)
                if tid:
                    self._obj_store[tid] = obj
            except Exception as e:
                logger.debug(f"Mnemon: {self._source} EME async store failed — {e}")
