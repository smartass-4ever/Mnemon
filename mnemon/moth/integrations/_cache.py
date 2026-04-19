"""
Bounded TTL cache for moth integrations.
LRU eviction at maxsize. Entries expire after ttl seconds.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Optional


class BoundedTTLCache:
    """
    Thread-unsafe LRU cache with per-entry TTL.
    Default: 500 entries, 1-hour TTL.
    Thread safety not needed — all integrations run in a single agent loop.
    """

    def __init__(self, maxsize: int = 500, ttl: float = 3600.0) -> None:
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._times: dict[str, float] = {}
        self.maxsize = maxsize
        self.ttl = ttl

    def _expired(self, key: str) -> bool:
        return time.time() - self._times.get(key, 0) > self.ttl

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str) or key not in self._cache:
            return False
        if self._expired(key):
            del self._cache[key]
            del self._times[key]
            return False
        self._cache.move_to_end(key)
        return True

    def __getitem__(self, key: str) -> Any:
        self._cache.move_to_end(key)
        return self._cache[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        self._times[key] = time.time()
        if len(self._cache) > self.maxsize:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
            self._times.pop(oldest, None)

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self else default

    def clear(self) -> None:
        self._cache.clear()
        self._times.clear()

    def __len__(self) -> int:
        return len(self._cache)
