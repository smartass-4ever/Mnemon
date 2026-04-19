"""
Mnemon moth — Anthropic SDK integration.

Patches Messages.create and AsyncMessages.create to:
  1. Check MothCache (hash fast-path + EME semantic) — zero API call on hit
  2. Inject recalled memory into system prompt (protein bond gated)
  3. After real API call: store in MothCache + EME for future semantic hits
  4. Record outcome to experience bus

Cache key hashes the ORIGINAL messages and system — not the patched system
(which includes injected memories and would change every run as memories grow).
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from mnemon.moth import MnemonIntegration
from ._utils import (
    extract_query, inject_into_system, prompt_hash,
    recall_as_context, record_outcome, track_cache_hit,
)
from ._eme_bridge import MothCache

logger = logging.getLogger(__name__)


class AnthropicIntegration(MnemonIntegration):
    name = "anthropic"

    def __init__(self) -> None:
        self._original_sync:  Optional[Any] = None
        self._original_async: Optional[Any] = None
        self._mnemon:         Optional[Any] = None

    def is_available(self) -> bool:
        import importlib.util
        return importlib.util.find_spec("anthropic") is not None

    def patch(self, mnemon: Any) -> None:
        import anthropic.resources.messages as _mod

        self._mnemon = mnemon
        self._original_sync  = _mod.Messages.create
        self._original_async = _mod.AsyncMessages.create

        m         = mnemon
        orig_sync  = self._original_sync
        orig_async = self._original_async
        cache      = MothCache(m, "anthropic")

        def patched_create(
            _self: Any,
            *,
            messages: list,
            model: str,
            system: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            query    = extract_query(messages, system)
            hash_key = prompt_hash(messages, system, model)  # original, not patched

            cached = cache.check(query, [model], hash_key)
            if cached is not None:
                track_cache_hit(m, "anthropic", _anthropic_tokens(cached))
                return cached

            context       = recall_as_context(m, query, source="anthropic") if query else ""
            patched_system = inject_into_system(system, context)

            response = orig_sync(
                _self, messages=messages, model=model,
                system=patched_system if patched_system else system,
                **kwargs,
            )

            text = _anthropic_text(response)
            cache.store(query, [model], hash_key, response, text)
            record_outcome(m, query, text)
            return response

        async def patched_async_create(
            _self: Any,
            *,
            messages: list,
            model: str,
            system: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            query    = extract_query(messages, system)
            hash_key = prompt_hash(messages, system, model)

            cached = await cache.async_check(query, [model], hash_key)
            if cached is not None:
                track_cache_hit(m, "anthropic", _anthropic_tokens(cached))
                return cached

            context       = recall_as_context(m, query, source="anthropic") if query else ""
            patched_system = inject_into_system(system, context)

            response = await orig_async(
                _self, messages=messages, model=model,
                system=patched_system if patched_system else system,
                **kwargs,
            )

            text = _anthropic_text(response)
            await cache.async_store(query, [model], hash_key, response, text)
            record_outcome(m, query, text)
            return response

        _mod.Messages.create      = patched_create
        _mod.AsyncMessages.create = patched_async_create

    def unpatch(self) -> None:
        if self._original_sync is None:
            return
        try:
            import anthropic.resources.messages as _mod
            _mod.Messages.create      = self._original_sync
            _mod.AsyncMessages.create = self._original_async
        except Exception as e:
            logger.debug(f"Mnemon: Anthropic unpatch failed — {e}")
        finally:
            self._original_sync  = None
            self._original_async = None


def _anthropic_text(response: Any) -> str:
    try:
        if hasattr(response, "content") and response.content:
            block = response.content[0]
            return getattr(block, "text", str(block))
    except Exception:
        pass
    return str(response)[:400]


def _anthropic_tokens(response: Any) -> Optional[int]:
    try:
        usage = getattr(response, "usage", None)
        if usage:
            return getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
    except Exception:
        pass
    return None
