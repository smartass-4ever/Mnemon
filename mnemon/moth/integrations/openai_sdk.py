"""
Mnemon moth — OpenAI SDK integration.

Same pattern as Anthropic: MothCache (hash + EME semantic), memory injection,
outcome recording. Cache key hashes original messages — not patched.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from mnemon.moth import MnemonIntegration
from ._utils import (
    extract_query, inject_into_openai_messages, prompt_hash,
    recall_as_context, record_outcome, track_cache_hit,
)
from ._eme_bridge import MothCache

logger = logging.getLogger(__name__)


class OpenAIIntegration(MnemonIntegration):
    name = "openai"

    def __init__(self) -> None:
        self._original_sync:  Optional[Any] = None
        self._original_async: Optional[Any] = None
        self._mnemon:         Optional[Any] = None

    def is_available(self) -> bool:
        import importlib.util
        return importlib.util.find_spec("openai") is not None

    def patch(self, mnemon: Any) -> None:
        import openai.resources.chat.completions as _mod

        self._mnemon = mnemon
        self._original_sync  = _mod.Completions.create
        self._original_async = _mod.AsyncCompletions.create

        m         = mnemon
        orig_sync  = self._original_sync
        orig_async = self._original_async
        cache      = MothCache(m, "openai")

        def patched_create(
            _self: Any,
            *,
            messages: List[Dict],
            model: str,
            **kwargs: Any,
        ) -> Any:
            query    = extract_query(messages)
            hash_key = prompt_hash(messages, None, model)  # original, not patched

            cached = cache.check(query, [model], hash_key)
            if cached is not None:
                track_cache_hit(m, "openai", _openai_tokens(cached))
                return cached

            context         = recall_as_context(m, query, source="openai") if query else ""
            patched_messages = inject_into_openai_messages(messages, context)

            response = orig_sync(_self, messages=patched_messages, model=model, **kwargs)

            text = _openai_text(response)
            cache.store(query, [model], hash_key, response, text)
            record_outcome(m, query, text)
            return response

        async def patched_async_create(
            _self: Any,
            *,
            messages: List[Dict],
            model: str,
            **kwargs: Any,
        ) -> Any:
            query    = extract_query(messages)
            hash_key = prompt_hash(messages, None, model)

            cached = await cache.async_check(query, [model], hash_key)
            if cached is not None:
                track_cache_hit(m, "openai", _openai_tokens(cached))
                return cached

            context         = recall_as_context(m, query, source="openai") if query else ""
            patched_messages = inject_into_openai_messages(messages, context)

            response = await orig_async(_self, messages=patched_messages, model=model, **kwargs)

            text = _openai_text(response)
            await cache.async_store(query, [model], hash_key, response, text)
            record_outcome(m, query, text)
            return response

        _mod.Completions.create      = patched_create
        _mod.AsyncCompletions.create = patched_async_create

    def unpatch(self) -> None:
        if self._original_sync is None:
            return
        try:
            import openai.resources.chat.completions as _mod
            _mod.Completions.create      = self._original_sync
            _mod.AsyncCompletions.create = self._original_async
        except Exception as e:
            logger.debug(f"Mnemon: OpenAI unpatch failed — {e}")
        finally:
            self._original_sync  = None
            self._original_async = None


def _openai_text(response: Any) -> str:
    try:
        choices = getattr(response, "choices", [])
        if choices:
            msg = getattr(choices[0], "message", None)
            return getattr(msg, "content", "") or ""
    except Exception:
        pass
    return str(response)[:400]


def _openai_tokens(response: Any) -> Optional[int]:
    try:
        usage = getattr(response, "usage", None)
        if usage:
            return getattr(usage, "total_tokens", None)
    except Exception:
        pass
    return None
