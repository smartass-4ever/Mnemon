"""
Mnemon moth — Anthropic SDK integration.

Patches Messages.create and AsyncMessages.create to:
  1. Check MothCache (hash fast-path + EME semantic) — zero API call on hit
  2. Inject recalled memory into system prompt (protein bond gated)
  3. After real API call: store in MothCache + EME for future semantic hits
  4. Record outcome to experience bus

Streaming (stream=True):
  Cache hit  → returns _SyntheticAnthropicStream (yields compatible events from
               cached text — no API call, memory injection still fires)
  Cache miss → wraps real stream in _CapturingAnthropicStream which passes all
               events through while collecting text; stores to cache on exhaustion

Cache key hashes the ORIGINAL messages and system — not the patched system
(which includes injected memories and would change every run as memories grow).
"""

from __future__ import annotations

import logging
import types
from typing import Any, List, Optional

from mnemon.moth import MnemonIntegration
from ._utils import (
    extract_query, inject_into_system, prompt_hash,
    recall_as_context, recall_as_context_async, record_outcome, track_cache_hit,
)
from ._eme_bridge import MothCache
from ._cache import BoundedTTLCache

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

        m               = mnemon
        orig_sync        = self._original_sync
        orig_async       = self._original_async
        cache            = MothCache(m, "anthropic")
        stream_txt_cache = BoundedTTLCache(maxsize=500, ttl=3600)

        def patched_create(
            _self: Any,
            *,
            messages: list,
            model: str,
            system: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            is_stream = kwargs.get("stream", False)
            query     = extract_query(messages, system)
            hash_key  = prompt_hash(messages, system, model)  # original, not patched

            # ── streaming path ────────────────────────────────────────────────
            if is_stream:
                cached_text = stream_txt_cache.get(hash_key)
                if cached_text is not None:
                    track_cache_hit(m, "anthropic")
                    return _SyntheticAnthropicStream(cached_text, model)

                context       = recall_as_context(m, query, source="anthropic") if query else ""
                patched_system = inject_into_system(system, context)

                real_stream = orig_sync(
                    _self, messages=messages, model=model,
                    system=patched_system if patched_system else system,
                    **kwargs,
                )
                return _CapturingAnthropicStream(
                    real_stream, stream_txt_cache, hash_key, m, query
                )

            # ── non-streaming path ────────────────────────────────────────────
            cached = cache.check(query, [model], hash_key)
            if cached is not None:
                if isinstance(cached, str):
                    cached = _synthetic_anthropic_response(cached, model)
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
            is_stream = kwargs.get("stream", False)
            query     = extract_query(messages, system)
            hash_key  = prompt_hash(messages, system, model)

            # ── streaming path ────────────────────────────────────────────────
            if is_stream:
                cached_text = stream_txt_cache.get(hash_key)
                if cached_text is not None:
                    track_cache_hit(m, "anthropic")
                    return _SyntheticAnthropicStream(cached_text, model)

                context       = await recall_as_context_async(m, query, source="anthropic") if query else ""
                patched_system = inject_into_system(system, context)

                real_stream = await orig_async(
                    _self, messages=messages, model=model,
                    system=patched_system if patched_system else system,
                    **kwargs,
                )
                return _AsyncCapturingAnthropicStream(
                    real_stream, stream_txt_cache, hash_key, m, query
                )

            # ── non-streaming path ────────────────────────────────────────────
            cached = await cache.async_check(query, [model], hash_key)
            if cached is not None:
                if isinstance(cached, str):
                    cached = _synthetic_anthropic_response(cached, model)
                track_cache_hit(m, "anthropic", _anthropic_tokens(cached))
                return cached

            context       = await recall_as_context_async(m, query, source="anthropic") if query else ""
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


# ── Streaming helpers ─────────────────────────────────────────────────────────

def _make_event(type_: str, **attrs: Any) -> Any:
    return types.SimpleNamespace(type=type_, **attrs)


class _SyntheticAnthropicStream:
    """
    Fake Anthropic stream returned on a cache hit with stream=True.
    Yields just enough events to satisfy the two most common patterns:
      • for event in stream: event.type / event.delta.text
      • for text in stream.text_stream: ...
    """

    def __init__(self, text: str, model: str) -> None:
        self._text  = text
        self._model = model

    def _events(self):
        yield _make_event("message_start",
                          message=types.SimpleNamespace(model=self._model, role="assistant"))
        yield _make_event("content_block_start", index=0,
                          content_block=types.SimpleNamespace(type="text", text=""))
        yield _make_event("content_block_delta", index=0,
                          delta=types.SimpleNamespace(type="text_delta", text=self._text))
        yield _make_event("content_block_stop", index=0)
        yield _make_event("message_delta",
                          delta=types.SimpleNamespace(stop_reason="end_turn"),
                          usage=types.SimpleNamespace(output_tokens=len(self._text.split())))
        yield _make_event("message_stop")

    def __iter__(self):
        return self._events()

    async def __aiter__(self):
        for event in self._events():
            yield event

    @property
    def text_stream(self):
        yield self._text

    async def atext_stream(self):
        yield self._text

    def get_final_message(self) -> Any:
        return types.SimpleNamespace(
            role="assistant",
            model=self._model,
            content=[types.SimpleNamespace(type="text", text=self._text)],
            stop_reason="end_turn",
            usage=types.SimpleNamespace(input_tokens=0, output_tokens=0),
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class _CapturingAnthropicStream:
    """
    Wraps a real sync Anthropic stream. Passes all events through and captures
    text deltas. On exhaustion stores the assembled text to cache.
    """

    def __init__(
        self,
        stream: Any,
        cache: BoundedTTLCache,
        hash_key: str,
        m: Any,
        query: str,
    ) -> None:
        self._stream   = stream
        self._cache    = cache
        self._hash_key = hash_key
        self._m        = m
        self._query    = query
        self._chunks: list = []

    def __iter__(self):
        try:
            for event in self._stream:
                if getattr(event, "type", None) == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    text  = getattr(delta, "text", None)
                    if text:
                        self._chunks.append(text)
                yield event
        finally:
            self._flush()

    def _flush(self) -> None:
        text = "".join(self._chunks)
        if text:
            try:
                self._cache[self._hash_key] = text
                record_outcome(self._m, self._query, text[:400])
            except Exception:
                pass

    @property
    def text_stream(self):
        for event in self:
            if getattr(event, "type", None) == "content_block_delta":
                delta = getattr(event, "delta", None)
                text  = getattr(delta, "text", None)
                if text:
                    yield text

    def get_final_message(self) -> Any:
        return getattr(self._stream, "get_final_message", lambda: None)()

    def __enter__(self):
        self._stream.__enter__() if hasattr(self._stream, "__enter__") else None
        return self

    def __exit__(self, *args):
        if hasattr(self._stream, "__exit__"):
            self._stream.__exit__(*args)


class _AsyncCapturingAnthropicStream:
    """Async variant of _CapturingAnthropicStream."""

    def __init__(
        self,
        stream: Any,
        cache: BoundedTTLCache,
        hash_key: str,
        m: Any,
        query: str,
    ) -> None:
        self._stream   = stream
        self._cache    = cache
        self._hash_key = hash_key
        self._m        = m
        self._query    = query
        self._chunks: list = []

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        try:
            async for event in self._stream:
                if getattr(event, "type", None) == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    text  = getattr(delta, "text", None)
                    if text:
                        self._chunks.append(text)
                yield event
        finally:
            text = "".join(self._chunks)
            if text:
                try:
                    self._cache[self._hash_key] = text
                    record_outcome(self._m, self._query, text[:400])
                except Exception:
                    pass

    async def get_final_message(self) -> Any:
        fn = getattr(self._stream, "get_final_message", None)
        if fn:
            import inspect
            result = fn()
            if inspect.isawaitable(result):
                return await result
            return result
        return None


# ── Non-streaming helpers ─────────────────────────────────────────────────────

def _anthropic_text(response: Any) -> str:
    try:
        if hasattr(response, "content") and response.content:
            block = response.content[0]
            return getattr(block, "text", str(block))
    except Exception:
        pass
    return str(response)[:400]


def _synthetic_anthropic_response(text: str, model: str) -> Any:
    """Reconstruct a minimal Anthropic response from cached text (cold-start path)."""
    return types.SimpleNamespace(
        id="mnemon-cached-0",
        type="message",
        role="assistant",
        model=model,
        content=[types.SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=types.SimpleNamespace(input_tokens=0, output_tokens=0),
    )


def _anthropic_tokens(response: Any) -> Optional[int]:
    try:
        usage = getattr(response, "usage", None)
        if usage:
            return getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
    except Exception:
        pass
    return None
