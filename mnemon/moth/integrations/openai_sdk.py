"""
Mnemon moth — OpenAI SDK integration.

Patches ChatCompletion.create (sync + async):
  Cache hit  → returns cached response, zero API call
  Cache miss → calls real API, stores result for next run

Streaming (stream=True):
  Cache hit  → returns _SyntheticOpenAIStream (compatible ChatCompletionChunk iterator)
  Cache miss → wraps real stream in _CapturingOpenAIStream; stores text on exhaustion

Cache key hashes original messages — never any patched version.
"""

from __future__ import annotations

import logging
import types
from typing import Any, Dict, List, Optional

from mnemon.moth import MnemonIntegration
from ._utils import extract_query, prompt_hash, track_cache_hit
from ._eme_bridge import MothCache
from ._cache import BoundedTTLCache

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

        m                = mnemon
        orig_sync        = self._original_sync
        orig_async       = self._original_async
        cache            = MothCache(m, "openai")
        stream_txt_cache = BoundedTTLCache(maxsize=500, ttl=3600)

        def patched_create(
            _self: Any,
            *,
            messages: List[Dict],
            model: str,
            **kwargs: Any,
        ) -> Any:
            is_stream = kwargs.get("stream", False)
            query    = extract_query(messages)
            hash_key = prompt_hash(messages, None, model)

            # ── streaming path ────────────────────────────────────────────────
            if is_stream:
                cached_text = stream_txt_cache.get(hash_key)
                if cached_text is not None:
                    track_cache_hit(m, "openai")
                    return _SyntheticOpenAIStream(cached_text, model)

                real_stream = orig_sync(_self, messages=messages, model=model, **kwargs)
                return _CapturingOpenAIStream(real_stream, stream_txt_cache, hash_key, m)

            # ── non-streaming path ────────────────────────────────────────────
            cached = cache.check(query, [model], hash_key)
            if cached is not None:
                if isinstance(cached, str):
                    cached = _synthetic_openai_response(cached, model)
                total, inp, out = _openai_tokens(cached)
                track_cache_hit(m, "openai", total, model, inp, out)
                return cached

            response = orig_sync(_self, messages=messages, model=model, **kwargs)
            text = _openai_text(response)
            cache.store(query, [model], hash_key, response, text)
            return response

        async def patched_async_create(
            _self: Any,
            *,
            messages: List[Dict],
            model: str,
            **kwargs: Any,
        ) -> Any:
            is_stream = kwargs.get("stream", False)
            query    = extract_query(messages)
            hash_key = prompt_hash(messages, None, model)

            # ── streaming path ────────────────────────────────────────────────
            if is_stream:
                cached_text = stream_txt_cache.get(hash_key)
                if cached_text is not None:
                    track_cache_hit(m, "openai")
                    return _SyntheticOpenAIStream(cached_text, model)

                real_stream = await orig_async(_self, messages=messages, model=model, **kwargs)
                return _AsyncCapturingOpenAIStream(real_stream, stream_txt_cache, hash_key, m)

            # ── non-streaming path ────────────────────────────────────────────
            cached = await cache.async_check(query, [model], hash_key)
            if cached is not None:
                if isinstance(cached, str):
                    cached = _synthetic_openai_response(cached, model)
                total, inp, out = _openai_tokens(cached)
                track_cache_hit(m, "openai", total, model, inp, out)
                return cached

            response = await orig_async(_self, messages=messages, model=model, **kwargs)
            text = _openai_text(response)
            await cache.async_store(query, [model], hash_key, response, text)
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


# ── Streaming helpers ─────────────────────────────────────────────────────────

def _chunk(content: Optional[str], finish_reason: Optional[str], model: str) -> Any:
    return types.SimpleNamespace(
        id="cached-0", object="chat.completion.chunk", model=model,
        choices=[types.SimpleNamespace(
            index=0,
            delta=types.SimpleNamespace(role="assistant" if content else None, content=content),
            finish_reason=finish_reason,
        )],
    )


class _SyntheticOpenAIStream:
    """Fake OpenAI stream returned on a cache hit with stream=True."""

    def __init__(self, text: str, model: str) -> None:
        self._text  = text
        self._model = model

    def _chunks(self):
        yield _chunk(self._text, None, self._model)
        yield _chunk(None, "stop", self._model)

    def __iter__(self):
        return self._chunks()

    async def __aiter__(self):
        for chunk in self._chunks():
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class _CapturingOpenAIStream:
    """Wraps a real sync OpenAI stream. Captures delta content, stores to cache on exhaustion."""

    def __init__(self, stream: Any, cache: BoundedTTLCache, hash_key: str, m: Any) -> None:
        self._stream   = stream
        self._cache    = cache
        self._hash_key = hash_key
        self._m        = m
        self._chunks: list = []

    def __iter__(self):
        try:
            for chunk in self._stream:
                try:
                    content = chunk.choices[0].delta.content
                    if content:
                        self._chunks.append(content)
                except (AttributeError, IndexError):
                    pass
                yield chunk
        finally:
            self._flush()

    def _flush(self) -> None:
        text = "".join(self._chunks)
        if text:
            try:
                self._cache[self._hash_key] = text
            except Exception:
                pass

    def __enter__(self):
        self._stream.__enter__() if hasattr(self._stream, "__enter__") else None
        return self

    def __exit__(self, *args):
        if hasattr(self._stream, "__exit__"):
            self._stream.__exit__(*args)


class _AsyncCapturingOpenAIStream:
    """Async variant of _CapturingOpenAIStream."""

    def __init__(self, stream: Any, cache: BoundedTTLCache, hash_key: str, m: Any) -> None:
        self._stream   = stream
        self._cache    = cache
        self._hash_key = hash_key
        self._m        = m
        self._chunks: list = []

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        try:
            async for chunk in self._stream:
                try:
                    content = chunk.choices[0].delta.content
                    if content:
                        self._chunks.append(content)
                except (AttributeError, IndexError):
                    pass
                yield chunk
        finally:
            text = "".join(self._chunks)
            if text:
                try:
                    self._cache[self._hash_key] = text
                except Exception:
                    pass


# ── Non-streaming helpers ─────────────────────────────────────────────────────

def _synthetic_openai_response(text: str, model: str) -> Any:
    import time as _time
    return types.SimpleNamespace(
        id="mnemon-cached-0", object="chat.completion",
        created=int(_time.time()), model=model, system_fingerprint=None,
        choices=[types.SimpleNamespace(
            index=0,
            message=types.SimpleNamespace(role="assistant", content=text),
            finish_reason="stop", logprobs=None,
        )],
        usage=types.SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )


def _openai_text(response: Any) -> str:
    try:
        choices = getattr(response, "choices", [])
        if choices:
            msg = getattr(choices[0], "message", None)
            return getattr(msg, "content", "") or ""
    except Exception:
        pass
    return str(response)[:400]


def _openai_tokens(response: Any) -> tuple:
    try:
        usage = getattr(response, "usage", None)
        if usage:
            inp = getattr(usage, "prompt_tokens", 0) or 0
            out = getattr(usage, "completion_tokens", 0) or 0
            total = getattr(usage, "total_tokens", None) or (inp + out)
            return total, inp, out
    except Exception:
        pass
    return None, None, None
