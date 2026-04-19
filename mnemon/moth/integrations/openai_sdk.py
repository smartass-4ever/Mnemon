"""
Mnemon moth — OpenAI SDK integration.

Patches openai.resources.chat.completions.Completions.create (sync) and
AsyncCompletions.create (async) to:
  1. Inject recalled memory as a system message
  2. Return a cached response on System 1 EME hit
  3. Cache the response after a real API call
  4. Record the outcome to the experience bus

Zero user code changes required.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from mnemon.moth import MnemonIntegration
from ._utils import (
    extract_query, inject_into_openai_messages, prompt_hash,
    recall_as_context, record_outcome,
)

logger = logging.getLogger(__name__)

_response_cache: Dict[str, Any] = {}


class OpenAIIntegration(MnemonIntegration):
    """
    Instruments the OpenAI Python SDK.

    Hooks into Completions.create and AsyncCompletions.create so every
    OpenAI chat call gets Mnemon intelligence.
    """

    name = "openai"

    def __init__(self) -> None:
        self._original_sync: Optional[Any] = None
        self._original_async: Optional[Any] = None
        self._mnemon: Optional[Any] = None

    def is_available(self) -> bool:
        import importlib.util
        return importlib.util.find_spec("openai") is not None

    def patch(self, mnemon: Any) -> None:
        import openai.resources.chat.completions as _mod

        self._mnemon = mnemon
        self._original_sync = _mod.Completions.create
        self._original_async = _mod.AsyncCompletions.create

        m = mnemon
        orig_sync = self._original_sync
        orig_async = self._original_async

        def patched_create(
            _self: Any,
            *,
            messages: List[Dict],
            model: str,
            **kwargs: Any,
        ) -> Any:
            query = extract_query(messages)
            context = recall_as_context(m, query) if query else ""
            patched_messages = inject_into_openai_messages(messages, context)

            key = prompt_hash(patched_messages, None, model)
            if key in _response_cache:
                logger.debug("Mnemon: OpenAI System 1 cache hit")
                return _response_cache[key]

            response = orig_sync(_self, messages=patched_messages, model=model, **kwargs)

            _response_cache[key] = response
            _record_openai_outcome(m, query, response)
            return response

        async def patched_async_create(
            _self: Any,
            *,
            messages: List[Dict],
            model: str,
            **kwargs: Any,
        ) -> Any:
            query = extract_query(messages)
            context = recall_as_context(m, query) if query else ""
            patched_messages = inject_into_openai_messages(messages, context)

            key = prompt_hash(patched_messages, None, model)
            if key in _response_cache:
                logger.debug("Mnemon: OpenAI async System 1 cache hit")
                return _response_cache[key]

            response = await orig_async(_self, messages=patched_messages, model=model, **kwargs)

            _response_cache[key] = response
            _record_openai_outcome(m, query, response)
            return response

        _mod.Completions.create = patched_create
        _mod.AsyncCompletions.create = patched_async_create

    def unpatch(self) -> None:
        if self._original_sync is None:
            return
        try:
            import openai.resources.chat.completions as _mod
            _mod.Completions.create = self._original_sync
            _mod.AsyncCompletions.create = self._original_async
        except Exception as e:
            logger.debug(f"Mnemon: OpenAI unpatch failed — {e}")
        finally:
            self._original_sync = None
            self._original_async = None


def _record_openai_outcome(m: Any, query: str, response: Any) -> None:
    try:
        text = ""
        choices = getattr(response, "choices", [])
        if choices:
            msg = getattr(choices[0], "message", None)
            text = getattr(msg, "content", "") or ""
        record_outcome(m, query, text)
    except Exception as e:
        logger.debug(f"Mnemon: OpenAI outcome record failed — {e}")
