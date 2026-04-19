"""
Mnemon moth — Anthropic SDK integration.

Patches anthropic.resources.messages.Messages.create (sync) and
AsyncMessages.create (async) to:
  1. Inject recalled memory into the system prompt
  2. Return a cached response on System 1 EME hit (skip API call entirely)
  3. Cache the response after a real API call
  4. Record the outcome to the experience bus

Zero user code changes required.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from mnemon.moth import MnemonIntegration
from ._utils import (
    extract_query, inject_into_system, prompt_hash, recall_as_context, record_outcome
)

logger = logging.getLogger(__name__)

# Module-level cache: hash → raw response object (System 1 only at client level)
_response_cache: Dict[str, Any] = {}


class AnthropicIntegration(MnemonIntegration):
    """
    Instruments the Anthropic Python SDK.

    Hooks into Messages.create and AsyncMessages.create so every
    Anthropic call — regardless of framework — gets Mnemon intelligence.
    """

    name = "anthropic"

    def __init__(self) -> None:
        self._original_sync: Optional[Any] = None
        self._original_async: Optional[Any] = None
        self._mnemon: Optional[Any] = None

    def is_available(self) -> bool:
        import importlib.util
        return importlib.util.find_spec("anthropic") is not None

    def patch(self, mnemon: Any) -> None:
        import anthropic.resources.messages as _mod

        self._mnemon = mnemon
        self._original_sync = _mod.Messages.create
        self._original_async = _mod.AsyncMessages.create

        m = mnemon
        orig_sync = self._original_sync
        orig_async = self._original_async

        def patched_create(
            _self: Any,
            *,
            messages: list,
            model: str,
            system: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            query = extract_query(messages, system)
            context = recall_as_context(m, query) if query else ""
            patched_system = inject_into_system(system, context)

            # System 1 EME — return cached response if available
            key = prompt_hash(messages, patched_system, model)
            if key in _response_cache:
                logger.debug("Mnemon: Anthropic System 1 cache hit")
                return _response_cache[key]

            response = orig_sync(
                _self,
                messages=messages,
                model=model,
                system=patched_system if patched_system else system,
                **kwargs,
            )

            _response_cache[key] = response
            _record_anthropic_outcome(m, query, response)
            return response

        async def patched_async_create(
            _self: Any,
            *,
            messages: list,
            model: str,
            system: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            query = extract_query(messages, system)
            context = recall_as_context(m, query) if query else ""
            patched_system = inject_into_system(system, context)

            key = prompt_hash(messages, patched_system, model)
            if key in _response_cache:
                logger.debug("Mnemon: Anthropic async System 1 cache hit")
                return _response_cache[key]

            response = await orig_async(
                _self,
                messages=messages,
                model=model,
                system=patched_system if patched_system else system,
                **kwargs,
            )

            _response_cache[key] = response
            _record_anthropic_outcome(m, query, response)
            return response

        _mod.Messages.create = patched_create
        _mod.AsyncMessages.create = patched_async_create

    def unpatch(self) -> None:
        if self._original_sync is None:
            return
        try:
            import anthropic.resources.messages as _mod
            _mod.Messages.create = self._original_sync
            _mod.AsyncMessages.create = self._original_async
        except Exception as e:
            logger.debug(f"Mnemon: Anthropic unpatch failed — {e}")
        finally:
            self._original_sync = None
            self._original_async = None


def _record_anthropic_outcome(m: Any, query: str, response: Any) -> None:
    try:
        text = ""
        if hasattr(response, "content") and response.content:
            block = response.content[0]
            text = getattr(block, "text", str(block))
        record_outcome(m, query, text)
    except Exception as e:
        logger.debug(f"Mnemon: Anthropic outcome record failed — {e}")
