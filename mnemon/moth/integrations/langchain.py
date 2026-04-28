"""
Mnemon moth — LangChain integration.

System 2 EME via per-step patching:
  RunnableSequence.invoke is patched to iterate each step individually.
  Each step is a segment. If a step's output is cached (same input hash),
  it is returned directly and the LLM is not called for that step.
  Only changed steps call the LLM.

BaseChatModel.invoke / ainvoke:
  LLM-step caching for all providers (Groq, Anthropic, OpenAI, etc.)
  Cache keyed on original input — never patched versions.

Legacy Chain.__call__ (LangChain v0.1) gets chain-level System 1 cache.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Dict, List, Optional

from mnemon.moth import MnemonIntegration
from ._utils import extract_query, prompt_hash, track_cache_hit
from ._cache import BoundedTTLCache
from ._eme_bridge import MothCache

logger = logging.getLogger(__name__)


class LangChainIntegration(MnemonIntegration):
    name = "langchain"

    def __init__(self) -> None:
        self._original_runnable_invoke: Optional[Any] = None
        self._original_chain_call:      Optional[Any] = None
        self._original_chat_invoke:     Optional[Any] = None
        self._original_chat_ainvoke:    Optional[Any] = None
        self._mnemon:                   Optional[Any] = None

    def is_available(self) -> bool:
        return (
            importlib.util.find_spec("langchain_core") is not None
            or importlib.util.find_spec("langchain") is not None
        )

    def patch(self, mnemon: Any) -> None:
        self._mnemon = mnemon
        m = mnemon

        # Patch 1: LCEL RunnableSequence — per-step System 2
        try:
            from langchain_core.runnables.base import RunnableSequence
            self._original_runnable_invoke = RunnableSequence.invoke
            orig_invoke = self._original_runnable_invoke
            step_cache  = BoundedTTLCache(maxsize=500, ttl=3600)
            self._step_cache = step_cache

            def patched_invoke(_self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
                steps = getattr(_self, "steps", None)
                if not steps:
                    return orig_invoke(_self, input, config, **kwargs)

                current = input
                try:
                    for i, step in enumerate(steps):
                        if _is_chat_model(step):
                            current = step.invoke(current, config) if config is not None else step.invoke(current)
                            continue

                        step_key = _step_cache_key(step, current)
                        cached = step_cache.get(step_key)
                        if cached is not None:
                            track_cache_hit(m, f"langchain:step_{i}")
                            current = cached
                            continue

                        current = step.invoke(current, config) if config is not None else step.invoke(current)
                        step_cache[step_key] = current

                except Exception as e:
                    logger.debug(f"Mnemon: LangChain per-step failed at step {i} — {e}, falling back to whole-chain")
                    current = orig_invoke(_self, input, config, **kwargs)

                return current

            RunnableSequence.invoke = patched_invoke

        except Exception as e:
            logger.debug(f"Mnemon: LangChain RunnableSequence patch failed — {e}")

        # Patch 2: BaseChatModel.invoke — persistent EME cache for all providers
        # Uses MothCache (hash + EME semantic) instead of BoundedTTLCache so that
        # cache survives process restarts and System 2 semantic matching works.
        try:
            from langchain_core.language_models.chat_models import BaseChatModel
            self._original_chat_invoke = BaseChatModel.invoke
            orig_chat_invoke = self._original_chat_invoke
            llm_cache = MothCache(m, "langchain:llm")

            def patched_chat_invoke(_self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
                hash_key = _step_cache_key(_self, input)
                query = _extract_lc_messages(input)
                model_name = getattr(_self, "model_name", type(_self).__name__)
                try:
                    cached = llm_cache.check(query, [model_name], hash_key)
                    if cached is not None:
                        cached = _ensure_ai_message(cached)
                        track_cache_hit(m, "langchain:llm")
                        return cached
                except Exception:
                    pass

                result = orig_chat_invoke(_self, input, config, **kwargs)

                try:
                    text = result.content if hasattr(result, "content") else str(result)
                    llm_cache.store(query, [model_name], hash_key, result, text[:400])
                except Exception:
                    pass

                return result

            BaseChatModel.invoke = patched_chat_invoke

            # Patch 2b: BaseChatModel.ainvoke — async chains with persistent EME
            self._original_chat_ainvoke = BaseChatModel.ainvoke
            orig_chat_ainvoke = self._original_chat_ainvoke
            ainvoke_cache = MothCache(m, "langchain:llm_async")

            async def patched_chat_ainvoke(
                _self: Any, input: Any, config: Any = None, **kwargs: Any
            ) -> Any:
                hash_key = _step_cache_key(_self, input)
                query = _extract_lc_messages(input)
                model_name = getattr(_self, "model_name", type(_self).__name__)
                try:
                    cached = await ainvoke_cache.async_check(query, [model_name], hash_key)
                    if cached is not None:
                        cached = _ensure_ai_message(cached)
                        track_cache_hit(m, "langchain:llm_async")
                        return cached
                except Exception:
                    pass

                result = await orig_chat_ainvoke(_self, input, config, **kwargs)

                try:
                    text = result.content if hasattr(result, "content") else str(result)
                    await ainvoke_cache.async_store(query, [model_name], hash_key, result, text[:400])
                except Exception:
                    pass

                return result

            BaseChatModel.ainvoke = patched_chat_ainvoke

        except Exception as e:
            logger.debug(f"Mnemon: LangChain BaseChatModel patch failed — {e}")

        # Patch 3: Legacy Chain.__call__ — chain-level System 1
        try:
            from langchain.chains.base import Chain
            self._original_chain_call = Chain.__call__
            orig_call    = self._original_chain_call
            legacy_cache = MothCache(m, "langchain")

            def patched_chain_call(_self: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
                query    = _extract_lc_query(inputs)
                hash_key = _chain_cache_key(_self, inputs)

                cached = legacy_cache.check(query, [type(_self).__name__], hash_key)
                if cached is not None:
                    track_cache_hit(m, "langchain")
                    return cached

                result = orig_call(_self, inputs, *args, **kwargs)
                legacy_cache.store(query, [type(_self).__name__], hash_key, result, str(result)[:400])
                return result

            Chain.__call__ = patched_chain_call

        except Exception as e:
            logger.debug(f"Mnemon: LangChain Chain patch failed — {e}")

    def unpatch(self) -> None:
        try:
            if self._original_runnable_invoke is not None:
                from langchain_core.runnables.base import RunnableSequence
                RunnableSequence.invoke = self._original_runnable_invoke

            if self._original_chat_invoke is not None:
                from langchain_core.language_models.chat_models import BaseChatModel
                BaseChatModel.invoke = self._original_chat_invoke

            if self._original_chat_ainvoke is not None:
                from langchain_core.language_models.chat_models import BaseChatModel
                BaseChatModel.ainvoke = self._original_chat_ainvoke

            if self._original_chain_call is not None:
                from langchain.chains.base import Chain
                Chain.__call__ = self._original_chain_call
        except Exception as e:
            logger.debug(f"Mnemon: LangChain unpatch failed — {e}")
        finally:
            self._original_runnable_invoke = None
            self._original_chat_invoke     = None
            self._original_chat_ainvoke    = None
            self._original_chain_call      = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_lc_query(input: Any) -> str:
    if isinstance(input, str):
        return input[:300]
    if isinstance(input, dict):
        for key in ("input", "query", "question", "human_input", "messages"):
            val = input.get(key)
            if isinstance(val, str):
                return val[:300]
            if isinstance(val, list):
                return extract_query(val)
    return str(input)[:200]


def _step_cache_key(step: Any, input: Any) -> str:
    step_type  = type(step).__name__
    input_hash = prompt_hash([{"role": "user", "content": str(input)}], None, step_type)
    return f"step:{step_type}:{input_hash}"


def _chain_cache_key(chain: Any, input: Any) -> str:
    chain_type = type(chain).__name__
    input_hash = prompt_hash([{"role": "user", "content": str(input)}], None, chain_type)
    return f"{chain_type}:{input_hash}"


def _is_chat_model(step: Any) -> bool:
    try:
        from langchain_core.language_models.chat_models import BaseChatModel
        return isinstance(step, BaseChatModel)
    except Exception:
        return False


def _extract_lc_messages(input: Any) -> str:
    """
    Extract the user query string from any LangChain input format.
    Handles: plain strings, dicts, ChatPromptValue, list of BaseMessage objects.
    """
    if isinstance(input, str):
        return input[:300]
    if isinstance(input, dict):
        return _extract_lc_query(input)
    # ChatPromptValue or any object with a .messages attribute
    msgs = None
    if hasattr(input, "messages"):
        msgs = input.messages
    elif isinstance(input, (list, tuple)) and input:
        msgs = input
    if msgs:
        for msg in reversed(msgs):
            msg_type = getattr(msg, "type", "") or type(msg).__name__.lower()
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                if "human" in msg_type or "user" in msg_type:
                    return content[:300]
        # Fallback: last message content regardless of role
        last = msgs[-1] if msgs else None
        if last:
            content = getattr(last, "content", "")
            if isinstance(content, str):
                return content[:300]
    return str(input)[:200]


def _ensure_ai_message(cached: Any) -> Any:
    """
    Cold-start path: EME returns plain text (object store cleared on restart).
    Wrap it in AIMessage so LangChain callers get the right type back.
    """
    if not isinstance(cached, str):
        return cached
    try:
        from langchain_core.messages import AIMessage
        return AIMessage(content=cached)
    except Exception:
        return cached
