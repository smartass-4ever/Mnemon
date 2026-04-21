"""
Mnemon moth — LangChain integration.

System 2 EME via per-step patching:
  RunnableSequence.invoke is patched to iterate each step individually.
  Each step is a segment. If a step's output is cached (same input hash),
  it is returned directly and the LLM is not called for that step.
  Only changed steps call the LLM — identical to LangGraph per-node cache.

  Memory recall fires once per chain invocation (first step) so the
  context is available to guide the chain without per-step noise.

  Falls back to whole-chain invocation if per-step execution errors.

Legacy Chain.__call__ (LangChain v0.1) gets chain-level System 1 cache.

MnemonMemory:
  Factory for a LangChain BaseMemory backed by Mnemon. One line to add
  persistent memory to any ConversationChain.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Dict, List, Optional

from mnemon.moth import MnemonIntegration
from ._utils import extract_query, prompt_hash, record_outcome, track_cache_hit, recall_as_context
from ._cache import BoundedTTLCache
from ._eme_bridge import MothCache

logger = logging.getLogger(__name__)


class LangChainIntegration(MnemonIntegration):
    """
    Instruments LangChain at the per-step level for true System 2 EME.
    """

    name = "langchain"

    def __init__(self) -> None:
        self._original_runnable_invoke: Optional[Any] = None
        self._original_chain_call:      Optional[Any] = None
        self._original_chat_invoke:     Optional[Any] = None
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
            orig_invoke  = self._original_runnable_invoke
            # Per-step uses hash-only cache — EME semantic lookup causes cross-step
            # false positives (step N's output mistaken for step N+1's cache hit).
            step_cache   = BoundedTTLCache(maxsize=500, ttl=3600)
            self._step_cache = step_cache
            chain_cache  = BoundedTTLCache(maxsize=500, ttl=3600)  # legacy fallback

            def patched_invoke(_self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
                query = _extract_lc_query(input)

                steps = getattr(_self, "steps", None)
                if not steps:
                    result = orig_invoke(_self, input, config, **kwargs)
                    record_outcome(m, query, str(result)[:400])
                    return result

                # Per-step execution with System 2 cache.
                # LLM steps (BaseChatModel) are intentionally NOT step-cached here —
                # they flow through patched_chat_invoke which handles both caching
                # and memory injection in the right order.
                current = input
                try:
                    for i, step in enumerate(steps):
                        if _is_chat_model(step):
                            # Let BaseChatModel patch handle cache + injection
                            if config is not None:
                                current = step.invoke(current, config)
                            else:
                                current = step.invoke(current)
                            continue

                        step_key = _step_cache_key(step, current)
                        cached = step_cache.get(step_key)
                        if cached is not None:
                            track_cache_hit(m, f"langchain:step_{i}")
                            current = cached
                            continue

                        if config is not None:
                            current = step.invoke(current, config)
                        else:
                            current = step.invoke(current)

                        step_cache[step_key] = current

                except Exception as e:
                    logger.debug(
                        f"Mnemon: LangChain per-step failed at step {i} — {e}, "
                        f"falling back to whole-chain"
                    )
                    current = orig_invoke(_self, input, config, **kwargs)
                    record_outcome(m, query, str(current)[:400])

                return current

            RunnableSequence.invoke = patched_invoke

        except Exception as e:
            logger.debug(f"Mnemon: LangChain RunnableSequence patch failed — {e}")

        # Patch 2: BaseChatModel.invoke — cache + memory injection for ALL providers
        # (Groq, Anthropic via langchain-anthropic, OpenAI via langchain-openai, etc.)
        # Owns LLM-step caching so RunnableSequence doesn't bypass injection.
        # Cache is keyed on ORIGINAL messages (before injection) so memory growth
        # doesn't invalidate cached LLM responses.
        try:
            from langchain_core.language_models.chat_models import BaseChatModel
            self._original_chat_invoke = BaseChatModel.invoke
            orig_chat_invoke = self._original_chat_invoke
            llm_cache = BoundedTTLCache(maxsize=500, ttl=3600)

            def patched_chat_invoke(_self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
                query = ""
                recall_query = ""
                hash_key = None
                try:
                    messages     = _messages_from_prompt_value(input)
                    # recall_query = system + human (full intent signal for injection)
                    # query        = human only (clean goal for memory storage)
                    recall_query = _query_from_messages(messages)
                    query        = _human_query_from_messages(messages) or recall_query
                    hash_key     = _step_cache_key(_self, input)

                    cached = llm_cache.get(hash_key)
                    if cached is not None:
                        track_cache_hit(m, "langchain:llm")
                        return cached

                    context = recall_as_context(m, recall_query, source="langchain") if recall_query else ""
                    if context:
                        input = _inject_context_into_prompt_value(input, messages, context)
                except Exception:
                    pass

                result = orig_chat_invoke(_self, input, config, **kwargs)

                try:
                    if hash_key is not None:
                        llm_cache[hash_key] = result
                    if query:
                        text = getattr(result, "content", None) or str(result)
                        if isinstance(text, list):
                            text = " ".join(str(b) for b in text)
                        record_outcome(m, query, str(text)[:400])
                except Exception:
                    pass

                return result

            BaseChatModel.invoke = patched_chat_invoke

        except Exception as e:
            logger.debug(f"Mnemon: LangChain BaseChatModel patch failed — {e}")

        # Patch 3: Legacy Chain.__call__ — chain-level System 1
        try:
            from langchain.chains.base import Chain
            self._original_chain_call = Chain.__call__
            orig_call   = self._original_chain_call
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
                record_outcome(m, query, str(result)[:400])
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

            if self._original_chain_call is not None:
                from langchain.chains.base import Chain
                Chain.__call__ = self._original_chain_call
        except Exception as e:
            logger.debug(f"Mnemon: LangChain unpatch failed — {e}")
        finally:
            self._original_runnable_invoke = None
            self._original_chat_invoke = None
            self._original_chain_call = None


# ── Helpers ──────────────────────────────────────────────────────────────────

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
    step_type = type(step).__name__
    input_hash = prompt_hash(
        [{"role": "user", "content": str(input)}], None, step_type
    )
    return f"step:{step_type}:{input_hash}"


def _chain_cache_key(chain: Any, input: Any) -> str:
    chain_type = type(chain).__name__
    input_hash = prompt_hash(
        [{"role": "user", "content": str(input)}], None, chain_type
    )
    return f"{chain_type}:{input_hash}"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_chat_model(step: Any) -> bool:
    try:
        from langchain_core.language_models.chat_models import BaseChatModel
        return isinstance(step, BaseChatModel)
    except Exception:
        return False


# ── BaseChatModel injection helpers ──────────────────────────────────────────

def _messages_from_prompt_value(input: Any) -> List[Any]:
    """Extract BaseMessage list from a PromptValue or raw message list."""
    try:
        from langchain_core.prompt_values import PromptValue
        if isinstance(input, PromptValue):
            return input.to_messages()
    except Exception:
        pass
    if isinstance(input, list):
        return input
    return []


def _human_query_from_messages(messages: List[Any]) -> str:
    """Return only the last human message — used as clean goal for memory storage."""
    try:
        from langchain_core.messages import HumanMessage
        human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if human and isinstance(human.content, str):
            return human.content[:300].strip()
    except Exception:
        pass
    return ""


def _query_from_messages(messages: List[Any]) -> str:
    """Pull last human/user message as the recall query."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
        )
        system = next(
            (m for m in messages if isinstance(m, SystemMessage)), None
        )
        parts = []
        if system and isinstance(system.content, str):
            parts.append(system.content[:200])
        if human and isinstance(human.content, str):
            parts.append(human.content[:200])
        return " ".join(parts).strip()[:350]
    except Exception:
        return ""


def _inject_context_into_prompt_value(input: Any, messages: List[Any], context: str) -> Any:
    """Prepend context to the system message (or insert one) in a PromptValue."""
    try:
        from langchain_core.messages import SystemMessage
        from langchain_core.prompt_values import ChatPromptValue

        new_messages = list(messages)
        if new_messages and isinstance(new_messages[0], SystemMessage):
            existing = new_messages[0].content if isinstance(new_messages[0].content, str) else ""
            new_messages[0] = SystemMessage(content=f"{context}\n\n{existing}" if existing else context)
        else:
            new_messages.insert(0, SystemMessage(content=context))

        try:
            from langchain_core.prompt_values import PromptValue
            if isinstance(input, PromptValue):
                return ChatPromptValue(messages=new_messages)
        except Exception:
            pass
        return new_messages
    except Exception:
        return input


# ── Public helper: MnemonMemory ───────────────────────────────────────────────

def make_mnemon_memory(mnemon: Any, memory_key: str = "history", top_k: int = 5) -> Any:
    """
    Return a LangChain-compatible BaseMemory backed by Mnemon.

    Usage:
        from mnemon.moth.integrations.langchain import make_mnemon_memory
        chain = ConversationChain(llm=llm, memory=make_mnemon_memory(m))
    """
    try:
        from langchain_core.memory import BaseMemory
    except ImportError:
        try:
            from langchain.schema import BaseMemory
        except ImportError:
            raise ImportError("langchain-core is required: pip install langchain-core")

    class _MnemonMemory(BaseMemory):  # type: ignore[misc]
        mnemon_inst: Any = mnemon
        _memory_key: str = memory_key
        _top_k: int = top_k

        class Config:
            arbitrary_types_allowed = True

        @property
        def memory_variables(self) -> List[str]:
            return [self._memory_key]

        def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            query = _extract_lc_query(inputs)
            context = recall_as_context(self.mnemon_inst, query, top_k=self._top_k)
            return {self._memory_key: context}

        def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
            input_str = _extract_lc_query(inputs)
            output_str = str(outputs)[:400]
            record_outcome(self.mnemon_inst, input_str, output_str, importance=0.7)

        def clear(self) -> None:
            pass  # persistent by design

    return _MnemonMemory()
