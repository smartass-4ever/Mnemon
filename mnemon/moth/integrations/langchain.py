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
from ._utils import extract_query, prompt_hash, recall_as_context, record_outcome

logger = logging.getLogger(__name__)

_step_cache:  Dict[str, Any] = {}   # per-step System 2
_chain_cache: Dict[str, Any] = {}   # chain-level System 1 (legacy)


class LangChainIntegration(MnemonIntegration):
    """
    Instruments LangChain at the per-step level for true System 2 EME.
    """

    name = "langchain"

    def __init__(self) -> None:
        self._original_runnable_invoke: Optional[Any] = None
        self._original_chain_call: Optional[Any] = None
        self._mnemon: Optional[Any] = None

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

            def patched_invoke(_self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
                query = _extract_lc_query(input)
                context = recall_as_context(m, query) if query else ""

                # Inject context into the input for the first step only
                first_input = (
                    {**input, "_mnemon_context": context}
                    if context and isinstance(input, dict)
                    else input
                )

                steps = getattr(_self, "steps", None)
                if not steps:
                    # No steps exposed — fall through to original
                    result = orig_invoke(_self, first_input, config, **kwargs)
                    record_outcome(m, query, str(result)[:400])
                    return result

                # Per-step execution with System 2 cache
                current = first_input
                try:
                    for i, step in enumerate(steps):
                        step_key = _step_cache_key(step, current)
                        if step_key in _step_cache:
                            logger.debug(
                                f"Mnemon: LangChain step {i} "
                                f"({type(step).__name__}) System 2 cache hit"
                            )
                            current = _step_cache[step_key]
                            continue

                        if config is not None:
                            current = step.invoke(current, config)
                        else:
                            current = step.invoke(current)

                        _step_cache[step_key] = current

                except Exception as e:
                    logger.debug(
                        f"Mnemon: LangChain per-step failed at step {i} — {e}, "
                        f"falling back to whole-chain"
                    )
                    current = orig_invoke(_self, first_input, config, **kwargs)

                record_outcome(m, query, str(current)[:400])
                return current

            RunnableSequence.invoke = patched_invoke

        except Exception as e:
            logger.debug(f"Mnemon: LangChain RunnableSequence patch failed — {e}")

        # Patch 2: Legacy Chain.__call__ — chain-level System 1
        try:
            from langchain.chains.base import Chain
            self._original_chain_call = Chain.__call__
            orig_call = self._original_chain_call

            def patched_chain_call(_self: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
                query = _extract_lc_query(inputs)
                context = recall_as_context(m, query) if query else ""

                if context and isinstance(inputs, dict):
                    inputs = {**inputs, "_mnemon_context": context}

                cache_key = _chain_cache_key(_self, inputs)
                if cache_key in _chain_cache:
                    logger.debug("Mnemon: LangChain Chain cache hit")
                    return _chain_cache[cache_key]

                result = orig_call(_self, inputs, *args, **kwargs)
                _chain_cache[cache_key] = result
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

            if self._original_chain_call is not None:
                from langchain.chains.base import Chain
                Chain.__call__ = self._original_chain_call
        except Exception as e:
            logger.debug(f"Mnemon: LangChain unpatch failed — {e}")
        finally:
            self._original_runnable_invoke = None
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
