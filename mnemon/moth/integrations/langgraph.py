"""
Mnemon moth — LangGraph integration.

Patches at two levels:

Level 1 — CompiledGraph.invoke / ainvoke  (graph-level, System 1)
  On every graph invocation: cache the full result keyed by (graph_id +
  state hash). Cache hit returns immediately — zero LLM calls.

Level 2 — Per-node wrapping  (node-level, System 2)
  After compile(), wraps each node function individually. Each node
  becomes a segment. If 4 of 5 nodes have cached outputs for the
  current state, only 1 node calls the LLM — full System 2.

Memory injection:
  Not done at the state/graph level. The Anthropic and OpenAI intercepts
  already inject recalled memory directly into every LLM call. Injecting
  _mnemon_context into state would silently pollute state dicts that
  nothing reads — removed.

Version detection:
  Tested against langgraph 0.1.x–0.2.x. If per-node wrapping fails,
  falls back to graph-level only. Original behavior always preserved.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Callable, Dict, Optional

from mnemon.moth import MnemonIntegration
from ._utils import prompt_hash, record_outcome
from ._cache import BoundedTTLCache

logger = logging.getLogger(__name__)

_graph_cache = BoundedTTLCache(maxsize=500, ttl=3600)
_node_cache  = BoundedTTLCache(maxsize=500, ttl=3600)


class LangGraphIntegration(MnemonIntegration):
    """
    Instruments LangGraph at both graph and node level.
    Node-level patching gives full System 2 EME.
    Memory injection delegated to Anthropic/OpenAI intercepts.
    """

    name = "langgraph"

    def __init__(self) -> None:
        self._original_invoke:  Optional[Any] = None
        self._original_ainvoke: Optional[Any] = None
        self._original_compile: Optional[Any] = None
        self._mnemon: Optional[Any] = None

    def is_available(self) -> bool:
        return importlib.util.find_spec("langgraph") is not None

    def patch(self, mnemon: Any) -> None:
        from langgraph.graph.graph import CompiledGraph
        try:
            from langgraph.graph.state import StateGraph
            self._patch_compile(StateGraph, mnemon)
        except Exception as e:
            logger.debug(f"Mnemon: LangGraph compile patch failed — {e}")

        self._mnemon = mnemon
        self._original_invoke  = CompiledGraph.invoke
        self._original_ainvoke = CompiledGraph.ainvoke

        m = mnemon
        orig_invoke  = self._original_invoke
        orig_ainvoke = self._original_ainvoke

        def patched_invoke(_self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
            cache_key = _graph_cache_key(_self, input)
            if cache_key in _graph_cache:
                logger.debug("Mnemon: LangGraph graph-level cache hit")
                return _graph_cache[cache_key]

            result = orig_invoke(_self, input, config, **kwargs)
            _graph_cache[cache_key] = result
            record_outcome(m, _extract_graph_goal(input), _extract_graph_outcome(result))
            return result

        async def patched_ainvoke(_self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
            cache_key = _graph_cache_key(_self, input)
            if cache_key in _graph_cache:
                logger.debug("Mnemon: LangGraph async graph-level cache hit")
                return _graph_cache[cache_key]

            result = await orig_ainvoke(_self, input, config, **kwargs)
            _graph_cache[cache_key] = result
            record_outcome(m, _extract_graph_goal(input), _extract_graph_outcome(result))
            return result

        CompiledGraph.invoke  = patched_invoke
        CompiledGraph.ainvoke = patched_ainvoke

    def _patch_compile(self, StateGraph: Any, mnemon: Any) -> None:
        self._original_compile = StateGraph.compile
        m = mnemon
        orig_compile = self._original_compile

        def patched_compile(_self: Any, *args: Any, **kwargs: Any) -> Any:
            compiled = orig_compile(_self, *args, **kwargs)
            try:
                _wrap_nodes(compiled, m)
                logger.debug(f"Mnemon: LangGraph wrapped {len(compiled.nodes)} nodes")
            except Exception as e:
                logger.debug(f"Mnemon: LangGraph node wrap failed — {e}")
            return compiled

        StateGraph.compile = patched_compile

    def unpatch(self) -> None:
        try:
            from langgraph.graph.graph import CompiledGraph
            if self._original_invoke is not None:
                CompiledGraph.invoke  = self._original_invoke
                CompiledGraph.ainvoke = self._original_ainvoke
            if self._original_compile is not None:
                from langgraph.graph.state import StateGraph
                StateGraph.compile = self._original_compile
        except Exception as e:
            logger.debug(f"Mnemon: LangGraph unpatch failed — {e}")
        finally:
            self._original_invoke  = None
            self._original_ainvoke = None
            self._original_compile = None


# ── Per-node System 2 ─────────────────────────────────────────────────────────

def _wrap_nodes(compiled: Any, m: Any) -> None:
    nodes = getattr(compiled, "nodes", {})
    if not nodes:
        return
    for node_name, node_runnable in list(nodes.items()):
        if node_name in ("__start__", "__end__"):
            continue
        nodes[node_name] = _make_node_wrapper(node_name, node_runnable, m)


def _make_node_wrapper(node_name: str, original: Any, m: Any) -> Any:
    def wrapped_invoke(state: Any, config: Any = None, **kwargs: Any) -> Any:
        cache_key = f"{node_name}:{prompt_hash([{'content': str(state)}], None, node_name)}"
        if cache_key in _node_cache:
            logger.debug(f"Mnemon: LangGraph node '{node_name}' System 2 cache hit")
            return _node_cache[cache_key]

        result = _safe_node_invoke(original, state, config, **kwargs)
        _node_cache[cache_key] = result
        record_outcome(
            m,
            f"{node_name}: {_extract_graph_goal(state)}",
            _extract_graph_outcome(result),
            importance=0.6,
        )
        return result

    async def wrapped_ainvoke(state: Any, config: Any = None, **kwargs: Any) -> Any:
        cache_key = f"{node_name}:{prompt_hash([{'content': str(state)}], None, node_name)}"
        if cache_key in _node_cache:
            logger.debug(f"Mnemon: LangGraph node '{node_name}' async System 2 cache hit")
            return _node_cache[cache_key]

        result = await _safe_node_ainvoke(original, state, config, **kwargs)
        _node_cache[cache_key] = result
        record_outcome(
            m,
            f"{node_name}: {_extract_graph_goal(state)}",
            _extract_graph_outcome(result),
            importance=0.6,
        )
        return result

    try:
        original.invoke  = wrapped_invoke
        original.ainvoke = wrapped_ainvoke
    except (AttributeError, TypeError):
        return wrapped_invoke

    return original


def _safe_node_invoke(node: Any, state: Any, config: Any, **kwargs: Any) -> Any:
    if callable(node) and not hasattr(node, "invoke"):
        return node(state)
    invoke = getattr(node, "invoke", None)
    if invoke:
        return invoke(state, config, **kwargs) if config is not None else invoke(state)
    return node(state)


async def _safe_node_ainvoke(node: Any, state: Any, config: Any, **kwargs: Any) -> Any:
    ainvoke = getattr(node, "ainvoke", None)
    if ainvoke:
        import inspect
        result = ainvoke(state, config, **kwargs) if config is not None else ainvoke(state)
        if inspect.isawaitable(result):
            return await result
        return result
    return _safe_node_invoke(node, state, config, **kwargs)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_graph_goal(state: Any) -> str:
    if isinstance(state, dict):
        for key in ("messages", "input", "query", "goal", "task", "question"):
            val = state.get(key)
            if isinstance(val, str):
                return val[:300]
            if isinstance(val, list) and val:
                last = val[-1]
                content = getattr(last, "content", None) or (
                    last.get("content") if isinstance(last, dict) else str(last)
                )
                return str(content)[:300]
    return str(state)[:200]


def _graph_cache_key(graph: Any, state: Any) -> str:
    graph_id = getattr(graph, "name", type(graph).__name__)
    return f"{graph_id}:{prompt_hash([{'content': str(state)}], None, graph_id)}"


def _extract_graph_outcome(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("messages", "output", "result", "response", "answer"):
            val = result.get(key)
            if val:
                if isinstance(val, list) and val:
                    last = val[-1]
                    return str(getattr(last, "content", last))[:400]
                return str(val)[:400]
    return str(result)[:400]
