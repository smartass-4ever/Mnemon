"""
Mnemon moth — LangGraph integration.

Level 1 — CompiledGraph.invoke / ainvoke  (graph-level, System 1)
  Full graph output cached by (graph_id + state hash). Cache hit returns
  immediately — zero LLM calls.

Level 2 — Per-node System 2
  After compile(), wraps each node individually. Each node is a segment.
  MothCache checks hash first, then EME semantic similarity — same node
  task with slightly different state can still hit if semantically close.

Memory injection:
  Not done at state level — the Anthropic/OpenAI intercepts handle injection
  directly into every LLM call with protein bond gating. No state pollution.

Caches are per-patch (per MnemonSync instance) not module-level.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Optional

from mnemon.moth import MnemonIntegration
from ._utils import prompt_hash, record_outcome, track_cache_hit
from ._eme_bridge import MothCache

logger = logging.getLogger(__name__)


class LangGraphIntegration(MnemonIntegration):
    name = "langgraph"

    def __init__(self) -> None:
        self._original_invoke:  Optional[Any] = None
        self._original_ainvoke: Optional[Any] = None
        self._original_compile: Optional[Any] = None
        self._mnemon:           Optional[Any] = None

    def is_available(self) -> bool:
        return importlib.util.find_spec("langgraph") is not None

    def patch(self, mnemon: Any) -> None:
        # langgraph 1.x: CompiledGraph removed, Pregel is the common base
        try:
            from langgraph.pregel.main import Pregel as _GraphBase
        except ImportError:
            try:
                from langgraph.graph.graph import CompiledGraph as _GraphBase  # type: ignore
            except ImportError:
                logger.debug("Mnemon: LangGraph — cannot find graph base class")
                return

        try:
            from langgraph.graph.state import StateGraph
            self._patch_compile(StateGraph, mnemon)
        except Exception as e:
            logger.debug(f"Mnemon: LangGraph compile patch failed — {e}")

        self._mnemon           = mnemon
        self._original_invoke  = _GraphBase.invoke
        self._original_ainvoke = _GraphBase.ainvoke
        self._graph_base       = _GraphBase

        m           = mnemon
        orig_invoke  = self._original_invoke
        orig_ainvoke = self._original_ainvoke
        graph_cache  = MothCache(m, "langgraph")
        self._graph_cache = graph_cache

        def patched_invoke(_self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
            goal      = _extract_graph_goal(input)
            hash_key  = _graph_cache_key(_self, input)

            cached = graph_cache.check(goal, [], hash_key)
            if cached is not None:
                track_cache_hit(m, "langgraph")
                return cached

            result = orig_invoke(_self, input, config, **kwargs)
            text   = _extract_graph_outcome(result)
            graph_cache.store(goal, [], hash_key, result, text)
            record_outcome(m, goal, text)
            return result

        async def patched_ainvoke(_self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
            goal     = _extract_graph_goal(input)
            hash_key = _graph_cache_key(_self, input)

            cached = await graph_cache.async_check(goal, [], hash_key)
            if cached is not None:
                track_cache_hit(m, "langgraph")
                return cached

            result = await orig_ainvoke(_self, input, config, **kwargs)
            text   = _extract_graph_outcome(result)
            await graph_cache.async_store(goal, [], hash_key, result, text)
            record_outcome(m, goal, text)
            return result

        _GraphBase.invoke  = patched_invoke
        _GraphBase.ainvoke = patched_ainvoke

    def _patch_compile(self, StateGraph: Any, mnemon: Any) -> None:
        self._original_compile = StateGraph.compile
        m            = mnemon
        orig_compile = self._original_compile

        def patched_compile(_self: Any, *args: Any, **kwargs: Any) -> Any:
            compiled   = orig_compile(_self, *args, **kwargs)
            node_cache = MothCache(m, "langgraph")
            self._node_cache = node_cache
            try:
                _wrap_nodes(compiled, m, node_cache)
                logger.debug(f"Mnemon: LangGraph wrapped {len(compiled.nodes)} nodes")
            except Exception as e:
                logger.debug(f"Mnemon: LangGraph node wrap failed — {e}")
            return compiled

        StateGraph.compile = patched_compile

    def unpatch(self) -> None:
        try:
            base = getattr(self, "_graph_base", None)
            if base and self._original_invoke is not None:
                base.invoke  = self._original_invoke
                base.ainvoke = self._original_ainvoke
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

def _wrap_nodes(compiled: Any, m: Any, node_cache: MothCache) -> None:
    nodes = getattr(compiled, "nodes", {})
    if not nodes:
        return
    for node_name, node_runnable in list(nodes.items()):
        if node_name in ("__start__", "__end__"):
            continue
        nodes[node_name] = _make_node_wrapper(node_name, node_runnable, m, node_cache)


def _make_node_wrapper(node_name: str, original: Any, m: Any, node_cache: MothCache) -> Any:
    # Capture real invoke/ainvoke before we replace them
    _real_invoke  = getattr(original, "invoke",  None)
    _real_ainvoke = getattr(original, "ainvoke", None)

    def wrapped_invoke(state: Any, config: Any = None, **kwargs: Any) -> Any:
        goal     = f"{node_name}: {_extract_graph_goal(state)}"
        hash_key = f"{node_name}:{prompt_hash([{'content': str(state)}], None, node_name)}"

        cached = node_cache.check(goal, [node_name], hash_key)
        if cached is not None:
            track_cache_hit(m, f"langgraph:{node_name}")
            return cached

        if _real_invoke is not None:
            result = _real_invoke(state, config, **kwargs) if config is not None else _real_invoke(state)
        elif callable(original):
            result = original(state)
        else:
            raise RuntimeError(f"Mnemon: node {node_name!r} has no invoke method")

        text = _extract_graph_outcome(result)
        node_cache.store(goal, [node_name], hash_key, result, text)
        record_outcome(m, goal, text, importance=0.6)
        return result

    async def wrapped_ainvoke(state: Any, config: Any = None, **kwargs: Any) -> Any:
        goal     = f"{node_name}: {_extract_graph_goal(state)}"
        hash_key = f"{node_name}:{prompt_hash([{'content': str(state)}], None, node_name)}"

        cached = await node_cache.async_check(goal, [node_name], hash_key)
        if cached is not None:
            track_cache_hit(m, f"langgraph:{node_name}")
            return cached

        if _real_ainvoke is not None:
            import inspect
            r = _real_ainvoke(state, config, **kwargs) if config is not None else _real_ainvoke(state)
            result = await r if inspect.isawaitable(r) else r
        elif _real_invoke is not None:
            result = _real_invoke(state, config, **kwargs) if config is not None else _real_invoke(state)
        elif callable(original):
            result = original(state)
        else:
            raise RuntimeError(f"Mnemon: node {node_name!r} has no invoke method")

        text = _extract_graph_outcome(result)
        await node_cache.async_store(goal, [node_name], hash_key, result, text)
        record_outcome(m, goal, text, importance=0.6)
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
                last    = val[-1]
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
