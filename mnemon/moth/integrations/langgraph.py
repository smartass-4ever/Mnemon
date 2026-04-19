"""
Mnemon moth — LangGraph integration.

LangGraph compiles agent workflows into a CompiledGraph. This integration
patches CompiledGraph.invoke and ainvoke to:

  1. Inject recalled memory into the initial graph state before any node runs
  2. Apply System 2 EME at the graph level — if the same workflow with the
     same inputs was seen before, return the cached final state
  3. Record the outcome to the experience bus after the graph completes

System 2 EME note:
  Each unique (graph_id + input_hash) is treated as an execution segment.
  On a cache hit, the entire graph execution is skipped. For per-node EME
  (finer-grained), users can add MnemonNode to their graph manually.

Zero user code changes required for graph-level instrumentation.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Dict, Optional

from mnemon.moth import MnemonIntegration
from ._utils import prompt_hash, recall_as_context, record_outcome

logger = logging.getLogger(__name__)

_graph_cache: Dict[str, Any] = {}


class LangGraphIntegration(MnemonIntegration):
    """
    Instruments LangGraph's CompiledGraph at the invoke level.
    """

    name = "langgraph"

    def __init__(self) -> None:
        self._original_invoke: Optional[Any] = None
        self._original_ainvoke: Optional[Any] = None
        self._mnemon: Optional[Any] = None

    def is_available(self) -> bool:
        return importlib.util.find_spec("langgraph") is not None

    def patch(self, mnemon: Any) -> None:
        from langgraph.graph.graph import CompiledGraph

        self._mnemon = mnemon
        self._original_invoke = CompiledGraph.invoke
        self._original_ainvoke = CompiledGraph.ainvoke

        m = mnemon
        orig_invoke = self._original_invoke
        orig_ainvoke = self._original_ainvoke

        def patched_invoke(
            _self: Any,
            input: Any,
            config: Any = None,
            **kwargs: Any,
        ) -> Any:
            goal = _extract_graph_goal(input)
            context = recall_as_context(m, goal) if goal else ""

            # Inject context into state
            patched_input = _inject_into_state(input, context)

            # System 2 EME: skip entire graph if cached
            cache_key = _graph_cache_key(_self, input)
            if cache_key in _graph_cache:
                logger.debug("Mnemon: LangGraph invoke cache hit")
                return _graph_cache[cache_key]

            result = orig_invoke(_self, patched_input, config, **kwargs)

            _graph_cache[cache_key] = result
            record_outcome(m, goal, _extract_graph_outcome(result))
            return result

        async def patched_ainvoke(
            _self: Any,
            input: Any,
            config: Any = None,
            **kwargs: Any,
        ) -> Any:
            goal = _extract_graph_goal(input)
            context = recall_as_context(m, goal) if goal else ""

            patched_input = _inject_into_state(input, context)

            cache_key = _graph_cache_key(_self, input)
            if cache_key in _graph_cache:
                logger.debug("Mnemon: LangGraph ainvoke cache hit")
                return _graph_cache[cache_key]

            result = await orig_ainvoke(_self, patched_input, config, **kwargs)

            _graph_cache[cache_key] = result
            record_outcome(m, goal, _extract_graph_outcome(result))
            return result

        CompiledGraph.invoke = patched_invoke
        CompiledGraph.ainvoke = patched_ainvoke

    def unpatch(self) -> None:
        if self._original_invoke is None:
            return
        try:
            from langgraph.graph.graph import CompiledGraph
            CompiledGraph.invoke = self._original_invoke
            CompiledGraph.ainvoke = self._original_ainvoke
        except Exception as e:
            logger.debug(f"Mnemon: LangGraph unpatch failed — {e}")
        finally:
            self._original_invoke = None
            self._original_ainvoke = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_graph_goal(state: Any) -> str:
    if isinstance(state, dict):
        for key in ("messages", "input", "query", "goal", "task"):
            val = state.get(key)
            if isinstance(val, str):
                return val[:300]
            if isinstance(val, list) and val:
                last = val[-1]
                if isinstance(last, dict):
                    return last.get("content", str(last))[:300]
                return str(last)[:300]
    return str(state)[:200]


def _inject_into_state(state: Any, context: str) -> Any:
    if not context or not isinstance(state, dict):
        return state
    return {**state, "_mnemon_context": context}


def _graph_cache_key(graph: Any, state: Any) -> str:
    graph_id = getattr(graph, "name", type(graph).__name__)
    state_hash = prompt_hash(
        [{"role": "user", "content": str(state)}], None, graph_id
    )
    return f"{graph_id}:{state_hash}"


def _extract_graph_outcome(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("messages", "output", "result", "response"):
            val = result.get(key)
            if val:
                if isinstance(val, list) and val:
                    last = val[-1]
                    return str(getattr(last, "content", last))[:400]
                return str(val)[:400]
    return str(result)[:400]
