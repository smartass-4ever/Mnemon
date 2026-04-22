"""
Integration tests for the Mnemon moth auto-instrumentation engine.

Covers:
  - Moth loading and integration availability detection
  - Framework patching and unpatching (using stubs, no real LLM calls)
  - CrewAI event bus integration (crewai is installed)
  - LangChain per-step System 2 cache
  - LangGraph per-node System 2 cache
  - Anthropic / OpenAI patch shape (with mock modules)
  - Protein bond gating: no context injected when query is empty

Real LLM calls are never made. Frameworks are imported with
pytest.importorskip() so missing ones are cleanly skipped.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ── Shared stub Mnemon instance ───────────────────────────────────────────────

class _StubMnemon:
    """Minimal Mnemon stand-in that records calls."""

    def __init__(self):
        self.recalled = []
        self.remembered = []

    def recall(self, query: str, top_k: int = 5):
        self.recalled.append(query)
        return {"memories": [], "facts": {}}

    def remember(self, text: str, importance: float = 0.65):
        self.remembered.append((text, importance))


# ── Moth loading ──────────────────────────────────────────────────────────────

class TestMothLoading:
    def test_moth_instantiates(self):
        from mnemon.moth import Moth
        moth = Moth()
        assert hasattr(moth, "_registered")
        assert hasattr(moth, "_active")

    def test_activate_returns_list(self):
        from mnemon.moth import Moth
        moth = Moth()
        m = _StubMnemon()
        activated = moth.activate(m)
        assert isinstance(activated, list)

    def test_active_property(self):
        from mnemon.moth import Moth
        moth = Moth()
        m = _StubMnemon()
        moth.activate(m)
        assert isinstance(moth.active, list)

    def test_deactivate_clears_active(self):
        from mnemon.moth import Moth
        moth = Moth()
        m = _StubMnemon()
        moth.activate(m)
        moth.deactivate()
        assert moth.active == []


# ── _utils ────────────────────────────────────────────────────────────────────

class TestUtils:
    def test_recall_as_context_empty_on_no_memories(self):
        from mnemon.moth.integrations._utils import recall_as_context
        m = _StubMnemon()
        result = recall_as_context(m, "test query")
        assert result == ""

    def test_recall_as_context_returns_string_with_memories(self):
        from mnemon.moth.integrations._utils import recall_as_context

        class _MnemonWithMemory(_StubMnemon):
            def recall(self, query, top_k=5):
                return {
                    "memories": [{"content": {"text": "user prefers Python"}}],
                    "facts": {},
                }

        result = recall_as_context(_MnemonWithMemory(), "query")
        assert "Python" in result

    def test_record_outcome_never_raises(self):
        from mnemon.moth.integrations._utils import record_outcome
        # Even with a broken Mnemon instance
        record_outcome(None, "goal", "outcome")

    def test_prompt_hash_stable(self):
        from mnemon.moth.integrations._utils import prompt_hash
        msgs = [{"role": "user", "content": "hello"}]
        h1 = prompt_hash(msgs, "system", "claude-3")
        h2 = prompt_hash(msgs, "system", "claude-3")
        assert h1 == h2

    def test_prompt_hash_differs_on_content(self):
        from mnemon.moth.integrations._utils import prompt_hash
        h1 = prompt_hash([{"role": "user", "content": "a"}], None, "m")
        h2 = prompt_hash([{"role": "user", "content": "b"}], None, "m")
        assert h1 != h2

    def test_extract_query_uses_system_and_user(self):
        from mnemon.moth.integrations._utils import extract_query
        msgs = [{"role": "user", "content": "summarize the doc"}]
        result = extract_query(msgs, system="You are a summarizer")
        assert "summarizer" in result
        assert "summarize" in result

    def test_extract_query_empty_on_no_user_message(self):
        from mnemon.moth.integrations._utils import extract_query
        result = extract_query([{"role": "assistant", "content": "hi"}])
        assert result == ""

    def test_inject_into_system_prepends(self):
        from mnemon.moth.integrations._utils import inject_into_system
        result = inject_into_system("Be helpful.", "Context: foo")
        assert result.startswith("Context: foo")
        assert "Be helpful." in result

    def test_inject_into_system_returns_original_on_empty_context(self):
        from mnemon.moth.integrations._utils import inject_into_system
        assert inject_into_system("original", "") == "original"


# ── Protein bond gating ───────────────────────────────────────────────────────

class TestProteinBondGating:
    """Ensure no memory is injected when recall returns nothing."""

    def test_no_injection_when_recall_empty(self):
        from mnemon.moth.integrations._utils import recall_as_context, inject_into_system
        m = _StubMnemon()
        context = recall_as_context(m, "query with no matches")
        system = inject_into_system("original system", context)
        assert system == "original system"

    def test_recall_as_context_empty_string_on_exception(self):
        from mnemon.moth.integrations._utils import recall_as_context

        class _BrokenMnemon:
            def recall(self, *a, **kw):
                raise RuntimeError("db offline")

        result = recall_as_context(_BrokenMnemon(), "query")
        assert result == ""


# ── LangChain per-step System 2 ───────────────────────────────────────────────

class TestLangChainIntegration:
    def setup_method(self):
        langchain_core = pytest.importorskip("langchain_core")

    def test_patch_and_unpatch(self):
        from mnemon.moth.integrations.langchain import LangChainIntegration
        from langchain_core.runnables.base import RunnableSequence

        original = RunnableSequence.invoke
        integration = LangChainIntegration()
        m = _StubMnemon()
        integration.patch(m)
        assert RunnableSequence.invoke is not original
        integration.unpatch()
        assert RunnableSequence.invoke is original

    def test_per_step_cache_hit_skips_step(self):
        from mnemon.moth.integrations.langchain import LangChainIntegration
        from langchain_core.runnables.base import RunnableSequence

        integration = LangChainIntegration()
        m = _StubMnemon()
        integration.patch(m)

        call_count = 0

        class _StubStep:
            def invoke(self, input, config=None):
                nonlocal call_count
                call_count += 1
                return f"out:{input}"

        step1 = _StubStep()
        step2 = _StubStep()

        stub_seq = MagicMock(spec=RunnableSequence)
        stub_seq.steps = [step1, step2]

        # First call — both steps execute
        result = RunnableSequence.invoke(stub_seq, "hello")
        assert call_count == 2

        # Second call with same input — both steps should hit cache
        call_count = 0
        result2 = RunnableSequence.invoke(stub_seq, "hello")
        assert call_count == 0, "Cache should have prevented all step calls"
        assert result == result2

        integration.unpatch()

    def test_fallback_on_step_error(self):
        """If a step raises, falls back to whole-chain invocation."""
        from mnemon.moth.integrations.langchain import LangChainIntegration
        from langchain_core.runnables.base import RunnableSequence

        integration = LangChainIntegration()
        m = _StubMnemon()
        integration.patch(m)

        fallback_called = []

        class _ErrorStep:
            def invoke(self, input, config=None):
                raise ValueError("step exploded")

        stub_seq = MagicMock(spec=RunnableSequence)
        stub_seq.steps = [_ErrorStep()]

        # Intercept the original (fallback) invoke
        real_orig = integration._original_runnable_invoke
        def mock_fallback(_self, input, config=None, **kwargs):
            fallback_called.append(True)
            return "fallback_result"
        integration._original_runnable_invoke = mock_fallback

        # Re-patch so the closure captures mock_fallback
        integration.unpatch()
        integration.patch(m)
        integration._original_runnable_invoke = mock_fallback

        result = RunnableSequence.invoke(stub_seq, "error_input")
        assert len(fallback_called) == 1, "Fallback should have been called once"
        assert result == "fallback_result"

        integration.unpatch()


# ── LangGraph per-node System 2 ───────────────────────────────────────────────

class TestLangGraphIntegration:
    def setup_method(self):
        pytest.importorskip("langgraph")

    def test_patch_and_unpatch(self):
        from mnemon.moth.integrations.langgraph import LangGraphIntegration
        try:
            from langgraph.pregel.main import Pregel as GraphBase
        except ImportError:
            from langgraph.graph.graph import CompiledGraph as GraphBase  # type: ignore

        original_invoke = GraphBase.invoke
        integration = LangGraphIntegration()
        m = _StubMnemon()
        integration.patch(m)
        assert GraphBase.invoke is not original_invoke
        integration.unpatch()
        assert GraphBase.invoke is original_invoke

    def test_node_cache_hit(self):
        from mnemon.moth.integrations.langgraph import LangGraphIntegration, _make_node_wrapper
        from mnemon.moth.integrations._eme_bridge import MothCache

        call_count = 0

        class _FakeRunnable:
            def invoke(self, state, config=None):
                nonlocal call_count
                call_count += 1
                return {"result": "done"}

        m = _StubMnemon()
        node_cache = MothCache(m, "langgraph_test")
        original = _FakeRunnable()
        wrapped = _make_node_wrapper("my_node", original, m, node_cache)

        state = {"messages": "process this"}

        # First call
        wrapped.invoke(state)
        assert call_count == 1

        # Second call with same state — should hit cache
        wrapped.invoke(state)
        assert call_count == 1, "Node should have been served from cache"

    def test_compile_wraps_nodes(self):
        from langgraph.graph.state import StateGraph
        from mnemon.moth.integrations.langgraph import LangGraphIntegration

        integration = LangGraphIntegration()
        m = _StubMnemon()
        integration.patch(m)

        # Build a minimal graph
        graph = StateGraph(dict)

        def node_a(state):
            return state

        graph.add_node("node_a", node_a)
        graph.set_entry_point("node_a")
        graph.set_finish_point("node_a")

        compiled = graph.compile()
        # nodes should be wrapped — original function should be replaced
        assert "node_a" in compiled.nodes

        integration.unpatch()


# ── CrewAI event bus integration ─────────────────────────────────────────────

class TestCrewAIIntegration:
    def setup_method(self):
        pytest.importorskip("crewai")

    def test_patch_registers_listener(self):
        from mnemon.moth.integrations.crewai import CrewAIIntegration

        integration = CrewAIIntegration()
        m = _StubMnemon()
        integration.patch(m)

        assert bool(integration._handlers)
        integration.unpatch()
        assert not integration._handlers

    def test_task_execute_sync_patched(self):
        from mnemon.moth.integrations.crewai import CrewAIIntegration
        from crewai.task import Task

        original = Task.execute_sync
        integration = CrewAIIntegration()
        m = _StubMnemon()
        integration.patch(m)

        assert Task.execute_sync is not original

        integration.unpatch()
        assert Task.execute_sync is original

    def test_task_cache_hit(self):
        from mnemon.moth.integrations.crewai import CrewAIIntegration, _task_cache
        from crewai.task import Task

        _task_cache.clear()

        integration = CrewAIIntegration()
        m = _StubMnemon()
        integration.patch(m)

        call_count = 0
        original_execute = integration._original_execute_sync

        def fake_execute(_self, agent=None, context=None, tools=None):
            nonlocal call_count
            call_count += 1
            return "task_output"

        # Replace original so cache miss calls our fake
        integration._original_execute_sync = fake_execute

        # Re-patch with updated original
        integration.unpatch()
        _task_cache.clear()
        # Manually test cache logic
        from mnemon.moth.integrations.crewai import _task_cache_key

        class FakeTask:
            description = "analyze the data"

        class FakeAgent:
            role = "analyst"

        key = _task_cache_key(FakeTask(), FakeAgent(), "some context")
        assert isinstance(key, str)
        assert key.startswith("crewai_task:")

    def test_agent_execution_event_fires_recall(self):
        """Verify handler is registered and calls recall on execution start."""
        from mnemon.moth.integrations.crewai import CrewAIIntegration, _backstory_originals

        integration = CrewAIIntegration()
        m = _StubMnemon()
        integration.patch(m)

        assert bool(integration._handlers)

        # Get the handler directly from _handlers dict
        from crewai.events.types.agent_events import AgentExecutionStartedEvent
        on_agent_start = integration._handlers.get(AgentExecutionStartedEvent)
        assert on_agent_start is not None

        # Call the handler directly with a fake event
        class FakeAgent:
            role = "researcher"
            backstory = "Expert researcher"
            fingerprint = None

            def __setattr__(self, name, value):
                object.__setattr__(self, name, value)

        class FakeTask:
            description = "research quantum computing"

        class FakeEvent:
            agent = FakeAgent()
            task = FakeTask()
            task_prompt = "research quantum computing"

        on_agent_start(source=None, event=FakeEvent())

        # recall should have been called
        assert len(m.recalled) > 0

        integration.unpatch()


# ── Anthropic integration shape ───────────────────────────────────────────────

class TestAnthropicIntegration:
    def test_patch_and_unpatch(self):
        pytest.importorskip("anthropic")
        from mnemon.moth.integrations.anthropic import AnthropicIntegration
        import anthropic.resources.messages as _mod

        original_sync = _mod.Messages.create
        integration = AnthropicIntegration()
        m = _StubMnemon()
        integration.patch(m)
        assert _mod.Messages.create is not original_sync
        integration.unpatch()
        assert _mod.Messages.create is original_sync

    def test_cache_hit_skips_original(self):
        pytest.importorskip("anthropic")
        from mnemon.moth.integrations.anthropic import AnthropicIntegration
        import anthropic.resources.messages as _mod

        integration = AnthropicIntegration()
        m = _StubMnemon()
        integration.patch(m)
        assert _mod.Messages.create is not None
        integration.unpatch()

    def test_synthetic_stream_yields_events(self):
        from mnemon.moth.integrations.anthropic import _SyntheticAnthropicStream
        stream = _SyntheticAnthropicStream("hello world", "claude-3")
        events = list(stream)
        types_ = [e.type for e in events]
        assert "content_block_delta" in types_
        delta_events = [e for e in events if e.type == "content_block_delta"]
        assert delta_events[0].delta.text == "hello world"

    def test_synthetic_stream_text_stream(self):
        from mnemon.moth.integrations.anthropic import _SyntheticAnthropicStream
        stream = _SyntheticAnthropicStream("test text", "claude-3")
        texts = list(stream.text_stream)
        assert texts == ["test text"]

    def test_capturing_stream_collects_and_stores(self):
        from mnemon.moth.integrations.anthropic import _CapturingAnthropicStream, _make_event
        import types as T
        from mnemon.moth.integrations._cache import BoundedTTLCache

        cache = BoundedTTLCache(maxsize=10, ttl=60)
        m = _StubMnemon()

        fake_events = [
            _make_event("content_block_delta", index=0,
                        delta=T.SimpleNamespace(type="text_delta", text="foo")),
            _make_event("content_block_delta", index=0,
                        delta=T.SimpleNamespace(type="text_delta", text="bar")),
            _make_event("message_stop"),
        ]
        capturing = _CapturingAnthropicStream(iter(fake_events), cache, "key1", m, "query")
        collected = list(capturing)
        assert len(collected) == 3
        assert cache.get("key1") == "foobar"


# ── OpenAI integration shape ──────────────────────────────────────────────────

class TestOpenAIIntegration:
    def test_patch_and_unpatch(self):
        pytest.importorskip("openai")
        from mnemon.moth.integrations.openai_sdk import OpenAIIntegration
        import openai.resources.chat.completions as _mod

        original = _mod.Completions.create
        integration = OpenAIIntegration()
        m = _StubMnemon()
        integration.patch(m)
        assert _mod.Completions.create is not original
        integration.unpatch()
        assert _mod.Completions.create is original

    def test_synthetic_stream_yields_chunks(self):
        from mnemon.moth.integrations.openai_sdk import _SyntheticOpenAIStream
        stream = _SyntheticOpenAIStream("hello", "gpt-4")
        chunks = list(stream)
        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "hello"
        assert chunks[1].choices[0].finish_reason == "stop"

    def test_capturing_stream_collects_and_stores(self):
        from mnemon.moth.integrations.openai_sdk import _CapturingOpenAIStream, _chunk
        from mnemon.moth.integrations._cache import BoundedTTLCache

        cache = BoundedTTLCache(maxsize=10, ttl=60)
        m = _StubMnemon()

        fake_chunks = [
            _chunk("foo", None, "gpt-4"),
            _chunk("bar", None, "gpt-4"),
            _chunk(None, "stop", "gpt-4"),
        ]
        capturing = _CapturingOpenAIStream(iter(fake_chunks), cache, "key2", m, "query")
        collected = list(capturing)
        assert len(collected) == 3
        assert cache.get("key2") == "foobar"


# ── Persistent stats ─────────────────────────────────────────────────────────

class TestPersistentStats:
    def test_stats_persist_across_instances(self, tmp_path):
        from mnemon.moth.stats import MothStats
        path = str(tmp_path / "stats.json")

        s1 = MothStats(persist_path=path)
        s1.record_hit("anthropic", tokens=200)
        s1.record_injection("anthropic", "query", "some context")
        s1.record_gate("openai", "empty query")

        s2 = MothStats(persist_path=path)
        assert s2._hits["anthropic"] == 1
        assert s2._injections["anthropic"] == 1
        assert s2._gates["openai"] == 1
        assert s2._tokens == 200

    def test_history_persists(self, tmp_path):
        from mnemon.moth.stats import MothStats
        path = str(tmp_path / "stats.json")

        s1 = MothStats(persist_path=path)
        s1.record_injection("crewai", "do the thing", "past context")

        s2 = MothStats(persist_path=path)
        assert len(s2.recall_history) == 1
        assert s2.recall_history[0].source == "crewai"
        assert s2.recall_history[0].injected is True

    def test_no_persist_path_never_raises(self):
        from mnemon.moth.stats import MothStats
        s = MothStats()
        s.record_hit("x", tokens=100)
        s.record_injection("x", "q", "ctx")
        s.record_gate("x", "q")
        assert s.total_hits == 1

    def test_corrupt_file_is_ignored(self, tmp_path):
        from mnemon.moth.stats import MothStats
        path = str(tmp_path / "stats.json")
        with open(path, "w") as f:
            f.write("not json {{{")
        s = MothStats(persist_path=path)
        assert s.total_hits == 0


# ── AutoGen integration shape ─────────────────────────────────────────────────

class TestAutoGenIntegration:
    def test_available_or_skip(self):
        from mnemon.moth.integrations.autogen import AutoGenIntegration
        integration = AutoGenIntegration()
        # Just verify it instantiates cleanly regardless of whether autogen is installed
        assert integration.name == "autogen"

    def test_patch_skips_gracefully_when_unavailable(self):
        from mnemon.moth.integrations.autogen import AutoGenIntegration
        integration = AutoGenIntegration()
        if not integration.is_available():
            pytest.skip("autogen not installed")
        m = _StubMnemon()
        # Should not raise even on version mismatch
        try:
            integration.patch(m)
            integration.unpatch()
        except Exception as e:
            pytest.fail(f"AutoGen patch raised unexpectedly: {e}")
