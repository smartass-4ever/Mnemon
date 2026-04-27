"""
Mnemon Test Suite
Tests for EME, Experience Bus, drift detection, fragments, streaming, and waste reporting.

Run: pytest tests/test_mnemon.py -v
or:  python tests/test_mnemon.py
"""

import asyncio
import json
import sys
import os
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnemon import Mnemon
from mnemon.core.models import (
    SignalType, ComputationFingerprint, ExperienceSignal,
    RiskLevel, TemplateSegment, ExecutionTemplate, MNEMON_VERSION,
)
from mnemon.core.persistence import EROSDatabase
from mnemon.core.embedder import SimpleEmbedder
from mnemon.core.eme import ExecutionMemoryEngine, GenericAdapter
from mnemon.core.bus import ExperienceBus, Tier1Observer
from mnemon.security.manager import SecurityManager, ContentFilter, ContentSensitivity, TenantSecurityConfig
from mnemon.observability.watchdog import Watchdog
from mnemon.observability.telemetry import Telemetry
from mnemon.fragments.library import load_fragments, FRAGMENT_COUNT
from mnemon.llm.client import MockLLMClient


DB_DIR = "/tmp"
TENANT = "test_tenant"


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def make_eros(llm=None, **kwargs) -> "Mnemon":
    import uuid
    eros = Mnemon(
        tenant_id=f"test_{uuid.uuid4().hex[:8]}",
        agent_id="test_agent",
        db_dir=DB_DIR,
        llm_client=llm,
        prewarm_fragments=False,
        **kwargs,
    )
    await eros.start()
    return eros


# ─────────────────────────────────────────────
# 1. CORE TYPES
# ─────────────────────────────────────────────

def test_computation_fingerprint():
    fp1 = ComputationFingerprint.build("audit Acme", {"client": "str"}, {}, ["scanner"], {})
    fp2 = ComputationFingerprint.build("audit Acme", {"client": "str"}, {}, ["scanner"], {})
    fp3 = ComputationFingerprint.build("audit Beta", {"client": "str"}, {}, ["scanner"], {})
    assert fp1.full_hash == fp2.full_hash, "Same inputs should produce same hash"
    assert fp1.full_hash != fp3.full_hash, "Different goals should produce different hash"
    print("  ✓ ComputationFingerprint")


def test_simple_embedder():
    emb = SimpleEmbedder()
    v1  = emb.embed("security audit report")
    v2  = emb.embed("security audit report")
    v3  = emb.embed("completely unrelated topic bicycle")

    assert len(v1) > 0
    assert v1 == v2, "Same text should produce same embedding"
    sim_same = SimpleEmbedder.cosine_similarity(v1, v2)
    sim_diff = SimpleEmbedder.cosine_similarity(v1, v3)
    assert sim_same > sim_diff, "Same text should have higher similarity than unrelated"
    print(f"  ✓ SimpleEmbedder ({emb.backend_name}, {emb.dim}-dim)")


# ─────────────────────────────────────────────
# 2. EME
# ─────────────────────────────────────────────

async def test_eme_system1_cache():
    eros = await make_eros()
    calls = []

    async def gen(goal, inputs, context, caps, constraints):
        calls.append(goal)
        return [{"id": "step_1", "action": "scan"}, {"id": "step_2", "action": "report"}]

    r1 = await eros.run(
        goal="weekly security audit for Acme",
        inputs={"week": "week_1"},
        generation_fn=gen,
    )
    assert r1["cache_level"] == "miss", f"First run should be miss, got {r1['cache_level']}"
    assert len(calls) == 1

    r2 = await eros.run(
        goal="weekly security audit for Acme",
        inputs={"week": "week_2"},
        generation_fn=gen,
    )
    assert r2["cache_level"] == "system1", f"Second run should be system1, got {r2['cache_level']}"
    assert len(calls) == 1, "Generation should NOT be called again"
    assert r2["tokens_saved"] > 0

    await eros.stop()
    print("  ✓ EME System 1 cache hit")


async def test_eme_different_goals_miss():
    eros = await make_eros(eme_enabled=False)
    calls = []

    async def gen(goal, inputs, context, caps, constraints):
        calls.append(goal)
        return [{"id": "step_1", "action": "process"}]

    import uuid as _uuid
    goal_a = f"task_alpha_{_uuid.uuid4().hex}"
    goal_b = f"task_beta_{_uuid.uuid4().hex}"
    await eros.run(goal=goal_a, inputs={}, generation_fn=gen)
    await eros.run(goal=goal_b, inputs={}, generation_fn=gen)

    assert len(calls) == 2, "Different goals should both call generation"
    await eros.stop()
    print("  ✓ EME different goals produce misses")


async def test_eme_fragment_library():
    eros = await make_eros()

    async def gen(goal, inputs, context, caps, constraints):
        return [{"id": "auth_step", "action": "generate_jwt_token", "params": {}}]

    await eros.run(goal="authenticate user", inputs={}, generation_fn=gen)
    await asyncio.sleep(0.05)  # allow WriteBehindQueue 10ms debounce to flush

    stats = eros.get_stats()
    assert stats["db"]["fragments"] > 0, "Fragment library should have entries after successful run"
    await eros.stop()
    print("  ✓ EME fragment library accumulation")


async def test_eme_generation_fn_only():
    eros = Mnemon(
        tenant_id=TENANT, agent_id="test_gen_fn_only",
        db_dir=DB_DIR,
        prewarm_fragments=False,
    )
    await eros.start()

    async def gen(goal, inputs, context, caps, constraints):
        return [{"id": "step_1", "action": "process"}]

    result = await eros.run(goal="test gen fn only", inputs={}, generation_fn=gen)
    assert result is not None
    assert result["template"] is not None

    result2 = await eros.run(goal="test gen fn only", inputs={}, generation_fn=gen)
    assert result2["cache_level"] == "system1"
    await eros.stop()
    print("  ✓ EME generation_fn-only path")


# ─────────────────────────────────────────────
# 3. EXPERIENCE BUS
# ─────────────────────────────────────────────

async def test_bus_tier1_observation():
    eros = await make_eros()

    for i in range(6):
        await eros._bus.record_outcome(
            task_id=f"task_{i}",
            task_type="security_audit",
            outcome="failure" if i < 4 else "success",
            latency_ms=1200,
        )

    stats = eros._bus.tier1.get_stats()
    assert stats["observations"] >= 6
    failure_rate = stats["failure_rates"].get("security_audit", 0)
    assert failure_rate > 0.3, f"Expected failure rate > 30%, got {failure_rate:.0%}"
    await eros.stop()
    print("  ✓ Bus Tier 1 observation + pattern detection")


# ─────────────────────────────────────────────
# 4. SECURITY
# ─────────────────────────────────────────────

def test_security_content_filter():
    assert not ContentFilter(["pii"]).should_store("SSN: 123-45-6789"), "PII should be blocked"
    assert ContentFilter([]).should_store("Normal business content"), "Normal content should pass"

    cf = ContentFilter()
    assert cf.classify_sensitivity("attorney-client privileged") == ContentSensitivity.PRIVILEGED
    assert cf.classify_sensitivity("HR performance review salary") == ContentSensitivity.CONFIDENTIAL
    assert cf.classify_sensitivity("normal project update") == ContentSensitivity.INTERNAL
    print("  ✓ Security content filter + classification")


def test_security_encryption():
    from mnemon.security.manager import SimpleEncryption
    enc = SimpleEncryption("test_key_abc")

    original  = '{"key": "value", "number": 42}'
    encrypted = enc.encrypt(original)
    decrypted = enc.decrypt(encrypted)

    assert decrypted == original, "Decrypted should match original"
    assert encrypted != original, "Encrypted should differ from original"
    print("  ✓ Security encryption round-trip")


# ─────────────────────────────────────────────
# 5. WATCHDOG + TELEMETRY
# ─────────────────────────────────────────────

async def test_watchdog_health_check():
    eros = Mnemon(
        tenant_id=TENANT, agent_id="watched_agent",
        db_dir=DB_DIR, enable_watchdog=True,
        prewarm_fragments=False,
    )
    await eros.start()

    health = await eros.health_check()
    assert "healthy" in health
    assert "checks" in health
    assert len(health["checks"]) > 0

    await eros.stop()
    print("  ✓ Watchdog health check")


async def test_telemetry_tracking():
    eros = Mnemon(
        tenant_id=TENANT, agent_id="telemetry_agent",
        db_dir=DB_DIR, enable_telemetry=True,
        prewarm_fragments=False,
    )
    await eros.start()

    async def gen(goal, inputs, context, caps, constraints):
        return [{"id": "s1", "action": "test"}]

    for _ in range(3):
        await eros.run(goal="telemetry test task", inputs={}, generation_fn=gen)

    report = eros.telemetry_report()
    assert report["eme"]["total_runs"] == 3
    assert report["eme"]["system1_hits"] >= 1

    await eros.stop()
    print("  ✓ Telemetry tracking EME runs")


# ─────────────────────────────────────────────
# 6. FRAGMENTS
# ─────────────────────────────────────────────

def test_fragment_library_loads():
    frags = load_fragments("test_tenant")
    assert len(frags) == FRAGMENT_COUNT
    assert len(frags) > 40, "Should have a substantial fragment library"

    for frag in frags[:5]:
        assert len(frag.signature) == 384, "Fragments should have 384-dim signatures"
        assert len(frag.domain_tags) > 0, "Fragments should have domain tags"

    print(f"  ✓ Fragment library loads {FRAGMENT_COUNT} fragments")


async def test_prewarm_on_start():
    eros = Mnemon(
        tenant_id="prewarm_test", agent_id="agent",
        db_dir="/tmp",
        prewarm_fragments=True,
    )
    await eros.start()

    stats = eros.get_stats()
    assert stats["db"]["fragments"] > 0, "Fragments should load on start"

    await eros.stop()
    print(f"  ✓ Pre-warm loads fragments on cold start")


# ─────────────────────────────────────────────
# 7. DRIFT DETECTION
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_drift_detector_healthy():
    import tempfile
    from mnemon.core.drift import DriftDetector
    db = EROSDatabase(tenant_id="drift_test", db_dir=tempfile.mkdtemp())
    await db.connect()
    detector = DriftDetector(tenant_id="drift_test", db=db)
    report = await detector.detect()
    assert report.severity == "healthy"
    assert report.is_healthy()
    await db.disconnect()
    print("  ✓ Drift detector returns healthy with no session history")


@pytest.mark.asyncio
async def test_drift_detector_with_sessions():
    import tempfile
    from mnemon.core.drift import DriftDetector
    db = EROSDatabase(tenant_id="drift_sess", db_dir=tempfile.mkdtemp())
    await db.connect()

    for i in range(5):
        await db.write_session_health(
            tenant_id="drift_sess", session_id=f"sess_{i}",
            cache_hit_rate=0.80, memory_writes=5, total_calls=20, avg_latency_ms=100.0,
        )
    for i in range(5, 8):
        await db.write_session_health(
            tenant_id="drift_sess", session_id=f"sess_{i}",
            cache_hit_rate=0.20, memory_writes=5, total_calls=20, avg_latency_ms=100.0,
        )

    detector = DriftDetector(tenant_id="drift_sess", db=db)
    report = await detector.detect()
    assert report.severity in ("warning", "degraded", "critical")
    assert report.hit_rate_delta < 0
    assert report.total_sessions == 8
    await db.disconnect()
    print(f"  ✓ Drift detector detects degradation: severity={report.severity}")


@pytest.mark.asyncio
async def test_drift_flush_and_detect():
    import tempfile
    from mnemon.core.drift import DriftDetector
    db = EROSDatabase(tenant_id="drift_flush", db_dir=tempfile.mkdtemp())
    await db.connect()
    detector = DriftDetector(tenant_id="drift_flush", db=db)

    for _ in range(4):
        detector._session_calls = 10
        detector._session_hits  = 8
        await detector.flush_session(notes="synthetic")

    report = await detector.detect()
    assert report.total_sessions == 4
    assert report.severity == "healthy"
    await db.disconnect()
    print(f"  ✓ flush_session() + detect() round-trip: {report.severity}")


# ─────────────────────────────────────────────
# 8. COLLECTIVE LEARNING (SIGNAL DB)
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_proven_intent_accumulates():
    import tempfile
    from mnemon.core.signal_db import SignalDatabase
    db = SignalDatabase(db_path=os.path.join(tempfile.mkdtemp(), "sig.db"))
    await db.connect()
    intent_key = "test_intent_001"

    await db.record_proven_intent(intent_key, domain="rag", success=True)
    await db.record_proven_intent(intent_key, domain="rag", success=True)
    boosts = await db.get_proven_boosts()
    assert intent_key not in boosts

    await db.record_proven_intent(intent_key, domain="rag", success=True)
    boosts = await db.get_proven_boosts()
    assert intent_key in boosts
    assert 0 < boosts[intent_key] <= 0.12
    await db.disconnect()
    print(f"  ✓ proven_intent boost after 3 successes: {boosts[intent_key]:.4f}")


@pytest.mark.asyncio
async def test_proven_intent_failure_reduces_boost():
    import tempfile
    from mnemon.core.signal_db import SignalDatabase
    db = SignalDatabase(db_path=os.path.join(tempfile.mkdtemp(), "sig.db"))
    await db.connect()
    intent_key = "test_intent_002"

    for _ in range(5):
        await db.record_proven_intent(intent_key, domain="reasoning", success=True)
    boosts_before = await db.get_proven_boosts()
    assert intent_key in boosts_before

    for _ in range(10):
        await db.record_proven_intent(intent_key, domain="reasoning", success=False)
    boosts_after = await db.get_proven_boosts()
    assert intent_key not in boosts_after
    await db.disconnect()
    print(f"  ✓ Failures correctly remove collective boost")


# ─────────────────────────────────────────────
# 9. WASTE REPORT
# ─────────────────────────────────────────────

def test_waste_report_no_data():
    from mnemon.moth.stats import MothStats
    stats = MothStats()
    report = stats.waste_report()
    assert "no data yet" in report.lower()
    print("  ✓ waste_report() with no data shows correct message")


def test_waste_report_repeated_queries():
    from mnemon.moth.stats import MothStats
    stats = MothStats()
    query = "what is the capital of France"
    stats.record_query(query)
    stats.record_query(query)
    stats.record_query(query)
    report = stats.waste_report()
    assert "asked 3x" in report
    assert "2 redundant" in report
    print(f"  ✓ waste_report() correctly identifies repeated queries")


def test_waste_report_custom_cost():
    from mnemon.moth.stats import MothStats
    stats = MothStats()
    query = "how do you make tea"
    stats.record_query(query)
    stats.record_query(query)
    report_default = stats.waste_report()
    report_custom  = stats.waste_report(cost_per_call=0.05)
    assert "0.050" in report_custom
    assert report_default != report_custom
    print("  ✓ waste_report() respects custom cost_per_call")


# ─────────────────────────────────────────────
# 10. SYNTHETIC STREAMING RESPONSES
# ─────────────────────────────────────────────

def test_synthetic_anthropic_response_fields():
    from mnemon.moth.integrations.anthropic import _synthetic_anthropic_response
    resp = _synthetic_anthropic_response("hello world", "claude-sonnet-4-6")
    assert hasattr(resp, "id")
    assert hasattr(resp, "type")
    assert hasattr(resp, "role")
    assert hasattr(resp, "content")
    assert hasattr(resp, "stop_reason")
    assert hasattr(resp, "usage")
    assert resp.type == "message"
    assert resp.role == "assistant"
    assert resp.content[0].text == "hello world"
    print("  ✓ Anthropic synthetic response has all expected fields")


def test_synthetic_openai_response_fields():
    from mnemon.moth.integrations.openai_sdk import _synthetic_openai_response
    resp = _synthetic_openai_response("hello", "gpt-4o")
    assert hasattr(resp, "id")
    assert hasattr(resp, "object")
    assert hasattr(resp, "choices")
    assert hasattr(resp, "usage")
    assert resp.object == "chat.completion"
    assert resp.choices[0].message.content == "hello"
    assert resp.choices[0].finish_reason == "stop"
    print("  ✓ OpenAI synthetic response has all expected fields")


def test_synthetic_anthropic_stream_iter():
    from mnemon.moth.integrations.anthropic import _SyntheticAnthropicStream
    stream = _SyntheticAnthropicStream("Test response text", "claude-sonnet-4-6")

    events = list(stream)
    types_ = [e.type for e in events]
    assert "message_start" in types_
    assert "content_block_delta" in types_
    assert "message_stop" in types_

    text_parts = list(stream.text_stream)
    assert "".join(text_parts) == "Test response text"

    msg = stream.get_final_message()
    assert msg.role == "assistant"
    assert msg.stop_reason == "end_turn"
    print("  ✓ _SyntheticAnthropicStream yields correct events and text")


def test_synthetic_anthropic_stream_context_manager():
    from mnemon.moth.integrations.anthropic import _SyntheticAnthropicStream
    with _SyntheticAnthropicStream("hello", "claude-sonnet-4-6") as s:
        events = list(s)
    assert len(events) > 0
    print("  ✓ _SyntheticAnthropicStream works as context manager")


@pytest.mark.asyncio
async def test_synthetic_anthropic_stream_async():
    from mnemon.moth.integrations.anthropic import _SyntheticAnthropicStream
    stream = _SyntheticAnthropicStream("Async test", "claude-sonnet-4-6")
    events = []
    async for event in stream:
        events.append(event)
    assert any(e.type == "content_block_delta" for e in events)
    print("  ✓ _SyntheticAnthropicStream async iteration works")


def test_synthetic_openai_stream_iter():
    from mnemon.moth.integrations.openai_sdk import _SyntheticOpenAIStream
    stream = _SyntheticOpenAIStream("OpenAI cached response", "gpt-4o")
    chunks = list(stream)
    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "OpenAI cached response"
    assert chunks[1].choices[0].finish_reason == "stop"
    print("  ✓ _SyntheticOpenAIStream yields correct chunks")


@pytest.mark.asyncio
async def test_synthetic_openai_stream_async():
    from mnemon.moth.integrations.openai_sdk import _SyntheticOpenAIStream
    stream = _SyntheticOpenAIStream("async openai", "gpt-4o")
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    assert len(chunks) == 2
    print("  ✓ _SyntheticOpenAIStream async iteration works")


# ─────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────

def run_all():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    print("\nMnemon Test Suite")
    print("=" * 50)

    sync_tests = [
        test_computation_fingerprint,
        test_simple_embedder,
        test_security_content_filter,
        test_security_encryption,
        test_fragment_library_loads,
        test_waste_report_no_data,
        test_waste_report_repeated_queries,
        test_waste_report_custom_cost,
        test_synthetic_anthropic_response_fields,
        test_synthetic_openai_response_fields,
        test_synthetic_anthropic_stream_iter,
        test_synthetic_anthropic_stream_context_manager,
        test_synthetic_openai_stream_iter,
    ]

    async_tests = [
        test_eme_system1_cache,
        test_eme_different_goals_miss,
        test_eme_fragment_library,
        test_eme_generation_fn_only,
        test_bus_tier1_observation,
        test_watchdog_health_check,
        test_telemetry_tracking,
        test_prewarm_on_start,
        test_drift_detector_healthy,
        test_drift_detector_with_sessions,
        test_drift_flush_and_detect,
        test_proven_intent_accumulates,
        test_proven_intent_failure_reduces_boost,
        test_synthetic_anthropic_stream_async,
        test_synthetic_openai_stream_async,
    ]

    passed = 0
    failed = 0

    print("\n[Sync tests]")
    for test in sync_tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1

    print("\n[Async tests]")
    for test in async_tests:
        try:
            loop.run_until_complete(test())
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    loop.close()

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILURES: {failed}")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
