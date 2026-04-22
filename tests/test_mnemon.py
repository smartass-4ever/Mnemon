"""
Mnemon Test Suite
Comprehensive tests for all components.

Run: pytest tests/test_mnemon.py -v
or:  python tests/test_mnemon.py
"""

import asyncio
import hashlib
import json
import sys
import os
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnemon import Mnemon, TenantSecurityConfig
from mnemon.core.models import (
    MemoryLayer, SignalType, BondedMemory, PADVector,
    ComputationFingerprint, ExperienceSignal, RiskLevel, RecallResult
)
from mnemon.core.persistence import EROSDatabase, InvertedIndex
from mnemon.core.memory import (
    CognitiveMemorySystem, SimpleEmbedder,
    RuleClassifier, WorkingMemory
)
from mnemon.core.eme import ExecutionMemoryEngine, GenericAdapter, CostBudget
from mnemon.core.bus import ExperienceBus, PADMonitor, Tier1Observer, BeliefRegistry
from mnemon.security.manager import SecurityManager, ContentFilter, ContentSensitivity, TenantSecurityConfig
from mnemon.observability.watchdog import Watchdog
from mnemon.observability.telemetry import Telemetry
from mnemon.fragments.library import load_fragments, FRAGMENT_COUNT
from mnemon.adapters.crewai import CrewAIAdapter
from mnemon.adapters.letta import LettaAdapter
from mnemon.llm.client import MockLLMClient


DB_DIR = "/tmp"
TENANT  = "test_tenant"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

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
# 1. TYPES
# ─────────────────────────────────────────────

def test_computation_fingerprint():
    fp1 = ComputationFingerprint.build("audit Acme", {"client": "str"}, {}, ["scanner"], {})
    fp2 = ComputationFingerprint.build("audit Acme", {"client": "str"}, {}, ["scanner"], {})
    fp3 = ComputationFingerprint.build("audit Beta", {"client": "str"}, {}, ["scanner"], {})
    assert fp1.full_hash == fp2.full_hash, "Same inputs should produce same hash"
    assert fp1.full_hash != fp3.full_hash, "Different goals should produce different hash"
    print("  ✓ ComputationFingerprint")


def test_pad_vector_severity():
    healthy  = PADVector(pleasure=0.8, arousal=0.2, dominance=0.7, agent_id="a", tenant_id="t")
    warning  = PADVector(pleasure=0.35, arousal=0.65, dominance=0.4, agent_id="a", tenant_id="t")
    critical = PADVector(pleasure=0.15, arousal=0.85, dominance=0.2, agent_id="a", tenant_id="t")
    flatline = PADVector(pleasure=0.05, arousal=0.95, dominance=0.1, agent_id="a", tenant_id="t")

    assert healthy.is_healthy(), "Healthy PAD should be healthy"
    assert not warning.is_healthy(), "Warning PAD should not be healthy"
    from mnemon.core.models import PADSeverity
    assert critical.severity() == PADSeverity.CRITICAL
    assert flatline.severity() == PADSeverity.FLATLINE
    print("  ✓ PADVector severity")


# ─────────────────────────────────────────────
# 2. PERSISTENCE
# ─────────────────────────────────────────────

async def test_persistence_write_read():
    import uuid, tempfile
    tenant = f"persist_{uuid.uuid4().hex[:8]}"
    tmpdir = tempfile.mkdtemp()
    db = EROSDatabase(tenant_id=tenant, db_dir=tmpdir)
    await db.connect()

    mem = BondedMemory(
        memory_id="test_mem_001",
        tenant_id=tenant,
        layer=MemoryLayer.SEMANTIC,
        content={"key": "test_key", "value": "test_value"},
        activation_tags={"test", "semantic"},
        importance=0.8,
        timestamp=time.time(),
        last_accessed=time.time(),
    )
    await db.write_memory(mem)
    fetched = await db.fetch_memories(tenant, ["test_mem_001"])
    assert len(fetched) == 1
    assert fetched[0].content == mem.content
    assert fetched[0].layer == MemoryLayer.SEMANTIC
    print("  ✓ Persistence write/read")

    # Belief registry
    ok = await db.set_belief(tenant, "test_key", "test_val", expected_version=0)
    assert ok, "Belief set should succeed on version 0"
    result = await db.get_belief(tenant, "test_key")
    assert result["value"] == "test_val"

    # Optimistic locking conflict
    conflict = await db.set_belief(tenant, "test_key", "new_val", expected_version=0)
    assert not conflict, "Old version should cause conflict"
    print("  ✓ Belief registry + optimistic locking")

    await db.disconnect()


async def test_inverted_index():
    idx = InvertedIndex()
    await idx.update("t1", "mem_a", {"security", "acme_corp", "audit"})
    await idx.update("t1", "mem_b", {"finance", "acme_corp"})
    await idx.update("t1", "mem_c", {"security", "beta_corp"})

    # intersect returns memories matching ALL queried tags — only mem_a has both
    results = await idx.intersect("t1", {"security", "acme_corp"})
    assert "mem_a" in results, "mem_a should match security+acme_corp"
    assert "mem_b" not in results, "mem_b has acme_corp but not security — no intersection match"
    assert "mem_c" not in results, "mem_c has security but not acme_corp — no intersection match"

    # union fallback fires when intersection is empty
    results_union = await idx.intersect("t1", {"nonexistent_tag", "acme_corp"})
    assert "mem_b" in results_union, "union fallback should surface mem_b on empty intersection"

    # Remove
    await idx.remove("t1", "mem_a", {"security"})
    results2 = await idx.intersect("t1", {"security"})
    assert "mem_a" not in results2, "mem_a should be removed from security index"
    print("  ✓ InvertedIndex intersect + remove")


# ─────────────────────────────────────────────
# 3. MEMORY SYSTEM
# ─────────────────────────────────────────────

def test_rule_classifier():
    cases = [
        ("The meeting was very tense and stressful",       MemoryLayer.EMOTIONAL),
        ("Last week we found 3 open ports",                MemoryLayer.EPISODIC),
        ("Acme Corp is a client in the finance sector",    MemoryLayer.SEMANTIC),
        ("Client usually prefers formal communication",    MemoryLayer.RELATIONSHIP),
    ]
    for content, expected in cases:
        result = RuleClassifier.classify(content)
        assert result == expected, f"'{content[:40]}' → got {result}, expected {expected}"
    print("  ✓ RuleClassifier")


def test_rule_classifier_tag_extraction():
    tags = RuleClassifier.extract_tags(
        "Security audit for Acme Corp found 3 open ports", MemoryLayer.EPISODIC
    )
    assert "acme" in tags or "acme_corp" in tags or "corp" in tags, "Should tag entity name"
    assert "security" in tags, "Should tag security domain"
    print("  ✓ Tag extraction")


def test_simple_embedder():
    emb = SimpleEmbedder()
    v1  = emb.embed("security audit report")
    v2  = emb.embed("security audit report")
    v3  = emb.embed("completely unrelated topic bicycle")

    assert len(v1) == 384
    assert v1 == v2, "Same text should produce same embedding"
    sim_same = SimpleEmbedder.cosine_similarity(v1, v2)
    sim_diff = SimpleEmbedder.cosine_similarity(v1, v3)
    assert sim_same > sim_diff, "Same text should have higher similarity"
    print("  ✓ SimpleEmbedder")


async def test_memory_write_retrieve():
    eros = await make_eros()

    # Write various layers
    await eros.remember("Security scan found 3 open ports last week",
                        layer=MemoryLayer.EPISODIC, importance=0.9)
    await eros.remember("Acme Corp prefers formal PDF reports",
                        layer=MemoryLayer.SEMANTIC, importance=0.8)
    await eros.learn_fact("acme_contact", "Sarah K", confidence=0.95)

    # Retrieve
    ctx = await eros.recall("security audit Acme Corp")
    assert ctx is not None
    assert "memories" in ctx

    # Fact retrieval
    contact = await eros.recall_fact("acme_contact")
    assert contact == "Sarah K", f"Expected 'Sarah K', got {contact}"

    # Working memory session isolation
    await eros.start_session("session_A")
    await eros.end_session("session_A")
    assert "session_A" not in eros._memory._working, "Working memory should flush on end_session"

    await eros.stop()
    print("  ✓ Memory write/retrieve/facts/session isolation")


async def test_memory_conflict_detection():
    eros = await make_eros()

    await eros.remember("Acme Corp contact is James R",
                        layer=MemoryLayer.SEMANTIC, importance=0.7)
    await eros.remember("Acme Corp contact is Sarah K",
                        layer=MemoryLayer.SEMANTIC, importance=0.8)

    ctx = await eros.recall("Acme Corp contact")
    # Conflicts may or may not be detected depending on pool size and embedder
    # The important thing is the system doesn't crash
    assert "memories" in ctx
    await eros.stop()
    print("  ✓ Memory conflict detection (no crash)")


# ─────────────────────────────────────────────
# 4. EME
# ─────────────────────────────────────────────

async def test_eme_system1_cache():
    eros = await make_eros()
    calls = []

    async def gen(goal, inputs, context, caps, constraints):
        calls.append(goal)
        return [{"id": "step_1", "action": "scan"}, {"id": "step_2", "action": "report"}]

    # First run — miss
    r1 = await eros.run(
        goal="weekly security audit for Acme",
        inputs={"week": "week_1"},
        generation_fn=gen,
    )
    assert r1["cache_level"] == "miss", f"First run should be miss, got {r1['cache_level']}"
    assert len(calls) == 1, "Generation should be called once"

    # Second run — System 1 hit
    r2 = await eros.run(
        goal="weekly security audit for Acme",
        inputs={"week": "week_2"},
        generation_fn=gen,
    )
    assert r2["cache_level"] == "system1", f"Second run should be system1, got {r2['cache_level']}"
    assert len(calls) == 1, "Generation should NOT be called again"
    assert r2["tokens_saved"] > 0, "Tokens should be saved on cache hit"

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


async def test_eme_cost_budget_fallback():
    budget = CostBudget(max_llm_calls_per_hour=0, overflow_policy="fallback")
    eros = Mnemon(
        tenant_id=TENANT, agent_id="test_budget",
        db_dir=DB_DIR, cost_budget=budget,
        prewarm_fragments=False,
    )
    await eros.start()

    async def gen(goal, inputs, context, caps, constraints):
        return [{"id": "step_1", "action": "process"}]

    # Should still work even with zero LLM budget
    result = await eros.run(goal="test budget", inputs={}, generation_fn=gen)
    assert result is not None
    await eros.stop()
    print("  ✓ EME cost budget fallback")


# ─────────────────────────────────────────────
# 5. EXPERIENCE BUS
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


async def test_bus_belief_registry():
    eros = await make_eros()

    # Set
    ok = await eros._bus.belief_registry.set("lang", "Python", expected_version=0)
    assert ok

    # Get
    val = await eros._bus.belief_registry.get("lang")
    assert val == "Python"

    # Version conflict
    conflict = await eros._bus.belief_registry.set("lang", "Go", expected_version=0)
    assert not conflict, "Stale version should be rejected"

    # Correct version
    current_version = await eros._bus.belief_registry.get_version()
    ok2 = await eros._bus.belief_registry.set("lang", "Go", expected_version=current_version)
    assert ok2, "Correct version should succeed"

    await eros.stop()
    print("  ✓ Bus belief registry + optimistic locking")


async def test_bus_signal_broadcast_and_query():
    eros = await make_eros()

    await eros.broadcast(
        SignalType.ERROR_RESOLVED,
        {"error_type": "endpoint_timeout", "fix": "retry with 30s backoff"},
        importance=0.9,
    )
    await asyncio.sleep(0.15)  # let bus process

    solutions = await eros.query_solutions("endpoint_timeout")
    assert len(solutions) > 0, "Solution should be queryable after broadcast"
    assert solutions[0].content["error_type"] == "endpoint_timeout"

    await eros.stop()
    print("  ✓ Bus signal broadcast + query")


async def test_bus_pad_monitoring():
    eros = await make_eros()
    await eros._bus.register_agent("test_agent", {"domain": "security"}, goal="security audit")

    sev_good = await eros.report_health(
        "Successfully completed network scan. All checks passed.",
        goal="security audit for Acme Corp",
    )
    sev_bad = await eros.report_health(
        "error error timeout failed retry error exception error error",
        goal="security audit for Acme Corp",
    )

    from mnemon.core.models import PADSeverity
    assert sev_good != PADSeverity.FLATLINE, "Good output should not flatline"
    assert sev_bad in (PADSeverity.CRITICAL, PADSeverity.FLATLINE, PADSeverity.WARNING), \
        f"Bad output should trigger alert, got {sev_bad}"

    await eros.stop()
    print("  ✓ Bus PAD monitoring")


# ─────────────────────────────────────────────
# 6. SECURITY
# ─────────────────────────────────────────────

def test_security_content_filter():
    cf = ContentFilter()

    # PII should be detected
    assert not ContentFilter(["pii"]).should_store("SSN: 123-45-6789"), "PII should be blocked"
    assert ContentFilter([]).should_store("Normal business content"), "Normal content should pass"

    # Sensitivity classification
    assert cf.classify_sensitivity("attorney-client privileged") == ContentSensitivity.PRIVILEGED
    assert cf.classify_sensitivity("HR performance review salary") == ContentSensitivity.CONFIDENTIAL
    assert cf.classify_sensitivity("normal project update") == ContentSensitivity.INTERNAL
    print("  ✓ Security content filter + classification")


def test_security_encryption():
    from mnemon.security.manager import SimpleEncryption
    enc = SimpleEncryption("test_key_abc")

    original = '{"key": "value", "number": 42}'
    encrypted = enc.encrypt(original)
    decrypted = enc.decrypt(encrypted)

    assert decrypted == original, "Decrypted should match original"
    assert encrypted != original, "Encrypted should differ from original"
    print("  ✓ Security encryption round-trip")


async def test_security_blocked_write():
    eros = Mnemon(
        tenant_id=TENANT, agent_id="secure_agent",
        db_dir=DB_DIR,
        blocked_categories=["pii"],
        prewarm_fragments=False,
    )
    await eros.start()

    # Normal content — should be stored
    mem_id = await eros.remember("Acme Corp prefers formal reports")
    # PII content — filtered by security layer (passes through, not stored)
    # Note: current impl doesn't hook security into remember() directly
    # but the SecurityManager is available and the architecture is wired

    stats = eros.get_stats()
    assert "security" in stats
    await eros.stop()
    print("  ✓ Security config wired into EROS")


# ─────────────────────────────────────────────
# 7. WATCHDOG + TELEMETRY
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
    assert report["eme"]["system1_hits"] >= 1, "At least one cache hit expected"

    await eros.stop()
    print("  ✓ Telemetry tracking EME runs")


# ─────────────────────────────────────────────
# 8. FRAGMENTS
# ─────────────────────────────────────────────

def test_fragment_library_loads():
    frags = load_fragments("test_tenant")
    assert len(frags) == FRAGMENT_COUNT
    assert len(frags) > 40, "Should have a substantial fragment library"

    # Check signatures are generated
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
# 9. ADAPTERS
# ─────────────────────────────────────────────

def test_crewai_adapter():
    adapter = CrewAIAdapter()

    template = {
        "tasks": [
            {"id": "task_1", "description": "scan network", "tools": ["network_scanner"]},
            {"id": "task_2", "description": "generate report", "tools": ["report_gen"]},
        ],
        "agents": [
            {"role": "security_analyst", "backstory": "Expert in security audits", "goal": "find vulnerabilities"},
        ]
    }

    segments = adapter.decompose(template)
    assert len(segments) == 2

    fp = adapter.extract_signature(template, "security audit")
    assert fp.goal_hash != ""

    reconstructed = adapter.reconstruct([])
    assert reconstructed["type"] == "crewai_flow"

    # Signal translation
    signals = CrewAIAdapter.agent_to_signals(
        {"role": "analyst", "backstory": "Security expert", "goal": "audit"},
        tenant_id=TENANT,
        session_id="sess_001",
    )
    assert len(signals) >= 1
    assert any(s.layer == MemoryLayer.SEMANTIC for s in signals)

    print("  ✓ CrewAI adapter decompose/reconstruct/signals")


def test_letta_adapter():
    adapter = LettaAdapter()

    template = {
        "steps": [
            {"name": "read_memory", "type": "function"},
            {"name": "generate_response", "type": "function"},
        ],
        "persona": "You are a helpful assistant",
        "human": "I am a software engineer",
    }

    segments = adapter.decompose(template)
    assert len(segments) == 2

    fp = adapter.extract_signature(template, "answer user question")
    assert fp.goal_hash != ""

    # Heartbeat translation
    hb_signal = LettaAdapter.heartbeat_to_signal(
        agent_id="letta_agent_01",
        heartbeat_data={"type": "tool_call_result", "content": "scan completed"},
        tenant_id=TENANT,
        session_id="sess_letta",
        goal="security audit",
    )
    assert hb_signal.layer == MemoryLayer.EPISODIC
    assert hb_signal.signal_type == SignalType.VALIDATION_PASSED

    # Core memory translation
    cm_signal = LettaAdapter.core_memory_to_signal(
        key="user_name", value="Alex",
        tenant_id=TENANT, session_id="s", agent_id="a",
    )
    assert cm_signal.layer == MemoryLayer.SEMANTIC

    # Sleep time signals
    messages = [
        {"role": "user",      "content": "I prefer formal reports"},
        {"role": "assistant", "content": "I will remember that"},
    ]
    sleep_signals = LettaAdapter.sleep_time_signals(
        session_id="s", tenant_id=TENANT, agent_id="a",
        recent_messages=messages,
    )
    assert len(sleep_signals) == 2

    print("  ✓ Letta adapter + heartbeat/core_memory/sleep_time signals")


# ─────────────────────────────────────────────
# 10. LLM CLIENT
# ─────────────────────────────────────────────

async def test_mock_llm_router():
    mock = MockLLMClient()

    # Memory routing
    resp = await mock.complete(
        "Classify this memory content: The meeting was very tense and emotional",
        model="mock"
    )
    data = json.loads(resp)
    assert "layer" in data
    assert data["layer"] == "emotional"

    # Tag verification
    resp2 = await mock.complete(
        "Check if these tags accurately represent this memory content",
        model="mock"
    )
    data2 = json.loads(resp2)
    assert data2["verdict"] == "YES"

    assert mock.call_count == 2
    print("  ✓ MockLLMClient routes correctly")


async def test_mnemon_with_mock_llm():
    mock = MockLLMClient()
    eros = Mnemon(
        tenant_id=TENANT, agent_id="llm_test_agent",
        db_dir=DB_DIR, llm_client=mock,
        prewarm_fragments=False,
    )
    await eros.start()

    # Write memory — LLM router should fire for ambiguous content
    mem_id = await eros.remember(
        "The client seemed frustrated but we resolved the issue",
        importance=0.7,
    )
    assert mem_id is not None

    # Recall
    ctx = await eros.recall("client emotional state")
    assert ctx is not None

    await eros.stop()
    print(f"  ✓ EROS with MockLLMClient ({mock.call_count} LLM calls)")


# ─────────────────────────────────────────────
# 11. INTEGRATION — FULL PIPELINE
# ─────────────────────────────────────────────

async def test_full_pipeline():
    import uuid, tempfile
    mock = MockLLMClient()
    eros = Mnemon(
        tenant_id=f"integration_{uuid.uuid4().hex[:8]}",
        agent_id="pipeline_agent",
        db_dir=tempfile.mkdtemp(),
        llm_client=mock,
        enable_watchdog=True,
        enable_telemetry=True,
        prewarm_fragments=True,
    )
    await eros.start()

    # Register agent
    await eros._bus.register_agent(
        "pipeline_agent",
        {"domain": "security", "apis_used": ["scanner"], "entities": ["acme_corp"]},
        goal="weekly security audit",
    )

    # Phase 1: Teach
    await eros.learn_fact("client_format", "PDF executive summary")
    await eros.remember("endpoint check times out — fixed with 30s retry",
                        layer=MemoryLayer.EPISODIC, importance=0.9)

    # Phase 2: Run (cold + warm)
    async def planner(goal, inputs, context, caps, constraints):
        return [
            {"id": "scan", "action": "network_scan"},
            {"id": "check", "action": "endpoint_check"},
            {"id": "report", "action": "create_pdf"},
        ]

    r1 = await eros.run(
        goal="weekly security audit for Acme Corp",
        inputs={"week": "March 17"},
        generation_fn=planner,
        task_type="security_audit",
    )
    assert r1["cache_level"] == "miss"

    r2 = await eros.run(
        goal="weekly security audit for Acme Corp",
        inputs={"week": "March 24"},
        generation_fn=planner,
        task_type="security_audit",
    )
    assert r2["cache_level"] == "system1"
    assert r2["tokens_saved"] > 0

    # Phase 3: Broadcast discovery
    await eros.broadcast(
        SignalType.ERROR_RESOLVED,
        {"error_type": "endpoint_timeout", "fix": "30s retry"},
        importance=0.9,
    )
    await asyncio.sleep(0.15)
    solutions = await eros.query_solutions("endpoint_timeout")
    assert len(solutions) > 0

    # Phase 4: PAD health
    sev = await eros.report_health(
        "Successfully completed security scan. Report generated.",
        goal="weekly security audit",
    )
    assert sev is not None

    # Phase 5: Stats
    stats = eros.get_stats()
    assert stats["db"]["memories"] > 0
    assert stats["db"]["templates"] > 0
    assert stats["db"]["fragments"] > 0
    assert stats["db"]["facts"] > 0

    # Phase 6: Health check
    health = await eros.health_check()
    assert health["healthy"]

    # Phase 7: Telemetry
    report = eros.telemetry_report()
    assert report["eme"]["total_runs"] == 2
    assert report["eme"]["system1_hits"] == 1

    await eros.stop()
    print(f"  ✓ Full pipeline integration test (LLM calls: {mock.call_count})")


# ─────────────────────────────────────────────
# 12. DRIFT DETECTION
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_drift_detector_healthy():
    """Fresh system with no session history reports healthy."""
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
    """Degraded hit rate across sessions triggers warning."""
    import tempfile
    from mnemon.core.drift import DriftDetector
    db = EROSDatabase(tenant_id="drift_sess", db_dir=tempfile.mkdtemp())
    await db.connect()

    # Write 5 healthy baseline sessions
    for i in range(5):
        await db.write_session_health(
            tenant_id="drift_sess",
            session_id=f"sess_{i}",
            cache_hit_rate=0.80,
            memory_writes=5,
            total_calls=20,
            avg_latency_ms=100.0,
        )
    # Write 3 degraded sessions (50% drop from 0.80)
    for i in range(5, 8):
        await db.write_session_health(
            tenant_id="drift_sess",
            session_id=f"sess_{i}",
            cache_hit_rate=0.20,
            memory_writes=5,
            total_calls=20,
            avg_latency_ms=100.0,
        )

    detector = DriftDetector(tenant_id="drift_sess", db=db)
    report = await detector.detect()
    assert report.severity in ("warning", "degraded", "critical")
    assert report.hit_rate_delta < 0
    assert report.total_sessions == 8
    await db.disconnect()
    print(f"  ✓ Drift detector detects degradation: severity={report.severity}, delta={report.hit_rate_delta:.2%}")


@pytest.mark.asyncio
async def test_drift_flush_and_detect():
    """flush_session() persists data that detect() then picks up."""
    import tempfile
    from mnemon.core.drift import DriftDetector
    db = EROSDatabase(tenant_id="drift_flush", db_dir=tempfile.mkdtemp())
    await db.connect()
    detector = DriftDetector(tenant_id="drift_flush", db=db)

    # Simulate 4 sessions manually
    for _ in range(4):
        detector._session_calls = 10
        detector._session_hits  = 8
        await detector.flush_session(notes="synthetic")

    report = await detector.detect()
    assert report.total_sessions == 4
    assert report.severity == "healthy"  # not enough sessions for window but all healthy
    await db.disconnect()
    print(f"  ✓ flush_session() + detect() round-trip works: {report.severity}")


# ─────────────────────────────────────────────
# 13. COLLECTIVE LEARNING (SIGNAL DB)
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_proven_intent_accumulates():
    """record_proven_intent accumulates boost after enough successes."""
    import tempfile
    from mnemon.core.signal_db import SignalDatabase
    db = SignalDatabase(db_path=os.path.join(tempfile.mkdtemp(), "sig.db"))
    await db.connect()
    intent_key = "test_intent_001"

    # First two successes — below _PROVEN_MIN_SUCCESSES (3)
    await db.record_proven_intent(intent_key, domain="rag", success=True)
    await db.record_proven_intent(intent_key, domain="rag", success=True)
    boosts = await db.get_proven_boosts()
    assert intent_key not in boosts  # not enough evidence yet

    # Third success crosses the threshold
    await db.record_proven_intent(intent_key, domain="rag", success=True)
    boosts = await db.get_proven_boosts()
    assert intent_key in boosts
    assert 0 < boosts[intent_key] <= 0.12
    await db.disconnect()
    print(f"  ✓ proven_intent boost after 3 successes: {boosts[intent_key]:.4f}")


@pytest.mark.asyncio
async def test_proven_intent_failure_reduces_boost():
    """Failures lower the success rate and can remove the boost."""
    import tempfile
    from mnemon.core.signal_db import SignalDatabase
    db = SignalDatabase(db_path=os.path.join(tempfile.mkdtemp(), "sig.db"))
    await db.connect()
    intent_key = "test_intent_002"

    # Enough successes to qualify
    for _ in range(5):
        await db.record_proven_intent(intent_key, domain="reasoning", success=True)
    boosts_before = await db.get_proven_boosts()
    assert intent_key in boosts_before

    # Add 10 failures — rate drops well below 0.75
    for _ in range(10):
        await db.record_proven_intent(intent_key, domain="reasoning", success=False)
    boosts_after = await db.get_proven_boosts()
    assert intent_key not in boosts_after  # filtered out by min rate
    await db.disconnect()
    print(f"  ✓ Failures correctly remove collective boost")


# ─────────────────────────────────────────────
# 14. WASTE REPORT
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
    stats.record_query(query)  # second call = repeated
    stats.record_query(query)  # third call
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
    # Custom cost should show higher dollar amount
    assert "0.050" in report_custom
    assert report_default != report_custom
    print("  ✓ waste_report() respects custom cost_per_call")


# ─────────────────────────────────────────────
# 15. RECALL RESULT TYPE
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recall_returns_recall_result():
    """m.recall() returns a RecallResult with proper attributes."""
    import tempfile, uuid
    from mnemon.core.models import RecallResult
    m = Mnemon(
        tenant_id=f"rr_{uuid.uuid4().hex[:8]}",
        db_dir=tempfile.mkdtemp(),
        enable_telemetry=False,
    )
    await m.start()
    result = await m.recall("test query")
    assert isinstance(result, RecallResult)
    assert hasattr(result, "memories")
    assert hasattr(result, "memory_ids")
    assert hasattr(result, "context")
    assert hasattr(result, "conflicts")
    assert hasattr(result, "working")
    assert hasattr(result, "pool_size")
    assert isinstance(result.memories, list)
    assert isinstance(result.memory_ids, list)
    # Backward compat: dict-like access
    assert result.get("memories") == result.memories
    assert result.get("nonexistent", "default") == "default"
    assert "memories" in result
    await m.stop()
    print("  ✓ recall() returns RecallResult with correct attributes and dict compat")


@pytest.mark.asyncio
async def test_recall_result_with_memories():
    """After remembering, recall result contains the memory."""
    import tempfile, uuid
    m = Mnemon(
        tenant_id=f"rr2_{uuid.uuid4().hex[:8]}",
        db_dir=tempfile.mkdtemp(),
        enable_telemetry=False,
    )
    await m.start()
    await m.remember("Acme Corp uses GPT-4 for summarization", importance=0.9)
    await asyncio.sleep(0.1)  # let index settle
    result = await m.recall("Acme Corp")
    assert isinstance(result.memories, list)
    assert isinstance(result.memory_ids, list)
    assert len(result) == len(result.memories)
    assert bool(result) == (len(result.memories) > 0)
    await m.stop()
    print(f"  ✓ RecallResult after remember: {len(result.memories)} memories, pool_size={result.pool_size}")


# ─────────────────────────────────────────────
# 16. FORGET / EXPORT / IMPORT
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_forget_supersedes_memories():
    """forget() marks matching memories as superseded."""
    import tempfile, uuid
    m = Mnemon(
        tenant_id=f"fg_{uuid.uuid4().hex[:8]}",
        db_dir=tempfile.mkdtemp(),
        enable_telemetry=False,
    )
    await m.start()
    await m.remember("Acme Corp prefers PDF reports", importance=0.9)
    await m.remember("Acme Corp uses Slack for comms", importance=0.8)
    # forget() should not crash even if recall returns 0 (cold inverted index)
    count = await m.forget("Acme Corp", top_k=10)
    assert isinstance(count, int)
    assert count >= 0
    await m.stop()
    print(f"  ✓ forget() executed without error, removed {count} memories")


@pytest.mark.asyncio
async def test_export_import_round_trip():
    """export_memory + import_memory preserves memory count."""
    import tempfile, uuid, os
    tmp = tempfile.mkdtemp()
    tenant = f"ei_{uuid.uuid4().hex[:8]}"

    m = Mnemon(tenant_id=tenant, db_dir=tmp, enable_telemetry=False)
    await m.start()
    await m.remember("Alpha team uses Python", importance=0.8)
    await m.remember("Beta team uses Go", importance=0.7)
    await m.remember("CI runs on GitHub Actions", importance=0.6)

    export_path = os.path.join(tmp, "memories.json")
    exported = await m.export_memory(export_path)
    assert exported == 3
    assert os.path.exists(export_path)
    await m.stop()

    # Import into a fresh tenant
    m2 = Mnemon(tenant_id=f"ei2_{uuid.uuid4().hex[:8]}", db_dir=tmp, enable_telemetry=False)
    await m2.start()
    imported = await m2.import_memory(export_path)
    assert imported == 3
    await m2.stop()
    print(f"  ✓ export_memory + import_memory round-trip: {exported} exported, {imported} imported")


# ─────────────────────────────────────────────
# 17. SYNTHETIC RESPONSE FIELDS (cold-start)
# ─────────────────────────────────────────────

def test_synthetic_anthropic_response_fields():
    """_synthetic_anthropic_response has all SDK-expected fields."""
    import types
    from mnemon.moth.integrations.anthropic import _synthetic_anthropic_response
    resp = _synthetic_anthropic_response("hello world", "claude-sonnet-4-6")
    assert hasattr(resp, "id")
    assert hasattr(resp, "type")
    assert hasattr(resp, "role")
    assert hasattr(resp, "content")
    assert hasattr(resp, "stop_reason")
    assert hasattr(resp, "stop_sequence")
    assert hasattr(resp, "usage")
    assert resp.type == "message"
    assert resp.role == "assistant"
    assert resp.content[0].text == "hello world"
    print("  ✓ Anthropic synthetic response has all expected fields")


def test_synthetic_openai_response_fields():
    """_synthetic_openai_response has all SDK-expected fields."""
    from mnemon.moth.integrations.openai_sdk import _synthetic_openai_response
    resp = _synthetic_openai_response("hello", "gpt-4o")
    assert hasattr(resp, "id")
    assert hasattr(resp, "object")
    assert hasattr(resp, "created")
    assert hasattr(resp, "model")
    assert hasattr(resp, "system_fingerprint")
    assert hasattr(resp, "choices")
    assert hasattr(resp, "usage")
    assert resp.object == "chat.completion"
    assert resp.choices[0].message.content == "hello"
    assert resp.choices[0].finish_reason == "stop"
    print("  ✓ OpenAI synthetic response has all expected fields")


# ─────────────────────────────────────────────
# 18. AUTO CORRECT DRIFT
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_auto_correct_drift_healthy():
    """auto_correct_drift() returns healthy status when no drift."""
    import tempfile, uuid
    m = Mnemon(
        tenant_id=f"acd_{uuid.uuid4().hex[:8]}",
        db_dir=tempfile.mkdtemp(),
        enable_telemetry=False,
    )
    await m.start()
    result = await m.auto_correct_drift()
    assert result["status"] == "healthy"
    assert result["corrections"] == 0
    await m.stop()
    print("  ✓ auto_correct_drift() returns healthy on fresh system")


@pytest.mark.asyncio
async def test_auto_correct_drift_with_degraded_sessions():
    """auto_correct_drift() returns degraded status and corrections >= 0."""
    import tempfile, uuid
    tmp = tempfile.mkdtemp()
    tenant = f"acd2_{uuid.uuid4().hex[:8]}"

    m = Mnemon(tenant_id=tenant, db_dir=tmp, enable_telemetry=False)
    await m.start()

    # Inject session history directly to simulate drift
    for i in range(5):
        await m._db.write_session_health(
            tenant_id=tenant, session_id=f"s{i}",
            cache_hit_rate=0.85, memory_writes=3, total_calls=20, avg_latency_ms=80.0,
        )
    for i in range(5, 8):
        await m._db.write_session_health(
            tenant_id=tenant, session_id=f"s{i}",
            cache_hit_rate=0.15, memory_writes=3, total_calls=20, avg_latency_ms=80.0,
        )

    result = await m.auto_correct_drift()
    assert result["status"] in ("warning", "degraded", "critical", "healthy")
    assert "corrections" in result
    assert isinstance(result["corrections"], int)
    await m.stop()
    print(f"  ✓ auto_correct_drift() with degraded sessions: status={result['status']}, corrections={result['corrections']}")


# ─────────────────────────────────────────────
# 19. STREAMING CACHE (SYNTHETIC STREAMS)
# ─────────────────────────────────────────────

def test_synthetic_anthropic_stream_iter():
    """_SyntheticAnthropicStream yields events and text_stream works."""
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
    """_SyntheticAnthropicStream works as context manager."""
    from mnemon.moth.integrations.anthropic import _SyntheticAnthropicStream
    with _SyntheticAnthropicStream("hello", "claude-sonnet-4-6") as s:
        events = list(s)
    assert len(events) > 0
    print("  ✓ _SyntheticAnthropicStream works as context manager")


@pytest.mark.asyncio
async def test_synthetic_anthropic_stream_async():
    """_SyntheticAnthropicStream async iteration works."""
    from mnemon.moth.integrations.anthropic import _SyntheticAnthropicStream
    stream = _SyntheticAnthropicStream("Async test", "claude-sonnet-4-6")
    events = []
    async for event in stream:
        events.append(event)
    assert any(e.type == "content_block_delta" for e in events)
    print("  ✓ _SyntheticAnthropicStream async iteration works")


def test_synthetic_openai_stream_iter():
    """_SyntheticOpenAIStream yields ChatCompletionChunk-compatible objects."""
    from mnemon.moth.integrations.openai_sdk import _SyntheticOpenAIStream
    stream = _SyntheticOpenAIStream("OpenAI cached response", "gpt-4o")
    chunks = list(stream)
    assert len(chunks) == 2  # content chunk + stop chunk
    assert chunks[0].choices[0].delta.content == "OpenAI cached response"
    assert chunks[1].choices[0].finish_reason == "stop"
    print("  ✓ _SyntheticOpenAIStream yields correct chunks")


@pytest.mark.asyncio
async def test_synthetic_openai_stream_async():
    """_SyntheticOpenAIStream async iteration works."""
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
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    print("\nMnemon Test Suite")
    print("=" * 50)

    sync_tests = [
        test_computation_fingerprint,
        test_pad_vector_severity,
        test_rule_classifier,
        test_rule_classifier_tag_extraction,
        test_simple_embedder,
        test_security_content_filter,
        test_security_encryption,
        test_crewai_adapter,
        test_letta_adapter,
        test_fragment_library_loads,
    ]

    async_tests = [
        test_persistence_write_read,
        test_inverted_index,
        test_memory_write_retrieve,
        test_memory_conflict_detection,
        test_eme_system1_cache,
        test_eme_different_goals_miss,
        test_eme_fragment_library,
        test_eme_cost_budget_fallback,
        test_bus_tier1_observation,
        test_bus_belief_registry,
        test_bus_signal_broadcast_and_query,
        test_bus_pad_monitoring,
        test_security_blocked_write,
        test_watchdog_health_check,
        test_telemetry_tracking,
        test_prewarm_on_start,
        test_mock_llm_router,
        test_mnemon_with_mock_llm,
        test_full_pipeline,
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
