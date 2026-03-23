"""
Mnemon Example — Full Pipeline Demo

Demonstrates all three components working together:
- Cognitive memory with five layers
- Execution Memory Engine (EME) with System 1/2
- Two-tier experience bus with PAD monitoring

Run: python example.py
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnemon import Mnemon
from mnemon.core.models import MemoryLayer, SignalType


# ─────────────────────────────────────────────
# Simulate an expensive planning function
# In production this calls your LLM
# ─────────────────────────────────────────────

async def planning_function(goal, inputs, context, capabilities, constraints):
    """Simulates an expensive LLM planning call (~15-30s in production)."""
    await asyncio.sleep(0.1)  # simulate latency
    return {
        "nodes": [
            {"id": "scan_network",    "action": "network_scan",    "params": inputs},
            {"id": "check_endpoints", "action": "endpoint_check",  "params": {"retry": True}},
            {"id": "generate_report", "action": "create_pdf",      "params": {"format": "executive_summary"}},
        ],
        "type": "security_audit_dag"
    }


async def main():
    print("=" * 60)
    print("EROS Intelligence Infrastructure Layer — Demo")
    print("=" * 60)

    async with Mnemon(
        tenant_id="demo_company",
        agent_id="security_agent",
        db_path="/tmp/eros_demo.db",
    ) as mnemon:

        # ── Setup: Register agent with experience bus ──
        await eros._bus.register_agent(
            "security_agent",
            context={
                "domain":    "security",
                "apis_used": ["network_scanner", "endpoint_checker"],
                "entities":  ["acme_corp"],
            },
            goal="weekly security audit",
            task_type="security_audit",
        )

        # ── Phase 1: Teach EROS about Acme Corp ────────
        print("\n[Phase 1] Writing memories...")

        await eros.learn_fact("acme_contact",       "Sarah K",               confidence=0.95)
        await eros.learn_fact("acme_report_format", "PDF executive summary", confidence=1.0)
        await eros.learn_fact("acme_network_scope", "3 subnets",             confidence=1.0)

        await eros.remember(
            "Acme Corp endpoint check times out — fixed with 30s retry",
            layer=MemoryLayer.EPISODIC,
            importance=0.9,
        )
        await eros.remember(
            "Acme Corp stakeholder James R prefers bullet-point summaries",
            layer=MemoryLayer.RELATIONSHIP,
            importance=0.7,
        )
        await eros.remember(
            "Last Acme Corp audit found 3 open ports on subnet 192.168.1.x",
            layer=MemoryLayer.EPISODIC,
            importance=0.85,
        )

        print("  ✓ 6 memories written across 3 layers")

        # ── Phase 2: First run (cache miss — plan generated) ──
        print("\n[Phase 2] First run — cold start...")
        import time
        t0 = time.time()

        result1 = await eros.run(
            goal="weekly security audit for Acme Corp",
            inputs={"client": "Acme Corp", "week": "March 17-21", "scope": "network+endpoints"},
            generation_fn=planning_function,
            capabilities=["network_scanner", "endpoint_checker"],
            task_type="security_audit",
        )
        t1 = time.time()

        print(f"  Cache level:     {result1['cache_level']}")
        print(f"  Latency:         {(t1-t0)*1000:.0f}ms")
        print(f"  Memory context:  {len(result1['memory_context'].get('memories', []))} memories retrieved")
        if result1['memory_context'].get('conflicts'):
            print(f"  Conflicts:       {len(result1['memory_context']['conflicts'])} flagged")

        # ── Phase 3: Second run (System 1 cache hit) ──
        print("\n[Phase 3] Second run — same task, different week...")
        t2 = time.time()

        result2 = await eros.run(
            goal="weekly security audit for Acme Corp",
            inputs={"client": "Acme Corp", "week": "March 24-28", "scope": "network+endpoints"},
            generation_fn=planning_function,
            capabilities=["network_scanner", "endpoint_checker"],
            task_type="security_audit",
        )
        t3 = time.time()

        print(f"  Cache level:     {result2['cache_level']}")
        print(f"  Latency:         {(t3-t2)*1000:.0f}ms")
        print(f"  Tokens saved:    {result2['tokens_saved']}")
        print(f"  Segments reused: {result2['segments_reused']}")

        # ── Phase 4: PAD Health Monitoring ────────────
        print("\n[Phase 4] PAD health monitoring...")

        sev_good = await eros.report_health(
            "Successfully completed network scan. Found 1 open port. Generating PDF report.",
            goal="weekly security audit for Acme Corp",
        )
        print(f"  Healthy output PAD severity: {sev_good}")

        sev_bad = await eros.report_health(
            "error error timeout failed retry error error exception error timeout",
            goal="weekly security audit for Acme Corp",
        )
        print(f"  Stressed output PAD severity: {sev_bad}")

        # ── Phase 5: Experience Bus — Collective Immunity ──
        print("\n[Phase 5] Broadcasting discovery to swarm...")

        await eros.broadcast(
            SignalType.ERROR_RESOLVED,
            {
                "error_type": "endpoint_timeout",
                "fix":        "retry with 30s exponential backoff",
                "applies_to": ["endpoint_checker", "acme_corp"],
            },
            importance=0.9,
        )

        await asyncio.sleep(0.1)  # let bus process

        solutions = await eros.query_solutions("endpoint_timeout")
        print(f"  Solutions available for endpoint_timeout: {len(solutions)}")

        # ── Phase 6: Belief Registry ──────────────────
        print("\n[Phase 6] Belief registry — shared swarm truth...")

        await eros._bus.belief_registry.set("coding_standard", "PEP8",       expected_version=0)
        await eros._bus.belief_registry.set("output_language", "Python",     expected_version=0)
        await eros._bus.belief_registry.set("api_version",     "v3",         expected_version=0)

        beliefs = await eros._bus.belief_registry.sync()
        print(f"  Registry entries: {len(beliefs)}")
        for k, v in beliefs.items():
            print(f"    {k}: {v}")

        # ── Phase 7: Tier 1 System Learning ──────────
        print("\n[Phase 7] Tier 1 system learning loop...")

        for i in range(5):
            await eros._bus.record_outcome(
                task_id=f"task_{i}", task_type="security_audit",
                outcome="success" if i != 2 else "failure",
                latency_ms=1200 + i * 100,
            )

        tier1_stats = eros._bus.tier1.get_stats()
        print(f"  Observations recorded: {tier1_stats['observations']}")
        print(f"  Task types tracked:    {tier1_stats['task_types']}")
        if tier1_stats.get('failure_rates'):
            for task, rate in tier1_stats['failure_rates'].items():
                print(f"  Failure rate ({task}): {rate:.0%}")

        # ── Final Stats ───────────────────────────────
        print("\n[Stats] Final system state:")
        stats = eros.get_stats()
        print(f"  Memories stored:   {stats['db']['memories']}")
        print(f"  Templates cached:  {stats['db']['templates']}")
        print(f"  Fragment library:  {stats['db']['fragments']}")
        print(f"  Semantic facts:    {stats['db']['facts']}")
        print(f"  Bus broadcasts:    {stats['bus']['tier2']['broadcasts_sent']}")
        print(f"  Immunizations:     {stats['bus']['tier2']['immunizations']}")
        print(f"  Memory pool size:  {stats['memory']['pool_size']}")

    print("\n" + "=" * 60)
    print("Demo complete. EROS is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
