"""
Mnemon - Quick Start Example

Demonstrates all three components working together:
- Cognitive memory with five layers
- Execution Memory Engine (EME) with System 1/2 caching
- Two-tier experience bus with PAD health monitoring

Run: python example.py
"""

import asyncio
import time
from mnemon import Mnemon
from mnemon.core.models import MemoryLayer, SignalType


# ---------------------------------------------------------
# Simulate an expensive planning function.
# In production this calls your LLM (~15-30s per run).
# ---------------------------------------------------------

async def planning_function(goal, inputs, context, capabilities, constraints):
    await asyncio.sleep(0.1)  # simulate LLM latency (~15-30s in production)
    return {
        "nodes": [
            {"id": "scan_network",    "action": "network_scan",   "params": inputs},
            {"id": "check_endpoints", "action": "endpoint_check", "params": {"retry": True}},
            {"id": "generate_report", "action": "create_pdf",     "params": {"format": "executive_summary"}},
        ],
        "type": "security_audit_dag",
    }


async def main():
    print("=" * 60)
    print("Mnemon - Intelligence Layer Demo")
    print("=" * 60)

    async with Mnemon(tenant_id="my_company", agent_id="agent_01") as m:

        # ── Phase 1: Write memories ─────────────────────
        print("\n[Phase 1] Writing memories...")

        await m.remember("Acme Corp prefers formal PDF reports")
        await m.learn_fact("acme_contact", "Sarah K")
        await m.remember(
            "Acme Corp endpoint check times out — fixed with 30 s retry",
            layer=MemoryLayer.EPISODIC,
            importance=0.9,
        )
        await m.remember(
            "Last Acme Corp audit found 3 open ports on subnet 192.168.1.x",
            layer=MemoryLayer.EPISODIC,
            importance=0.85,
        )
        print("  [ok] 4 memories written across episodic and semantic layers")

        # ── Phase 2: Recall relevant memories ──────────
        print("\n[Phase 2] Recalling memories...")

        context = await m.recall("weekly security audit for Acme Corp")
        print(f"  Memories retrieved: {len(context.get('memories', []))}")
        if context.get("conflicts"):
            print(f"  Conflicts flagged:  {len(context['conflicts'])}")

        # ── Phase 3: First run — cold start ────────────
        print("\n[Phase 3] First run - cold start (cache miss expected)...")

        t0 = time.time()
        result = await m.run(
            goal="weekly security audit for Acme Corp",
            inputs={"client": "Acme Corp", "week": "March 17-21"},
            generation_fn=planning_function,
        )
        t1 = time.time()

        print(f"  Cache level:  {result['cache_level']}")
        print(f"  Tokens saved: {result['tokens_saved']}")
        print(f"  Latency:      {(t1 - t0) * 1000:.0f}ms")

        # ── Phase 4: Second run — cache hit ─────────────
        print("\n[Phase 4] Second run - same goal, different week...")

        t2 = time.time()
        result2 = await m.run(
            goal="weekly security audit for Acme Corp",
            inputs={"client": "Acme Corp", "week": "March 24-28"},
            generation_fn=planning_function,
        )
        t3 = time.time()

        print(f"  Cache level:     {result2['cache_level']}")
        print(f"  Tokens saved:    {result2['tokens_saved']}")
        print(f"  Segments reused: {result2['segments_reused']}")
        print(f"  Latency:         {(t3 - t2) * 1000:.0f}ms")

        # ── Phase 5: PAD health monitoring ─────────────
        print("\n[Phase 5] PAD health monitoring...")

        sev_good = await m.report_health(
            "Successfully completed network scan. Found 1 open port. Generating PDF report.",
            goal="weekly security audit for Acme Corp",
        )
        sev_bad = await m.report_health(
            "error error timeout failed retry error error exception error timeout",
            goal="weekly security audit for Acme Corp",
        )
        print(f"  Healthy output:  {sev_good}")
        print(f"  Stressed output: {sev_bad}")

        # ── Phase 6: Collective immunity (swarm broadcast)
        print("\n[Phase 6] Broadcasting discovery to swarm...")

        await m.broadcast(
            SignalType.ERROR_RESOLVED,
            {
                "error_type": "endpoint_timeout",
                "fix":        "retry with 30 s exponential backoff",
                "applies_to": ["endpoint_checker"],
            },
            importance=0.9,
        )
        await asyncio.sleep(0.1)
        solutions = await m.query_solutions("endpoint_timeout")
        print(f"  Solutions available for endpoint_timeout: {len(solutions)}")

        # ── Final stats ─────────────────────────────────
        print("\n[Stats] System state:")
        stats = m.get_stats()
        db    = stats.get("db", {})
        print(f"  Memories stored:  {db.get('memories', 0)}")
        print(f"  Templates cached: {db.get('templates', 0)}")
        print(f"  Fragment library: {db.get('fragments', 0)}")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
