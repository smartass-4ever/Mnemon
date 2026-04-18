"""
Test: Full Mnemon on Claude — all three subsystems.

Scenario: Claude is a coding assistant helping with the Mnemon project.
It uses Mnemon to:
  1. Remember what it learned about the codebase (CognitiveMemory)
  2. Cache its planning responses for repeated question types (EME)
  3. Monitor its own health and broadcast fixes it found (ExperienceBus)

Run: python test_claude_memory.py
"""

import asyncio
import os
import sys
import time

# UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from mnemon import Mnemon
from mnemon.core.models import MemoryLayer, SignalType


# ── Simulate Claude generating a "plan" for how to answer a code question ───
# In real use this is the expensive LLM call (~15-30s, 1000s of tokens).
# EME caches it so the second identical question type is free.

async def claude_answer_plan(goal, inputs, context, capabilities, constraints):
    """Simulates Claude planning an answer to a codebase question."""
    await asyncio.sleep(0.05)   # simulate LLM latency
    topic = inputs.get("topic", "unknown")
    return {
        "steps": [
            {"id": "locate_file",  "action": "grep_codebase",  "params": {"query": topic}},
            {"id": "read_context", "action": "read_file",       "params": {"lines": 50}},
            {"id": "explain",      "action": "generate_answer", "params": {"style": "concise"}},
        ],
        "type": "code_explanation_plan",
        "topic": topic,
    }


def separator(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


async def main():
    separator("Full Mnemon on Claude — all three subsystems")

    db_path = "./mnemon_tenant_claude_session.db"
    for suffix in ["", "-shm", "-wal"]:
        if os.path.exists(db_path + suffix):
            os.remove(db_path + suffix)
    print("  (fresh DB for this session)")

    async with Mnemon(
        tenant_id="claude_session",
        agent_id="claude",
        db_dir=".",
        memory_enabled=True,
        eme_enabled=True,
        bus_enabled=True,
        resonance_floor=0.40,       # tuned for code knowledge base (best empirical score ~0.46)
        enable_telemetry=True,
    ) as m:

        session_id = "session_mahika_01"
        await m.start_session(session_id)

        # ════════════════════════════════════════════════
        # SUBSYSTEM 1 — COGNITIVE MEMORY
        # Claude learns things during the session and stores them
        # ════════════════════════════════════════════════
        separator("1. Cognitive Memory — learning the codebase")

        # Facts: precise lookups, no similarity needed
        await m.learn_fact("project_name",   "Mnemon — intelligence layer for AI agents")
        await m.learn_fact("author",         "Mahika Jadhav (smartass-4ever)")
        await m.learn_fact("entry_point",    "mnemon/__init__.py — Mnemon class")
        await m.learn_fact("three_systems",  "CognitiveMemory + EME + ExperienceBus")
        await m.learn_fact("embedder",       "sentence-transformers all-MiniLM-L6-v2, 384-dim")
        await m.learn_fact("db_type",        "SQLite, per-tenant files, EROSDatabase in persistence.py")
        await m.learn_fact("benchmark",      "LongMemEval 64.6% retrieval accuracy")

        # Episodic memories: things Claude "experienced" during this session
        episodes = [
            ("User asked about the database. Told them Mnemon uses per-tenant SQLite files "
             "managed by EROSDatabase. TenantConnectionPool handles connections.",
             MemoryLayer.EPISODIC, 0.85),

            ("Found a bug: RESONANCE_FLOOR was hardcoded at 0.70 in memory.py, blocking "
             "all retrieval for knowledge-base style queries. Fixed by adding resonance_floor "
             "param to CognitiveMemorySystem and Mnemon.",
             MemoryLayer.EPISODIC, 0.95),

            ("User asked how EME caching works. Explained three levels: L1 exact fingerprint "
             "match (zero LLM), L2 template reuse with segment diff, L3 gap-fill with LLM.",
             MemoryLayer.EPISODIC, 0.85),

            ("Ran benchmark_longmemeval.py. Result: 64.6% retrieval accuracy. "
             "Honest about limitations — hash projection fallback drops to ~56%.",
             MemoryLayer.EPISODIC, 0.75),

            ("User wants to test all three Mnemon subsystems together with Claude as subject.",
             MemoryLayer.EPISODIC, 0.90),
        ]

        for text, layer, importance in episodes:
            await m.remember(text, layer=layer, importance=importance, session_id=session_id)

        print(f"  Stored {len(episodes)} episodic memories + 7 facts")

        # Recall test
        print("\n  Recall queries:")
        recall_tests = [
            ("how does the database work",           "db_type"),
            ("what happened with the resonance bug", None),
            ("what did the benchmark show",          "benchmark"),
        ]

        for query, fact_key in recall_tests:
            t0 = time.perf_counter()
            result = await m.recall(query, session_id=session_id, top_k=3)
            ms = (time.perf_counter() - t0) * 1000
            mems = result.get("memories", [])

            print(f"\n  Q: '{query}'")
            print(f"     Memories retrieved: {len(mems)}  ({ms:.1f}ms)")
            for mem in mems[:2]:
                snippet = mem.get("text", str(mem))[:75]
                print(f"     - {snippet}...")

            if fact_key:
                val = await m.recall_fact(fact_key)
                print(f"     Fact[{fact_key}]: {val}")

        # ════════════════════════════════════════════════
        # SUBSYSTEM 2 — EXECUTION MEMORY ENGINE (EME)
        # Cache Claude's planning for repeated question types
        # ════════════════════════════════════════════════
        separator("2. EME — caching Claude's answer plans")

        print("  Run 1: 'explain the database layer' — cold start (cache miss)")
        t0 = time.perf_counter()
        r1 = await m.run(
            goal="explain the database layer to the user",
            inputs={"topic": "database", "user": "mahika"},
            generation_fn=claude_answer_plan,
            task_type="code_explanation",
        )
        ms1 = (time.perf_counter() - t0) * 1000
        print(f"     cache_level:  {r1['cache_level']}")
        print(f"     tokens_saved: {r1['tokens_saved']}")
        print(f"     latency:      {ms1:.1f}ms")

        print("\n  Run 2: 'explain the database layer' — same goal, second user")
        t0 = time.perf_counter()
        r2 = await m.run(
            goal="explain the database layer to the user",
            inputs={"topic": "database", "user": "priya"},
            generation_fn=claude_answer_plan,
            task_type="code_explanation",
        )
        ms2 = (time.perf_counter() - t0) * 1000
        print(f"     cache_level:    {r2['cache_level']}")
        print(f"     tokens_saved:   {r2['tokens_saved']}")
        print(f"     segments_reused:{r2['segments_reused']}")
        print(f"     latency:        {ms2:.1f}ms")

        print("\n  Run 3: 'explain the EME caching system' — different goal")
        t0 = time.perf_counter()
        r3 = await m.run(
            goal="explain the EME caching system to the user",
            inputs={"topic": "EME caching", "user": "mahika"},
            generation_fn=claude_answer_plan,
            task_type="code_explanation",
        )
        ms3 = (time.perf_counter() - t0) * 1000
        print(f"     cache_level:  {r3['cache_level']}")
        print(f"     tokens_saved: {r3['tokens_saved']}")
        print(f"     latency:      {ms3:.1f}ms")

        # ════════════════════════════════════════════════
        # SUBSYSTEM 3 — EXPERIENCE BUS
        # Claude monitors its own health + broadcasts fixes
        # ════════════════════════════════════════════════
        separator("3. Experience Bus — health + collective learning")

        print("  PAD health monitoring:")
        healthy_output = (
            "Successfully explained the database layer. User understood. "
            "Retrieval worked correctly. All facts accurate."
        )
        stressed_output = (
            "error error retrieval failed timeout exception "
            "error retry failed error timeout error"
        )
        sev_good = await m.report_health(healthy_output, goal="explain the database layer")
        sev_bad  = await m.report_health(stressed_output, goal="explain the database layer")
        print(f"     Healthy run:  severity = {sev_good}")
        print(f"     Stressed run: severity = {sev_bad}")

        print("\n  Broadcasting fix to swarm (resonance_floor bug resolved):")
        await m.broadcast(
            SignalType.ERROR_RESOLVED,
            {
                "error_type":  "resonance_floor_too_high",
                "component":   "CognitiveMemorySystem",
                "fix":         "expose resonance_floor as constructor param, default 0.70",
                "workaround":  "use resonance_floor=0.50 for knowledge-base workloads",
                "applies_to":  ["CognitiveMemorySystem", "Mnemon"],
            },
            importance=0.95,
        )
        await asyncio.sleep(0.3)   # let process loop flush the queue
        solutions = await m.query_solutions("resonance_floor_too_high")
        print(f"     Solutions retrievable by other agents: {len(solutions)}")

        # ════════════════════════════════════════════════
        # FINAL STATS
        # ════════════════════════════════════════════════
        separator("Final stats")

        stats = m.get_stats()
        db    = stats.get("db", {})
        eme   = stats.get("eme", {})
        tel   = stats.get("telemetry", {})

        print(f"  Memories stored:   {db.get('memories', 0)}")
        print(f"  Facts stored:      {db.get('facts', 0)}")
        print(f"  Templates cached:  {db.get('templates', 0)}")
        print(f"  EME cache hits:    {eme.get('cache_hits', 0)}")
        print(f"  EME total tokens saved: {tel.get('total_tokens_saved', 0)}")
        print(f"  Embedder backend:  {m._embedder.backend_name}")

        await m.end_session(session_id)

    separator("Done")
    print("  All three subsystems exercised with Claude as the agent.")


if __name__ == "__main__":
    asyncio.run(main())
