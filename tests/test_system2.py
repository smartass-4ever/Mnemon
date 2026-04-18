import asyncio
from mnemon.core.eme import ExecutionMemoryEngine, CostBudget
from mnemon.core.persistence import EROSDatabase
from mnemon.core.memory import SimpleEmbedder
from mnemon.adapters.crewai import CrewAIAdapter

async def fake_generation(goal, inputs, context, capabilities, constraints):
    """Simulates an expensive LLM planning call."""
    return {
        "tasks": [
            {"id": "research",  "action": "search",   "topic": goal, "tools": ["serper"]},
            {"id": "summarise", "action": "summarise", "context": ["research"]},
            {"id": "report",    "action": "write",     "context": ["summarise"]},
        ]
    }

async def main():
    db = EROSDatabase(tenant_id="test_tenant", db_dir=".")
    await db.connect()

    eme = ExecutionMemoryEngine(
        tenant_id="test_tenant",
        db=db,
        embedder=SimpleEmbedder(),
        adapter=CrewAIAdapter(),
        cost_budget=CostBudget(max_llm_calls_per_hour=100),
    )
    await eme.warm()

    common_args = dict(
        inputs={"client": "Acme"},
        context={},
        capabilities=["serper", "file_write"],
        constraints={},
        generation_fn=fake_generation,
    )

    print("\n--- Step 1: Cold run (expect: miss) ---")
    r1 = await eme.run(goal="weekly security audit for Acme", **common_args)
    print(f"cache_level:     {r1.cache_level}")
    print(f"tokens_saved:    {r1.tokens_saved}")
    print(f"segments_reused: {r1.segments_reused}")

    print("\n--- Step 2: Exact same goal (expect: system1) ---")
    r2 = await eme.run(goal="weekly security audit for Acme", **common_args)
    print(f"cache_level:     {r2.cache_level}")
    print(f"tokens_saved:    {r2.tokens_saved}  ← should be > 0")
    print(f"segments_reused: {r2.segments_reused}  ← should be > 0")

    print("\n--- Step 3: Similar goal (expect: system2) ---")
    r3 = await eme.run(goal="monthly security review for Acme", **common_args)
    print(f"cache_level:     {r3.cache_level}")
    print(f"tokens_saved:    {r3.tokens_saved}  ← should be > 0 now (BUG-1 fix)")
    print(f"segments_reused: {r3.segments_reused}  ← should be > 0 now (BUG-3 fix)")

    print("\n--- Step 4: Same similar goal again (expect: system1 now, not system2) ---")
    r4 = await eme.run(goal="monthly security review for Acme", **common_args)
    print(f"cache_level:     {r4.cache_level}  ← should be system1 now (BUG-2 fix)")
    print(f"tokens_saved:    {r4.tokens_saved}  ← should be > 0")

    print("\n--- Stats ---")
    print(eme.get_stats())

    await eme.shutdown()
    await db.disconnect()

asyncio.run(main())
