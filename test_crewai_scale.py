import asyncio
from mnemon.core.eme import ExecutionMemoryEngine, CostBudget
from mnemon.core.persistence import EROSDatabase
from mnemon.core.memory import SimpleEmbedder
from mnemon.adapters.crewai import CrewAIAdapter

async def fake_crew_plan(goal, inputs, context, capabilities, constraints):
    return {
        "tasks": [
            {"id": "t1", "action": "research", "goal": goal, "tools": capabilities},
            {"id": "t2", "action": "analyse",  "context": ["t1"]},
            {"id": "t3", "action": "report",   "context": ["t2"]},
        ]
    }

async def run_agent(eme, agent_id, goal):
    result = await eme.run(
        goal=goal,
        inputs={"agent": agent_id},
        context={},
        capabilities=["search", "write"],
        constraints={},
        generation_fn=fake_crew_plan,
        task_id=f"agent_{agent_id}",
    )
    return agent_id, result.cache_level, result.tokens_saved

async def main():
    db = EROSDatabase("test_scale.db")
    await db.connect()

    eme = ExecutionMemoryEngine(
        tenant_id="swarm_test",
        db=db,
        embedder=SimpleEmbedder(),
        adapter=CrewAIAdapter(),
    )
    await eme.warm()

    # Seed the cache with one run first
    await eme.run(
        goal="audit security posture",
        inputs={"agent": "seed"},
        context={}, capabilities=["search", "write"], constraints={},
        generation_fn=fake_crew_plan,
    )

    # 20 concurrent agents with similar goals
    goals = [
        "audit security posture",          # exact → system1
        "review security posture",          # similar → system2
        "assess security posture",          # similar → system2
        "audit security posture",           # exact → system1
        "evaluate security posture",        # similar → system2
    ] * 4   # 20 total agents

    import time
    start = time.time()
    tasks = [run_agent(eme, i, goals[i]) for i in range(len(goals))]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    s1 = sum(1 for _, level, _ in results if level == "system1")
    s2 = sum(1 for _, level, _ in results if level == "system2")
    miss = sum(1 for _, level, _ in results if level == "miss")
    total_saved = sum(t for _, _, t in results)

    print(f"\n20 concurrent agents in {elapsed:.2f}s")
    print(f"system1 hits:  {s1}")
    print(f"system2 hits:  {s2}")
    print(f"misses:        {miss}")
    print(f"total tokens saved: {total_saved}")
    print(f"\nStats: {eme.get_stats()}")

    await eme.shutdown()
    await db.disconnect()

asyncio.run(main())
