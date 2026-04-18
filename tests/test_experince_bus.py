import asyncio
import time
from mnemon.core.eme import ExecutionMemoryEngine
from mnemon.core.persistence import EROSDatabase
from mnemon.core.memory import SimpleEmbedder
from mnemon.adapters.crewai import CrewAIAdapter

# Simulated 'Successful' execution that the Bus will broadcast
async def simulate_experience_feedback(eme, task_id, goal, plan):
    """
    Simulates the 'Experience Bus' confirming a success.
    This strengthens the Protein Bond in the DB.
    """
    await eme.record_success(
        task_id=task_id,
        goal=goal,
        plan=plan,
        metrics={"duration": 1.5, "accuracy": 1.0}
    )

async def main():
    db = EROSDatabase(tenant_id="bus_stress_test", db_dir=".")
    await db.connect()

    eme = ExecutionMemoryEngine(
        tenant_id="bus_stress_test",
        db=db,
        embedder=SimpleEmbedder(),
        adapter=CrewAIAdapter(),
    )
    await eme.warm()

    # THE SCENARIO: 
    # 1. We have a generic goal.
    # 2. Multiple agents are performing different variations.
    # 3. The Bus must ingest 50 'Success' signals concurrently.
    
    print("🚀 Launching Experience Bus Stress Test...")
    
    goal = "optimize rust compiler flags"
    fake_plan = {"steps": ["check architecture", "set opt-level=3"]}
    
    start = time.time()

    # Create 50 concurrent experience 'feedbacks' (The Bus Ingestion)
    bus_tasks = []
    for i in range(50):
        t_id = f"task_success_{i}"
        bus_tasks.append(simulate_experience_feedback(eme, t_id, goal, fake_plan))

    # Await the Bus to process all incoming experience bonds
    await asyncio.gather(*bus_tasks)
    
    elapsed = time.time() - start

    # VERIFY: Check if the 'Protein Bond' for this goal is now ultra-stable
    # A single retrieval should now be a perfect System 1/2 hit
    test_result = await eme.run(
        goal="optimize rust flags", # Slightly different text
        capabilities=["rustc"],
        generation_fn=None # Should NOT need a generator if Bond is strong
    )

    print(f"✅ Bus processed 50 experiences in {elapsed:.4f}s")
    print(f"📈 Resulting Cache Level: {test_result.cache_level}")
    print(f"🔗 Bond Stability: {eme.get_stats().get('bonded_count', 'N/A')}")

    await eme.shutdown()
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
