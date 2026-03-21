"""
EROS CLI
Interactive setup and tooling.

Commands:
  eros init     — guided setup, installs adapter, loads fragments, runs benchmark
  eros eval     — run eval suite
  eros health   — check system health
  eros stats    — show telemetry report
  eros explore  — interactive documentation

Usage:
  python -m eros.cli.main init
  python -m eros.cli.main eval --suite standard
  python -m eros.cli.main health
"""

import argparse
import asyncio
import json
import os
import sys
import time


def detect_framework() -> str:
    """Detect which agent framework is installed."""
    try:
        import crewai
        return "crewai"
    except ImportError:
        pass
    try:
        import langchain
        return "langchain"
    except ImportError:
        pass
    try:
        import letta
        return "letta"
    except ImportError:
        pass
    return "generic"


def print_banner():
    print("""
╔═══════════════════════════════════════════════════════╗
║          EROS — Intelligence Infrastructure           ║
║          Memory · Execution Cache · Learning          ║
╚═══════════════════════════════════════════════════════╝
""")


async def cmd_init(args):
    """Interactive setup."""
    print_banner()
    print("Setting up EROS...\n")

    # Detect framework
    framework = detect_framework()
    if framework != "generic":
        print(f"  Detected framework: {framework}")
        print(f"  Recommended adapter: eros-{framework}")
    else:
        print("  No known framework detected — using generic adapter")

    # Persistence choice
    print("\nPersistence backend:")
    print("  1. SQLite (local, zero config) — recommended for development")
    print("  2. Redis (production ready)    — recommended for production")
    choice = input("Choose [1]: ").strip() or "1"
    db_path = "eros.db" if choice == "1" else None
    if db_path:
        print(f"  Using SQLite: {db_path}")
    else:
        print("  Configure EROS_REDIS_URL environment variable")

    # Load pre-warmed fragments
    print("\nLoading pre-warmed fragment library...")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from eros.fragments.library import load_fragments, FRAGMENT_COUNT
    print(f"  {FRAGMENT_COUNT} curated fragments loaded")
    print(f"  Covers: auth, API, data, reports, error handling, file processing,")
    print(f"          web scraping, database, notifications, scheduling, security")

    # Write config
    config = {
        "tenant_id":    args.tenant_id or "my_company",
        "db_path":      db_path or "eros.db",
        "framework":    framework,
        "version":      "1.0.0",
        "created_at":   time.time(),
    }
    config_path = args.config or "./eros.config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration saved to {config_path}")

    # Quick benchmark
    print("\nRunning quick benchmark...")
    from eros import EROS

    async def dummy_gen(goal, inputs, context, caps, constraints):
        await asyncio.sleep(0.05)
        return [{"id": "step_1", "action": "scan"}, {"id": "step_2", "action": "report"}]

    async with EROS(tenant_id=config["tenant_id"], db_path=db_path) as eros:
        # Pre-load fragments
        frags = load_fragments(config["tenant_id"])
        for frag in frags[:20]:  # load first 20 for demo
            await eros.db.write_fragment(frag)

        t0 = time.time()
        r1 = await eros.run(
            goal="generate weekly security report",
            inputs={"client": "Demo Corp", "week": "this week"},
            generation_fn=dummy_gen,
        )
        t1 = time.time()

        r2 = await eros.run(
            goal="generate weekly security report",
            inputs={"client": "Demo Corp", "week": "next week"},
            generation_fn=dummy_gen,
        )
        t2 = time.time()

    first_run_ms  = (t1 - t0) * 1000
    second_run_ms = (t2 - t1) * 1000

    print(f"\n  First run:  {first_run_ms:.0f}ms  (cache: {r1['cache_level']})")
    print(f"  Second run: {second_run_ms:.0f}ms (cache: {r2['cache_level']})")
    if r2["cache_level"] in ("system1", "system2"):
        saved = r2.get("tokens_saved", 0)
        print(f"  Tokens saved on second run: {saved}")
        print(f"  ✓ EROS is working correctly")

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ready. Add to your agent:

  from eros import EROS

  eros = EROS.from_config("{config_path}")
  await eros.start()

  result = await eros.run(
      goal="your task description",
      inputs={{...}},
      generation_fn=your_planning_function,
  )
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


async def cmd_eval(args):
    """Run eval suite."""
    print_banner()
    suite = args.suite or "standard"
    print(f"Running eval suite: {suite}\n")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from eros.eval.harness import run_eval

    report = await run_eval(suite=suite)
    print(report.summary())

    if not report.passed:
        sys.exit(1)


async def cmd_health(args):
    """Check system health."""
    print_banner()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from eros import EROS
    from eros.observability.watchdog import Watchdog

    config_path = args.config or "./eros.config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        tenant_id = config.get("tenant_id", "default")
        db_path   = config.get("db_path", "eros.db")
    else:
        tenant_id = "default"
        db_path   = "eros.db"

    async with EROS(tenant_id=tenant_id, db_path=db_path) as eros:
        watchdog = Watchdog(
            tenant_id=tenant_id,
            db=eros._db,
            index=eros._index,
            bus=eros._bus,
            eme=eros._eme,
        )
        result = await watchdog.health_check()

    print(f"Health check — tenant: {tenant_id}")
    print(f"Overall: {'✓ HEALTHY' if result['healthy'] else '✗ UNHEALTHY'}\n")
    for check in result["checks"]:
        status = "✓" if check["passed"] else "✗"
        print(f"  {status} {check['name']:<25} {check['message']}")

    stats = result["stats"]
    print(f"\nUptime: {stats['uptime_hours']:.1f}h | Checks: {stats['checks_run']} | Alerts: {stats['alerts_fired']}")


async def cmd_stats(args):
    """Show telemetry report."""
    print_banner()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from eros import EROS

    config_path = args.config or "./eros.config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        tenant_id = config.get("tenant_id", "default")
        db_path   = config.get("db_path", "eros.db")
    else:
        tenant_id = "default"
        db_path   = "eros.db"

    async with EROS(tenant_id=tenant_id, db_path=db_path) as eros:
        stats = eros.get_stats()

    print(f"EROS Stats — tenant: {tenant_id}\n")
    db = stats.get("db", {})
    print(f"  Memories:   {db.get('memories', 0)}")
    print(f"  Facts:      {db.get('facts', 0)}")
    print(f"  Templates:  {db.get('templates', 0)}")
    print(f"  Fragments:  {db.get('fragments', 0)}")

    eme = stats.get("eme", {})
    print(f"\n  EME cache:  {eme.get('system1_entries', 0)} System 1 entries")

    mem = stats.get("memory", {})
    print(f"  Pool size:  {mem.get('pool_size', 0)} active memories")

    bus = stats.get("bus", {})
    t2  = bus.get("tier2", {})
    t1  = bus.get("tier1", {})
    print(f"\n  Bus broadcasts:    {t2.get('broadcasts_sent', 0)}")
    print(f"  Bus immunizations: {t2.get('immunizations', 0)}")
    print(f"  Tier 1 observed:   {t1.get('observations', 0)}")


def main():
    parser = argparse.ArgumentParser(
        prog="eros",
        description="EROS — Intelligence Infrastructure Layer"
    )
    parser.add_argument("--config", help="Path to eros.config.json")
    subparsers = parser.add_subparsers(dest="command")

    init_p = subparsers.add_parser("init", help="Interactive setup")
    init_p.add_argument("--tenant-id", dest="tenant_id", default="my_company")

    eval_p = subparsers.add_parser("eval", help="Run eval suite")
    eval_p.add_argument("--suite", default="standard", choices=["standard", "retrieval", "eme", "bus"])

    subparsers.add_parser("health", help="Check system health")
    subparsers.add_parser("stats", help="Show telemetry report")

    args = parser.parse_args()

    if args.command == "init":
        asyncio.run(cmd_init(args))
    elif args.command == "eval":
        asyncio.run(cmd_eval(args))
    elif args.command == "health":
        asyncio.run(cmd_health(args))
    elif args.command == "stats":
        asyncio.run(cmd_stats(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
