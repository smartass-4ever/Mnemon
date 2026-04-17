"""
Mnemon CLI
Interactive setup and tooling.

Commands:
  mnemon demo     — instant demo, no API key needed
  mnemon init     — guided setup, installs adapter, loads fragments, runs benchmark
  mnemon eval     — run eval suite
  mnemon health   — check system health
  mnemon stats    — show telemetry report

Usage:
  mnemon demo
  python -m mnemon.cli.main init
  python -m mnemon.cli.main eval --suite standard
  python -m mnemon.cli.main health
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
║        Mnemon — Intelligence Layer for Agents         ║
║          Memory · Execution Cache · Learning          ║
╚═══════════════════════════════════════════════════════╝
""")


async def cmd_demo(args):
    """Instant demo — no API key, no config, no user input required."""
    import tempfile
    print_banner()
    print("No API key. No config. Just run it.\n")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from mnemon import Mnemon
    from mnemon.fragments.library import load_fragments

    async def mock_gen(goal, inputs, context, caps, constraints):
        """Simulates an LLM planning call — only runs on cache miss."""
        await asyncio.sleep(0.25)  # realistic ~250ms LLM planning latency
        return [
            {"id": "step_1", "action": "authenticate",    "tool": "auth_service"},
            {"id": "step_2", "action": "fetch_data",       "tool": "data_api",    "depends_on": ["step_1"]},
            {"id": "step_3", "action": "analyse",          "tool": "analysis_fn", "depends_on": ["step_2"]},
            {"id": "step_4", "action": "generate_report",  "tool": "report_fn",   "depends_on": ["step_3"]},
            {"id": "step_5", "action": "send_output",      "tool": "notify_fn",   "depends_on": ["step_4"]},
        ]

    with tempfile.TemporaryDirectory() as tmp:
        async with Mnemon(tenant_id="demo", db_dir=tmp, enable_telemetry=False) as m:
            frags = load_fragments("demo")
            for frag in frags:
                await m._db.write_fragment(frag)
            if m._eme:
                m._eme._fragments = frags

            print("  Scenario: your agent runs the same class of task repeatedly.")
            print("  Goal: 'generate weekly security report'\n")

            # Run 1 — cold miss, LLM is called
            t0 = time.time()
            r1 = await m.run(
                goal="generate weekly security report",
                inputs={"client": "Acme Corp", "week": "2026-W15"},
                generation_fn=mock_gen,
            )
            ms1 = (time.time() - t0) * 1000

            # Run 2 — same goal, different week → System 1 hit (served from cache)
            t1 = time.time()
            r2 = await m.run(
                goal="generate weekly security report",
                inputs={"client": "Acme Corp", "week": "2026-W16"},
                generation_fn=mock_gen,
            )
            ms2 = (time.time() - t1) * 1000

            # Run 3 — different client, same goal → System 1 hit
            t2 = time.time()
            r3 = await m.run(
                goal="generate weekly security report",
                inputs={"client": "Beta LLC", "week": "2026-W17"},
                generation_fn=mock_gen,
            )
            ms3 = (time.time() - t2) * 1000

    def cache_label(r):
        level = r.get("cache_level", "miss")
        if level == "system1": return "CACHE HIT   (exact match — LLM skipped)"
        if level == "system2": return "CACHE HIT   (partial match — LLM skipped)"
        return "cache miss  (LLM called)"

    total_tokens  = r1.get("tokens_saved", 0) + r2.get("tokens_saved", 0) + r3.get("tokens_saved", 0)
    total_lat_saved = r1.get("latency_saved_ms", 0) + r2.get("latency_saved_ms", 0) + r3.get("latency_saved_ms", 0)

    print(f"  Run 1  {ms1:5.0f}ms   {cache_label(r1)}")
    print(f"  Run 2  {ms2:5.0f}ms   {cache_label(r2)}")
    print(f"  Run 3  {ms3:5.0f}ms   {cache_label(r3)}")
    print()

    if total_tokens > 0:
        lat_s = total_lat_saved / 1000
        print(f"  Tokens saved       : {total_tokens:,}")
        print(f"  LLM calls avoided  : {sum(1 for r in [r1,r2,r3] if r.get('cache_level') != 'miss')}")
        print(f"  Simulated time saved: {lat_s:.0f}s of LLM latency")
    print()
    print("  Your agent learns. Every repeated task gets faster.")
    print()
    print("  Get started:")
    print("    pip install mnemon-ai")
    print("    https://github.com/smartass-4ever/Mnemon")
    print()


async def cmd_init(args):
    """Interactive setup."""
    print_banner()
    print("Setting up Mnemon...\n")

    # Detect framework
    framework = detect_framework()
    if framework != "generic":
        print(f"  Detected framework: {framework}")
        print(f"  Recommended adapter: mnemon-{framework}")
    else:
        print("  No known framework detected — using generic adapter")

    # Persistence choice
    print("\nPersistence backend:")
    print("  1. SQLite (local, zero config) — recommended for development")
    print("  2. Redis (production ready)    — recommended for production")
    choice = input("Choose [1]: ").strip() or "1"
    db_dir = "." if choice == "1" else None
    if db_dir:
        print(f"  Using SQLite: mnemon_tenant_<tenant_id>.db in current directory")
    else:
        print("  Configure MNEMON_REDIS_URL environment variable")

    # Load pre-warmed fragments
    print("\nLoading pre-warmed fragment library...")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from mnemon.fragments.library import load_fragments, FRAGMENT_COUNT
    print(f"  {FRAGMENT_COUNT} curated fragments loaded")
    print(f"  Covers: auth, API, data, reports, error handling, file processing,")
    print(f"          web scraping, database, notifications, scheduling, security")

    # Write config
    config = {
        "tenant_id":          args.tenant_id or "my_company",
        "db_dir":             db_dir or ".",
        "framework":          framework,
        "version":            "1.0.0",
        "created_at":         time.time(),
        # Subsystem toggles
        "memory_enabled":     True,
        "eme_enabled":        True,
        "bus_enabled":        True,
        "prewarm_fragments":  True,
        # Observability
        "enable_watchdog":    False,
        "enable_telemetry":   True,
        # Models
        "router_model":       "claude-haiku-4-5-20251001",
        "gap_fill_model":     "claude-sonnet-4-6",
        "drone_model":        "claude-haiku-4-5-20251001",
        # Retrieval
        "similarity_threshold": 0.70,
        "data_region":        "default",
        "blocked_categories": [],
    }
    config_path = args.config or "./mnemon.config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration saved to {config_path}")

    # Quick benchmark
    print("\nRunning quick benchmark...")
    from mnemon import Mnemon

    async def dummy_gen(goal, inputs, context, caps, constraints):
        await asyncio.sleep(0.05)
        return [{"id": "step_1", "action": "scan"}, {"id": "step_2", "action": "report"}]

    async with Mnemon(tenant_id=config["tenant_id"], db_dir=db_dir) as mnemon:
        # Pre-load fragments
        frags = load_fragments(config["tenant_id"])
        for frag in frags[:20]:  # load first 20 for demo
            await mnemon._db.write_fragment(frag)

        t0 = time.time()
        r1 = await mnemon.run(
            goal="generate weekly security report",
            inputs={"client": "Demo Corp", "week": "this week"},
            generation_fn=dummy_gen,
        )
        t1 = time.time()

        r2 = await mnemon.run(
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
        print(f"  Mnemon is working correctly")

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ready. Add to your agent:

  from mnemon import Mnemon

  mnemon = Mnemon.from_config("{config_path}")
  await mnemon.start()

  result = await mnemon.run(
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
    from mnemon.eval.harness import run_eval

    report = await run_eval(suite=suite)
    print(report.summary())

    if not report.passed:
        sys.exit(1)


async def cmd_health(args):
    """Check system health."""
    print_banner()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from mnemon import Mnemon
    from mnemon.observability.watchdog import Watchdog

    config_path = args.config or "./mnemon.config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        tenant_id = config.get("tenant_id", "default")
        db_dir    = config.get("db_dir", config.get("db_path", "."))
    else:
        tenant_id = "default"
        db_dir    = "."

    async with Mnemon(tenant_id=tenant_id, db_dir=db_dir) as mnemon:
        watchdog = Watchdog(
            tenant_id=tenant_id,
            db=mnemon._db,
            index=mnemon._index,
            bus=mnemon._bus,
            eme=mnemon._eme,
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
    from mnemon import Mnemon

    config_path = args.config or "./mnemon.config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        tenant_id = config.get("tenant_id", "default")
        db_dir    = config.get("db_dir", config.get("db_path", "."))
    else:
        tenant_id = "default"
        db_dir    = "."

    async with Mnemon(tenant_id=tenant_id, db_dir=db_dir) as mnemon:
        stats = mnemon.get_stats()

    print(f"Mnemon Stats — tenant: {tenant_id}\n")
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
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="mnemon",
        description="Mnemon — The intelligence layer between your agents and oblivion"
    )
    parser.add_argument("--config", help="Path to mnemon.config.json")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("demo", help="Instant demo — no API key needed")
    init_p = subparsers.add_parser("init", help="Interactive setup")
    init_p.add_argument("--tenant-id", dest="tenant_id", default="my_company")

    eval_p = subparsers.add_parser("eval", help="Run eval suite")
    eval_p.add_argument("--suite", default="standard", choices=["standard", "retrieval", "eme", "bus"])

    subparsers.add_parser("health", help="Check system health")
    subparsers.add_parser("stats", help="Show telemetry report")

    args = parser.parse_args()

    if args.command == "demo":
        asyncio.run(cmd_demo(args))
    elif args.command == "init":
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
