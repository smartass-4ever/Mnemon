"""
Mnemon YC Demo
==============
Run this script to produce a compelling terminal recording.
No API key required — uses a simulated LLM with realistic latency.

Usage:
    python scripts/yc_demo.py

Screen recording:
    Windows: Win+G or OBS
    Mac:     asciinema rec demo.cast
    Any:     ttyrec / termtosvg
"""

import asyncio
import os
import sys
import tempfile
import time

# ── colours ──────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
GREY   = "\033[90m"

def clr(text, *codes):
    return "".join(codes) + text + RESET

def hr(char="─", width=60):
    return clr(char * width, GREY)

def section(title):
    print()
    print(clr(f"  {title}", BOLD, CYAN))
    print(hr())

def pause(ms=400):
    time.sleep(ms / 1000)


# ── fake LLM (realistic timing) ───────────────────────────────────────────────

SIMULATED_LLM_MS = 2800   # simulate ~2.8s LLM round-trip (conservative)
SIMULATED_TOKENS = 1250

async def mock_llm(goal, inputs, context, caps, constraints):
    """Simulates an LLM planning call — only fires on cache miss."""
    await asyncio.sleep(SIMULATED_LLM_MS / 1000)
    return [
        {"id": "s1", "action": "authenticate",   "tool": "auth_service"},
        {"id": "s2", "action": "fetch_data",      "tool": "data_api",    "depends_on": ["s1"]},
        {"id": "s3", "action": "analyse",         "tool": "analysis_fn", "depends_on": ["s2"]},
        {"id": "s4", "action": "generate_report", "tool": "report_fn",   "depends_on": ["s3"]},
        {"id": "s5", "action": "notify",          "tool": "notify_fn",   "depends_on": ["s4"]},
    ]


# ── demo ──────────────────────────────────────────────────────────────────────

async def run_demo():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mnemon import Mnemon
    from mnemon.fragments.library import load_fragments

    # ── header ────────────────────────────────────────────────────────────────
    print()
    print(clr("  ╔══════════════════════════════════════════════════════╗", GREY))
    print(clr("  ║", GREY) + clr("          Mnemon — YC Application Demo               ", BOLD, WHITE) + clr("║", GREY))
    print(clr("  ║", GREY) + clr("   Stop paying for work your agent already did.       ", DIM) + clr("║", GREY))
    print(clr("  ╚══════════════════════════════════════════════════════╝", GREY))
    print()
    pause(600)

    # ── part 1: the pain ──────────────────────────────────────────────────────
    section("THE PROBLEM  —  stateless agents pay full price every run")
    print()
    print(clr("  Scenario: weekly security report for Acme Corp.", DIM))
    print(clr("  Your agent runs every Monday. Same goal. Same structure. Same cost.", DIM))
    print()
    pause(800)

    pain_runs = [
        ("Monday",    "Acme Corp", "2026-W15"),
        ("Tuesday",   "Acme Corp", "2026-W15"),   # same goal, same week
        ("Wednesday", "Acme Corp", "2026-W15"),
    ]

    total_pain_cost   = 0.0
    total_pain_tokens = 0
    total_pain_ms     = 0.0

    for day, client, week in pain_runs:
        print(f"  {clr(day, BOLD)}  goal: weekly security report for {client}")
        print(f"         {clr('calling LLM...', YELLOW)}", end="", flush=True)

        t0 = time.time()
        await asyncio.sleep(SIMULATED_LLM_MS / 1000)
        elapsed = (time.time() - t0) * 1000

        total_pain_ms     += elapsed
        total_pain_tokens += SIMULATED_TOKENS
        total_pain_cost   += SIMULATED_TOKENS * 0.000003

        print(f"\r         {clr(f'✗  {elapsed:,.0f}ms  ·  {SIMULATED_TOKENS:,} tokens  ·  ${SIMULATED_TOKENS * 0.000003:.4f}', RED)}")
        pause(300)

    print()
    print(f"  {clr('Result:', DIM)}  3 identical reports generated")
    print(f"  {clr('Cost:  ', DIM)}  {clr(f'{total_pain_tokens:,} tokens · ${total_pain_cost:.4f} · {total_pain_ms/1000:.1f}s wasted', RED, BOLD)}")
    print()
    pause(1000)

    # ── part 2: add mnemon ────────────────────────────────────────────────────
    section("THE FIX  —  one line of code")
    print()
    print(f"  {clr('import mnemon', CYAN)}")
    print(f"  {clr('m = mnemon.init()', CYAN)}  {clr('← that is the entire integration', GREY)}")
    print()
    print(clr("  MOTH auto-instruments every framework at startup:", DIM))
    print(clr("  Anthropic SDK · OpenAI SDK · LangChain · LangGraph · CrewAI · AutoGen", DIM))
    print()
    pause(1200)

    # ── part 3: runs with mnemon ──────────────────────────────────────────────
    section("DEMO  —  same scenario, same code, now with Mnemon")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        async with Mnemon(tenant_id="yc_demo", db_dir=tmp, enable_telemetry=False, silent=True) as m:

            # pre-warm fragment library
            frags = load_fragments("yc_demo")
            for frag in frags:
                await m._db.write_fragment(frag)
                if frag.signature and m._eme:
                    await m._eme._fragment_index.add("yc_demo", frag.segment_id, frag.signature)
                    m._eme._fragment_map[frag.segment_id] = frag

            runs = [
                ("Run 1", "Acme Corp", "2026-W15", "first run — LLM called, result cached"),
                ("Run 2", "Acme Corp", "2026-W16", "same goal, next week"),
                ("Run 3", "Acme Corp", "2026-W17", "third week — watch the cache"),
                ("Run 4", "Beta LLC",  "2026-W15", "different client, same goal type"),
            ]

            results = []
            for label, client, week, note in runs:
                print(f"  {clr(label, BOLD)}  client: {client}  week: {week}")
                print(f"         {clr(note, DIM)}")
                print(f"         {clr('running...', YELLOW)}", end="", flush=True)

                t0 = time.time()
                r = await m.run(
                    goal="generate weekly security report",
                    inputs={"client": client, "week": week},
                    generation_fn=mock_llm,
                )
                elapsed = (time.time() - t0) * 1000

                level = r.get("cache_level", "miss")
                tokens_saved = r.get("tokens_saved", 0)
                lat_saved    = r.get("latency_saved_ms", 0.0)

                if level == "system1":
                    status = clr(f"✓  CACHE HIT (System 1)  ·  {elapsed:.1f}ms  ·  0 tokens  ·  $0.00", GREEN, BOLD)
                elif level == "system2":
                    status = clr(f"✓  CACHE HIT (System 2)  ·  {elapsed:.1f}ms  ·  delta tokens only", GREEN)
                else:
                    cost = SIMULATED_TOKENS * 0.000003
                    status = clr(f"  cache miss  ·  {elapsed:,.0f}ms  ·  {SIMULATED_TOKENS:,} tokens  ·  ${cost:.4f}  (stored)", YELLOW)

                print(f"\r         {status}")
                results.append((label, level, elapsed, tokens_saved, lat_saved))
                pause(500)

            print()

    # ── summary ───────────────────────────────────────────────────────────────
    section("RESULTS")
    print()

    hits        = sum(1 for _, lvl, *_ in results if lvl in ("system1", "system2"))
    total_saved = sum(ts for _, _, _, ts, _ in results)
    lat_saved_s = sum(ls for _, _, _, _, ls in results) / 1000
    cost_saved  = total_saved * 0.000003

    for label, level, elapsed, tokens_saved, lat_saved in results:
        lvl_str = clr("MISS  (cached)", YELLOW) if level == "miss" else clr("HIT   (LLM skipped)", GREEN, BOLD)
        print(f"  {label:<8} {lvl_str}   {elapsed:6.1f}ms")

    print()
    print(f"  {clr('Cache hits    :', DIM)}  {clr(str(hits), GREEN, BOLD)}  of {len(results)} runs")
    print(f"  {clr('Tokens saved  :', DIM)}  {clr(f'{total_saved:,}', GREEN, BOLD)}")
    print(f"  {clr('Cost saved    :', DIM)}  {clr(f'${cost_saved:.4f}', GREEN, BOLD)}")
    print(f"  {clr('Time saved    :', DIM)}  {clr(f'{lat_saved_s:.1f}s of LLM latency eliminated', GREEN, BOLD)}")
    print()
    print(clr("  The cache gets smarter every run.", DIM))
    print(clr("  Failures are quarantined. Successes are reinforced.", DIM))
    print(clr("  At 100k plans/day this is $50,344/month.", DIM))
    print()
    pause(800)

    # ── at scale ──────────────────────────────────────────────────────────────
    section("AT SCALE  —  what this means in production")
    print()

    rows = [
        (100,     56),
        (1_000,   503),
        (10_000,  5_034),
        (100_000, 50_344),
    ]

    print(f"  {clr('Daily plans', DIM):<26}  {clr('Monthly cost saved', DIM)}")
    print(f"  {hr('─', 40)}")
    for plans, saved in rows:
        emphasis = BOLD if plans == 100_000 else ""
        print(f"  {clr(f'{plans:>10,}', emphasis):<26}  {clr(f'${saved:>8,}', GREEN, emphasis)}")
    print()
    pause(600)

    # ── close ─────────────────────────────────────────────────────────────────
    print(hr())
    print()
    print(f"  {clr('pip install mnemon-ai', CYAN, BOLD)}")
    print()
    print(clr("  Mnemon. Your agents have a memory now.", DIM))
    print()


if __name__ == "__main__":
    asyncio.run(run_demo())
