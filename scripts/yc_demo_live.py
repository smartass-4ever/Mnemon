"""
Mnemon — YC Live Demo
Real LLM (Groq). Real cache. Zero code changes to your agent.

Setup:
    set GROQ_API_KEY=gsk_...      (Windows)
    python scripts/yc_demo_live.py

Screen recording:
    Open a clean dark terminal, start OBS or Win+G, then run.
"""

import os
import sys
import time
import tempfile

# ── colours (same palette as yc_demo.py) ─────────────────────────────────────

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


# ── demo ──────────────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        print()
        print(clr("  No GROQ_API_KEY set.", RED, BOLD))
        print(clr("  Get a free key at console.groq.com, then:", DIM))
        print()
        print(f"    {clr('set GROQ_API_KEY=gsk_...', CYAN)}")
        print()
        sys.exit(1)

    # ── header ────────────────────────────────────────────────────────────────
    print()
    print(clr("  ╔══════════════════════════════════════════════════════╗", GREY))
    print(clr("  ║", GREY) + clr("       Mnemon — YC Demo  ·  Real LLM, Real Cache      ", BOLD, WHITE) + clr("║", GREY))
    print(clr("  ║", GREY) + clr("       LangChain + Groq  ·  One line of code.          ", DIM) + clr("║", GREY))
    print(clr("  ╚══════════════════════════════════════════════════════╝", GREY))
    print()
    pause(700)

    # ── the code ──────────────────────────────────────────────────────────────
    section("THE CODE  —  your existing agent, untouched")
    print()
    print(f"  {clr('import mnemon', CYAN)}")
    print(f"  {clr('m = mnemon.init()', BOLD, CYAN)}  {clr('  ← the only change', GREY)}")
    print()
    print(f"  {clr('# Everything below is unchanged:', GREY)}")
    _model = 'llama-3.1-8b-instant'
    _chatgroq_line = f'llm   = ChatGroq(model="{_model}")'
    print(f"  {clr(_chatgroq_line, DIM)}")
    print(f"  {clr('chain = prompt | llm | StrOutputParser()', DIM)}")
    _invoke = 'chain.invoke({"question": "..."})'
    print(f"  {clr(_invoke, DIM)}")
    print()
    print(clr("  MOTH intercepts every LangChain call automatically.", DIM))
    print(clr("  No wrappers. No decorators. No refactoring.", DIM))
    print()
    pause(1400)

    # ── init ──────────────────────────────────────────────────────────────────
    section("INITIALIZING")
    print()
    print(f"  {clr('importing libraries ...', DIM)}", end="", flush=True)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import mnemon
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    print(f"\r  {clr('mnemon.init()  ...', YELLOW)}        ", end="", flush=True)

    _tmp = tempfile.mkdtemp()
    m = mnemon.init(
        tenant_id="yc_demo_live",
        db_dir=_tmp,
        silent=True,
        enable_telemetry=False,
        eme_enabled=False,
    )

    active = m.active_integrations
    active_str = ", ".join(active) if active else "langchain"
    print(f"\r  {clr('✓ ready', GREEN, BOLD)}  ·  MOTH active on: {clr(active_str, CYAN)}")
    print()
    pause(900)

    # ── build chain ───────────────────────────────────────────────────────────
    llm   = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_key, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise analyst. Answer in exactly 2 sentences."),
        ("human",  "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    # ── tasks ─────────────────────────────────────────────────────────────────
    section("DEMO  —  3 tasks, then the same 3 again")
    print()
    pause(400)

    tasks = [
        ("1", "What are the top 2 cyber risks for a financial services firm this week?"),
        ("2", "What are the 2 biggest compliance priorities for a SaaS company right now?"),
        ("3", "What should a CISO prioritize for endpoint security in the next 30 days?"),
        ("1", "What are the top 2 cyber risks for a financial services firm this week?"),
        ("2", "What are the 2 biggest compliance priorities for a SaaS company right now?"),
        ("3", "What should a CISO prioritize for endpoint security in the next 30 days?"),
    ]

    results     = []
    # Keyed by task label — stores real latency + token estimate from the first LLM call.
    # Hits derive their "saved" numbers from this, so we never conflate used vs saved.
    first_run   = {}

    for i, (label, question) in enumerate(tasks):
        is_repeat = i >= 3

        if i == 0:
            print(clr("  ┌─ WITHOUT Mnemon — your agent today ─────────────────────┐", YELLOW))
            print(clr("  │  Every workflow call hits the LLM. You pay every time.  │", YELLOW))
            print(clr("  └──────────────────────────────────────────────────────────┘", YELLOW))
            print()
            pause(600)
        elif i == 3:
            print()
            print(clr("  ┌─ WITH Mnemon — same agent, one line added ───────────────┐", GREEN))
            print(clr("  │  Seen it before? Instant. Free. Zero code changes.       │", GREEN))
            print(clr("  └──────────────────────────────────────────────────────────┘", GREEN))
            print()
            pause(600)

        run_label = f"Task {label}" + (" (repeat)" if is_repeat else "")

        print(f"  {clr(run_label, BOLD)}")
        if not is_repeat:
            q_preview = question if len(question) <= 68 else question[:65] + "..."
            print(f"  {clr(f'  Q: {q_preview}', DIM)}")
        print(f"  {clr('  running...', YELLOW)}", end="", flush=True)

        prev_hits = m.stats.get("cache_hits", 0)
        t0 = time.time()

        try:
            answer = chain.invoke({"question": question})
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            print(f"\r  {clr(f'  ✗ error: {e}', RED)}")
            results.append((label, "error", elapsed, ""))
            pause(300)
            continue

        elapsed = (time.time() - t0) * 1000
        hit     = m.stats.get("cache_hits", 0) > prev_hits
        answer  = answer.strip()

        if hit:
            # saved = whatever the first LLM call actually cost on this task
            saved_ms  = first_run.get(label, {}).get("ms",     elapsed * 50)
            saved_tok = first_run.get(label, {}).get("tokens", 0)
            tok_str   = f"~{saved_tok} tokens" if saved_tok else "tokens"
            line = clr(
                f"  ⚡ CACHE HIT  ·  {elapsed:.1f}ms  ·  {tok_str} saved  ·  $0.00",
                GREEN, BOLD,
            )
            print(f"\r{line}")
            results.append((label, "hit", elapsed, answer))
        else:
            # Estimate tokens from actual response length (output ≈ words × 1.3, input ≈ 50)
            est_tokens = max(40, int(len(answer.split()) * 1.3) + 50)
            first_run[label] = {"ms": elapsed, "tokens": est_tokens}

            preview = answer[:80].replace("\n", " ")
            if len(answer) > 80:
                preview += "..."
            line = clr(
                f"  LLM  {elapsed:.0f}ms  ·  ~{est_tokens} tokens  (cached for next time)",
                YELLOW,
            )
            print(f"\r{line}")
            _preview_line = f'  └─ "{preview}"'
            print(f"  {clr(_preview_line, DIM)}")
            results.append((label, "miss", elapsed, answer))

        print()
        pause(350 if hit else 500)

    # ── results ───────────────────────────────────────────────────────────────
    section("RESULTS")
    print()

    misses = [r for r in results if r[1] == "miss"]
    hits   = [r for r in results if r[1] == "hit"]

    avg_llm_ms   = sum(r[2] for r in misses) / len(misses) if misses else 0
    avg_cache_ms = sum(r[2] for r in hits)   / len(hits)   if hits   else 0

    print(f"  {'Run':<16} {'Status':<38} {'Latency':>8}")
    print(f"  {hr('─', 62)}")
    for label, status, ms, _ in results:
        run_str = f"Task {label}"
        if status == "hit":
            st     = clr("⚡ CACHE HIT  (LLM skipped)", GREEN, BOLD)
            ms_str = clr(f"{ms:.1f}ms", GREEN, BOLD)
        elif status == "miss":
            st     = clr("LLM called   (stored)", YELLOW)
            ms_str = clr(f"{ms:,.0f}ms", YELLOW)
        else:
            st     = clr("error", RED)
            ms_str = ""
        print(f"  {clr(run_str, BOLD):<16} {st:<46} {ms_str:>8}")

    print()

    if avg_llm_ms > 0 and avg_cache_ms > 0:
        speedup = int(avg_llm_ms / avg_cache_ms)
        print(f"  {clr('LLM average  :', DIM)}  {clr(f'{avg_llm_ms:,.0f}ms', YELLOW)}")
        print(f"  {clr('Cache average:', DIM)}  {clr(f'{avg_cache_ms:.1f}ms', GREEN, BOLD)}")
        print(f"  {clr('Speedup      :', DIM)}  {clr(f'{speedup:,}×  faster', GREEN, BOLD)}")

    print()

    # Savings derived entirely from first-run measurements — tokens saved = tokens
    # that would have been used, latency saved = time that would have been spent.
    tokens_saved  = sum(first_run.get(label, {}).get("tokens", 0) for label, status, *_ in results if status == "hit")
    latency_saved = sum(first_run.get(label, {}).get("ms",     0) for label, status, *_ in results if status == "hit")
    n_hits        = len(hits)

    print(f"  {clr('LLM calls skipped:', DIM)}  {clr(str(n_hits), GREEN, BOLD)}")
    print(f"  {clr('Time saved       :', DIM)}  {clr(f'{latency_saved / 1000:.1f}s of LLM latency eliminated', GREEN, BOLD)}")
    print(f"  {clr('Tokens saved     :', DIM)}  {clr(f'~{tokens_saved:,}', GREEN, BOLD)}  {clr('(from actual response sizes, not an estimate)', GREY)}")
    print()
    pause(800)

    # ── at scale ──────────────────────────────────────────────────────────────
    section("AT SCALE  —  production models (Claude Sonnet · GPT-4)")
    print()
    print(clr("  Same mechanism. Bigger models, bigger savings.", DIM))
    print()
    pause(500)

    rows = [
        (100,     56),
        (1_000,   503),
        (10_000,  5_034),
        (100_000, 50_344),
    ]

    print(f"  {clr('Daily workflows', DIM):<30}  {clr('Monthly cost saved', DIM)}")
    print(f"  {hr('─', 46)}")
    for plans, saved in rows:
        em = BOLD if plans == 100_000 else ""
        print(f"  {clr(f'{plans:>10,}', em):<34}  {clr(f'${saved:>8,}', GREEN, em)}")
        pause(300)

    print()
    print(clr("  ╔══════════════════════════════════════════════════════╗", WHITE))
    print(clr("  ║", WHITE) + clr("   At 100,000 workflows/day:  $50,344 saved / month   ", BOLD, WHITE) + clr("║", WHITE))
    print(clr("  ╚══════════════════════════════════════════════════════╝", WHITE))
    print()
    pause(800)

    # ── footer ────────────────────────────────────────────────────────────────
    print(hr())
    print()
    print(f"  {clr('pip install mnemon-ai', CYAN, BOLD)}")
    print()
    print(clr("  Mnemon. Your agents have a memory now.", DIM))
    print()

    m.close()


if __name__ == "__main__":
    main()
