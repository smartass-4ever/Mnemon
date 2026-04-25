![Mnemon](banner.png)

# Mnemon

**The persistent cognitive context layer for AI agents.**

---

## The Problem

Your agent is expensive, slow, and amnesiac — by design.

Every run, it starts from zero. It re-plans what it already planned last Tuesday. It repeats the mistake it made three sessions ago. It charges you full LLM price for a plan it already generated yesterday. It asks your customer the same onboarding questions it asked last month.

This isn't a bug in your code. It's structural. LangChain, CrewAI, AutoGen, LangGraph — every major agent framework is **stateless by default**. Memory was bolted on later, requiring managed cloud services, external vector databases, and new infrastructure just to remember a name between sessions.

Meanwhile, you're paying for the same tokens over and over. Every repeated plan. Every redundant reasoning step. Every re-generated report.

**You built a smart agent. You got an amnesiac that invoices you twice.**

---

## The Solution

```python
import mnemon
m = mnemon.init()
```

One line. Your existing agent — whatever framework it uses — now has:

- **Persistent memory** across sessions
- **Execution caching** that skips the LLM entirely on repeated work
- **Cross-session learning** that gets smarter the longer it runs
- **Drift detection** that catches silent degradation before it costs you

No config file. No server. No new infrastructure. No changes to your existing agent code. Everything lives in a local SQLite file next to your code. Your data never leaves your machine.

At the end of every session, Mnemon tells you exactly what it saved:

```
Mnemon: ~1,250 tokens saved · ~$0.0038 · 20.0s faster
```

---

## If You Already Have an Agent — Read This First

Mnemon automatically patches whatever frameworks you have installed. You do not need to change your agent code.

**Before Mnemon — your existing LangChain agent:**
```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6")
response = llm.invoke("Generate weekly security report for Acme Corp")
# Every call hits the LLM. No memory. No caching. Full cost every time.
```

**After Mnemon — same code, one new line at the top:**
```python
import mnemon
m = mnemon.init()  # ← this is the only change

from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6")
response = llm.invoke("Generate weekly security report for Acme Corp")
# First call: normal LLM call, result cached
# Every repeat: served in 2.66ms, zero tokens, zero cost
```

Same thing works for the Anthropic SDK, OpenAI SDK, CrewAI, LangGraph, and AutoGen. Install Mnemon, add one line, done.

**Framework notes:**
- **LangGraph** — call `mnemon.init()` *before* compiling your graph. If you compile first, Mnemon patches the graph after the fact and per-node caching won't activate.
- **CrewAI** — import `crewai` *before* calling `mnemon.init()`. Mnemon hooks into CrewAI's event bus at init time; reversing the order means the hook misses registration.

---

## Install

```bash
pip install mnemon-ai
```

For production-quality memory retrieval (recommended):
```bash
pip install mnemon-ai[embeddings]   # sentence-transformers only — retrieval upgrade
pip install mnemon-ai[full]         # embeddings + all LLM providers
```

Set one environment variable for your LLM (used for gap-fill and memory extraction — retrieval never calls it):

```bash
export GROQ_API_KEY=gsk_...      # pip install mnemon-ai[groq]   ← free tier, start here
export ANTHROPIC_API_KEY=sk-...  # pip install mnemon-ai[anthropic]
export OPENAI_API_KEY=sk-...     # pip install mnemon-ai[openai]
export GOOGLE_API_KEY=AIza...    # pip install mnemon-ai[google]
```

Mnemon detects the key automatically. Use Groq to try it for free right now.

**Try it in 30 seconds — no API key needed:**
```bash
mnemon demo
```

---

## Getting Started

### Step 1 — Initialize

```python
import mnemon

m = mnemon.init()
```

This auto-detects your project name, creates a local SQLite database, and patches any installed frameworks (LangChain, Anthropic SDK, OpenAI SDK, CrewAI, LangGraph, AutoGen). You can access the same instance from anywhere in your codebase:

```python
m = mnemon.get()
```

### Step 2 — Give your agent memory

```python
# Store facts that persist across sessions
m.remember("Acme Corp prefers formal PDF reports, not dashboards")
m.remember("Always flag open CVEs before recommendations")
m.learn_fact("acme_contact", "Sarah K — all decisions go through her")
m.learn_fact("acme_sla", "4h response window")
```

### Step 3 — Recall relevant context before any task

```python
context = m.recall("weekly security audit for Acme Corp")
# Returns ranked, relevant memories in ~15ms — no LLM call
```

### Step 4 — Let the cache work

After the first run of any repeated task, Mnemon caches the execution plan. Every subsequent run with the same goal — or a semantically similar one — is served from cache:

```python
result = m.run(
    goal="weekly security audit for Acme Corp",
    inputs={"client": "Acme Corp", "week": "Apr 21-25"},
    generation_fn=your_planning_function,
)

print(result["cache_level"])       # "system1" on a hit — exact match in 2.66ms
print(result["tokens_saved"])      # 1250
print(result["latency_saved_ms"])  # 20000.0
```

The `generation_fn` receives `(goal, inputs, context, capabilities, constraints)` and returns your plan. It's only called on a cache miss.

---

## What Makes Mnemon Different

Everyone else in this space builds **flat memory** — store a memory, retrieve it, done. The agent still calls the LLM every single time. The memory just makes the prompt slightly better.

Mnemon does three things nobody else does together:

### 1. Execution caching — skips the LLM entirely

If your agent generates a security report for Acme Corp every Monday, Mnemon recognizes the pattern after the first run and serves the cached plan in **2.66ms** — skipping 20 seconds of LLM generation and 1,250 tokens of cost.

Partial matches work too: if 4 of 5 plan segments match a new request, only the delta goes to the LLM. You pay for the difference, not the whole thing.

No other agent memory library does this.

### 2. Zero-code auto-instrumentation across 6 frameworks

Most memory libraries require you to wire them in manually — call `memory.add()`, call `memory.search()`, restructure your agent. Mnemon patches your existing code at the framework level. One import, zero restructuring.

Supported: **Anthropic SDK, OpenAI SDK, LangChain, LangGraph, CrewAI, AutoGen**

### 3. Drift detection — catches silent degradation

Every session, Mnemon measures your agent's performance against a rolling baseline. When cache hit rates drop, latency climbs, or memory retrieval degrades — it tells you before your users notice.

```python
report = m.drift_report()
print(report)  # severity, what degraded, since when

# Auto-correct: finds conflicting memories written during degradation and resolves them
result = m.auto_correct_drift()
```

---

## The Numbers

### Memory retrieval — LoCoMo benchmark (1,986 questions)

| | Accuracy |
|---|---|
| Mnemon | **70.0%** |
| Last-session baseline | 27.6% |
| Random baseline | 25.2% |
| Null (no memory) | 23.4% |

### Execution cache — EME benchmarks

| Scenario | Result |
|---|---|
| System 1 hit latency (exact match) | **2.66ms** |
| Fresh LLM generation | ~20,000ms |
| Latency reduction | **7,500×** |
| 50 concurrent agents, burst | **0 LLM calls, 0.18s total** |
| Tokens saved (50 agents) | 62,500 |
| Cost saved (50 agents, Sonnet pricing) | $0.94 |

### At scale (80% System 1 + 15% System 2 hit rate)

| Daily plans | Monthly cost saved |
|---|---|
| 100 | $56 |
| 1,000 | $503 |
| 10,000 | $5,034 |
| 100,000 | $50,344 |

Full benchmark runs, methodology, and raw data: [`reports/`](reports/)

---

## What's Inside

### Memory — Stratified, not flat

Five memory layers, each with a purpose:

| Layer | What it holds | Lifetime |
|---|---|---|
| Working | scratchpad for current task | flushes at task end |
| Episodic | chronological experiences, importance-scored | permanent |
| Semantic | stable facts, versioned key-value vault | permanent |
| Relationship | per-user interaction patterns | permanent |
| Emotional | affective context, time-decayed | decays |

Retrieval uses resonance weighting — no LLM call, ~15ms average. Intent-based reranking fires only when needed.

### Cache — Execution Memory Engine (EME)

- **System 1** — exact fingerprint match → 2.66ms, zero tokens
- **System 2** — partial segment match → only the delta goes to the LLM
- **Retrospector** — quarantines failed plan fragments so bad patterns don't recycle

Ships with 49 pre-warmed segments from real enterprise runs.

### Bus — Experience Loop

Always-on learning. Records outcomes, detects patterns, feeds both memory and EME automatically. Your agent improves on every run without any instrumentation from you.

---

## Advanced Usage

### Memory and recall

```python
# Forget a topic (supersedes matching memories)
m.forget("Acme Corp contact details")

# Export / import memory (JSON round-trip)
m.export_memory("backup.json")
m.import_memory("backup.json")

# Specific facts
m.learn_fact("preferred_format", "PDF")
value = m.recall_fact("preferred_format")

# Waste report — which queries your agent repeated and what they cost
print(m.waste_report)
```

### Use only what you need

```python
# Via init() shorthand
m = mnemon.init(use="memory")          # memory only — no caching, no bus
m = mnemon.init(use="cache")           # EME caching only
m = mnemon.init(use="bus")             # experience bus only
m = mnemon.init(use=["memory","cache"])# memory + EME

# Or directly with the async class
async with Mnemon(tenant_id="x", eme_enabled=False, bus_enabled=False) as m:
    ...  # memory only

async with Mnemon(tenant_id="x", memory_enabled=False, bus_enabled=False) as m:
    ...  # cache only
```

### Production — multi-tenant with security

```python
from mnemon import Mnemon
from mnemon.security.manager import TenantSecurityConfig

m = Mnemon(
    tenant_id="acme_corp",
    security_config=TenantSecurityConfig(
        tenant_id="acme_corp",
        blocked_categories=["pii", "medical_records"],
        encrypt_privileged=True,
    ),
    enable_watchdog=True,
    enable_telemetry=True,
)
```

Each `tenant_id` gets an isolated SQLite database — no cross-tenant leakage.

### Suppress the session summary

```python
m = mnemon.init(silent=True)
```

### Async API

```python
import asyncio
from mnemon import Mnemon

async def main():
    async with Mnemon(tenant_id="my_company") as m:
        await m.remember("Acme Corp prefers formal PDF reports")
        result = await m.run(
            goal="weekly security audit for Acme Corp",
            inputs={"client": "Acme Corp", "week": "Apr 21-25"},
            generation_fn=my_planning_function,
        )
        print(result["cache_level"])
        print(result["tokens_saved"])

asyncio.run(main())
```

### Health and diagnostics

```python
# CLI health check
# mnemon doctor

# Programmatic
report = m.drift_report()       # cross-session degradation analysis
result = m.auto_correct_drift() # auto-resolve conflicting memories
stats  = m.get_stats()          # memory counts, cache stats, security config
```

---

## Fail-Safe

Mnemon never crashes the system it serves.

- Memory retrieval fails → agent runs without context
- EME fails → `generation_fn` called directly
- Bus fails → agent continues unmonitored
- Database unavailable → in-memory fallback

All failures are logged, never raised. You can't break your agent by adding Mnemon.

---

## The Problem, Filed Against the Frameworks

These aren't hypothetical — we documented the issues on the repos before building:

- [CrewAI #4415](https://github.com/crewAIInc/crewAI/issues/4415) — context pollution and DB write contention in multi-agent runs
- [Dify #32306](https://github.com/langgenius/dify/issues/32306) — redundant reasoning tax in agent nodes
- [Kimi CLI #1058](https://github.com/MoonshotAI/kimi-cli/issues/1058) — context saturation in 100-agent swarms
- [E2B #1207](https://github.com/e2b-dev/E2B/issues/1207) — environmental amnesia across sandbox restarts

---

## License

MIT. Free to use, free to build on.

---

*Mnemon was Alexander the Great's personal historian — the one whose only job was to ensure nothing was ever forgotten, so every campaign built on the total accumulated knowledge of every campaign before it. Your agents have a Mnemon now.*
