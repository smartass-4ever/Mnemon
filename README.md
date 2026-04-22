![Mnemon](banner.png)

# Mnemon

**Your agents forget everything. Every run. Every time. Mnemon fixes that.**

---

## The Problem

Every agent you've shipped starts from zero.

It re-plans what it already planned last Tuesday. It repeats the mistake it made three runs ago. It asks the customer the same questions it asked last month. It re-generates the same expensive plan it generated yesterday — and charges you again for the privilege.

This isn't a configuration issue. It's structural. CrewAI, LangChain, Dify, Letta — every major agent framework is stateless by default. Memory was bolted on later, as an afterthought, requiring managed services, external vector databases, and new infrastructure just to remember a name between sessions.

You built a smart agent. You got an amnesiac.

---

## The Answer

```python
import mnemon

m = mnemon.init()
```

That's it. One line. Your agent now has persistent memory, execution caching, and cross-session learning. No config file. No server. No new infrastructure. Everything lives in a local SQLite file next to your code. Your data never leaves your machine.

```python
# Remember across sessions
m.remember("Acme Corp prefers formal PDF reports, not dashboards")
m.learn_fact("acme_contact", "Sarah K — decisions go through her")

# Recall relevant context in 15ms, zero LLM calls
context = m.recall("weekly security audit for Acme Corp")

# Cache expensive plans — skip the LLM entirely on repeats
result = m.run(
    goal="weekly security audit for Acme Corp",
    inputs={"client": "Acme Corp", "week": "Apr 14-18"},
    generation_fn=my_planning_function,
)

print(f"Cache:        {result['cache_level']}")    # system1 — exact hit
print(f"Tokens saved: {result['tokens_saved']}")   # 1,250
print(f"Time saved:   {result['latency_saved_ms']:.0f}ms")  # 20,000ms
```

Access the same instance from anywhere in your codebase:

```python
m = mnemon.get()
```

No context manager required. Cleans up when your process exits.

---

## Why Mnemon Is Different

Every other memory library for agents is one of two things: a managed cloud service that owns your data, or a thin wrapper around a vector database that only does retrieval.

Mnemon is neither.

**It runs entirely local.** SQLite, no server, no API calls to a memory service. Your agent's memory is a file you own.

**Retrieval doesn't call the LLM.** Most memory systems embed your query, do a cosine search, and return chunks. Mnemon uses resonance-weighted retrieval — 15ms average, zero tokens, zero cost per recall.

**It caches execution, not just context.** No other agent memory library does this. If your agent plans a security audit for Acme Corp every Monday, Mnemon recognizes the pattern and serves the cached plan in 2.66ms — skipping 20 seconds of LLM generation entirely. Partial matches work too: if 4 of 5 plan segments match, only the delta goes to the LLM.

**It's a library, not a framework.** Drop it into CrewAI, LangChain, raw API calls, or anything else. You don't adopt a new runtime. You add one import.

**It gets smarter over time.** Every run feeds the experience bus. Patterns that work get reinforced. Plans that fail get quarantined. A pattern learned from one run improves cache hit rates on all future runs.

---

## The Numbers

### Memory retrieval — LoCoMo benchmark (1,986 questions)

| | Accuracy |
|---|---|
| Mnemon | **70.0%** |
| Last-session baseline | 27.6% |
| Random baseline | 25.2% |
| Null (no memory) | 23.4% |

### Memory retrieval — LongMemEval

| Metric | Score |
|---|---|
| Retrieval accuracy | **64.6%** |

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

## Install

```bash
pip install mnemon-ai
```

Set one environment variable for your LLM (used for gap-fill only — retrieval never calls it):

```bash
export GROQ_API_KEY=gsk_...      # pip install mnemon-ai[groq]   ← free tier, start here
export ANTHROPIC_API_KEY=sk-...  # pip install mnemon-ai[anthropic]
export OPENAI_API_KEY=sk-...     # pip install mnemon-ai[openai]
export GOOGLE_API_KEY=AIza...    # pip install mnemon-ai[google]
```

Mnemon detects the key automatically. Use Groq if you want to try it right now for free.

---

## Complete Example

```python
import mnemon

# Initialize once — auto-detects your project name
m = mnemon.init()

# First run: agent learns client preferences
m.remember("Acme Corp wants bullet-point summaries, not prose")
m.remember("always flag open CVEs before recommendations")
m.learn_fact("acme_sla", "4h response window")

# Second run: instant recall, no LLM
context = m.recall("security audit for Acme Corp")
print(context)  # returns ranked memories in ~15ms

# Third run: plan is fully cached, 0 tokens spent
def plan_audit(goal, inputs, context, capabilities, constraints):
    # your actual planning logic here
    return {"steps": ["enumerate assets", "check CVEs", "draft report"]}

result = m.run(
    goal="security audit for Acme Corp",
    inputs={"client": "Acme Corp", "week": "Apr 21-25"},
    generation_fn=plan_audit,
)

print(result["cache_level"])      # system1
print(result["tokens_saved"])     # 1250
print(result["latency_saved_ms"]) # 20000.0
```

---

## What's Inside

### `mnemon.memory` — Stratified Memory

Five memory layers, each with a purpose:

| Layer | What it holds | Lifetime |
|---|---|---|
| Working | scratchpad for current task | flushes at task end |
| Episodic | chronological experiences, importance-scored | permanent |
| Semantic | stable facts, versioned key-value vault | permanent |
| Relationship | per-user interaction patterns | permanent |
| Emotional | affective context, time-decayed | decays |

Retrieval uses resonance weighting — no LLM, ~15ms. Intent-based reranking fires only when needed.

### `mnemon.cache` — Execution Memory Engine (EME)

Template cache for structured agent workflows.

- **System 1** — exact fingerprint match → 2.66ms, zero LLM
- **System 2** — partial segment match → only the delta goes to the LLM
- **Retrospector** — quarantines failed plan fragments so bad patterns don't recycle

Ships with 49 pre-warmed segments from real enterprise runs.

### `mnemon.bus` — Experience Bus

Always-on learning loop. Records outcomes, detects patterns, feeds both memory and EME automatically. Your agent improves on every run without any instrumentation from you.

---

## Production Features

```python
from mnemon import Mnemon
from mnemon.security.manager import TenantSecurityConfig

m = Mnemon(
    tenant_id="my_company",
    security_config=TenantSecurityConfig(
        tenant_id="my_company",
        blocked_categories=["pii", "medical_records"],
        encrypt_privileged=True,
    ),
    enable_watchdog=True,
    enable_telemetry=True,
)
```

Multi-tenant out of the box. Each `tenant_id` gets an isolated SQLite database — no cross-tenant leakage. Cross-tenant signal sharing (opt-in) lets cache patterns learned by one tenant improve hit rates across your platform.

---

## Fail-Safe

Mnemon never crashes the system it serves.

- Memory retrieval fails → agent runs without context
- EME fails → `generation_fn` called directly
- Bus fails → agent continues unmonitored
- Database unavailable → in-memory fallback

All failures are logged, never raised. You can't break your agent by adding Mnemon.

---

## Use Only What You Need

```python
# Memory only
m = Mnemon(tenant_id="x", eme_enabled=False, bus_enabled=False)

# Execution cache only
m = Mnemon(tenant_id="x", memory_enabled=False, bus_enabled=False)

# Specific memory layers only
from mnemon.core.models import MemoryLayer
m = Mnemon(
    tenant_id="x",
    enabled_layers=[MemoryLayer.EPISODIC, MemoryLayer.SEMANTIC],
    eme_enabled=False,
    bus_enabled=False,
)
```

---

## Framework Adapters

`mnemon.init()` auto-detects CrewAI if it's installed. For manual control:

```python
from mnemon.adapters.crewai import CrewAIAdapter

m = Mnemon(tenant_id="my_company", adapter=CrewAIAdapter())
```

Write your own by subclassing `TemplateAdapter`:

```python
from mnemon.core.eme import TemplateAdapter

class MyAdapter(TemplateAdapter):
    def decompose(self, template): ...
    def reconstruct(self, segments): ...
    def extract_signature(self, template, goal): ...
```

---

## Async API

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
