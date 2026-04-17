![Mnemon](banner.png)
# Mnemon

**The intelligence layer between your agents and oblivion.**

---

## Install

```bash
pip install mnemon-ai
```

Connect your LLM — set **one** environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # pip install mnemon-ai[anthropic]
export OPENAI_API_KEY=sk-...          # pip install mnemon-ai[openai]
export GOOGLE_API_KEY=AIza...         # pip install mnemon-ai[google]
export GROQ_API_KEY=gsk_...           # pip install mnemon-ai[groq]  ← free tier
```

Mnemon detects the key automatically. No config file needed.

---

## One line to add memory to any agent

```python
import mnemon

m = mnemon.init()
```

That's it. Mnemon auto-detects your project, connects your LLM, and is ready to use anywhere in your codebase:

```python
# Remember things
m.remember("Acme Corp prefers formal PDF reports")
m.learn_fact("acme_contact", "Sarah K")

# Recall relevant context
context = m.recall("weekly security audit for Acme Corp")

# Run with execution caching — LLM skipped on repeat tasks
result = m.run(
    goal="weekly security audit for Acme Corp",
    inputs={"client": "Acme Corp", "week": "March 17-21"},
    generation_fn=my_planning_function,
)

print(f"Cache:        {result['cache_level']}")
print(f"Tokens saved: {result['tokens_saved']}")
print(f"Time saved:   {result['latency_saved_ms']:.0f}ms")
```

Access the same instance from anywhere:

```python
m = mnemon.get()
```

No context manager required. Cleans up automatically when your process exits.

---

## The Problem

Every major agent framework — CrewAI, Letta, Dify, LangChain — treats agents as stateless by default. Every run starts from zero. Agents re-plan things they already planned. They repeat mistakes they already made. They forget what they learned last session. Parallel agents step on each other's state.

This isn't a small inconvenience. It's why production agent deployments are fragile, slow, and expensive at scale.

Mnemon fixes this. Drop it in and your agents stop being amnesiac.

> *Mnemon was Alexander the Great's personal historian — the one whose only job was to ensure nothing was ever forgotten, so every campaign built on the total accumulated knowledge of every campaign before it. Your agents have a Mnemon now.*

---

## Benchmarks

| Benchmark | Metric | Score |
|-----------|--------|-------|
| LongMemEval | Retrieval accuracy | 64.6% |
| LoCoMo | Recall | 0.619 |
| LoCoMo | F1 | 0.636 |
| EME (execution cache) | System 1 hit rate | varies by workload |

Retrieval improved from 0.273 → 0.619 recall after the v1.0 overhaul. Full benchmark runs are in [`reports/`](reports/).

---

## Three Components

### `mnemon.memory` — Cognitive Memory System
Five-layer stratified memory with protein bond activation retrieval.

- **Working** — ephemeral scratchpad, flushes at task end (no context bleed)
- **Episodic** — chronological experiences, importance-scored
- **Semantic** — stable facts, versioned key-value vault
- **Relationship** — per-user interaction patterns
- **Emotional** — emotional context, time-decayed

Retrieval: protein bond pattern assembly (~15ms, zero LLM) followed by conditional intent drone (only when needed).

### `mnemon.cache` — Execution Memory Engine (EME)
Template cache for expensive recurring agent computations.

- **System 1** — exact fingerprint match → sub-millisecond, zero LLM
- **System 2** — partial segment match → gap fill with windowed context
- **Fragment library** — 49 pre-warmed segments from real enterprise runs

Works for agent plans, RAG pipelines, data pipelines, any structured workflow.

### `mnemon.bus` — Two-Tier Experience Bus
**Tier 1** — always-on learning loop. Records outcomes, detects patterns, feeds EME and memory.

**Tier 2** — agent swarm layer. PAD health monitoring (Pleasure/Arousal/Dominance), collective immunity, atomic belief registry for shared state.

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
            inputs={"client": "Acme Corp", "week": "March 17-21"},
            generation_fn=my_planning_function,
        )

        print(f"Cache level:  {result['cache_level']}")
        print(f"Tokens saved: {result['tokens_saved']}")
        print(f"Latency saved: {result['latency_saved_ms']:.0f}ms")

asyncio.run(main())
```

---

## Modular — Use Only What You Need

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

---

## Fail-Safe Design

Mnemon **never crashes the system it serves**.

- Memory retrieval fails → agent runs without context
- EME fails → `generation_fn` called directly
- Bus fails → agent continues unmonitored
- Database unavailable → in-memory fallback mode

All failures are logged, never raised.

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

## Issues Filed — Problems Mnemon Solves

Documented on the frameworks themselves:

- [CrewAI #4415](https://github.com/crewAIInc/crewAI/issues/4415) — context pollution and DB write contention
- [Dify #32306](https://github.com/langgenius/dify/issues/32306) — redundant reasoning tax in agent nodes
- [Kimi CLI #1058](https://github.com/MoonshotAI/kimi-cli/issues/1058) — context saturation in 100-agent swarms
- [E2B #1207](https://github.com/e2b-dev/E2B/issues/1207) — environmental amnesia in sandboxes

---

## License

MIT — free to use, free to build on.
