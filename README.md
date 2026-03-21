# Mnemon

**The intelligence layer between your agents and oblivion.**

By Mahika Jadhav ([@smartass-4ever](https://github.com/smartass-4ever))

---

## The Problem

Every major agent framework — CrewAI, Letta, Dify, LangChain — treats agents as stateless by default. Every run starts from zero. Agents re-plan things they already planned. They repeat mistakes they already made. They forget what they learned last session. Parallel agents step on each other's state.

This isn't a small inconvenience. It's why production agent deployments are fragile, slow, and expensive at scale.

Mnemon fixes this. Drop it in and your agents stop being amnesiac.

> *Mnemon was Alexander the Great's personal historian — the one whose only job was to ensure nothing was ever forgotten, so every campaign built on the total accumulated knowledge of every campaign before it. Your agents have a Mnemon now.*

---

## Three Components

### `mnemon.memory` — Cognitive Memory System
Five-layer stratified memory with protein bond activation retrieval and conditional intent drone curation.

- **Working** — ephemeral scratchpad, flushes at task end (no context bleed)
- **Episodic** — chronological experiences, importance-scored
- **Semantic** — stable facts, versioned key-value vault
- **Relationship** — per-user interaction patterns
- **Emotional** — emotional context, time-decayed

Retrieval: two-part — protein bond pattern assembly (zero LLM, ~15ms) followed by conditional intent drone (only above memory pool threshold).

### `mnemon.cache` — Execution Memory Engine (EME)
Generalised execution template cache for any expensive recurring computation.

- **System 1** — exact fingerprint match → zero LLM, sub-millisecond
- **System 2** — partial segment match → gap fill with windowed context
- **Fragment library** — 49 pre-warmed proven segments, grows with use
- Works for agent plans, RAG pipelines, data pipelines, any structured workflow

### `mnemon.bus` — Two-Tier Experience Bus
**Tier 1** — system learning loop, always on, no agents needed. Records outcomes, detects patterns, feeds EME and memory.

**Tier 2** — agent intelligence layer. PAD health monitoring (Pleasure/Arousal/Dominance), knowledge propagation (collective immunity), atomic belief registry (shared truth for swarms).

---

## Quick Start

```bash
pip install mnemon-ai
```

```python
import asyncio
from mnemon import Mnemon

async def main():
    async with Mnemon(tenant_id="my_company", agent_id="agent_01") as m:

        # Remember something
        await m.remember("Acme Corp prefers formal PDF reports")
        await m.learn_fact("acme_contact", "Sarah K")

        # Recall relevant memories
        context = await m.recall("weekly security audit for Acme Corp")

        # Run with full caching
        result = await m.run(
            goal="weekly security audit for Acme Corp",
            inputs={"client": "Acme Corp", "week": "March 17-21"},
            generation_fn=my_expensive_planning_function,
        )

        print(f"Cache level:  {result['cache_level']}")
        print(f"Tokens saved: {result['tokens_saved']}")
        print(f"Latency:      {result['latency_ms']:.0f}ms")

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
from mnemon.core.types import MemoryLayer
m = Mnemon(
    tenant_id="x",
    enabled_layers=[MemoryLayer.EPISODIC, MemoryLayer.SEMANTIC],
    eme_enabled=False,
    bus_enabled=False,
)
```

---

## Connect Your LLM

```python
from mnemon import Mnemon
from mnemon.llm.client import AnthropicClient

m = Mnemon(
    tenant_id="my_company",
    llm_client=AnthropicClient(api_key="sk-ant-..."),
)
```

Without a real LLM, Mnemon runs in rule-based mode — fully functional, slightly less intelligent memory routing.

---

## Framework Adapters

```python
from mnemon.adapters.crewai import CrewAIAdapter
from mnemon.adapters.letta import LettaAdapter

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

## Production Features

```python
from mnemon import Mnemon
from mnemon.security.manager import TenantSecurityConfig

m = Mnemon(
    tenant_id="my_company",
    llm_client=AnthropicClient(api_key="..."),

    # Security
    security_config=TenantSecurityConfig(
        tenant_id="my_company",
        blocked_categories=["pii", "medical_records"],
        encrypt_privileged=True,
    ),

    # Observability
    enable_watchdog=True,
    enable_telemetry=True,
    watchdog_webhook="https://hooks.slack.com/...",

    # Fragment pre-warming
    prewarm_fragments=True,
)
```

---

## Fail-Safe Design

Mnemon **never crashes the system it serves**.

- Memory retrieval fails → agent runs without context
- EME fails → `generation_fn` called directly
- Bus fails → agent continues unmonitored
- Database unavailable → in-memory fallback mode
- All failures logged, never raised

---

## CLI

```bash
# Guided setup — detects your framework, loads fragments, runs benchmark
mnemon init

# Run eval suite
mnemon eval --suite standard

# Health check
mnemon health

# Telemetry report
mnemon stats
```

---

## Architecture

```
Any AI system
      ↓ adapter translates
┌──────────────────────────────────────────┐
│  EME      Execution Memory Engine        │  S1 → S2 → generation
│  Memory   Five-layer cognitive memory    │  protein bonds → intent drone
│  Bus      Two-tier experience bus        │  Tier 1 always / Tier 2 for agents
└──────────────────────────────────────────┘
           ↓
    SQLite (local) / Redis (scale)
```

---

## Package Structure

```
Mnemon/
├── mnemon/
│   ├── __init__.py          ← Mnemon class — the unified API
│   ├── core/
│   │   ├── types.py         ← all shared dataclasses and enums
│   │   ├── persistence.py   ← SQLite + inverted index + migrations
│   │   ├── memory.py        ← five-layer memory + protein bonds + drone
│   │   ├── eme.py           ← execution memory engine S1/S2
│   │   └── bus.py           ← two-tier experience bus + PAD
│   ├── adapters/
│   │   ├── crewai.py        ← CrewAI reference adapter
│   │   └── letta.py         ← Letta/MemGPT adapter
│   ├── llm/
│   │   └── client.py        ← Anthropic, OpenAI, Mock clients
│   ├── security/
│   │   └── manager.py       ← content filtering, encryption, isolation
│   ├── observability/
│   │   ├── watchdog.py      ← health checks, self-healing, alerts
│   │   └── telemetry.py     ← structured metrics
│   ├── eval/
│   │   └── harness.py       ← 14-scenario test suite with scoring
│   ├── fragments/
│   │   └── library.py       ← 49 pre-warmed execution fragments
│   └── cli/
│       └── main.py          ← mnemon init/eval/health/stats
├── tests/
│   └── test_mnemon.py       ← 29 tests, all passing
├── example.py               ← runnable full demo
├── setup.py
├── requirements.txt
└── README.md
```

---

## Issues Filed — Problems Mnemon Solves

These are documented on the frameworks themselves:

- [CrewAI #4415](https://github.com/crewAIInc/crewAI/issues/4415) — context pollution and DB write contention
- [Dify #32306](https://github.com/langgenius/dify/issues/32306) — redundant reasoning tax in agent nodes
- [Kimi CLI #1058](https://github.com/MoonshotAI/kimi-cli/issues/1058) — context saturation in 100-agent swarms
- [E2B #1207](https://github.com/e2b-dev/E2B/issues/1207) — environmental amnesia in sandboxes
- [Letta RFC](https://github.com/letta-ai/letta) — heartbeat contention and sleep-time compute integration

---

## License

MIT — free to use, free to build on.
