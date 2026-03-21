# EROS — Intelligence Infrastructure Layer

**Memory, execution caching, and collective learning for any AI system.**

By Mahika Jadhav ([@smartass-4ever](https://github.com/smartass-4ever))

---

## The Problem

Every major agent framework — CrewAI, Letta, Dify, LangChain — treats agents as stateless by default. Every run starts from zero. Agents re-plan things they already planned. They repeat mistakes they already made. They forget what they learned last session. Parallel agents step on each other's state.

This isn't a small inconvenience. It's why production agent deployments are fragile, slow, and expensive at scale.

EROS fixes this. Drop it in and your agents stop being amnesiac.

---

## Three Components

### `eros.memory` — Cognitive Memory System
Five-layer stratified memory with protein bond activation retrieval and conditional intent drone curation.

- **Working** — ephemeral scratchpad, flushes at task end (no context bleed)
- **Episodic** — chronological experiences, importance-scored
- **Semantic** — stable facts, versioned key-value vault
- **Relationship** — per-user interaction patterns
- **Emotional** — emotional context, time-decayed

Retrieval: Two-part — protein bond pattern assembly (zero LLM, ~15ms) followed by conditional intent drone (only above memory pool threshold).

### `eros.cache` — Execution Memory Engine (EME)
Generalised execution template cache for any expensive recurring computation.

- **System 1** — exact fingerprint match → zero LLM, sub-millisecond
- **System 2** — partial segment match → gap fill with windowed context
- **Fragment library** — proven segments accumulate, reused automatically
- Works for agent plans, RAG pipelines, data pipelines, any structured workflow

### `eros.bus` — Two-Tier Experience Bus
**Tier 1** — system learning loop, always on, no agents needed. Records outcomes, detects patterns, feeds EME and memory.

**Tier 2** — agent intelligence layer. PAD health monitoring (Pleasure/Arousal/Dominance), knowledge propagation (collective immunity), atomic belief registry (shared truth for swarms).

---

## Quick Start

```python
pip install eros-ai
```

```python
import asyncio
from eros import EROS

async def main():
    async with EROS(tenant_id="my_company", agent_id="agent_01") as eros:

        # Remember something
        await eros.remember("Acme Corp prefers formal PDF reports")
        await eros.learn_fact("acme_contact", "Sarah K")

        # Recall relevant memories
        context = await eros.recall("weekly security audit for Acme Corp")

        # Run with full caching
        result = await eros.run(
            goal="weekly security audit for Acme Corp",
            inputs={"client": "Acme Corp", "week": "March 17-21"},
            generation_fn=my_expensive_planning_function,
        )

        print(f"Cache level: {result['cache_level']}")
        print(f"Tokens saved: {result['tokens_saved']}")
        print(f"Latency: {result['latency_ms']:.0f}ms")

asyncio.run(main())
```

---

## Modular — Use Only What You Need

```python
# Memory only (no caching, no bus)
eros = EROS(tenant_id="x", eme_enabled=False, bus_enabled=False)

# EME only (no memory, no bus)
eros = EROS(tenant_id="x", memory_enabled=False, bus_enabled=False)

# Specific memory layers only
from eros.core.types import MemoryLayer
eros = EROS(
    tenant_id="x",
    enabled_layers=[MemoryLayer.EPISODIC, MemoryLayer.SEMANTIC],
    eme_enabled=False,
    bus_enabled=False,
)
```

---

## Framework Adapters

```python
from eros.adapters.crewai import CrewAIAdapter

eros = EROS(
    tenant_id="my_company",
    adapter=CrewAIAdapter(),
)
```

Write your own adapter by subclassing `TemplateAdapter`:

```python
from eros.core.eme import TemplateAdapter

class MyAdapter(TemplateAdapter):
    def decompose(self, template): ...
    def reconstruct(self, segments): ...
    def extract_signature(self, template, goal): ...
```

---

## Fail-Safe Design

EROS **never crashes the system it serves**.

- Every operation has a fallback
- Memory retrieval fails → agent runs without context
- EME fails → generation_fn called directly
- Bus fails → agent continues unmonitored
- Database unavailable → in-memory fallback mode
- All failures logged, never raised

---

## Architecture

```
Any AI system
      ↓ adapter translates
┌─────────────────────────────────┐
│  EME — Execution Memory Engine  │  System 1 → System 2 → generation
│  Memory — Cognitive 5 layers    │  protein bonds → intent drone
│  Bus — Two-tier experience      │  Tier 1 always / Tier 2 for agents
└─────────────────────────────────┘
      ↓
SQLite (local) / Redis (scale)
```

---

## Related Issues Filed

Problems this architecture addresses, documented on major AI repos:

- [CrewAI #4415](https://github.com/crewAIInc/crewAI/issues/4415) — context pollution and DB write contention
- [Dify #32306](https://github.com/langgenius/dify/issues/32306) — redundant reasoning tax in agent nodes
- [Kimi CLI #1058](https://github.com/MoonshotAI/kimi-cli/issues/1058) — context saturation in 100-agent swarms
- [E2B #1207](https://github.com/e2b-dev/E2B/issues/1207) — environmental amnesia in sandboxes

---

## License

MIT — free to use, free to build on.
