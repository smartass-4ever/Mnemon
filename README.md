<div align="center">

![Mnemon](banner.png)

# Mnemon

**Stop paying for work your agent already did.**

[![PyPI](https://img.shields.io/pypi/v/mnemon-ai?color=blue&label=PyPI)](https://pypi.org/project/mnemon-ai/)
[![Python](https://img.shields.io/pypi/pyversions/mnemon-ai)](https://pypi.org/project/mnemon-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/mnemon-ai)](https://pypi.org/project/mnemon-ai/)

</div>

---

Your agent runs the same task every week. It pays full LLM price every time. It never remembers. It never gets faster.

Mnemon fixes that. It caches what your agent has already figured out, learns from every run, and makes each subsequent run cheaper and faster than the last.

```bash
pip install mnemon-ai
```

```python
import mnemon
mnemon.init()

# your existing code — completely unchanged
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-6")
response = llm.invoke("Generate weekly security report for Acme Corp")
# second call with same input: 2.66ms · 0 tokens · $0.00
```

Works with **LangChain · CrewAI · AutoGen · LangGraph · Anthropic SDK · OpenAI SDK** — no code changes.

```bash
mnemon demo     # see it working in 30 seconds, no API key needed
```

---

## What Mnemon does

Mnemon has three components. They work together automatically.

Mnemon uses two matching modes across both paths:

- **System 1** — exact match. Sub-millisecond. Zero tokens. No LLM call.
- **System 2** — semantic match. "Weekly security report" hits the cache for "generate security audit".

Both modes are active on both paths. MOTH uses them for response-level caching. The EME uses them for plan-level caching with gap fill on top.

### 1. Execution Memory Engine (EME) — the cache

The EME stores what your agent has done before. On repeat runs it skips the LLM entirely.

```
First run:    20,000ms · 1,250 tokens · full cost
Every repeat:  2.66ms  · 0 tokens    · $0.00
```

System 1 + System 2 matching, plus segment-level caching: only the parts of a plan that actually changed go to the LLM. Everything else comes from cache.

### 2. Experience Bus — the learning loop

The Bus watches every run in the background. You never call it directly — it's always on.

It detects patterns, flags failures, quarantines bad plans, and strengthens what works. The cache gets smarter every run, not just bigger.

You get this for free. Nothing to configure.

### 3. MOTH — the auto-instrumentation layer

MOTH patches your existing frameworks at startup. It's how Mnemon sees what your agent is doing without you changing any code.

Supported: **Anthropic SDK · OpenAI SDK · LangChain · LangGraph · CrewAI · AutoGen**

---

## Which path should I use?

There are two ways to use Mnemon. Pick one based on what you're building.

**Use Path 1 if** you want caching with zero code changes. Drop it into any existing project — Anthropic, OpenAI, LangChain, CrewAI. Mnemon watches your LLM calls and caches the responses. Same input, instant response next time. Good for chatbots, simple agents, quick experiments.
It does not track individual steps, quarantine bad plans, or learn which parts of a workflow are failing. If your input changes every run, it won't hit the cache.

**Use Path 2 if** you run structured recurring tasks — weekly reports, research pipelines, multi-step workflows. This gives you the full system: segment-level caching so only the parts that changed get regenerated, a learning loop that strengthens what works and quarantines what fails, and guided generation that tells your LLM exactly what to fill in. The more it runs, the smarter it gets.
It requires wrapping your generation logic in a function and calling `m.run()` — it's not zero code changes, but the payoff compounds with every run.

---

### Path 1 — Response caching (zero code changes)

**Use this if:** you call an LLM directly and want to cache the responses. Works for chatbots, simple agents, any direct SDK usage.

```python
import mnemon
mnemon.init()   # patches your installed frameworks automatically

# your existing code — completely unchanged
from anthropic import Anthropic
client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Generate weekly security report for Acme Corp"}]
)
```

First call: normal. Every call after with the same or similar input: instant, zero tokens.

**What you get:** System 1 + System 2 response caching. The Bus learning loop. All automatic.

---

### Path 2 — Execution plan caching with `m.run()` (full system)

**Use this if:** your agent runs structured recurring tasks — research workflows, recurring reports, multi-step pipelines. This gives you the full EME with segment-level caching, gap fill, and guided generation.

```python
import mnemon
from anthropic import Anthropic

client = Anthropic()
m = mnemon.init()

def generate_report(goal, inputs, context, capabilities, constraints):
    # only called on a cache miss — put your real LLM logic here
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": goal}],
    )
    return response.content[0].text

result = m.run(
    goal="weekly security audit for Acme Corp",
    inputs={"client": "Acme Corp", "week": "Apr 21-25"},
    generation_fn=generate_report,
)

print(result["output"])            # your result
print(result["cache_level"])       # "system1" | "system2" | "miss"
print(result["tokens_saved"])      # tokens saved on this run
print(result["latency_saved_ms"])  # ms saved on this run
```

**What you get:** full EME — exact + semantic caching at the segment level, gap fill for partially changed plans, guided generation so your LLM only generates what's new, Bus learning loop, Retrospector quarantine.

---

## Install

```bash
pip install mnemon-ai
```

**System 2 semantic matching** — enable one of:

```bash
pip install mnemon-ai[full]    # offline, no API key (recommended)
export OPENAI_API_KEY=sk-...   # or set this — auto-detected
```

Without either, Mnemon runs System 1 only (exact match). Still valuable, just no semantic matching.

**Optional — disable anonymous usage stats:**
```bash
export MNEMON_NO_TELEMETRY=1
```

---

## Pricing

Mnemon is free for individual use. Pro is for production workloads.

| | Free | Pro |
|---|:---:|:---:|
| Cache hits per day | 100 | Unlimited |
| All caching modes (System 1 + System 2) | ✅ | ✅ |
| Experience Bus learning loop | ✅ | ✅ |
| MOTH auto-instrumentation | ✅ | ✅ |
| Production workloads | ❌ | ✅ |
| Price | $0 | $29/month |

**Upgrade to Pro:**

```bash
pip install mnemon-ai
```

Add your license key to `mnemon.config.json`:

```json
{
  "tenant_id": "your_company",
  "license_key": "your-license-key-here"
}
```

Or pass it directly:

```python
m = mnemon.init(license_key="your-license-key-here")
```

Get a license key at **[mnemon.lemonsqueezy.com/checkout/buy/23828905-b5e2-4946-bd60-9d669f379b1e](https://mnemon.lemonsqueezy.com/checkout/buy/23828905-b5e2-4946-bd60-9d669f379b1e)**

No money? Email **mahikajadhav22@gmail.com** with what you're building — get 1 month free.

When the free tier limit is reached, your agent keeps running — it just calls the LLM normally instead of serving from cache. No crashes, no errors.

> **Already using Mnemon?** If you integrated before pricing was introduced, email mahikajadhav22@gmail.com and I'll sort you out.

---

## How value compounds over time

Mnemon does not save you money on day one. It saves you money on day 30.

Here is what actually happens:

**First run:** everything is a miss. Mnemon caches what your agent did.

**Runs 2–10:** common steps start hitting the cache. You see token savings in your terminal. The fragment library is building.

**After 10+ cache hits:** the fragment library has seen enough of your workflows. Common steps — "authenticate user", "validate input", "generate report", "check rate limits" — are cached from previous runs. Only genuinely novel work goes to the LLM.

**Month 2 onwards:** hit rates of 50–80% on recurring workflows. The cache covers your patterns. You pay for new work only.

The longer you run Mnemon, the more it knows about your agent's workflows, and the less you pay per run. A support bot that handles the same issue types daily reaches 70%+ hit rate within two weeks. A code review pipeline reaches 60%+ within a month as security patterns accumulate.

**You will see a message in your terminal when you hit 10 cache hits.** That's when the compounding begins.

---

## The numbers

| | |
|---|---|
| System 1 hit latency | **2.66ms** |
| Typical LLM call | ~20,000ms |
| Speedup on cache hit | **7,500×** |
| Token reduction | **93%** |

At scale (80% hit rate):

| Daily runs | Monthly savings |
|---|---|
| 100 | $56 |
| 1,000 | $503 |
| 10,000 | $5,034 |

> Stanford researchers published [*Agentic Plan Caching*](https://arxiv.org/abs/2506.14852) at NeurIPS 2025, measuring 50.31% cost reduction with the same approach. Mnemon is the production implementation — one import, works today.

---

## What prints on each run

**First run:**
```
Mnemon: first run — plan cached, next run will be instant
```

**Cache hit:**
```
Mnemon: cache hit · 1,250 tokens saved · ~$0.0038 · 20.0s faster
```

**New input (cached for next time):**
```
Mnemon: new input — cached, next run will be instant
```

---

## Diagnostics

```bash
mnemon doctor   # health check — DB, embedder, fragment library
mnemon demo     # live demo — no API key needed
```

```python
m = mnemon.get()         # retrieve running instance from anywhere
print(m.get_stats())     # EME, bus, DB stats
print(m.waste_report)    # repeated queries and their cost
```

---

## Configuration

```python
m = mnemon.init(tenant_id="acme_corp")   # isolate by tenant
m = mnemon.init(silent=True)             # suppress output
m = mnemon.init(eme_enabled=False)       # bus + MOTH only
m = mnemon.init(bus_enabled=False)       # EME + MOTH only
```

Multi-tenant — each `tenant_id` gets an isolated SQLite database:

```python
from mnemon import Mnemon
async with Mnemon(tenant_id="acme_corp") as m:
    result = await m.run(goal="...", inputs={...}, generation_fn=fn)
```

---

## Fail-safe

Mnemon never crashes the system it wraps.

| What fails | What happens |
|---|---|
| EME cache | `generation_fn` called directly — no disruption |
| Experience Bus | agent continues unmonitored |
| Database | in-memory fallback |

---

## vs. everything else

| | Mnemon | Mem0 | LangMem | Roll your own |
|---|:---:|:---:|:---:|:---:|
| Execution caching (skip LLM entirely) | ✅ | ❌ | ❌ | ❌ |
| System learning loop | ✅ | ❌ | ❌ | ❌ |
| Zero-code auto-instrumentation | ✅ | ❌ | ❌ | ❌ |
| Fully local (no cloud, no API) | ✅ | ❌ | ❌ | ✅ |
| One-line setup | ✅ | ❌ | ❌ | ❌ |

---

## License

MIT. Free to use, free to build on.

Questions or integration help: mahikajadhav22@gmail.com

---

<div align="center">
<sub>Mnemon was Alexander the Great's personal historian — the one whose only job was to ensure nothing was ever forgotten, so every campaign built on the total accumulated knowledge of every campaign before it.<br>Your agents have a Mnemon now.</sub>
</div>
