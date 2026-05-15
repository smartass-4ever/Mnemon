# LLM Agent Cost Savings — Mnemon

## The Problem: What Recurring Agent Workflows Actually Cost

Every time a stateless LLM agent runs the same workflow, you pay full price. No memory of what it computed last run. No reuse of previous reasoning. Full token cost, every time.

On claude-sonnet-4-6 pricing ($3.00/1M input, $15.00/1M output):

| Workflow | Tokens per run | Cost per run | Cost per month (daily) |
|---|---|---|---|
| Security audit | ~1,250 | $0.019 | $0.57 |
| Weekly report | ~1,250 | $0.019 | $0.57 |
| Invoice processing | ~1,250 | $0.019 | $0.57 |
| 50-agent burst | ~62,500 | $0.94 | $28.13 |

These are single-workflow numbers. At scale across teams and pipelines, redundant reasoning is the single largest preventable cost in LLM agent infrastructure.

---

## The Fix: Execution Caching with Mnemon

Mnemon caches the execution plan after the first run. Every subsequent run with the same or semantically similar goal skips the LLM entirely.

```
Before Mnemon:  1,250 tokens · $0.019 · 20,000ms — every single run
After Mnemon:   0 tokens    · $0.000 · 2.66ms    — every repeat run
```

---

## Measured Savings — Real Benchmark Data

**Test setup:** 45 runs across 3 recurring workflow types on claude-sonnet-4-6

| Metric | Baseline | With Mnemon | Reduction |
|---|---|---|---|
| Total tokens | 9,786 | 651 | **93.3%** |
| LLM API calls | 45 | 3 | **93%** |
| Total cost | $0.005774 | $0.000384 | **93.4%** |
| Avg latency per hit | ~20,000ms | 2.66ms | **7,500×** |

**50 concurrent agents, single burst:**

| Metric | Without Mnemon | With Mnemon |
|---|---|---|
| LLM calls | 50 | 0 |
| Tokens | 62,500 | 0 |
| Cost | $0.9375 | $0.00 |
| Wall time | ~1,000s | 0.18s |

Raw data: [`reports/benchmark_eme_results.json`](reports/benchmark_eme_results.json)

---

## Monthly Savings at Scale

Assumes 80% System 1 hits + 15% System 2 hits + 5% genuine cache misses:

| Daily agent runs | Monthly tokens saved | Monthly cost saved |
|---|---|---|
| 100 | 125,000 | **$56** |
| 1,000 | 1,118,750 | **$503** |
| 10,000 | 11,187,500 | **$5,034** |
| 100,000 | 111,875,000 | **$50,344** |

---

## Per-Model Cost Comparison

How much you save per 1,000 cache hits:

| Model | Cost per run (est.) | Saved per 1,000 hits |
|---|---|---|
| claude-sonnet-4-6 | $0.019 | **$19** |
| gpt-4o | $0.025 | **$25** |
| claude-opus-4 | $0.075 | **$75** |
| gpt-4-turbo | $0.030 | **$30** |
| gemini-1.5-pro | $0.018 | **$18** |

The more expensive the model, the more Mnemon saves.

---

## Latency Savings

LLM API latency for a typical agent workflow: 15,000ms–25,000ms

Mnemon System 1 cache hit: **2.66ms**

For user-facing agent applications, this is the difference between a product that feels instant and one that makes users wait 20 seconds.

---

## How to Start Saving

```bash
pip install mnemon-ai
mnemon demo    # see the savings in 30 seconds, no API key needed
```

```python
import mnemon
mnemon.init()  # patches LangChain, CrewAI, AutoGen, LangGraph automatically
```

Full documentation: [README.md](README.md)
Full benchmark data: [reports/](reports/)
