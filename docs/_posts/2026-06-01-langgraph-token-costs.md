---
layout: post
title: "How I Cut My LangGraph Agent's Token Costs by 93% with One Import"
date: 2026-06-01
description: "LangGraph is stateless by default — every invocation is a cold start. Here's how to cache execution plans so repeat runs cost nothing."
tags: [python, llm, langgraph, ai]
---

I run a LangGraph pipeline that processes competitor intelligence reports every week. Same graph, same nodes, same conditional edges — just slightly different inputs each time. I was paying full LLM price on every run.

After profiling it, I found that 90%+ of the graph traversal was identical across runs. The planner node always produced the same structure. The summarizer always took the same path. I was essentially paying to re-derive work my agent had already done.

This is the core problem with LangGraph at scale: **the graph is stateless by default**. Every invocation is a cold start.

---

## The pattern that bleeds money

If your LangGraph agent does any of the following, you're paying for redundant computation:

- Scheduled pipelines (weekly reports, daily digests, recurring audits)
- Multi-step research agents that hit the same sources
- Document processing graphs with consistent structure
- Customer-facing agents that handle similar queries repeatedly

Each run: full token cost. Full latency. Zero memory of previous executions.

---

## What I tried first

**Prompt caching** — Anthropic and OpenAI both offer it. It helps with repeated *prefixes*, not repeated *reasoning*. When your graph re-derives a plan from slightly different inputs, prompt caching doesn't fire. You still pay.

**Manual caching** — I added SQLite lookups at individual nodes. It worked but was brittle and broke every time I changed the graph structure.

---

## The fix: execution-level caching

[mnemon-ai](https://github.com/smartass-4ever/Mnemon) caches at the plan level — not the prompt level.

```bash
pip install mnemon-ai
```

Two lines. Your existing graph stays completely unchanged:

```python
import mnemon
mnemon.init()

# your existing LangGraph code — untouched
from langgraph.graph import StateGraph

workflow = StateGraph(MyState)
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "summarizer")

app = workflow.compile()
result = app.invoke({"goal": "Competitor analysis for Acme Corp Q2"})
# second run: 2.66ms · 0 tokens · $0.00
```

Mnemon auto-instruments LangGraph at import time. No wrappers, no graph restructuring.

---

## How it works

**Exact match (System 1)** — fingerprint of your goal + context. If your agent has solved this before, it returns the cached result in ~2.66ms. Zero LLM calls.

**Semantic match (System 2)** — if the goal is *similar* but not identical, Mnemon finds the closest prior execution and only regenerates the segments that actually changed. You pay for the delta, not the whole run.

Enable semantic matching:

```bash
pip install mnemon-ai[full]  # local model, no API key needed
```

---

## Results

| Metric | Before | After |
|---|---|---|
| Tokens per run (avg) | ~1,250 | ~84 |
| LLM calls per run | 4 | 0.27 |
| Latency (cache hit) | 18–22s | 2.66ms |
| Monthly cost (1k runs/day) | ~$503 | ~$34 |

**93.3% token reduction. 7,500× faster on cache hits.**

---

## Try it

```bash
pip install mnemon-ai
```

```python
import mnemon
mnemon.init()
# your LangGraph agent now has memory across runs
```

Run `mnemon demo` to see a live cache hit in 30 seconds — no API key needed.

GitHub: [smartass-4ever/Mnemon](https://github.com/smartass-4ever/Mnemon)
