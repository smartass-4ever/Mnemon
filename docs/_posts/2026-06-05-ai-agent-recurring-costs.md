---
layout: post
title: "Your AI Agent Is Paying for the Same Work Every Day. Here's How to Stop It."
date: 2026-06-05
description: "Stateless agents re-derive the same plans from scratch on every run. Execution caching eliminates this — here's what it is and how to add it in 2 lines."
tags: [python, ai, machinelearning, llm]
---

Here's a cost structure most developers don't think about until the invoice arrives.

You have an agent. It runs every day — a report, a pipeline, a scheduled workflow. Each run costs roughly the same in tokens. Multiply that by 30 days and you have a monthly LLM bill that scales linearly with volume.

The problem: your agent is re-deriving the same work every single run.

---

## What "stateless" actually costs you

Most LLM agent frameworks — LangChain, LangGraph, CrewAI, AutoGen — are stateless by default. Each invocation is a cold start. The agent doesn't know it ran yesterday. It doesn't remember it solved this exact problem last week. It plans, reasons, and generates from scratch.

For genuinely novel tasks, that's correct behavior. For recurring workflows, it's burning money.

A weekly competitor analysis. A nightly data digest. A customer support agent handling the same question types daily. These workflows are 70–90% identical run to run. You're paying full price for the 10–30% that's actually new.

---

## Prompt caching is not the answer

Provider-level prompt caching (Anthropic, OpenAI, Google) caches the start of your prompt at a discounted rate. It's worth enabling. But it doesn't help when the variation is in the *reasoning* — when your agent produces slightly different plans for slightly different inputs. You still pay for the LLM call. You still wait 20 seconds.

Execution caching is different: **if the output is largely the same, skip the LLM entirely**.

---

## Two lines

```bash
pip install mnemon-ai
```

```python
import mnemon
mnemon.init()

# the rest of your code — completely unchanged
# works with LangChain, LangGraph, CrewAI, AutoGen,
# Anthropic SDK, OpenAI SDK, Google Generative AI, Groq
```

Mnemon intercepts LLM calls before they hit the API. If the same or similar task has been solved before, it returns the cached result instantly.

- **2.66ms** response on cache hit vs ~20s for a live LLM call
- **0 tokens** used on cache hit
- **$0.00** API cost on cache hit

---

## How it handles variation

Exact-match caching alone isn't enough — "weekly report for Acme Corp, Jan 6" and "weekly report for Acme Corp, Jan 13" are different strings but the same task.

Mnemon's semantic layer embeds your inputs and compares by meaning. Similar goals hit the same cache entry. For structured workflows, segment-level caching means only the parts that actually changed go to the LLM — you pay for the delta, not the full run.

---

## What it looks like in your terminal

First run:
```
Mnemon: first run — plan cached, next run will be instant
```

Cache hit:
```
Mnemon: cache hit · 1,250 tokens saved · ~$0.0038 · 20.0s faster
```

---

## At scale

| Daily runs | Hit rate | Monthly savings |
|---|---|---|
| 100 | 80% | ~$56 |
| 1,000 | 80% | ~$503 |
| 10,000 | 80% | ~$5,034 |

Hit rates of 60–80% are typical for recurring workflows after ~10 runs.

---

## Try it

```bash
pip install mnemon-ai
mnemon demo     # see it working in 30 seconds, no API key needed
```

GitHub: [smartass-4ever/Mnemon](https://github.com/smartass-4ever/Mnemon)
