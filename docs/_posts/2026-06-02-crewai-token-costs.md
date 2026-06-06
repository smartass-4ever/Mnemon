---
layout: post
title: "How I Cut My CrewAI Agent's Token Costs by 93% with One Import"
date: 2026-06-02
description: "Every CrewAI kickoff is stateless — agents re-derive the same plans from scratch every run. Here's how to cache execution so repeat runs cost nothing."
tags: [python, ai, crewai, llm]
---

I have a CrewAI crew that generates product descriptions. Researcher agent, writer agent, editor agent — 20 runs a day, most with structurally similar inputs.

After a month I profiled the token usage. The researcher was re-deriving the same content structure on every run. The writer was regenerating the same outline logic. Across hundreds of runs, the crew was solving the same problems from scratch every time.

This is the hidden cost of CrewAI at production scale: **every kickoff is stateless**. Agents don't remember the last run. The reasoning starts over.

---

## Why prompt caching doesn't solve it

Prompt caching (Anthropic, OpenAI) helps when the input prefix is identical. In CrewAI, dynamic task allocation and inter-agent communication mean the prompts differ enough that caching rarely fires where you need it.

You still pay for the reasoning. Every time.

---

## The fix

```bash
pip install mnemon-ai
```

```python
import mnemon
mnemon.init()

# your existing CrewAI code — completely unchanged
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Research Specialist",
    goal="Find accurate product information",
    backstory="Expert at gathering and synthesizing product data"
)
writer = Agent(
    role="Content Writer",
    goal="Write compelling product descriptions",
    backstory="Skilled at converting research into persuasive copy"
)

research_task = Task(description="Research {product_name} features", agent=researcher)
write_task = Task(description="Write a product description based on the research", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
result = crew.kickoff(inputs={"product_name": "Widget Pro 3000"})
# second run with similar product: 2.66ms · 0 tokens · $0.00
```

Mnemon auto-instruments CrewAI at import. No crew restructuring, no wrapper classes.

---

## Two caching layers

**System 1 — Exact match**: fingerprint of the task goal + inputs. Same request → instant response, zero API cost.

**System 2 — Semantic match**: "Widget Pro 3000" and "Widget Pro 3001" are structurally the same task. Mnemon finds the closest prior run and only regenerates what actually changed. You pay for the diff, not the full crew run.

---

## Results

| Metric | Before | After |
|---|---|---|
| Tokens per kickoff | ~1,250 | ~84 |
| LLM API calls per run | 6 | 0.4 |
| Latency (cache hit) | 18–22s | 2.66ms |
| Monthly cost (1k runs/day) | ~$503 | ~$34 |

**93.3% token reduction. 7,500× faster on cache hits.**

---

## The compounding benefit

Mnemon observes every crew run. Failed executions get quarantined and don't pollute future cache hits. Successful patterns accumulate. Over time your crew's effective cost per run decreases and the failure rate drops.

---

## Try it

```bash
pip install mnemon-ai
mnemon demo   # live demo, no API key needed
```

GitHub: [smartass-4ever/Mnemon](https://github.com/smartass-4ever/Mnemon)
