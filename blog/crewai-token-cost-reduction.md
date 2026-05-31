# How I Cut My CrewAI Agent's Token Costs by 93% with One Import

I have a CrewAI crew that generates product descriptions. Researcher agent, writer agent, editor agent — all working together. 20 runs a day, most with structurally similar inputs.

I profiled the token usage after a month. The researcher was re-deriving the same content structure on every run. The writer was regenerating the same outline logic. The editor was applying the same rules. Across hundreds of runs, the crew was solving the same problems from scratch every time.

This is the hidden cost of CrewAI at production scale: **every crew kickoff is stateless**. Agents don't remember the last run. The task delegation logic runs fresh. The reasoning starts over.

---

## Who this affects

If your CrewAI setup involves any of these, you're paying for redundant computation:

- Content generation crews running daily or weekly
- Research crews hitting the same topics or industries repeatedly
- Data analysis crews with consistent task structures across similar inputs
- Customer support crews handling common request patterns

The more your crew runs, the more you pay — linearly. It never gets cheaper. It never gets faster.

---

## Why prompt caching doesn't solve it

Prompt caching (Anthropic, OpenAI) helps when the *input prefix* is identical. In CrewAI, even when the task structure is the same, the dynamic task allocation, inter-agent communication, and context passing means the prompts differ enough that caching rarely fires where you need it.

You still pay for the reasoning. Every time.

---

## The fix: cache the execution, not the prompt

[Mnemon](https://github.com/smartass-4ever/Mnemon) works at the execution plan level. It intercepts your crew runs before they hit the LLM and checks whether it's already solved this problem. If it has, it returns the result instantly.

```bash
pip install mnemon-ai
```

Two lines. Your crew stays exactly as-is:

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

research_task = Task(
    description="Research {product_name} features and market position",
    agent=researcher
)

write_task = Task(
    description="Write a product description based on the research",
    agent=writer
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
result = crew.kickoff(inputs={"product_name": "Widget Pro 3000"})
```

Mnemon auto-instruments CrewAI at import. No crew restructuring, no wrapper classes.

---

## How it works

Two caching layers:

**System 1 — Exact match**: SHA-256 fingerprint of the task goal + inputs. If your crew has handled this exact request before, it returns the cached result in ~2.66ms. Zero LLM calls, zero API cost.

**System 2 — Semantic match**: If the request is similar but not identical ("Widget Pro 3000" vs "Widget Pro 3001"), Mnemon finds the closest prior execution and only regenerates what actually changed. You pay for the diff, not the full crew run.

---

## Results across my content crew

45 runs across structurally similar product description tasks:

| Metric | Before | After |
|---|---|---|
| Tokens per kickoff (avg) | ~1,250 | ~84 |
| LLM API calls per run | 6 | 0.4 |
| Latency (cache hit) | 18–22s | 2.66ms |
| Monthly cost (1k runs/day) | ~$503 | ~$34 |

**93.3% token reduction. 7,500× faster on cache hits.**

First run of any new task still pays full cost. Every run after that doesn't.

---

## Explicit control per task

For crews with a mix of expensive and cheap tasks, you can cache specific tasks independently:

```python
import mnemon
from crewai import Agent, Task, Crew

m = mnemon.init()

def run_research_crew(product_name):
    crew = Crew(agents=[researcher], tasks=[research_task])
    return crew.kickoff(inputs={"product_name": product_name})

# Cache just the research phase — expensive, rarely changes for similar products
research_result = m.run(
    goal=f"Research product: {product_name}",
    inputs={"product_name": product_name, "category": "hardware"},
    generation_fn=lambda goal, inputs, *_: run_research_crew(inputs["product_name"])
)
```

---

## What it doesn't help with

- Crews handling genuinely novel requests every time
- Agents where real-time data freshness is critical
- Cold starts — first run of any new goal is always a full execution

---

## The compounding benefit

Mnemon observes every crew run. Failed executions get quarantined and don't pollute future cache hits. Successful patterns accumulate. Over time, your crew's effective cost per run decreases as the cache fills — and the failure rate drops as bad patterns get filtered out.

It's not just cheaper. It gets measurably better.

---

## Try it

```bash
pip install mnemon-ai
```

```python
import mnemon
mnemon.init()
# your CrewAI crew now has memory across runs
```

GitHub: [smartass-4ever/Mnemon](https://github.com/smartass-4ever/Mnemon)

If you're running CrewAI at any volume, the token savings in the first week will make the case for you.

---

*Tags: crewai, llm, python, ai, cost-optimization*
