# CrewAI Caching: Eliminate Redundant Token Costs on Recurring Agent Workflows

CrewAI has built-in tool-level caching. It's useful, but it caches tool *outputs* — not the LLM reasoning that produced them. Your agents still re-think, re-plan, and re-prompt the LLM from scratch every run.

If your CrewAI workflow runs on a schedule — daily reports, weekly audits, recurring pipelines — you're paying full LLM price every time for work that's 80–90% identical to the last run.

## Add execution caching to CrewAI in one line

```bash
pip install mnemon-ai
```

```python
import mnemon
mnemon.init()

# your existing CrewAI code — unchanged
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Research Analyst",
    goal="Research weekly AI news",
    llm="claude-sonnet-4-6",
)

task = Task(
    description="Summarize the top 5 AI developments this week",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
# second run same week: 2.66ms · 0 tokens · $0.00
```

Mnemon's MOTH layer patches CrewAI at startup before your crew runs. No changes to your agents, tasks, or crew definition.

## What gets cached

Mnemon operates at the LLM call level, below the CrewAI abstraction layer. Every `messages.create()` or `chat.completions.create()` call your crew makes is intercepted. On repeat runs:

- **System 1** — exact match. The same prompt returns instantly from a fingerprint lookup.
- **System 2** — semantic match. "Summarize top AI developments this week" matches "Summarize top AI news this week" because they're the same task. Only the genuinely new parts go to the LLM.

## For structured recurring tasks: Path 2

If your crew runs the same structured workflow repeatedly (same goal, different inputs per run), wrap it in `m.run()` for segment-level caching — only the parts that actually changed get regenerated:

```python
import mnemon
from crewai import Agent, Task, Crew

m = mnemon.init()

def run_research_crew(goal, inputs, context, capabilities, constraints):
    researcher = Agent(role="Research Analyst", goal=goal, llm="claude-sonnet-4-6")
    task = Task(description=goal, agent=researcher)
    crew = Crew(agents=[researcher], tasks=[task])
    return crew.kickoff().raw

result = m.run(
    goal="weekly AI news digest",
    inputs={"week": "Jun 2–6", "topics": ["LLMs", "agents", "open source"]},
    generation_fn=run_research_crew,
)

print(result["output"])
print(result["tokens_saved"])   # tokens saved vs. full run
print(result["cache_level"])    # "system1" | "system2" | "miss"
```

With segment-level caching, if 3 of 5 report sections are unchanged from last week, only the 2 changed sections go to the LLM. Token cost scales with how much actually changed, not with the size of the full workflow.

## Token savings at scale

| Hit rate | Daily runs | Monthly savings |
|---|---|---|
| 60% | 100 | ~$34 |
| 80% | 1,000 | ~$503 |
| 80% | 10,000 | ~$5,034 |

Hit rates of 60–80% are typical for recurring CrewAI workflows after ~10 runs. The cache gets smarter every run.

## Install

```bash
pip install mnemon-ai           # System 1 only (exact match)
pip install mnemon-ai[full]     # System 1 + System 2 (semantic, local model, no API key)
```

```python
import mnemon
mnemon.init()
```

That's it. Your existing CrewAI code runs unchanged.

GitHub: [github.com/smartass-4ever/Mnemon](https://github.com/smartass-4ever/Mnemon)
PyPI: [pypi.org/project/mnemon-ai](https://pypi.org/project/mnemon-ai/)
