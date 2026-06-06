---
layout: post
title: "How to Add Caching to Any AutoGen Workflow in 2 Lines"
date: 2026-06-04
description: "AutoGen has no built-in execution cache. Here's how to add one without changing your agent or conversation code."
tags: [python, ai, llm, automation]
---

AutoGen doesn't have a built-in execution cache. Every `GroupChat`, every `ConversableAgent` run starts fresh. If your multi-agent workflow runs similar tasks repeatedly — research pipelines, code review agents, scheduled reports — you're paying full LLM price every time.

Here's how to fix it without touching your AutoGen code.

---

## The setup

```bash
pip install mnemon-ai
```

```python
import mnemon
mnemon.init()

# your existing AutoGen code — completely unchanged
import autogen

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4o", "api_key": "..."},
)
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
)

user_proxy.initiate_chat(
    assistant,
    message="Analyze Q2 sales data for Acme Corp and generate a summary report",
)
# second run with same or similar message: 2.66ms · 0 tokens · $0.00
```

Mnemon's MOTH layer patches AutoGen at startup. No agent changes, no conversation changes.

---

## What gets cached

Every LLM call your agents make is intercepted. On repeat runs:

- **Exact match** — same message, instant response from cache
- **Semantic match** — "Analyze Q2 sales for Acme Corp" matches "Generate Q2 sales analysis for Acme" — same task, different phrasing

For multi-agent workflows, common sub-tasks (data parsing, formatting, summarization) hit the cache across different top-level goals.

---

## For structured recurring workflows

```python
import autogen, mnemon

m = mnemon.init()

def run_analysis(goal, inputs, context, capabilities, constraints):
    user_proxy.initiate_chat(assistant, message=goal)
    return user_proxy.last_message()["content"]

result = m.run(
    goal="Q2 sales analysis for Acme Corp",
    inputs={"quarter": "Q2", "client": "Acme Corp"},
    generation_fn=run_analysis,
)

print(result["tokens_saved"])   # tokens saved on this run
print(result["cache_level"])    # "system1" | "system2" | "miss"
```

---

## Numbers

| | First run | Cached run |
|---|---|---|
| Tokens | ~1,250 | 0 |
| Latency | ~20s | 2.66ms |
| Cost | full price | $0.00 |

At 80% hit rate: **93% token reduction**.

---

## Install

```bash
pip install mnemon-ai           # exact match only
pip install mnemon-ai[full]     # + semantic matching (local, no API key)
```

GitHub: [smartass-4ever/Mnemon](https://github.com/smartass-4ever/Mnemon)
