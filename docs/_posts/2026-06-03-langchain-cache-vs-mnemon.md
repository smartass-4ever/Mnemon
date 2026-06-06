---
layout: post
title: "LangChain Already Has a Cache. Here's Why I Replaced It."
date: 2026-06-03
description: "set_llm_cache works — until your inputs vary even slightly. Here's what happens when you need semantic matching and cross-framework caching."
tags: [python, llm, langchain, tutorial]
---

LangChain's built-in cache is real and it works:

```python
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
set_llm_cache(InMemoryCache())
```

Same input → instant response. I used it for months. Then I hit its ceiling.

---

## The exact-match problem

LangChain's cache is a key-value store. The key is the exact prompt string. Change one character — a date, a name, a number — and it's a cache miss.

For a scheduled pipeline running weekly:

```
"Generate security report for Acme Corp, week of Jan 6"   → miss
"Generate security report for Acme Corp, week of Jan 13"  → miss
"Generate security report for Acme Corp, week of Jan 20"  → miss
```

Three different strings. Three full LLM calls. The structure of that report is 90% identical every week. I was paying for the same reasoning seven times a month.

---

## What I needed: semantic matching

I switched to [mnemon-ai](https://github.com/smartass-4ever/Mnemon). Same two-line setup:

```bash
pip install mnemon-ai[full]   # includes local semantic embedder
```

```python
import mnemon
mnemon.init()

from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-6")
response = llm.invoke("Generate security report for Acme Corp, week of Jan 13")
# hits the Jan 6 cache entry: 2.66ms · 0 tokens · $0.00
```

"Week of Jan 13" matches "week of Jan 6" because they're semantically the same task. Only genuinely novel inputs miss the cache.

---

## Side-by-side comparison

| | `set_llm_cache` | mnemon-ai |
|---|:---:|:---:|
| Exact match caching | ✅ | ✅ |
| Semantic matching (similar inputs) | ❌ | ✅ |
| Segment-level plan caching | ❌ | ✅ |
| Zero code changes | ❌ | ✅ |
| Works with CrewAI, AutoGen, LangGraph | ❌ | ✅ |
| Learning loop | ❌ | ✅ |
| Local, no external service | ✅ | ✅ |

---

## When to use each

**Use `set_llm_cache`** if your inputs are truly identical across calls and you only use LangChain.

**Use mnemon-ai** if your inputs vary even slightly, you run recurring workflows, or you use multiple frameworks.

---

## Token savings at scale

At 80% hit rate (typical for recurring workflows after ~10 runs):

| Daily runs | Monthly savings |
|---|---|
| 100 | ~$56 |
| 1,000 | ~$503 |
| 10,000 | ~$5,034 |

---

## Try it

```bash
pip install mnemon-ai
```

```python
import mnemon
mnemon.init()
# drop-in for any existing LangChain code
```

GitHub: [smartass-4ever/Mnemon](https://github.com/smartass-4ever/Mnemon)
