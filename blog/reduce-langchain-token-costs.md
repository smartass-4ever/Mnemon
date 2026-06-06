# How to Reduce LangChain Token Costs by 93%

LangChain has a built-in cache. You've probably seen it:

```python
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
set_llm_cache(InMemoryCache())
```

It works — for exact matches. If your prompt is "generate weekly security report for Acme Corp", the cache hits only when the next call is byte-for-byte identical. Change one word and you pay full price again.

For chatbots where every user message is unique, that's fine. For agents running recurring workflows — weekly reports, scheduled pipelines, nightly data processing — you're paying full price every single run.

## The problem with exact-match caching

A weekly security report agent might call:

```
"Generate security report for Acme Corp, week of Jan 6"
"Generate security report for Acme Corp, week of Jan 13"
"Generate security report for Acme Corp, week of Jan 20"
```

Three different strings. Three full LLM calls. The structure of that report — the sections, the analysis approach, the format — is 90% identical every week. You're paying for the same work seven times a month.

## One import, zero code changes

```bash
pip install mnemon-ai
```

```python
import mnemon
mnemon.init()

# your existing LangChain code — completely unchanged
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-6")
response = llm.invoke("Generate weekly security report for Acme Corp, week of Jan 13")
# hits the cache from the Jan 6 run: 2.66ms · 0 tokens · $0.00
```

Mnemon patches LangChain at startup via its MOTH layer. You don't change any LangChain code. You don't swap out your LLM class. You don't configure a cache backend.

## How the semantic matching works

Mnemon uses two matching modes:

- **System 1** — exact fingerprint match. Sub-millisecond. Zero tokens.
- **System 2** — semantic match. Embeds your input and compares it to cached inputs by meaning. "Weekly security report for Acme Corp, week of Jan 13" matches the Jan 6 cache entry because they're semantically identical tasks with only the date changed.

Enable System 2:

```bash
pip install mnemon-ai[full]  # uses local sentence-transformers, no API key needed
```

## Results on a recurring workflow

| | First run | Subsequent runs |
|---|---|---|
| Latency | ~20,000ms | **2.66ms** |
| Tokens used | 1,250 | **0** |
| Cost | $0.0038 | **$0.00** |

At 80% hit rate (typical for recurring workflows after ~10 runs):

| Daily runs | Monthly token savings |
|---|---|
| 100 | ~$56 |
| 1,000 | ~$503 |
| 10,000 | ~$5,034 |

## How it compares to LangChain's built-in cache

| | `set_llm_cache` | mnemon-ai |
|---|:---:|:---:|
| Exact match caching | ✅ | ✅ |
| Semantic matching | ❌ | ✅ |
| Segment-level plan caching | ❌ | ✅ |
| Works without code changes | ❌ | ✅ |
| Works across frameworks (CrewAI, AutoGen) | ❌ | ✅ |
| Learning loop | ❌ | ✅ |

## Getting started

```bash
pip install mnemon-ai
```

```python
import mnemon
mnemon.init()
# rest of your code unchanged
```

Run `mnemon demo` to see a live cache hit in 30 seconds, no API key needed.

GitHub: [github.com/smartass-4ever/Mnemon](https://github.com/smartass-4ever/Mnemon)
PyPI: [pypi.org/project/mnemon-ai](https://pypi.org/project/mnemon-ai/)
