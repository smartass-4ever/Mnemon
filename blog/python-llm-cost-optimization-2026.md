# Python LLM Cost Optimization in 2026: Execution Caching vs Prompt Caching vs Model Routing

LLM API bills are predictable in one way: they keep growing as you scale. Three approaches dominate cost optimization in 2026. Here's what each one actually does and when to use it.

## Option 1: Prompt caching (provider-level)

Anthropic, OpenAI, and Google all offer prompt caching — you pay a discounted rate (typically 10–50% of full price) when the start of your prompt matches a previously processed prefix.

**When it helps:** Long system prompts that stay constant across requests. If your agent has a 2,000-token system prompt repeated on every call, you save on those 2,000 tokens.

**When it doesn't help:** The LLM still runs on every call. You're paying for reasoning, not just tokens. A 500ms LLM call with prompt caching enabled is still a 500ms LLM call. For recurring workflows where the *output* is largely the same, you're still paying in full.

## Option 2: Model routing

Route cheap requests to cheap models (GPT-4o-mini, Haiku) and expensive requests to expensive models (GPT-4o, Opus, Sonnet).

**When it helps:** High-volume pipelines with mixed complexity. Classification, extraction, and summarization tasks often don't need frontier models.

**When it doesn't help:** Adds latency and complexity. Routing decisions are hard to get right, and misrouting degrades output quality. Doesn't help at all if your requests are already on the right model.

## Option 3: Execution caching (skip the LLM entirely)

Cache the *output* of an LLM call and return it on repeat requests. If the same or similar task has been done before, serve the result from cache — zero tokens, sub-millisecond response.

**When it helps:** Any recurring workflow. Weekly reports, scheduled pipelines, support bots handling repeat question types, code review agents running the same checks repeatedly.

**When it doesn't help:** Genuinely novel requests every time. Random user conversations where no two inputs are similar.

The key distinction: prompt caching reduces token cost. Execution caching eliminates it.

```
Prompt caching:   pay 10–50% of tokens on cached prefixes
Execution caching: pay 0 tokens on cache hits
```

## Execution caching with mnemon-ai

```bash
pip install mnemon-ai
```

```python
import mnemon
mnemon.init()

# existing code unchanged — works with LangChain, CrewAI, AutoGen, LangGraph,
# Anthropic SDK, OpenAI SDK, Google Generative AI, Groq
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
response = llm.invoke("Generate weekly market summary for AAPL")
# second call: 2.66ms · 0 tokens · $0.00
```

Mnemon uses two matching modes:
- **Exact match** — same input, instant response from fingerprint lookup
- **Semantic match** — similar inputs ("market summary for AAPL this week" vs "AAPL weekly summary") hit the same cache entry

For structured recurring tasks, segment-level caching means only the parts that changed go to the LLM — token cost scales with what's actually new, not with the size of the full workflow.

## Benchmark comparison

| Approach | Token reduction | Latency impact | Code changes needed |
|---|---|---|---|
| Prompt caching | 10–50% on prefixes | None | Minimal |
| Model routing | 30–70% depending on mix | +50–200ms routing overhead | Significant |
| Execution caching (mnemon-ai) | **93% at 80% hit rate** | **−19,997ms on cache hits** | **Zero** |

## When to combine them

These aren't mutually exclusive:

- Use **prompt caching** on long system prompts regardless — it's free money from providers.
- Use **model routing** if you have high-volume mixed-complexity pipelines and engineering bandwidth.
- Use **execution caching** if you have recurring workflows — it has the highest ceiling and requires the least code.

For most teams running scheduled agent workflows, execution caching alone gets you to 90%+ cost reduction faster than anything else.

## Getting started

```bash
pip install mnemon-ai[full]  # includes local semantic embedder, no API key needed
```

```python
import mnemon
mnemon.init()
# your code runs unchanged
```

Check the savings: `mnemon demo` — live demo in 30 seconds, no API key needed.

GitHub: [github.com/smartass-4ever/Mnemon](https://github.com/smartass-4ever/Mnemon)
PyPI: [pypi.org/project/mnemon-ai](https://pypi.org/project/mnemon-ai/)
