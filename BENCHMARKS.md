# Mnemon Benchmarks — Token Cost, Latency, and Performance

## TL;DR

| Metric | Result |
|---|---|
| Token reduction on recurring workflows | **93.3%** |
| LLM API call reduction | **93%** |
| Cache hit latency | **2.66ms** |
| Fresh LLM generation latency | ~20,000ms |
| Speedup on cache hits | **7,500×** |
| Cost per cache hit | **$0.00** |

---

## Test Environment

- **Model:** claude-sonnet-4-6 ($3.00/1M input · $15.00/1M output)
- **Workflow types:** security audits, invoice processing, weekly reports
- **Total runs:** 45
- **Framework:** Mnemon v1.0.6
- **Embedder:** hash-projection (no sentence-transformers required for System 1)
- **Storage:** local SQLite

Raw data: [`reports/benchmark_eme_results.json`](reports/benchmark_eme_results.json)

---

## System 1 — Exact Cache Hit Performance

First run: goal is fingerprinted, plan stored as segments.
All subsequent exact-match runs: served from SQLite.

| Metric | Value |
|---|---|
| Average hit latency | **2.66ms** |
| LLM calls made | **0** |
| Tokens consumed | **0** |
| Tokens saved per hit | **1,250** |
| Cost saved per hit | **$0.019** |
| Speedup vs fresh generation | **7,500×** |

---

## System 2 — Semantic Cache Hit Performance

Similar goal, different inputs (same workflow type, different client/date).
Only changed segments regenerated.

| Segments matched | LLM calls avoided | Tokens saved | Cost saved |
|---|---|---|---|
| 5/5 | 5 | 1,250 | $0.019 |
| 4/5 | 4 | 1,000 | $0.015 |
| 3/5 | 3 | 750 | $0.011 |
| 2/5 | 2 | 500 | $0.008 |
| 1/5 | 1 | 250 | $0.004 |

---

## 45-Run End-to-End Benchmark

3 workflow types × 15 runs each. First run per type = cache miss (necessary). All subsequent runs cached.

| Run type | Count | Tokens | LLM calls |
|---|---|---|---|
| Cache miss (cold start) | 3 | 651 | 3 |
| System 2 hits | 12 | 0 | 0 |
| System 1 hits | 30 | 0 | 0 |
| **Total** | **45** | **651** | **3** |

**Baseline (no Mnemon):** 9,786 tokens · 45 LLM calls · $0.005774

**With Mnemon:** 651 tokens · 3 LLM calls · $0.000384

**Savings: 9,135 tokens · 42 LLM calls · $0.005390 · 93.3% reduction**

---

## Concurrency — 50 Agents Burst

50 concurrent agents all running the same workflow type simultaneously.

| Metric | Without Mnemon | With Mnemon |
|---|---|---|
| LLM API calls | 50 | **0** |
| Total tokens | 62,500 | **0** |
| Total cost | $0.9375 | **$0.00** |
| Wall-clock time | ~1,000s | **0.18s** |

---

## Latency Distribution

| Percentile | System 1 hit | Fresh LLM call |
|---|---|---|
| p50 | 1ms | ~18,000ms |
| p95 | 7ms | ~25,000ms |
| p99 | 12ms | ~30,000ms |

---

## Cost Projections — Monthly Savings

80% System 1 hit rate + 15% System 2 (avg 3 segments reused) + 5% miss.
Pricing: claude-sonnet-4-6.

| Daily runs | Daily tokens saved | Daily cost saved | Monthly cost saved |
|---|---|---|---|
| 10 | 12,500 | $0.19 | **$5.63** |
| 100 | 125,000 | $1.88 | **$56.25** |
| 1,000 | 1,118,750 | $16.78 | **$503.44** |
| 10,000 | 11,187,500 | $167.81 | **$5,034** |
| 100,000 | 111,875,000 | $1,678 | **$50,344** |

---

## What Drives the Savings

**Why is 93% achievable?**

For recurring workflows, the LLM re-derives the same plan structure on every run. The goal is the same. The context is the same. The capabilities are the same. Only the variable inputs change. Mnemon separates the stable structure from the variable inputs — caches the structure, updates only the inputs.

**Why is 2.66ms achievable?**

SQLite read latency on a local disk is sub-millisecond. The 2.66ms includes plan reconstruction from segments, cache lookup, and response assembly. No network call. No LLM inference. No token processing.

**Why does it compound?**

Every cache hit feeds the Experience Bus with a success signal. The Bus tracks per-workflow-type hit rates and latency baselines. The Retrospector quarantines failed segments. The cache improves on every run — not just grows.

---

## Running the Benchmarks Yourself

```bash
git clone https://github.com/smartass-4ever/Mnemon
cd Mnemon
pip install -e ".[dev]"
pytest tests/test_full_suite.py -v
```

Results will appear in `reports/`.

---

## Install

```bash
pip install mnemon-ai          # base install, no API key needed
mnemon demo                    # live benchmark in 30 seconds
```

[Full documentation](README.md) · [Cost savings breakdown](COST_SAVINGS.md) · [Raw benchmark data](reports/)
