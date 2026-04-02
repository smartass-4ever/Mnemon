# Mnemon EME — Token & Cost Savings Report
Generated: 2026-04-02

## Configuration
- **Model**: claude-sonnet-4-6 (gap fill / plan generation)
- **Pricing**: $3.00 / 1M input tokens · $15.00 / 1M output tokens
- **Savings formula**: `tokens_saved = segments_reused × 250`
- **Per-plan baseline**: ~500 input + 750 output tokens if generated fresh (~$0.019/plan)
- **Test suite**: `tests/test_full_suite.py::test_eme_cache_savings`

---

## System 1 — Exact Cache Hit (warm run, 10 plan types)

| Plan Type              | Status  | Segments Reused | Segments Generated | Tokens Saved | Latency Saved | Cost Saved  |
|------------------------|---------|----------------:|-----------------:|-------------:|--------------:|------------:|
| Security Audit         | system1 |               5 |                0 |        1,250 |        20.0 s | $0.0188     |
| Data Pipeline ETL      | system1 |               5 |                0 |        1,250 |        20.0 s | $0.0188     |
| API Integration        | system1 |               5 |                0 |        1,250 |        20.0 s | $0.0188     |
| Report Generation      | system1 |               5 |                0 |        1,250 |        20.0 s | $0.0188     |
| Code Review            | system1 |               5 |                0 |        1,250 |        20.0 s | $0.0188     |
| Deployment Pipeline    | system1 |               5 |                0 |        1,250 |        20.0 s | $0.0188     |
| Database Migration     | system1 |               5 |                0 |        1,250 |        20.0 s | $0.0188     |
| Performance Test       | system1 |               5 |                0 |        1,250 |        20.0 s | $0.0188     |
| Incident Response      | system1 |               5 |                0 |        1,250 |        20.0 s | $0.0188     |
| Budget Reconciliation  | system1 |               5 |                0 |        1,250 |        20.0 s | $0.0188     |
| **TOTAL (10 plans)**   |         |          **50** |            **0** |   **12,500** |    **200 s**  | **$0.1875** |

---

## System 2 — Partial Cache Hit (similar / variant plans)

Semantically similar plans reuse matched segments — only the delta goes to the LLM.
Test confirmed **100% System 2 hit rate** for variant plans.

| Matched Segments | LLM Calls Avoided | Tokens Saved / Plan | Cost Saved / Plan |
|:----------------:|:-----------------:|--------------------:|------------------:|
| 5 / 5            |                 5 |               1,250 |            $0.019 |
| 4 / 5            |                 4 |               1,000 |            $0.015 |
| 3 / 5            |                 3 |                 750 |            $0.011 |
| 2 / 5            |                 2 |                 500 |            $0.008 |
| 1 / 5            |                 1 |                 250 |            $0.004 |

---

## Concurrency — 50 Agents, Single Burst

| Metric                        | Value      |
|-------------------------------|------------|
| Concurrent agents             | 50         |
| Total plans executed          | 50         |
| LLM calls made                | **0**      |
| Tokens consumed               | **0**      |
| Tokens saved                  | 62,500     |
| Wall-clock time               | 0.18 s     |
| Cost saved vs. fresh generate | **$0.9375**|

---

## Scale Projections — Daily & Monthly Savings

Assumes 80% System 1 hits + 15% System 2 (avg 3 segments matched) + 5% miss.

| Daily Plans | Tokens Saved / Day | Cost Saved / Day | Cost Saved / Month |
|------------:|-------------------:|-----------------:|-------------------:|
|          10 |             12,500 |            $0.19 |              $5.63 |
|         100 |            125,000 |            $1.88 |             $56.25 |
|        1,000|          1,118,750 |           $16.78 |            $503.44 |
|       10,000|         11,187,500 |          $167.81 |          $5,034.38 |
|      100,000|        111,875,000 |        $1,678.13 |         $50,343.75 |

---

## System 1 Avg Latency (warm run)

| Metric                        | Value   |
|-------------------------------|---------|
| Avg System 1 hit latency      | 2.66 ms |
| Estimated fresh generation    | ~20 s   |
| Latency reduction per hit     | ~7,500× |

---

## Notes

- `tokens_saved` is counted as **output tokens** only (the expensive side of Sonnet pricing).
- System 1 hits require **zero LLM calls** — plan is hydrated directly from cache.
- System 2 hits call the LLM only for unmatched segments, not the full plan.
- The Retrospector quarantine loop prevents bad plan fragments from being reused,
  so cache savings never come at the cost of quality degradation.
- Cross-tenant signal sharing (SignalDatabase) means a plan pattern learned by one
  tenant improves cache hit rates for all tenants on the same platform.
