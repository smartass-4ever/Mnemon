"""
benchmark_toolbench_eme.py
==========================
Evaluates Mnemon's EME (Execution Memory Engine) on real production-like data.

Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset (HuggingFace)
         27,000 real customer support queries across 27 intents and 11 categories.
         Chosen because it mirrors real production workloads: same intent, natural
         variation in phrasing — exactly the pattern EME's System 2 is built for.

What this tests:
  Queries grouped by intent are fed through Mnemon EME in sequence.
  Within each intent group, queries share structure but vary in wording
  (e.g. "cancel my order" vs "I need to cancel purchase #1234").
  This mirrors real agent workloads where repetition drives cache hits.

  generation_fn uses pre-computed dataset responses — no live LLM calls.
  Benchmark is free, fast, and fully reproducible.

Metrics:
  - System 1 hit rate  (exact match)
  - System 2 hit rate  (partial segment reuse)
  - Miss rate
  - Tokens saved vs full generation
  - Avg latency per call

Usage:
  python benchmarks/benchmark_toolbench_eme.py
  python benchmarks/benchmark_toolbench_eme.py --n 500
  python benchmarks/benchmark_toolbench_eme.py --n 200 --category order
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mnemon
from mnemon import Mnemon

# Tokens per response (ToolBench responses avg ~300 output tokens on Sonnet pricing)
TOKENS_PER_RESPONSE = 300
COST_PER_1M_OUTPUT  = 15.00  # claude-sonnet-4-6


@dataclass
class BenchmarkResult:
    total:          int = 0
    system1_hits:   int = 0
    system2_hits:   int = 0
    misses:         int = 0
    tokens_saved:   int = 0
    total_latency:  float = 0.0
    by_category:    Dict[str, Dict] = field(default_factory=dict)


def load_dataset_rows(n: int, category_filter: Optional[str]) -> List[Dict]:
    """Load rows from cached Bitext CSV, sampled evenly across all intents."""
    import csv
    from collections import defaultdict

    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "benchmarks", "bitext_support.csv"
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Run: python -c \"import urllib.request; urllib.request.urlretrieve("
            "'https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv', "
            "'data/benchmarks/bitext_support.csv')\""
        )

    per_intent = max(10, n // 27)
    buckets: Dict[str, List] = defaultdict(list)

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            instruction = row.get("instruction", "").strip()
            response    = row.get("response", "").strip()
            category    = row.get("intent", row.get("category", "unknown")).strip()

            if not instruction or not response:
                continue
            if category_filter and category_filter.lower() not in category.lower():
                continue
            if len(buckets[category]) >= per_intent:
                continue

            buckets[category].append({
                "instruction": instruction,
                "response":    response,
                "category":    category,
            })

    rows = [row for bucket in buckets.values() for row in bucket]
    print(f"Loaded {len(rows)} rows across {len(buckets)} intents ({per_intent} per intent)")
    return rows


async def run_benchmark(rows: List[Dict]) -> BenchmarkResult:
    result = BenchmarkResult()

    m = Mnemon(
        tenant_id="toolbench_benchmark",
        db_dir=os.path.join(os.path.dirname(__file__), ".."),
        silent=True,
    )
    await m.start()

    categories = defaultdict(list)
    for row in rows:
        categories[row["category"]].append(row)

    print(f"\nRunning EME across {len(rows)} instructions, {len(categories)} categories...\n")

    for cat, items in sorted(categories.items(), key=lambda x: -len(x[1])):
        cat_hits1 = cat_hits2 = cat_misses = 0

        for item in items:
            instruction = item["instruction"]
            response    = item["response"]

            async def generation_fn(goal, inputs, context, capabilities, constraints, _resp=response):
                return _resp

            t0 = time.perf_counter()
            res = await m.run(
                goal=instruction,
                inputs={"category": item["category"]},
                generation_fn=generation_fn,
            )
            latency = time.perf_counter() - t0

            result.total         += 1
            result.total_latency += latency

            status = res.get("cache_level") or res.get("status") or "miss"

            if status == "system1":
                result.system1_hits += 1
                cat_hits1           += 1
                result.tokens_saved += TOKENS_PER_RESPONSE
            elif status == "system2":
                result.system2_hits += 1
                cat_hits2           += 1
                segs = res.get("segments_reused", 0)
                total_segs = res.get("total_segments", 5) or 5
                result.tokens_saved += int(TOKENS_PER_RESPONSE * segs / total_segs)
            else:
                result.misses += 1
                cat_misses    += 1

        if len(items) > 1:
            result.by_category[cat] = {
                "n":       len(items),
                "system1": cat_hits1,
                "system2": cat_hits2,
                "misses":  cat_misses,
                "hit_rate": round((cat_hits1 + cat_hits2) / len(items), 4),
            }

    return result


def print_report(result: BenchmarkResult):
    total      = result.total or 1
    hit_rate   = (result.system1_hits + result.system2_hits) / total
    cost_saved = result.tokens_saved * COST_PER_1M_OUTPUT / 1_000_000
    avg_lat    = result.total_latency / total * 1000

    print("=" * 60)
    print("  MNEMON EME — ToolBench Benchmark Results")
    print("=" * 60)
    print(f"  Dataset:         Bitext Customer Support (27k real queries, 27 intents)")
    print(f"  Total calls:     {total:,}")
    print()
    print(f"  System 1 hits:   {result.system1_hits:,}  ({result.system1_hits/total:.1%})")
    print(f"  System 2 hits:   {result.system2_hits:,}  ({result.system2_hits/total:.1%})")
    print(f"  Misses:          {result.misses:,}  ({result.misses/total:.1%})")
    print(f"  Overall hit rate:{hit_rate:.1%}")
    print()
    print(f"  Tokens saved:    {result.tokens_saved:,}")
    print(f"  Cost saved:      ${cost_saved:.4f}  (Sonnet output pricing)")
    print(f"  Avg latency:     {avg_lat:.1f} ms/call")
    print()

    top = sorted(result.by_category.items(), key=lambda x: -x[1]["hit_rate"])[:10]
    if top:
        print("  Top categories by hit rate:")
        for cat, stats in top:
            print(f"    {cat[:35]:<35} {stats['hit_rate']:.0%}  (n={stats['n']})")
    print("=" * 60)

    out = {
        "benchmark": "Mnemon EME — Real Production Workload Simulation",
        "dataset": "bitext/Bitext-customer-support-llm-chatbot-training-dataset (HuggingFace, 27k queries)",
        "summary": {
            "total_calls":    total,
            "system1_hits":   result.system1_hits,
            "system2_hits":   result.system2_hits,
            "misses":         result.misses,
            "hit_rate":       round(hit_rate, 4),
            "tokens_saved":   result.tokens_saved,
            "cost_saved_usd": round(cost_saved, 4),
            "avg_latency_ms": round(avg_lat, 1),
        },
        "by_category": result.by_category,
    }
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "reports", "benchmark_toolbench_eme_results.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to reports/benchmark_toolbench_eme_results.json")


async def main(n: int, category: Optional[str]):
    rows = load_dataset_rows(n, category)
    if not rows:
        print("No data loaded. Check your internet connection.")
        return
    result = await run_benchmark(rows)
    print_report(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",        type=int, default=300,  help="Number of instructions to test")
    parser.add_argument("--category", type=str, default=None, help="Filter by category name")
    args = parser.parse_args()
    asyncio.run(main(args.n, args.category))
