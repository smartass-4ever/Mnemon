"""
Fragment Assembly test — verifies Mnemon can serve results from the
pre-warmed fragment library when System 2 finds no matching template.

Fragment Assembly fires when:
  1. No cached template matches the goal (System 1 + System 2 both miss)
  2. capabilities[] describes the steps of the task
  3. The fragment library covers >= 50% of those steps

Uses workflows whose steps map to the pre-warmed library fragments:
  - RAG research pipeline  (retrieve, expand_query, generate_with_citations, summarize)
  - Reasoning workflow     (decompose_goal, chain_of_thought, evaluate_options, self_critique)
  - Tool orchestration     (select_tool, validate_output, cache_result)

Each test uses a FRESH tenant so there are no prior cached templates.
The only source of system2 hits is the fragment library.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from mnemon.core.embedder import _try_load_sentence_transformers
_try_load_sentence_transformers()

import mnemon

DIVIDER = "-" * 60


def fresh(tenant: str):
    mnemon._instance = None
    return mnemon.init(tenant_id=tenant, db_dir=".", prewarm_fragments=True)


def run_case(m, label: str, goal: str, capabilities: list, mock_output: str):
    t0 = time.time()

    def gen(g, inp, ctx, caps, con):
        time.sleep(0.5)   # simulate LLM call
        return mock_output

    result = m.run(goal=goal, inputs={}, generation_fn=gen,
                   capabilities=capabilities)
    elapsed = (time.time() - t0) * 1000
    level = result.get("cache_level", "miss")
    label_map = {
        "system1": "SYSTEM 1 HIT",
        "system2": "SYSTEM 2 HIT (fragment assembly or semantic)",
        "miss":    "miss",
    }
    print(f"  {label}")
    print(f"    -> {label_map.get(level, level)}  |  {elapsed:.0f}ms", end="")
    if result.get("tokens_saved"):
        print(f"  |  {result['tokens_saved']} tokens saved", end="")
    print()
    return level


def wait_for_fragments(m, timeout=30):
    """Wait until the prewarm thread has loaded fragments into the DB."""
    from mnemon.fragments.library import FRAGMENT_COUNT
    deadline = time.time() + timeout
    while time.time() < deadline:
        stats = m.get_stats()
        n = stats.get("db", {}).get("fragments", 0)
        if n >= FRAGMENT_COUNT:
            print(f"  [prewarm] {n} fragments ready")
            return True
        time.sleep(0.5)
    print(f"  [prewarm] timed out — fragments may be incomplete")
    return False


def test_rag_pipeline():
    print(f"\n{DIVIDER}")
    print("TEST 1: RAG Research Pipeline")
    print("Steps map to: retrieve_and_generate, semantic_query_expansion,")
    print("              generate_with_citations, summarize_conversation")
    print(DIVIDER)

    m = fresh("frag_rag")
    wait_for_fragments(m)

    # These capability strings should match library fragment actions semantically
    rag_caps = [
        "retrieve and generate answer from documents",
        "expand query semantically for better retrieval",
        "generate response with citations from sources",
        "summarize conversation and key findings",
    ]

    results = []
    for i, (goal, desc) in enumerate([
        ("Research and summarize recent advances in transformer architectures",
         "first run — cold"),
        ("Find and summarise key papers on attention mechanisms in neural networks",
         "similar goal — fragment assembly expected"),
        ("Research latest developments in large language model training techniques",
         "rephrased similar goal"),
    ], 1):
        level = run_case(m, f"Run {i}: {desc}", goal, rag_caps,
                         "Summary: Key findings from retrieved documents...")
        results.append(level)

    hits = sum(1 for r in results[1:] if r in ("system1", "system2"))
    print(f"\n  Fragment assembly hits: {hits}/2 (runs 2-3)")
    return hits


def test_reasoning_workflow():
    print(f"\n{DIVIDER}")
    print("TEST 2: Goal Decomposition + Reasoning Workflow")
    print("Steps map to: decompose_goal_to_tasks, chain_of_thought,")
    print("              evaluate_options, self_critique")
    print(DIVIDER)

    m = fresh("frag_reasoning")
    wait_for_fragments(m)

    reasoning_caps = [
        "decompose goal into concrete subtasks",
        "apply chain of thought reasoning step by step",
        "evaluate available options against criteria",
        "self-critique and refine the output",
    ]

    results = []
    for i, (goal, desc) in enumerate([
        ("Plan a migration strategy for moving a monolith to microservices",
         "first run — cold"),
        ("Create a step-by-step plan for refactoring legacy codebase to modern architecture",
         "similar planning goal"),
        ("Devise a strategy for breaking down a large application into smaller services",
         "rephrased same intent"),
    ], 1):
        level = run_case(m, f"Run {i}: {desc}", goal, reasoning_caps,
                         "Plan: Step 1 — assess dependencies, Step 2 — identify seams...")
        results.append(level)

    hits = sum(1 for r in results[1:] if r in ("system1", "system2"))
    print(f"\n  Fragment assembly hits: {hits}/2 (runs 2-3)")
    return hits


def test_tool_orchestration():
    print(f"\n{DIVIDER}")
    print("TEST 3: Tool Selection + Validation Workflow")
    print("Steps map to: select_and_call_tool, validate_tool_output,")
    print("              tool_result_cache, nl_to_sql_query")
    print(DIVIDER)

    m = fresh("frag_tools")
    wait_for_fragments(m)

    tool_caps = [
        "select appropriate tool from available tools",
        "validate and check tool output schema",
        "cache tool result for reuse",
        "translate natural language request to structured query",
    ]

    results = []
    for i, (goal, desc) in enumerate([
        ("Query the sales database for last quarter revenue by region",
         "first run — cold"),
        ("Get total revenue figures broken down by geographic region from database",
         "similar DB query goal"),
        ("Retrieve quarterly sales data grouped by territory from data warehouse",
         "rephrased same intent"),
    ], 1):
        level = run_case(m, f"Run {i}: {desc}", goal, tool_caps,
                         '{"region": "US", "revenue": 1240000, "quarter": "Q1"}')
        results.append(level)

    hits = sum(1 for r in results[1:] if r in ("system1", "system2"))
    print(f"\n  Fragment assembly hits: {hits}/2 (runs 2-3)")
    return hits


def main():
    print("=" * 60)
    print("FRAGMENT ASSEMBLY TEST")
    print("Testing: System 2 fallback via pre-warmed fragment library")
    print("All tenants are fresh — no prior cached templates")
    print("=" * 60)

    h1 = test_rag_pipeline()
    h2 = test_reasoning_workflow()
    h3 = test_tool_orchestration()

    total = h1 + h2 + h3
    print(f"\n{'=' * 60}")
    print(f"Total hits from fragment assembly: {total}/6")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
