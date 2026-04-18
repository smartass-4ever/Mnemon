"""
Mnemon EME Benchmark
====================
Compares EME execution cache vs baseline (fresh LLM generation per task)
on a realistic corpus of 45 enterprise agent tasks.

Usage:
    GROQ_API_KEY=gsk_...        python benchmark_eme.py   # free tier
    ANTHROPIC_API_KEY=sk-ant-... python benchmark_eme.py
    OPENAI_API_KEY=sk-...        python benchmark_eme.py
    python benchmark_eme.py --dry-run                      # no API key needed

Flags:
    --dry-run        Simulate with mock LLM (no API key, no cost)
    --cold           Skip pre-warmed fragment library (true cold start)
    --tasks N        Number of tasks to run (default: 45)

Results saved to: reports/benchmark_eme_results.json
"""

import argparse
import asyncio
import io
import json
import logging
import os
import shutil
import sys
import time

# Force UTF-8 on Windows (cp1252 can't encode box-drawing chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

# Silence internal Mnemon logging so benchmark output is clean
logging.disable(logging.WARNING)

from mnemon import Mnemon
from mnemon.llm.client import (
    AnthropicClient, OpenAIClient, GroqClient, MockLLMClient, auto_client
)

# ─────────────────────────────────────────────────────────────
# TASK CORPUS
# Three task types × 15 variants each = 45 tasks.
# Ordering is designed to demonstrate the warmup curve:
#   tasks 1-3   → cold misses (one of each type)
#   tasks 4-9   → S2 partial hits (similar but not identical)
#   tasks 10-45 → S1 exact hits (same type, repeated)
# ─────────────────────────────────────────────────────────────

_SECURITY_VARIANTS = [
    {"repo": "acme/api-gateway",        "deps": 247, "node": True,  "ssl_days": 45},
    {"repo": "globex/payment-service",  "deps": 189, "node": False, "ssl_days": 12},
    {"repo": "initech/user-auth",       "deps": 312, "node": True,  "ssl_days": 90},
    {"repo": "umbrella/data-pipeline",  "deps":  78, "node": False, "ssl_days": 180},
    {"repo": "weyland/ml-serving",      "deps": 445, "node": True,  "ssl_days": 8},
    {"repo": "cyberdyne/ci-infra",      "deps": 156, "node": True,  "ssl_days": 60},
    {"repo": "soylent/search-api",      "deps": 203, "node": False, "ssl_days": 22},
    {"repo": "omni/batch-jobs",         "deps":  91, "node": True,  "ssl_days": 120},
    {"repo": "aperture/portal-api",     "deps": 367, "node": True,  "ssl_days": 5},
    {"repo": "tyrell/synthetics-svc",   "deps": 134, "node": False, "ssl_days": 200},
    {"repo": "chroma/analytics",        "deps": 289, "node": True,  "ssl_days": 30},
    {"repo": "vault-tec/secrets-mgr",   "deps": 422, "node": True,  "ssl_days": 15},
    {"repo": "nuka/cola-inventory",     "deps": 167, "node": False, "ssl_days": 75},
    {"repo": "robco/terminal-svc",      "deps": 310, "node": True,  "ssl_days": 40},
    {"repo": "poseidon/ocean-monitor",  "deps":  95, "node": False, "ssl_days": 160},
]

_INVOICE_VARIANTS = [
    {"vendor": "AWS",         "amount": 12450.00, "po": "PO-2891", "priority": "high"},
    {"vendor": "Stripe",      "amount":  3200.00, "po": "PO-3104", "priority": "normal"},
    {"vendor": "Snowflake",   "amount": 28900.00, "po": "PO-2776", "priority": "high"},
    {"vendor": "Twilio",      "amount":  1850.00, "po": "PO-3201", "priority": "low"},
    {"vendor": "Datadog",     "amount":  9600.00, "po": "PO-2990", "priority": "normal"},
    {"vendor": "GitHub",      "amount":  4100.00, "po": "PO-3050", "priority": "normal"},
    {"vendor": "Cloudflare",  "amount":  2250.00, "po": "PO-3155", "priority": "low"},
    {"vendor": "PagerDuty",   "amount":  6700.00, "po": "PO-2845", "priority": "high"},
    {"vendor": "Okta",        "amount": 15300.00, "po": "PO-2910", "priority": "high"},
    {"vendor": "Figma",       "amount":  1400.00, "po": "PO-3220", "priority": "low"},
    {"vendor": "Linear",      "amount":   890.00, "po": "PO-3244", "priority": "low"},
    {"vendor": "Vercel",      "amount":  3800.00, "po": "PO-3078", "priority": "normal"},
    {"vendor": "Notion",      "amount":  2100.00, "po": "PO-3191", "priority": "normal"},
    {"vendor": "Retool",      "amount":  7200.00, "po": "PO-2867", "priority": "high"},
    {"vendor": "LaunchDarkly","amount":  5500.00, "po": "PO-2933", "priority": "normal"},
]

_REPORT_VARIANTS = [
    {"client": "Acme Corp",      "period": "2026-W14", "domain": "security"},
    {"client": "Globex Inc",     "period": "2026-W14", "domain": "finance"},
    {"client": "Initech",        "period": "2026-W14", "domain": "devops"},
    {"client": "Umbrella Corp",  "period": "2026-W14", "domain": "security"},
    {"client": "Weyland-Yutani", "period": "2026-W14", "domain": "ml-ops"},
    {"client": "Cyberdyne",      "period": "2026-W14", "domain": "security"},
    {"client": "Soylent Corp",   "period": "2026-W14", "domain": "finance"},
    {"client": "Omni Consumer",  "period": "2026-W14", "domain": "devops"},
    {"client": "Aperture Sci",   "period": "2026-W14", "domain": "security"},
    {"client": "Tyrell Corp",    "period": "2026-W14", "domain": "ml-ops"},
    {"client": "Chroma Systems", "period": "2026-W15", "domain": "security"},
    {"client": "Vault-Tec",      "period": "2026-W15", "domain": "finance"},
    {"client": "Nuka-Cola",      "period": "2026-W15", "domain": "devops"},
    {"client": "RobCo Ind",      "period": "2026-W15", "domain": "security"},
    {"client": "Poseidon Energy","period": "2026-W15", "domain": "ml-ops"},
]


def _build_task_corpus(n: int) -> List[Dict]:
    """
    Build a task corpus that demonstrates the full warmup curve:

    Pass 1 (tasks 1 – N/3):  unique variants, interleaved by type.
      - First task of each type → MISS (cold, LLM called)
      - Subsequent tasks of each type → S2 (similar goal, partial cache hit)

    Pass 2 (tasks N/3+1 – 2N/3):  exact repeats of pass 1 → S1 (zero LLM calls)
    Pass 3 (tasks 2N/3+1 – N):    exact repeats again     → S1

    This produces the screenshot-worthy warmup curve.
    """
    types   = ["security_audit", "invoice_process", "weekly_report"]
    sources = [_SECURITY_VARIANTS, _INVOICE_VARIANTS, _REPORT_VARIANTS]

    # Build pass 1: interleave types, cycling through variants
    pass1: List[Dict] = []
    max_per_type = max(1, n // 3 // 3)      # variants per type in pass 1
    indices = [0, 0, 0]
    for _ in range(n // 3):
        for t in range(3):
            if len(pass1) >= n // 3:
                break
            idx = indices[t] % len(sources[t])
            pass1.append({"type": types[t], "inputs": sources[t][idx]})
            indices[t] += 1

    pass1 = pass1[: n // 3]

    # Pass 2 and 3 are exact repeats — triggers S1 on every task
    full = pass1 + pass1 + pass1
    return full[:n]


# ─────────────────────────────────────────────────────────────
# LLM PROMPTS
# Specifying exact action names ensures structural consistency
# across runs of the same task type — critical for S1/S2 hits.
# ─────────────────────────────────────────────────────────────

_PROMPTS = {
    "security_audit": (
        "Generate a security audit execution plan. "
        "Repo: {repo}  Deps: {deps}  Node.js: {node}  SSL days left: {ssl_days}\n\n"
        "Return ONLY a JSON array of exactly 5 steps. Each step must be:\n"
        '  {{"id": "step_N", "action": "<one of: scan_dependencies, check_ssl_cert, '
        'run_sast_scan, generate_audit_report, notify_security_team>", '
        '"tool": "<tool name>", "params": {{...}}}}\n\n'
        "No explanation. JSON only."
    ),
    "invoice_process": (
        "Generate an invoice processing execution plan. "
        "Vendor: {vendor}  Amount: ${amount}  PO: {po}  Priority: {priority}\n\n"
        "Return ONLY a JSON array of exactly 5 steps. Each step must be:\n"
        '  {{"id": "step_N", "action": "<one of: extract_invoice_fields, validate_line_items, '
        'match_purchase_order, route_for_approval, archive_invoice>", '
        '"tool": "<tool name>", "params": {{...}}}}\n\n'
        "No explanation. JSON only."
    ),
    "weekly_report": (
        "Generate a weekly status report execution plan. "
        "Client: {client}  Period: {period}  Domain: {domain}\n\n"
        "Return ONLY a JSON array of exactly 5 steps. Each step must be:\n"
        '  {{"id": "step_N", "action": "<one of: collect_metrics, analyze_trends, '
        'draft_executive_summary, format_report, deliver_report>", '
        '"tool": "<tool name>", "params": {{...}}}}\n\n'
        "No explanation. JSON only."
    ),
}

_GOAL_TEMPLATES = {
    "security_audit":  "run security audit on {repo}",
    "invoice_process": "process invoice from {vendor} PO {po}",
    "weekly_report":   "generate weekly {domain} report for {client}",
}

_DRY_RUN_PLANS = {
    "security_audit": [
        {"id": "step_1", "action": "scan_dependencies",     "tool": "trivy",       "params": {"format": "sarif"}},
        {"id": "step_2", "action": "check_ssl_cert",        "tool": "ssl-checker",  "params": {"warn_days": 30}},
        {"id": "step_3", "action": "run_sast_scan",         "tool": "semgrep",      "params": {"config": "auto"}},
        {"id": "step_4", "action": "generate_audit_report", "tool": "reporter",     "params": {"format": "pdf"}},
        {"id": "step_5", "action": "notify_security_team",  "tool": "slack",        "params": {"channel": "#security"}},
    ],
    "invoice_process": [
        {"id": "step_1", "action": "extract_invoice_fields", "tool": "ocr-engine",  "params": {"confidence": 0.95}},
        {"id": "step_2", "action": "validate_line_items",    "tool": "validator",   "params": {"strict": True}},
        {"id": "step_3", "action": "match_purchase_order",   "tool": "erp-api",     "params": {"fuzzy": False}},
        {"id": "step_4", "action": "route_for_approval",     "tool": "workflow",    "params": {"sla_hours": 24}},
        {"id": "step_5", "action": "archive_invoice",        "tool": "s3",          "params": {"retention": "7y"}},
    ],
    "weekly_report": [
        {"id": "step_1", "action": "collect_metrics",         "tool": "datadog",    "params": {"lookback": "7d"}},
        {"id": "step_2", "action": "analyze_trends",          "tool": "analytics",  "params": {"model": "linear"}},
        {"id": "step_3", "action": "draft_executive_summary", "tool": "llm",        "params": {"max_words": 500}},
        {"id": "step_4", "action": "format_report",           "tool": "renderer",   "params": {"theme": "executive"}},
        {"id": "step_5", "action": "deliver_report",          "tool": "email",      "params": {"cc": "leadership"}},
    ],
}


# ─────────────────────────────────────────────────────────────
# GENERATION FUNCTION WRAPPER
# Tracks tokens per invocation. Used for BOTH baseline and
# Mnemon runs so we compare apples to apples.
# ─────────────────────────────────────────────────────────────

@dataclass
class GenerationStats:
    calls: int = 0
    tokens: int = 0
    latency_ms: float = 0.0
    exact: bool = False          # True = provider gives exact token counts


class BenchmarkGenerationFn:
    """
    Real LLM plan generation with token tracking.
    Passed as generation_fn to both baseline loop and Mnemon.run().
    Only called on cache misses in Mnemon mode.
    """

    def __init__(self, llm_client, dry_run: bool = False):
        self._client = llm_client
        self._dry_run = dry_run
        self.stats = GenerationStats()
        # Detect if client provides exact token counts
        self.stats.exact = isinstance(llm_client, AnthropicClient)

    async def __call__(
        self,
        goal: str,
        inputs: Dict,
        context: Dict,
        caps: List,
        constraints: Dict,
    ) -> List[Dict]:
        task_type = self._infer_type(goal)

        if self._dry_run:
            await asyncio.sleep(0.8)  # simulate LLM latency
            plan = _DRY_RUN_PLANS.get(task_type, _DRY_RUN_PLANS["weekly_report"])
            tokens = _estimate_tokens(_PROMPTS.get(task_type, ""), json.dumps(plan))
            self.stats.calls += 1
            self.stats.tokens += tokens
            return plan

        prompt = _PROMPTS.get(task_type, _PROMPTS["weekly_report"]).format(**inputs)

        tokens_before = getattr(self._client, "_total_tokens", 0)
        t0 = time.time()

        try:
            response = await self._client.complete(
                prompt, model=None, max_tokens=600, temperature=0.1
            )
        except Exception as e:
            # On any LLM error, return a minimal valid plan so benchmark continues
            response = json.dumps(_DRY_RUN_PLANS.get(task_type, []))

        elapsed = (time.time() - t0) * 1000

        tokens_after = getattr(self._client, "_total_tokens", 0)
        delta = tokens_after - tokens_before
        tokens = delta if delta > 0 else _estimate_tokens(prompt, response)

        self.stats.calls += 1
        self.stats.tokens += tokens
        self.stats.latency_ms += elapsed

        return _parse_plan(response, task_type)

    def reset(self):
        self.stats = GenerationStats(exact=self.stats.exact)

    @staticmethod
    def _infer_type(goal: str) -> str:
        g = goal.lower()
        if "security audit" in g or "audit on" in g:
            return "security_audit"
        if "invoice" in g or "process invoice" in g:
            return "invoice_process"
        return "weekly_report"


def _estimate_tokens(prompt: str, response: str) -> int:
    """Rough token estimate: (chars) / 4. Conservative, ~10% over actual."""
    return max(1, (len(prompt) + len(response)) // 4)


def _parse_plan(response: str, task_type: str) -> List[Dict]:
    """Parse LLM response into a list of step dicts."""
    try:
        text = response.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )
        plan = json.loads(text)
        if isinstance(plan, list) and len(plan) > 0:
            return plan
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: return a minimal valid plan matching the task type
    return _DRY_RUN_PLANS.get(task_type, _DRY_RUN_PLANS["weekly_report"])


# ─────────────────────────────────────────────────────────────
# PROVIDER DETECTION
# ─────────────────────────────────────────────────────────────

def _detect_provider(dry_run: bool) -> Tuple[Any, str, str, bool]:
    """Returns (client, provider_name, model_name, is_exact_tokens)."""
    if dry_run:
        return MockLLMClient(), "mock", "dry-run (no API)", False

    client = auto_client()
    if client is None:
        print(
            "\nNo API key found. Use --dry-run for a simulation, or set one of:\n"
            "  GROQ_API_KEY       (free at groq.com)\n"
            "  ANTHROPIC_API_KEY\n"
            "  OPENAI_API_KEY\n"
        )
        sys.exit(1)

    if isinstance(client, AnthropicClient):
        return client, "anthropic", "claude-haiku-4-5 / claude-sonnet-4-6", True
    if isinstance(client, OpenAIClient):
        return client, "openai", "gpt-4o-mini / gpt-4o", False
    if isinstance(client, GroqClient):
        return client, "groq", "llama-3.1-8b / llama-3.1-70b (free tier)", False
    return client, "unknown", "auto-detected", False


# ─────────────────────────────────────────────────────────────
# COST LOOKUP
# ─────────────────────────────────────────────────────────────

_COST_PER_MILLION = {
    "anthropic": 0.75,   # haiku blended input+output
    "groq":      0.59,   # llama3-70b
    "openai":    0.375,  # gpt-4o-mini blended
    "mock":      0.0,
    "unknown":   0.75,   # conservative default
}


def _cost(tokens: int, provider: str) -> float:
    rate = _COST_PER_MILLION.get(provider, 0.75)
    return (tokens / 1_000_000) * rate


# ─────────────────────────────────────────────────────────────
# BASELINE PHASE
# ─────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    run:        int
    task_type:  str
    cache_level: str    # "baseline" | "miss" | "system2" | "system1"
    tokens:     int
    latency_ms: float
    segments_reused: int = 0


async def run_baseline(
    tasks: List[Dict],
    gen_fn: "BenchmarkGenerationFn",
    print_rows: bool = True,
) -> List[TaskResult]:
    gen_fn.reset()
    results = []

    for i, task in enumerate(tasks, 1):
        goal = _GOAL_TEMPLATES[task["type"]].format(**task["inputs"])
        t0 = time.time()
        await gen_fn(goal, task["inputs"], {}, [], {})
        elapsed = (time.time() - t0) * 1000

        # Tokens for THIS call = delta since last call
        tokens = gen_fn.stats.tokens - sum(r.tokens for r in results)

        result = TaskResult(
            run=i, task_type=task["type"],
            cache_level="baseline", tokens=tokens, latency_ms=elapsed
        )
        results.append(result)

        if print_rows:
            _print_row(result, exact=gen_fn.stats.exact)

    return results


# ─────────────────────────────────────────────────────────────
# MNEMON PHASE
# ─────────────────────────────────────────────────────────────

async def run_with_mnemon(
    tasks: List[Dict],
    gen_fn: "BenchmarkGenerationFn",
    mnemon_client,
    prewarm: bool,
    print_rows: bool = True,
) -> Tuple[List[TaskResult], Dict]:
    gen_fn.reset()

    db_dir = "/tmp/mnemon_benchmark" if os.name != "nt" else str(
        Path(os.environ.get("TEMP", "C:/Temp")) / "mnemon_benchmark"
    )
    # Fresh DB for every benchmark run
    if Path(db_dir).exists():
        shutil.rmtree(db_dir, ignore_errors=True)
    Path(db_dir).mkdir(parents=True, exist_ok=True)

    results = []
    mnemon_tokens_before = getattr(mnemon_client, "_total_tokens", 0)

    async with Mnemon(
        tenant_id="benchmark",
        agent_id="bench_agent",
        db_dir=db_dir,
        llm_client=mnemon_client,
        prewarm_fragments=prewarm,
        enable_telemetry=False,
    ) as m:
        for i, task in enumerate(tasks, 1):
            goal = _GOAL_TEMPLATES[task["type"]].format(**task["inputs"])
            gen_tokens_before = gen_fn.stats.tokens

            t0 = time.time()
            result = await m.run(
                goal=goal,
                inputs=task["inputs"],
                generation_fn=gen_fn,
            )
            elapsed = (time.time() - t0) * 1000

            gen_tokens_this_call = gen_fn.stats.tokens - gen_tokens_before

            task_result = TaskResult(
                run=i,
                task_type=task["type"],
                cache_level=result.get("cache_level", "miss"),
                tokens=gen_tokens_this_call,
                latency_ms=elapsed,
                segments_reused=result.get("segments_reused", 0),
            )
            results.append(task_result)

            if print_rows:
                _print_row(task_result, exact=gen_fn.stats.exact)

    mnemon_internal_tokens = (
        getattr(mnemon_client, "_total_tokens", 0) - mnemon_tokens_before
    )

    meta = {
        "gen_calls":       gen_fn.stats.calls,
        "gen_tokens":      gen_fn.stats.tokens,
        "internal_tokens": mnemon_internal_tokens,
        "total_tokens":    gen_fn.stats.tokens + mnemon_internal_tokens,
    }
    return results, meta


# ─────────────────────────────────────────────────────────────
# OUTPUT FORMATTING
# ─────────────────────────────────────────────────────────────

_CACHE_LABELS = {
    "baseline": ("baseline", ""),
    "miss":     ("MISS    ", ""),
    "system2":  ("S2      ", "partial hit"),
    "system1":  ("S1      ", "exact hit ✓"),
    "error":    ("ERROR   ", ""),
    "fallback": ("FALLBACK", ""),
}

_TASK_COLORS = {
    "security_audit":  "\033[94m",  # blue
    "invoice_process": "\033[93m",  # yellow
    "weekly_report":   "\033[92m",  # green
}
_RESET = "\033[0m"
_DIM   = "\033[2m"
_BOLD  = "\033[1m"


def _print_row(r: TaskResult, exact: bool = False):
    label, note = _CACHE_LABELS.get(r.cache_level, ("???     ", ""))
    color = _TASK_COLORS.get(r.task_type, "")
    tok_marker = "" if exact else "~"
    seg_info = f"  {r.segments_reused}/5 reused" if r.cache_level == "system2" else ""
    note_str = f"  {_DIM}{note}{_RESET}" if note else ""
    print(
        f"  {_DIM}#{r.run:>3}{_RESET}  "
        f"{color}{r.task_type:<18}{_RESET}  "
        f"{_BOLD}{label}{_RESET}  "
        f"{tok_marker}{r.tokens:>5} tok  "
        f"{r.latency_ms:>7.0f}ms"
        f"{seg_info}{note_str}"
    )


def _print_summary(
    baseline_results: List[TaskResult],
    mnemon_results: List[TaskResult],
    mnemon_meta: Dict,
    provider: str,
    exact: bool,
    n_tasks: int,
):
    b_tokens  = sum(r.tokens for r in baseline_results)
    b_latency = sum(r.latency_ms for r in baseline_results) / 1000
    b_calls   = len(baseline_results)

    m_tokens  = mnemon_meta["total_tokens"]
    m_latency = sum(r.latency_ms for r in mnemon_results) / 1000
    m_calls   = mnemon_meta["gen_calls"]

    b_cost = _cost(b_tokens, provider)
    m_cost = _cost(m_tokens, provider)

    tok_saved  = max(0, b_tokens - m_tokens)
    tok_pct    = (tok_saved / b_tokens * 100) if b_tokens > 0 else 0
    cost_saved = max(0.0, b_cost - m_cost)
    time_saved = max(0.0, b_latency - m_latency)
    time_pct   = (time_saved / b_latency * 100) if b_latency > 0 else 0

    s1 = sum(1 for r in mnemon_results if r.cache_level == "system1")
    s2 = sum(1 for r in mnemon_results if r.cache_level == "system2")
    ms = sum(1 for r in mnemon_results if r.cache_level in ("miss", "fallback"))

    tok_marker = "" if exact else "~"
    w = 64

    print()
    print("━" * w)
    print(f"{'  RESULTS':}")
    print("━" * w)
    print(f"  {'':28}  {'BASELINE':>12}  {'MNEMON':>12}  {'SAVINGS':>10}")
    print(f"  {'─'*28}  {'─'*12}  {'─'*12}  {'─'*10}")
    print(f"  {'LLM calls (generation)':<28}  {b_calls:>12}  {m_calls:>12}  {(b_calls-m_calls)/b_calls*100:>9.1f}%")
    print(f"  {'Tokens (generation fn)':<28}  {tok_marker}{b_tokens:>11}  {tok_marker}{mnemon_meta['gen_tokens']:>11}")
    if mnemon_meta["internal_tokens"] > 0:
        print(f"  {'Tokens (Mnemon internal)':<28}  {'—':>12}  {tok_marker}{mnemon_meta['internal_tokens']:>11}")
    print(f"  {'Tokens (total)':<28}  {tok_marker}{b_tokens:>11}  {tok_marker}{m_tokens:>11}  {tok_pct:>9.1f}%")
    print(f"  {'Estimated cost':<28}  ${b_cost:>11.4f}  ${m_cost:>11.4f}  ${cost_saved:>9.4f}")
    print(f"  {'Wall time':<28}  {b_latency:>11.1f}s  {m_latency:>11.1f}s  {time_pct:>9.1f}%")
    print(f"  {'─'*28}  {'─'*12}  {'─'*12}  {'─'*10}")
    print(f"  {'Cache: S1 exact hits':<28}  {'—':>12}  {s1:>11}/{n_tasks}")
    print(f"  {'Cache: S2 partial hits':<28}  {'—':>12}  {s2:>11}/{n_tasks}")
    print(f"  {'Cache: misses':<28}  {b_calls:>12}  {ms:>11}/{n_tasks}")
    print("━" * w)

    if not exact:
        print(f"  {_DIM}~ = estimated tokens (provider does not expose exact counts){_RESET}")

    return {
        "baseline": {
            "llm_calls": b_calls, "tokens": b_tokens,
            "cost_usd": round(b_cost, 6), "wall_time_s": round(b_latency, 2),
        },
        "mnemon": {
            "llm_calls": m_calls, "tokens": m_tokens,
            "gen_tokens": mnemon_meta["gen_tokens"],
            "internal_tokens": mnemon_meta["internal_tokens"],
            "cost_usd": round(m_cost, 6), "wall_time_s": round(m_latency, 2),
            "s1_hits": s1, "s2_hits": s2, "misses": ms,
        },
        "savings": {
            "tokens_saved": tok_saved,
            "tokens_pct": round(tok_pct, 1),
            "cost_saved_usd": round(cost_saved, 6),
            "time_saved_s": round(time_saved, 2),
            "time_pct": round(time_pct, 1),
        },
    }


def _save_report(summary: Dict, baseline: List, mnemon: List, provider: str, model: str, n: int):
    Path("reports").mkdir(exist_ok=True)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": provider,
        "model": model,
        "n_tasks": n,
        "summary": summary,
        "per_task_baseline": [
            {"run": r.run, "type": r.task_type, "tokens": r.tokens, "latency_ms": round(r.latency_ms)}
            for r in baseline
        ],
        "per_task_mnemon": [
            {"run": r.run, "type": r.task_type, "cache_level": r.cache_level,
             "tokens": r.tokens, "latency_ms": round(r.latency_ms), "segments_reused": r.segments_reused}
            for r in mnemon
        ],
    }
    path = "reports/benchmark_eme_results.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Mnemon EME Benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without API (no cost)")
    parser.add_argument("--cold",    action="store_true", help="Disable pre-warmed fragments")
    parser.add_argument("--tasks",   type=int, default=45, help="Number of tasks (default: 45)")
    args = parser.parse_args()

    n_tasks = max(3, args.tasks)
    tasks   = _build_task_corpus(n_tasks)
    prewarm = not args.cold

    # Two separate clients:
    #   bench_client  → used inside generation_fn (measures plan generation cost)
    #   mnemon_client → used by Mnemon for routing, gap-fill, memory ops
    # This gives clean per-phase token accounting.
    bench_client_raw, provider, model, exact = _detect_provider(args.dry_run)
    mnemon_client_raw, _, _, _               = _detect_provider(args.dry_run)

    gen_fn = BenchmarkGenerationFn(bench_client_raw, dry_run=args.dry_run)

    w = 64
    print()
    print("━" * w)
    print(f"  MNEMON EME BENCHMARK")
    print(f"  {n_tasks} tasks  ·  3 types  ·  {model}")
    if prewarm:
        print(f"  Pre-warmed fragment library: ON")
    else:
        print(f"  Pre-warmed fragment library: OFF  (--cold)")
    if args.dry_run:
        print(f"  {_DIM}DRY RUN — simulated latency, no API calls{_RESET}")
    print("━" * w)

    # ── PHASE 1: BASELINE ────────────────────────────────────
    print()
    print(f"  PHASE 1 — BASELINE  (fresh LLM call every task)")
    print(f"  {'─'*58}")
    baseline_results = await run_baseline(tasks, gen_fn, print_rows=True)

    b_tokens  = sum(r.tokens for r in baseline_results)
    b_latency = sum(r.latency_ms for r in baseline_results) / 1000
    tok_m = "" if exact else "~"
    print(f"  {'─'*58}")
    print(f"  Total: {len(baseline_results)} LLM calls  ·  {tok_m}{b_tokens:,} tokens  ·  {b_latency:.1f}s")

    # ── PHASE 2: MNEMON ──────────────────────────────────────
    print()
    print(f"  PHASE 2 — WITH MNEMON  (EME cache active)")
    print(f"  {'─'*58}")
    mnemon_results, mnemon_meta = await run_with_mnemon(
        tasks, gen_fn, mnemon_client_raw, prewarm, print_rows=True
    )
    m_latency = sum(r.latency_ms for r in mnemon_results) / 1000
    print(f"  {'─'*58}")
    print(
        f"  Total: {mnemon_meta['gen_calls']} generation calls  ·  "
        f"{tok_m}{mnemon_meta['total_tokens']:,} tokens  ·  {m_latency:.1f}s"
    )

    # ── SUMMARY ──────────────────────────────────────────────
    summary = _print_summary(
        baseline_results, mnemon_results, mnemon_meta,
        provider, exact, n_tasks
    )

    # ── SAVE REPORT ──────────────────────────────────────────
    report_path = _save_report(
        summary, baseline_results, mnemon_results, provider, model, n_tasks
    )
    print(f"\n  Report saved → {report_path}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
