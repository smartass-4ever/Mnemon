"""
EME Workflow Benchmark — Real-world agent workflow stress test.

Tests System 1 (exact match) and System 2 (partial segment match) against
8 genuine enterprise workflow domains with realistic variation patterns.

Workflow domains (based on real CrewAI / LangGraph / AutoGen patterns):
  1. Security audit            — 8-step, network + CVE + report
  2. Customer support triage   — 6-step, classify → KB lookup → draft → escalate
  3. ETL data pipeline         — 7-step, extract → validate → transform → load
  4. Code review               — 6-step, checkout → lint → test → security → report
  5. Financial report          — 8-step, fetch → metrics → benchmark → narrative → PDF
  6. Content marketing         — 7-step, research → brief → draft → edit → schedule
  7. DevOps deployment         — 9-step, build → test → stage → health → promote → notify
  8. Research synthesis        — 6-step, search → filter → extract → synthesise → cite

Variation matrix per workflow:
  Run A — cold miss  (first run, plan gets cached)
  Run B — exact repeat, different inputs     → expect System 1
  Run C — same goal, rephrased              → expect System 1 or System 2
  Run D — same domain, different sub-task   → expect System 2 or miss
  Run E — cross-domain semantic neighbour   → expect miss (measures boundary)

Run:
  python benchmarks/benchmark_eme_workflows.py
  python benchmarks/benchmark_eme_workflows.py --domain security
  python benchmarks/benchmark_eme_workflows.py --verbose
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnemon import Mnemon
from mnemon.fragments.library import load_fragments


# ─────────────────────────────────────────────────────────────────────────────
# REALISTIC PLAN GENERATORS
# Each returns a list of step dicts with id, action, tool, depends_on, outputs.
# These simulate what a real LLM planner would return.
# ─────────────────────────────────────────────────────────────────────────────

async def _plan_security_audit(goal, inputs, context, caps, constraints):
    """8-step security audit — based on real CrewAI security crew patterns."""
    await asyncio.sleep(0.05)   # simulates real LLM latency (fast for benchmark)
    client = inputs.get("client", "Acme Corp")
    scope  = inputs.get("scope", "internal network")
    return [
        {"id": "s1", "action": "enumerate_assets",    "tool": "nmap_scanner",      "params": {"target": scope},           "depends_on": [],         "outputs": ["asset_list"]},
        {"id": "s2", "action": "port_scan",           "tool": "nmap_scanner",      "params": {"flags": "-sV -sC"},        "depends_on": ["s1"],     "outputs": ["open_ports"]},
        {"id": "s3", "action": "cve_lookup",          "tool": "nvd_api",           "params": {"severity": "HIGH,CRITICAL"},"depends_on": ["s2"],    "outputs": ["cve_list"]},
        {"id": "s4", "action": "exploit_check",       "tool": "metasploit_rpc",    "params": {"mode": "check_only"},      "depends_on": ["s3"],     "outputs": ["exploitable"]},
        {"id": "s5", "action": "auth_audit",          "tool": "ldap_inspector",    "params": {"check_mfa": True},         "depends_on": ["s1"],     "outputs": ["auth_findings"]},
        {"id": "s6", "action": "compliance_check",    "tool": "scap_scanner",      "params": {"profile": "NIST-800-53"},  "depends_on": ["s3","s5"],"outputs": ["compliance_gaps"]},
        {"id": "s7", "action": "risk_score",          "tool": "cvss_calculator",   "params": {"weights": "enterprise"},   "depends_on": ["s4","s6"],"outputs": ["risk_matrix"]},
        {"id": "s8", "action": "generate_report",     "tool": "report_engine",     "params": {"format": "pdf", "client": client}, "depends_on": ["s7"], "outputs": ["report_pdf"]},
    ]


async def _plan_support_triage(goal, inputs, context, caps, constraints):
    """6-step customer support triage — based on real AutoGen support patterns."""
    await asyncio.sleep(0.05)
    ticket_type = inputs.get("ticket_type", "billing")
    priority    = inputs.get("priority", "medium")
    return [
        {"id": "t1", "action": "classify_intent",    "tool": "intent_classifier",  "params": {"categories": ["billing","technical","account","general"]}, "depends_on": [], "outputs": ["intent", "confidence"]},
        {"id": "t2", "action": "fetch_customer_ctx", "tool": "crm_api",            "params": {"include_history": True},   "depends_on": ["t1"],     "outputs": ["customer_profile"]},
        {"id": "t3", "action": "kb_search",          "tool": "knowledge_base",     "params": {"top_k": 5, "intent": ticket_type}, "depends_on": ["t1"], "outputs": ["kb_articles"]},
        {"id": "t4", "action": "draft_response",     "tool": "response_generator", "params": {"tone": "empathetic", "priority": priority}, "depends_on": ["t2","t3"], "outputs": ["draft"]},
        {"id": "t5", "action": "policy_check",       "tool": "policy_engine",      "params": {"check_sla": True},         "depends_on": ["t4"],     "outputs": ["policy_flags"]},
        {"id": "t6", "action": "route_or_resolve",   "tool": "ticket_router",      "params": {"auto_resolve_threshold": 0.85}, "depends_on": ["t4","t5"], "outputs": ["resolution"]},
    ]


async def _plan_etl_pipeline(goal, inputs, context, caps, constraints):
    """7-step ETL pipeline — based on real data engineering agent patterns."""
    await asyncio.sleep(0.05)
    source = inputs.get("source", "postgres")
    dest   = inputs.get("destination", "snowflake")
    return [
        {"id": "e1", "action": "connect_source",     "tool": "db_connector",       "params": {"engine": source, "pool_size": 10}, "depends_on": [], "outputs": ["source_conn"]},
        {"id": "e2", "action": "schema_validation",  "tool": "schema_validator",   "params": {"strict": True},            "depends_on": ["e1"],     "outputs": ["schema_report"]},
        {"id": "e3", "action": "extract_incremental","tool": "cdc_extractor",      "params": {"watermark_col": "updated_at"},"depends_on": ["e1","e2"],"outputs": ["raw_records"]},
        {"id": "e4", "action": "data_quality_check", "tool": "great_expectations",  "params": {"suite": "standard"},       "depends_on": ["e3"],     "outputs": ["quality_report"]},
        {"id": "e5", "action": "transform",          "tool": "dbt_runner",         "params": {"target": "prod"},          "depends_on": ["e3","e4"],"outputs": ["transformed_records"]},
        {"id": "e6", "action": "load",               "tool": "db_writer",          "params": {"engine": dest, "batch_size": 5000}, "depends_on": ["e5"], "outputs": ["load_stats"]},
        {"id": "e7", "action": "notify_downstream",  "tool": "event_bus",          "params": {"event": "etl_complete"},   "depends_on": ["e6"],     "outputs": ["notification_id"]},
    ]


async def _plan_code_review(goal, inputs, context, caps, constraints):
    """6-step code review — based on real LangGraph CI agent patterns."""
    await asyncio.sleep(0.05)
    pr_id = inputs.get("pr_id", "PR-001")
    lang  = inputs.get("language", "python")
    return [
        {"id": "c1", "action": "checkout_diff",      "tool": "git_api",            "params": {"pr_id": pr_id, "include_context": 3}, "depends_on": [], "outputs": ["diff", "changed_files"]},
        {"id": "c2", "action": "static_analysis",    "tool": "sonarqube",          "params": {"language": lang, "rules": "strict"},    "depends_on": ["c1"], "outputs": ["lint_issues"]},
        {"id": "c3", "action": "run_tests",          "tool": "pytest_runner",      "params": {"coverage_threshold": 80},  "depends_on": ["c1"],     "outputs": ["test_results", "coverage"]},
        {"id": "c4", "action": "security_scan",      "tool": "bandit",             "params": {"severity": "medium"},      "depends_on": ["c1"],     "outputs": ["security_issues"]},
        {"id": "c5", "action": "dependency_audit",   "tool": "safety_checker",     "params": {"check_licenses": True},    "depends_on": ["c1"],     "outputs": ["dep_issues"]},
        {"id": "c6", "action": "post_review",        "tool": "github_api",         "params": {"auto_approve_threshold": 0}, "depends_on": ["c2","c3","c4","c5"], "outputs": ["review_comment"]},
    ]


async def _plan_financial_report(goal, inputs, context, caps, constraints):
    """8-step financial reporting — based on real enterprise finance agent patterns."""
    await asyncio.sleep(0.05)
    period  = inputs.get("period", "Q1-2026")
    entity  = inputs.get("entity", "Acme Corp")
    return [
        {"id": "f1", "action": "fetch_gl_data",      "tool": "erp_connector",      "params": {"period": period, "entity": entity}, "depends_on": [], "outputs": ["gl_entries"]},
        {"id": "f2", "action": "reconcile_accounts",  "tool": "reconciliation_engine","params": {"tolerance_bps": 1},            "depends_on": ["f1"],     "outputs": ["reconciled_tb"]},
        {"id": "f3", "action": "calculate_metrics",   "tool": "metric_engine",      "params": {"metrics": ["revenue","ebitda","fcf"]}, "depends_on": ["f2"], "outputs": ["kpis"]},
        {"id": "f4", "action": "benchmark_comparison","tool": "market_data_api",    "params": {"index": "SP500", "peers": 5},   "depends_on": ["f3"],     "outputs": ["benchmark_delta"]},
        {"id": "f5", "action": "variance_analysis",   "tool": "variance_engine",    "params": {"vs": "budget"},                "depends_on": ["f3"],     "outputs": ["variances"]},
        {"id": "f6", "action": "narrative_generation","tool": "llm_narrator",       "params": {"tone": "executive", "length": "2_page"}, "depends_on": ["f4","f5"], "outputs": ["narrative"]},
        {"id": "f7", "action": "audit_trail",         "tool": "audit_logger",       "params": {"standard": "SOX"},             "depends_on": ["f1","f2"],"outputs": ["audit_log"]},
        {"id": "f8", "action": "generate_pdf",        "tool": "report_renderer",    "params": {"template": "executive_pack"}, "depends_on": ["f6","f7"], "outputs": ["report_pdf"]},
    ]


async def _plan_content_marketing(goal, inputs, context, caps, constraints):
    """7-step content marketing — based on real CrewAI content crew patterns."""
    await asyncio.sleep(0.05)
    topic   = inputs.get("topic", "AI in enterprise")
    channel = inputs.get("channel", "linkedin")
    return [
        {"id": "m1", "action": "audience_research",  "tool": "audience_analyzer",  "params": {"channel": channel, "depth": "persona"}, "depends_on": [], "outputs": ["personas"]},
        {"id": "m2", "action": "competitor_scan",     "tool": "semrush_api",        "params": {"keywords": 20, "competitor_count": 5}, "depends_on": ["m1"], "outputs": ["gap_analysis"]},
        {"id": "m3", "action": "content_brief",       "tool": "brief_generator",    "params": {"word_count": 1500, "tone": "thought_leadership"}, "depends_on": ["m1","m2"], "outputs": ["brief"]},
        {"id": "m4", "action": "draft_content",       "tool": "llm_writer",         "params": {"model": "claude-sonnet-4-6", "style": "conversational"}, "depends_on": ["m3"], "outputs": ["draft"]},
        {"id": "m5", "action": "seo_optimize",        "tool": "surfer_seo",         "params": {"target_score": 85},        "depends_on": ["m4"],     "outputs": ["optimized_draft"]},
        {"id": "m6", "action": "brand_review",        "tool": "brand_checker",      "params": {"guidelines_version": "v3"}, "depends_on": ["m5"],    "outputs": ["approved_draft"]},
        {"id": "m7", "action": "schedule_publish",    "tool": "buffer_api",         "params": {"channel": channel, "optimal_time": True}, "depends_on": ["m6"], "outputs": ["post_id"]},
    ]


async def _plan_devops_deployment(goal, inputs, context, caps, constraints):
    """9-step DevOps deployment — based on real LangGraph CI/CD agent patterns."""
    await asyncio.sleep(0.05)
    service = inputs.get("service", "api-gateway")
    env     = inputs.get("environment", "production")
    return [
        {"id": "d1", "action": "build_image",        "tool": "docker_builder",     "params": {"push": True, "cache": True},         "depends_on": [],         "outputs": ["image_tag"]},
        {"id": "d2", "action": "run_unit_tests",      "tool": "pytest_runner",      "params": {"parallel": True},                    "depends_on": ["d1"],     "outputs": ["test_results"]},
        {"id": "d3", "action": "run_integration_tests","tool": "integration_runner","params": {"env": "staging"},                    "depends_on": ["d2"],     "outputs": ["integration_results"]},
        {"id": "d4", "action": "security_scan_image", "tool": "trivy_scanner",     "params": {"severity": "HIGH,CRITICAL"},         "depends_on": ["d1"],     "outputs": ["vuln_report"]},
        {"id": "d5", "action": "deploy_staging",      "tool": "kubectl",            "params": {"namespace": f"{service}-staging"},   "depends_on": ["d2","d4"],"outputs": ["staging_url"]},
        {"id": "d6", "action": "smoke_test",          "tool": "health_checker",     "params": {"timeout_s": 120, "retries": 3},      "depends_on": ["d5"],     "outputs": ["health_status"]},
        {"id": "d7", "action": "canary_deploy",       "tool": "kubectl",            "params": {"weight_pct": 10, "namespace": env},  "depends_on": ["d6"],     "outputs": ["canary_url"]},
        {"id": "d8", "action": "monitor_error_rate",  "tool": "datadog_api",        "params": {"window_min": 5, "threshold_pct": 1}, "depends_on": ["d7"],     "outputs": ["error_rate"]},
        {"id": "d9", "action": "full_promote_or_rollback","tool": "kubectl",        "params": {"strategy": "blue_green"},            "depends_on": ["d8"],     "outputs": ["deployment_status"]},
    ]


async def _plan_research_synthesis(goal, inputs, context, caps, constraints):
    """6-step research synthesis — based on real RAG agent patterns."""
    await asyncio.sleep(0.05)
    topic  = inputs.get("topic", "LLM evaluation methods")
    depth  = inputs.get("depth", "comprehensive")
    return [
        {"id": "r1", "action": "semantic_search",    "tool": "vector_store",       "params": {"top_k": 50, "rerank": True},         "depends_on": [],         "outputs": ["candidate_docs"]},
        {"id": "r2", "action": "web_search",         "tool": "serp_api",           "params": {"recency_days": 90, "count": 20},     "depends_on": [],         "outputs": ["web_results"]},
        {"id": "r3", "action": "filter_relevance",   "tool": "relevance_ranker",   "params": {"threshold": 0.75},                  "depends_on": ["r1","r2"],"outputs": ["filtered_docs"]},
        {"id": "r4", "action": "extract_claims",     "tool": "claim_extractor",    "params": {"mode": "factual"},                  "depends_on": ["r3"],     "outputs": ["claims"]},
        {"id": "r5", "action": "synthesise",         "tool": "llm_synthesiser",    "params": {"depth": depth, "max_tokens": 3000}, "depends_on": ["r4"],     "outputs": ["synthesis"]},
        {"id": "r6", "action": "cite_sources",       "tool": "citation_formatter", "params": {"style": "APA"},                     "depends_on": ["r4","r5"],"outputs": ["final_report"]},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# WORKFLOW REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

WORKFLOWS = {
    "security":    (_plan_security_audit,    "Weekly security audit"),
    "support":     (_plan_support_triage,    "Customer support ticket triage"),
    "etl":         (_plan_etl_pipeline,      "ETL data pipeline"),
    "codereview":  (_plan_code_review,       "Automated code review"),
    "finance":     (_plan_financial_report,  "Financial report generation"),
    "content":     (_plan_content_marketing, "Content marketing campaign"),
    "devops":      (_plan_devops_deployment, "DevOps deployment"),
    "research":    (_plan_research_synthesis,"Research synthesis"),
}


# ─────────────────────────────────────────────────────────────────────────────
# VARIATION MATRIX
# Each entry: (goal, inputs, description, expected_cache_level)
# expected_cache_level = "miss" | "system1" | "system2" | "?"
# ─────────────────────────────────────────────────────────────────────────────

def _variations(domain: str) -> List[Tuple[str, Dict, str, str]]:
    """Return (goal, inputs, label, expected) for one domain."""
    if domain == "security":
        base = "weekly security audit"
        return [
            (base, {"client": "Acme Corp",   "scope": "10.0.0.0/24"},   "A — cold miss (first run)",             "miss"),
            (base, {"client": "Acme Corp",   "scope": "10.0.0.0/24"},   "B — exact repeat, same inputs",         "system1"),
            (base, {"client": "Beta LLC",    "scope": "10.0.0.0/24"},   "C — same goal, different client",       "system1"),
            (base, {"client": "Gamma Inc",   "scope": "192.168.1.0/24"},"D — same goal, different client+scope", "system1"),
            ("monthly compliance security review",
                   {"client": "Acme Corp",   "scope": "10.0.0.0/24"},   "E — related goal, rephrased",           "?"),
            ("penetration test full engagement",
                   {"client": "Delta Corp",  "scope": "external"},       "F — same domain, different task",       "?"),
        ]
    if domain == "support":
        base = "customer support ticket triage"
        return [
            (base, {"ticket_type": "billing",   "priority": "high"},    "A — cold miss",                         "miss"),
            (base, {"ticket_type": "billing",   "priority": "high"},    "B — exact repeat",                      "system1"),
            (base, {"ticket_type": "technical", "priority": "medium"},  "C — same workflow, different category",  "system1"),
            (base, {"ticket_type": "account",   "priority": "low"},     "D — same workflow, low priority",        "system1"),
            ("handle inbound customer complaint",
                   {"ticket_type": "billing",   "priority": "high"},    "E — rephrased goal",                    "?"),
            ("escalate urgent customer issue to tier 2",
                   {"ticket_type": "technical", "priority": "critical"}, "F — different sub-task",               "?"),
        ]
    if domain == "etl":
        base = "run ETL data pipeline"
        return [
            (base, {"source": "postgres",    "destination": "snowflake"}, "A — cold miss",                       "miss"),
            (base, {"source": "postgres",    "destination": "snowflake"}, "B — exact repeat",                    "system1"),
            (base, {"source": "mysql",       "destination": "snowflake"}, "C — different source",                "system1"),
            (base, {"source": "postgres",    "destination": "bigquery"},  "D — different destination",           "system1"),
            ("ingest and load data from source to warehouse",
                   {"source": "postgres",    "destination": "snowflake"}, "E — rephrased goal",                  "?"),
            ("real-time streaming pipeline with kafka",
                   {"source": "kafka",       "destination": "snowflake"}, "F — different architecture",          "?"),
        ]
    if domain == "codereview":
        base = "automated code review for pull request"
        return [
            (base, {"pr_id": "PR-042", "language": "python"},            "A — cold miss",                        "miss"),
            (base, {"pr_id": "PR-042", "language": "python"},            "B — exact repeat",                     "system1"),
            (base, {"pr_id": "PR-043", "language": "python"},            "C — different PR",                     "system1"),
            (base, {"pr_id": "PR-044", "language": "typescript"},        "D — different language",               "system1"),
            ("review and assess incoming pull request quality",
                   {"pr_id": "PR-045", "language": "python"},            "E — rephrased goal",                   "?"),
            ("security-focused code audit for compliance",
                   {"pr_id": "PR-046", "language": "python"},            "F — security tilt, same structure",    "?"),
        ]
    if domain == "finance":
        base = "generate quarterly financial report"
        return [
            (base, {"period": "Q1-2026", "entity": "Acme Corp"},         "A — cold miss",                        "miss"),
            (base, {"period": "Q1-2026", "entity": "Acme Corp"},         "B — exact repeat",                     "system1"),
            (base, {"period": "Q2-2026", "entity": "Acme Corp"},         "C — next quarter",                     "system1"),
            (base, {"period": "Q1-2026", "entity": "Beta LLC"},          "D — different entity",                 "system1"),
            ("produce executive financial summary for board",
                   {"period": "Q1-2026", "entity": "Acme Corp"},         "E — rephrased goal",                   "?"),
            ("annual audit and financial close",
                   {"period": "FY-2025", "entity": "Acme Corp"},         "F — annual vs quarterly",              "?"),
        ]
    if domain == "content":
        base = "create content marketing campaign"
        return [
            (base, {"topic": "AI in enterprise",  "channel": "linkedin"}, "A — cold miss",                      "miss"),
            (base, {"topic": "AI in enterprise",  "channel": "linkedin"}, "B — exact repeat",                   "system1"),
            (base, {"topic": "data privacy",      "channel": "linkedin"}, "C — different topic",                "system1"),
            (base, {"topic": "AI in enterprise",  "channel": "twitter"},  "D — different channel",              "system1"),
            ("develop thought leadership content for social media",
                   {"topic": "AI in enterprise",  "channel": "linkedin"}, "E — rephrased goal",                 "?"),
            ("run paid ad campaign with A/B testing",
                   {"topic": "AI in enterprise",  "channel": "google"},   "F — paid vs organic, different tools","?"),
        ]
    if domain == "devops":
        base = "deploy microservice to production"
        return [
            (base, {"service": "api-gateway",   "environment": "production"}, "A — cold miss",                  "miss"),
            (base, {"service": "api-gateway",   "environment": "production"}, "B — exact repeat",               "system1"),
            (base, {"service": "auth-service",  "environment": "production"}, "C — different service",          "system1"),
            (base, {"service": "api-gateway",   "environment": "staging"},    "D — staging deploy",             "system1"),
            ("release new version of service to prod",
                   {"service": "api-gateway",   "environment": "production"}, "E — rephrased goal",             "?"),
            ("hotfix rollback deployment on production incident",
                   {"service": "api-gateway",   "environment": "production"}, "F — rollback vs deploy",         "?"),
        ]
    if domain == "research":
        base = "research synthesis report"
        return [
            (base, {"topic": "LLM evaluation methods", "depth": "comprehensive"}, "A — cold miss",              "miss"),
            (base, {"topic": "LLM evaluation methods", "depth": "comprehensive"}, "B — exact repeat",           "system1"),
            (base, {"topic": "RAG architectures",      "depth": "comprehensive"}, "C — different topic",        "system1"),
            (base, {"topic": "LLM evaluation methods", "depth": "brief"},         "D — different depth",        "system1"),
            ("literature review and synthesis on topic",
                   {"topic": "LLM evaluation methods", "depth": "comprehensive"}, "E — rephrased goal",         "?"),
            ("competitive intelligence gathering report",
                   {"topic": "AI agent frameworks",    "depth": "comprehensive"}, "F — different intent",       "?"),
        ]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# RESULT TYPE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    domain:          str
    label:           str
    goal:            str
    cache_level:     str
    expected:        str
    tokens_saved:    int
    latency_saved_ms: float
    latency_ms:      float
    segments_reused: int
    hit:             bool = field(init=False)

    def __post_init__(self):
        self.hit = self.cache_level in ("system1", "system2")

    @property
    def expected_met(self) -> bool:
        if self.expected == "?":
            return True  # boundary run — no expectation
        return self.cache_level == self.expected

    def row(self, verbose: bool = False) -> str:
        exp_marker = "" if self.expected == "?" else ("✓" if self.expected_met else "✗")
        level_str  = {
            "system1": "S1 HIT ",
            "system2": "S2 HIT ",
            "miss":    "miss   ",
            "error":   "ERROR  ",
        }.get(self.cache_level, self.cache_level[:7])

        parts = [
            f"  {level_str} {exp_marker:<2}",
            f"{self.latency_ms:6.0f}ms",
            f"saved {self.tokens_saved:4d}tok",
            f"reused {self.segments_reused:2d}seg",
            f"  {self.label}",
        ]
        if verbose:
            parts.append(f"\n         goal: {self.goal[:80]}")
        return "  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK RUNNER
# ─────────────────────────────────────────────────────────────────────────────

async def run_domain(
    domain: str,
    gen_fn: Callable,
    verbose: bool,
    tmpdir: str,
) -> List[RunResult]:
    results = []
    variations = _variations(domain)

    async with Mnemon(
        tenant_id=f"bench_{domain}",
        db_dir=tmpdir,
        enable_telemetry=False,
        prewarm_fragments=True,
        silent=True,
    ) as m:
        for goal, inputs, label, expected in variations:
            t0 = time.time()
            try:
                r = await m.run(
                    goal=goal,
                    inputs=inputs,
                    generation_fn=gen_fn,
                )
                latency_ms = (time.time() - t0) * 1000
                results.append(RunResult(
                    domain=domain,
                    label=label,
                    goal=goal,
                    cache_level=r["cache_level"],
                    expected=expected,
                    tokens_saved=r.get("tokens_saved", 0),
                    latency_saved_ms=r.get("latency_saved_ms", 0),
                    latency_ms=latency_ms,
                    segments_reused=r.get("segments_reused", 0),
                ))
            except Exception as e:
                latency_ms = (time.time() - t0) * 1000
                results.append(RunResult(
                    domain=domain,
                    label=label,
                    goal=goal,
                    cache_level="error",
                    expected=expected,
                    tokens_saved=0,
                    latency_saved_ms=0,
                    latency_ms=latency_ms,
                    segments_reused=0,
                ))
                if verbose:
                    print(f"    ERROR: {e}")

    return results


async def run_benchmark(domains: List[str], verbose: bool) -> Dict[str, List[RunResult]]:
    all_results: Dict[str, List[RunResult]] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for domain in domains:
            gen_fn, description = WORKFLOWS[domain]
            print(f"\n{'─' * 60}")
            print(f"  {description.upper()}")
            print(f"{'─' * 60}")

            results = await run_domain(domain, gen_fn, verbose, tmpdir)
            all_results[domain] = results

            for r in results:
                print(r.row(verbose=verbose))

    return all_results


def print_summary(all_results: Dict[str, List[RunResult]]):
    print(f"\n{'═' * 60}")
    print("  SUMMARY")
    print(f"{'═' * 60}")

    total_runs      = 0
    total_s1        = 0
    total_s2        = 0
    total_miss      = 0
    total_tokens    = 0
    total_expected  = 0
    total_correct   = 0
    boundary_s1     = 0
    boundary_s2     = 0
    boundary_miss   = 0

    for domain, results in all_results.items():
        s1    = sum(1 for r in results if r.cache_level == "system1")
        s2    = sum(1 for r in results if r.cache_level == "system2")
        miss  = sum(1 for r in results if r.cache_level == "miss")
        tok   = sum(r.tokens_saved for r in results)
        bound = [r for r in results if r.expected == "?"]
        exp   = [r for r in results if r.expected != "?"]
        corr  = sum(1 for r in exp if r.expected_met)

        print(f"\n  {domain:<12}  S1={s1}  S2={s2}  miss={miss}  "
              f"tokens_saved={tok:,}  "
              f"expected_met={corr}/{len(exp)}")

        for r in bound:
            label = {"system1": "→ S1 hit", "system2": "→ S2 hit", "miss": "→ miss"}.get(r.cache_level, r.cache_level)
            print(f"             boundary [{r.label[:40]}]: {label}")
            if r.cache_level == "system1": boundary_s1 += 1
            elif r.cache_level == "system2": boundary_s2 += 1
            else: boundary_miss += 1

        total_runs     += len(results)
        total_s1       += s1
        total_s2       += s2
        total_miss     += miss
        total_tokens   += tok
        total_expected += len(exp)
        total_correct  += corr

    print(f"\n{'─' * 60}")
    print(f"  Total runs       : {total_runs}")
    print(f"  System 1 hits    : {total_s1}  ({total_s1/total_runs*100:.0f}%)")
    print(f"  System 2 hits    : {total_s2}  ({total_s2/total_runs*100:.0f}%)")
    print(f"  Misses           : {total_miss}  ({total_miss/total_runs*100:.0f}%)")
    print(f"  Tokens saved     : {total_tokens:,}")
    print(f"  Expected met     : {total_correct}/{total_expected}")
    print(f"\n  Boundary runs (E+F) — where EME had to decide:")
    print(f"    System 1 hits  : {boundary_s1}")
    print(f"    System 2 hits  : {boundary_s2}")
    print(f"    Misses         : {boundary_miss}")
    total_boundary = boundary_s1 + boundary_s2 + boundary_miss
    if total_boundary:
        pct = (boundary_s1 + boundary_s2) / total_boundary * 100
        print(f"    Cache rate on boundaries: {pct:.0f}%")
    print(f"{'═' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="EME workflow benchmark")
    parser.add_argument("--domain", choices=list(WORKFLOWS.keys()),
                        help="Run only one domain (default: all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print goal text for each run")
    parser.add_argument("--save", metavar="PATH",
                        help="Save results JSON to path")
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    domains = [args.domain] if args.domain else list(WORKFLOWS.keys())

    print(f"\nEME Workflow Benchmark — {len(domains)} domain(s)")
    print(f"Testing System 1 / System 2 / miss across real workflow patterns")

    all_results = asyncio.run(run_benchmark(domains, args.verbose))
    print_summary(all_results)

    if args.save:
        out = {
            domain: [
                {
                    "label":           r.label,
                    "cache_level":     r.cache_level,
                    "expected":        r.expected,
                    "expected_met":    r.expected_met,
                    "tokens_saved":    r.tokens_saved,
                    "latency_ms":      r.latency_ms,
                    "segments_reused": r.segments_reused,
                }
                for r in results
            ]
            for domain, results in all_results.items()
        }
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.save}")


if __name__ == "__main__":
    main()
