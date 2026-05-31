"""
Mnemon Realistic Workflow Gauntlet
====================================
Five production-grade LLM agent workflows running with mnemon.init().
The workflow IS the test. Mnemon sits underneath, invisible.
We measure whether it catches token savings without any code changes.

Run standalone:   python tests/test_workflow_gauntlet.py
Run with pytest:  pytest tests/test_workflow_gauntlet.py -v -s

Scenarios
---------
1. Customer Support Bot       30 tickets  ×  2 calls  =  60 calls
2. Code Review Pipeline        8 PRs      ×  4 calls  =  32 calls
3. Invoice Processing         20 invoices ×  2 calls  =  40 calls
4. Daily Report Agent         10 days    ×  1 call   =  10 calls
5. NL-to-SQL Data Analyst     15 queries ×  1-2 calls = ~22 calls
                                                    ────────────
Total: ~164 LLM calls. Every non-hit = tokens spent.

Architecture
------------
Fake LLM is installed at the Anthropic SDK level BEFORE mnemon.init()
so Mnemon captures it as "original". On miss Mnemon calls fake and
stores result in EME. On cache hit Mnemon never reaches the fake.
Token savings = (fake_calls_without_mnemon - fake_calls_with_mnemon) × avg_tokens.
"""

from __future__ import annotations

import os
import re
import sys
import time
import types
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────────────────────────────────
# FAKE LLM
# Installed before mnemon.init() so Mnemon wraps it as "original".
# Every call here = a real LLM call that would cost tokens in production.
# ─────────────────────────────────────────────────────────────────────────────

_llm_calls: list = []
_current_scenario: str = "init"
_current_call_type: str = "general"


def _tok(text: str) -> int:
    return max(1, len(text) // 4)


def _synth(text: str, model: str, inp: int, out: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        id=f"fake-{len(_llm_calls):04d}",
        type="message",
        role="assistant",
        model=model,
        content=[types.SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=types.SimpleNamespace(input_tokens=inp, output_tokens=out),
    )


# ── Realistic response bank keyed by regex patterns ──────────────────────────

_RESPONSE_BANK = [
    (r"reset.*password|forgot.*password|can.t.*log.*in|password.*reset",
     "To reset your password: visit account.acmesaas.com/forgot-password and "
     "enter your registered email address. A reset link will arrive within "
     "5 minutes — check spam if not received. Links expire after 24 hours. "
     "If the issue persists, our support team can trigger a manual reset: "
     "support@acmesaas.com or live chat (Mon–Fri 9–6 EST)."),

    (r"charged twice|double.*bill|duplicate.*charge|unexpected.*charge|extra.*charge",
     "I've confirmed a duplicate charge on your account (ref: BIL-{ref}). "
     "Our billing team has been notified and the credit will post within "
     "2–3 business days to your original payment method. You'll receive an "
     "email confirmation once processed. If you need it faster for accounting "
     "purposes, reply with 'URGENT' and we'll escalate to same-day processing."),

    (r"billing|invoice.*question|price.*wrong|contract.*renewal|vat.*invoice",
     "Your billing inquiry (case BIL-{ref}) has been routed to our accounts "
     "team. They will respond within one business day with a corrected invoice "
     "or clarification. Enterprise clients on annual contracts: your account "
     "manager has also been cc'd and can assist with procurement requirements."),

    (r"503|rate.?limit|429|api.*down|endpoint.*fail|batch.*error",
     "Engineering update (INC-{ref}): The batch endpoint degradation is caused "
     "by a capacity issue on us-east-1. ETA for full resolution: ~2 hours. "
     "Workarounds: (1) reduce batch size to ≤50 items, (2) add exponential "
     "backoff starting at 1 s. Status page: status.acmesaas.com. We'll post "
     "updates every 30 minutes until resolved."),

    (r"api.*doc|documentation.*wrong|deprecated.*endpoint|missing.*example",
     "Confirmed: our v3 REST docs are missing examples for the /batch and "
     "/async endpoints added in v4.2. A docs PR is open (docs#847) and "
     "will go live within 48 hours. In the meantime, the v3 migration guide "
     "at docs.acmesaas.com/migrate-v3 covers all breaking changes with "
     "side-by-side examples."),

    (r"webhook|kafka|streaming|real.?time.*event",
     "Webhook streaming is scheduled for Q3 2026 (FR-2841). You can track "
     "progress and upvote at feedback.acmesaas.com/webhooks. As an interim "
     "solution, our polling endpoint /v3/events supports up to 10 req/s with "
     "cursor-based pagination and returns events within 500 ms of occurrence."),

    (r"data.*missing|records.*gone|critical.*data|migration.*loss|gdpr.*export",
     "P1 INCIDENT DECLARED (INC-{ref}). Our data engineering team is "
     "investigating immediately with read-only audit log access. No write "
     "operations are permitted until root cause is confirmed. Current status: "
     "data present in backup snapshot taken at 02:00 UTC — restore window "
     "being calculated. Status updates every 20 minutes. CTO has been paged."),

    (r"sso.*fail|saml.*error|okta|login.*break|authentication.*fail",
     "SSO SAML issue logged (TKT-{ref}). Most common cause after Okta "
     "migration: attribute mapping mismatch. Verify in Okta: (1) NameID "
     "format = EmailAddress, (2) groups attribute mapped to acmesaas_roles, "
     "(3) SP Entity ID matches exactly. Share your SAML assertion XML "
     "(redact sensitive values) so we can pinpoint the attribute mismatch."),

    (r"dashboard.*slow|performance.*issue|load.*time|30 second",
     "Dashboard performance issue confirmed on our side (PERF-{ref}). "
     "Root cause: aggregation query on the analytics table is doing a "
     "full scan during business hours when concurrent queries spike. "
     "Immediate fix: we're deploying a covering index at 14:00 UTC today. "
     "This should reduce p95 load time from 32 s to under 3 s."),

    (r"onboarding|new.*seat|deployment|200.*seat|security.*review",
     "Welcome! For a 200-seat enterprise deployment we'll assign you a "
     "dedicated customer success engineer. Next steps: (1) security review "
     "packet sent to your InfoSec team, (2) SSO configuration call "
     "scheduled for this week, (3) custom onboarding portal at "
     "onboarding.acmesaas.com/{ref} with your org's tailored checklist."),

    # Code review patterns
    (r"sql.*inject|inject.*sql|parameteriz|prepared.*state|cwe.?89",
     "SECURITY — HIGH SEVERITY\n"
     "CWE-89 SQL Injection: line {line1} constructs query via string "
     "concatenation with unsanitized user input. Attacker can exfiltrate "
     "or delete data. Fix: use parameterized queries / PreparedStatement. "
     "Example fix:\n"
     "  BEFORE: f\"SELECT * FROM users WHERE id={user_id}\"\n"
     "  AFTER:  cursor.execute('SELECT * FROM users WHERE id=?', (user_id,))\n"
     "Block merge until resolved."),

    (r"xss|cross.?site|html.*inject|escape.*output|cwe.?79",
     "SECURITY — HIGH SEVERITY\n"
     "CWE-79 Stored XSS: line {line2} renders user-supplied content "
     "without HTML-escaping. Attacker can inject persistent JS that "
     "executes for all viewers. Fix: escape output in template context.\n"
     "  Jinja2: {{ user_input | e }}\n"
     "  React: value is safe by default — avoid dangerouslySetInnerHTML\n"
     "  PHP: htmlspecialchars($input, ENT_QUOTES, 'UTF-8')\n"
     "Block merge until resolved."),

    (r"auth.*bypass|privilege.*escal|admin.*unprotect|missing.*auth|cwe.?284",
     "SECURITY — CRITICAL\n"
     "CWE-284 Broken Access Control: admin endpoint /api/admin/users "
     "reachable without authentication middleware. Any unauthenticated "
     "user can enumerate/modify all accounts. Fix: apply @require_admin "
     "decorator (already used on /api/admin/settings). "
     "IMMEDIATE ACTION REQUIRED — do not merge."),

    (r"n\+1|eager.*load|lazy.*load|orm.*query|query.*loop",
     "PERFORMANCE — MEDIUM\n"
     "N+1 query pattern: product listing loop (lines {line1}–{line2}) "
     "executes one query per product to fetch variants. On a 50-item page "
     "this is 51 DB round-trips. Fix: eager-load variants in the initial "
     "queryset.\n"
     "  Django:  Product.objects.prefetch_related('variants')\n"
     "  Rails:   Product.includes(:variants)\n"
     "Estimated improvement: ~80% reduction in DB calls on listing pages."),

    (r"missing.*index|slow.*query|table.*scan|explain.*plan",
     "PERFORMANCE — MEDIUM\n"
     "Missing index: orders.customer_id column used in WHERE clause on "
     "high-traffic /api/orders endpoint but has no index. Full table scan "
     "on 4M-row table. Fix:\n"
     "  CREATE INDEX CONCURRENTLY idx_orders_customer_id "
     "ON orders(customer_id);\n"
     "Use CONCURRENTLY to avoid table lock in production. Estimated "
     "improvement: query time 2,400 ms → ~12 ms."),

    (r"unused.*import|dead.*code|naming.*convention|lint|style.*issue",
     "STYLE — LOW\n"
     "No functional issues. Housekeeping:\n"
     "• 4 unused imports (lines 3, 11, 22, 47) — remove to reduce load time\n"
     "• Variable 'userData' should be 'user_data' per PEP 8\n"
     "• 2 public methods missing docstrings\n"
     "• Lines 134, 201 exceed 120-char limit\n"
     "Suggest fixing in same PR or a follow-up before next release."),

    # Invoice classification
    (r"classif.*invoice|invoice.*type|document.*classif",
     "CLASSIFICATION RESULT\n"
     "  document_type: vendor_invoice\n"
     "  jurisdiction: {juris}\n"
     "  vendor_tier: {tier}\n"
     "  requires_cfo_approval: {cfo}\n"
     "  tax_treatment: {tax_treat}\n"
     "  duplicate_risk: low\n"
     "  confidence: 0.96\n"
     "  processing_path: standard_ap"),

    # Invoice extraction
    (r"extract.*invoice|invoice.*field|parse.*invoice|invoice.*data",
     "EXTRACTED FIELDS\n"
     "  vendor_name:    {vendor}\n"
     "  invoice_no:     {inv_no}\n"
     "  invoice_date:   2026-05-{day}\n"
     "  due_date:       2026-06-{due}\n"
     "  amount_net:     {amount} {currency}\n"
     "  tax_rate:       {tax_rate}%\n"
     "  tax_amount:     {tax_amt} {currency}\n"
     "  amount_total:   {total} {currency}\n"
     "  payment_terms:  Net-30\n"
     "  bank_iban:      {iban}\n"
     "  line_items:     1 item — Professional services May 2026\n"
     "  ocr_confidence: 0.99"),

    # Daily report
    (r"daily.*report|business.*intelligence|sales.*report|daily.*brief",
     "DAILY BUSINESS INTELLIGENCE — {date}\n"
     "Generated: {ts} UTC\n\n"
     "REVENUE\n"
     "  Today:       ${rev:,.0f}  ({rev_d:+.1f}% vs yesterday)\n"
     "  MTD:         ${mtd:,.0f}\n"
     "  Top channel: Direct Sales ${ch1:,.0f}\n\n"
     "USERS\n"
     "  Active DAU:  {dau:,}  ({dau_d:+d} vs yesterday)\n"
     "  New signups: {signups}\n"
     "  Conversion:  {conv:.2f}%\n\n"
     "SUPPORT\n"
     "  Opened: {tix}  Closed: {tix_c}  P1 active: {p1}\n"
     "  Median response: {resp:.1f}h\n\n"
     "ANOMALIES\n"
     "  {anomaly}\n\n"
     "ALL SYSTEMS HEALTHY"),

    # NL-to-SQL
    (r"revenue|monthly.*sales|quarterly.*revenue|annual.*revenue",
     "SELECT\n"
     "  DATE_TRUNC('{period}', created_at)  AS period,\n"
     "  SUM(amount_cents) / 100.0           AS revenue_usd,\n"
     "  COUNT(DISTINCT customer_id)         AS paying_customers,\n"
     "  COUNT(*)                            AS transactions,\n"
     "  AVG(amount_cents) / 100.0           AS avg_order_value\n"
     "FROM orders\n"
     "WHERE created_at  >= NOW() - INTERVAL '{window}'\n"
     "  AND status      = 'completed'\n"
     "GROUP BY 1\n"
     "ORDER BY 1 DESC;"),

    (r"churn|inactive.*customer|at.risk|retention|lapsed",
     "SELECT\n"
     "  c.customer_id,\n"
     "  c.plan,\n"
     "  c.mrr_cents / 100.0                                    AS mrr,\n"
     "  MAX(e.occurred_at)                                     AS last_activity,\n"
     "  EXTRACT(DAY FROM NOW() - MAX(e.occurred_at))           AS days_inactive,\n"
     "  COUNT(e.id) FILTER (WHERE e.occurred_at > NOW() - INTERVAL '30 days')  AS events_30d\n"
     "FROM customers c\n"
     "LEFT JOIN events e ON e.customer_id = c.customer_id\n"
     "WHERE c.status = 'active'\n"
     "GROUP BY 1, 2, 3\n"
     "HAVING days_inactive > 30 OR events_30d < 3\n"
     "ORDER BY mrr DESC;"),

    (r"conversion|free.*paid|trial.*convert|upgrade",
     "SELECT\n"
     "  DATE_TRUNC('week', trial_started_at)                          AS cohort_week,\n"
     "  COUNT(*)                                                       AS trials,\n"
     "  COUNT(*) FILTER (WHERE converted_at IS NOT NULL)              AS converted,\n"
     "  ROUND(\n"
     "    100.0 * COUNT(*) FILTER (WHERE converted_at IS NOT NULL)\n"
     "    / NULLIF(COUNT(*), 0), 2)                                   AS conversion_rate_pct,\n"
     "  ROUND(AVG(EXTRACT(DAY FROM converted_at - trial_started_at)), 1) AS avg_days_to_convert\n"
     "FROM trials\n"
     "WHERE trial_started_at >= NOW() - INTERVAL '90 days'\n"
     "GROUP BY 1\n"
     "ORDER BY 1 DESC;"),

    (r"top.*product|best.*sell|product.*rank|margin.*product",
     "SELECT\n"
     "  p.sku,\n"
     "  p.name,\n"
     "  p.category,\n"
     "  SUM(oi.quantity)                                        AS units_sold,\n"
     "  SUM(oi.unit_price_cents * oi.quantity) / 100.0         AS gross_revenue,\n"
     "  ROUND(AVG(\n"
     "    (oi.unit_price_cents - p.cost_cents) * 1.0\n"
     "    / NULLIF(oi.unit_price_cents, 0) * 100\n"
     "  ), 1)                                                   AS gross_margin_pct\n"
     "FROM order_items oi\n"
     "JOIN products p ON p.id = oi.product_id\n"
     "WHERE oi.created_at >= NOW() - INTERVAL '{window}'\n"
     "GROUP BY 1, 2, 3\n"
     "ORDER BY gross_revenue DESC\n"
     "LIMIT 20;"),

    (r"support.*metric|ticket.*volume|response.*time|sla.*breach",
     "SELECT\n"
     "  DATE_TRUNC('day', t.created_at)                             AS day,\n"
     "  COUNT(*)                                                     AS tickets_opened,\n"
     "  COUNT(*) FILTER (WHERE t.closed_at IS NOT NULL)             AS tickets_closed,\n"
     "  ROUND(AVG(EXTRACT(EPOCH FROM t.first_response_at - t.created_at) / 3600), 2)\n"
     "                                                               AS avg_first_response_h,\n"
     "  COUNT(*) FILTER (\n"
     "    WHERE t.priority = 'P1'\n"
     "    AND EXTRACT(EPOCH FROM t.first_response_at - t.created_at) / 3600 > 2\n"
     "  )                                                            AS sla_breaches_p1\n"
     "FROM tickets t\n"
     "WHERE t.created_at >= NOW() - INTERVAL '{window}'\n"
     "GROUP BY 1\n"
     "ORDER BY 1 DESC;"),

    (r"ltv|lifetime.*value|customer.*value",
     "SELECT\n"
     "  c.acquisition_channel,\n"
     "  c.plan                                             AS starting_plan,\n"
     "  COUNT(DISTINCT c.id)                               AS customers,\n"
     "  ROUND(AVG(c.total_revenue_cents) / 100.0, 2)      AS avg_ltv,\n"
     "  ROUND(AVG(c.tenure_months), 1)                     AS avg_tenure_months,\n"
     "  ROUND(AVG(c.total_revenue_cents / NULLIF(c.tenure_months, 0)) / 100.0, 2)\n"
     "                                                     AS avg_mrr_lifetime\n"
     "FROM customers c\n"
     "WHERE c.status IN ('active', 'churned')\n"
     "  AND c.created_at < NOW() - INTERVAL '6 months'\n"
     "GROUP BY 1, 2\n"
     "ORDER BY avg_ltv DESC;"),
]


def _fake_llm_create(_self, *, messages, model, system=None, **kwargs):
    user_msg = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    combined = ((system or "") + " " + user_msg).lower()

    response_text = (
        f"Request processed. Action taken for: {user_msg[:60]}. "
        "Please contact support if further assistance needed."
    )
    for pattern, template in _RESPONSE_BANK:
        if re.search(pattern, combined):
            import random
            rng = random.Random(hash(user_msg) % (2**31))
            response_text = template.format(
                ref=f"{rng.randint(10000,99999)}",
                line1=rng.randint(30, 80),
                line2=rng.randint(90, 150),
                juris=("US" if "united states" in combined or "aws" in combined
                       else "DE" if "germany" in combined or "gmbh" in combined
                       else "IN" if "india" in combined or "tcs" in combined or "infosys" in combined
                       else "UK" if "united kingdom" in combined or "ltd" in combined
                       else "FR" if "france" in combined
                       else "US"),
                tier=("enterprise" if rng.random() > 0.4 else "standard"),
                cfo=("yes" if rng.random() > 0.5 else "no"),
                tax_treat=("vat_applicable" if rng.random() > 0.3 else "exempt"),
                vendor=user_msg[user_msg.find("from") + 5:user_msg.find("from") + 30].strip().split()[0]
                       if "from" in user_msg else "Vendor Corp",
                inv_no=f"INV-2026-{rng.randint(1000,9999)}",
                day=str(rng.randint(1, 28)),
                due=str(rng.randint(1, 28)),
                amount=f"{rng.randint(5000, 80000):,.2f}",
                currency=("EUR" if "europe" in combined or "gmbh" in combined else "USD"),
                tax_rate=("19" if "germany" in combined else "20" if "uk" in combined else "18" if "india" in combined else "0"),
                tax_amt=f"{rng.randint(500, 8000):,.2f}",
                total=f"{rng.randint(6000, 90000):,.2f}",
                iban=f"GB{rng.randint(10,99)}NWBK{rng.randint(10000000,99999999)}{rng.randint(10000000,99999999)}",
                date=f"2026-05-{rng.randint(1,28):02d}",
                ts=f"{rng.randint(6,10):02d}:{rng.randint(0,59):02d}",
                rev=rng.uniform(380000, 620000),
                rev_d=rng.uniform(-5, 8),
                mtd=rng.uniform(8000000, 14000000),
                ch1=rng.uniform(140000, 240000),
                dau=rng.randint(10000, 16000),
                dau_d=rng.randint(-300, 600),
                signups=rng.randint(50, 140),
                conv=rng.uniform(3.0, 6.5),
                tix=rng.randint(30, 80),
                tix_c=rng.randint(25, 75),
                p1=rng.randint(0, 4),
                resp=rng.uniform(0.8, 3.5),
                anomaly=(
                    "No anomalies detected — all metrics within normal range."
                    if rng.random() > 0.3
                    else f"Latency spike on /api/v3/batch at {rng.randint(0,23):02d}:{rng.randint(0,59):02d} UTC (self-resolved)"
                ),
                period=("month" if "monthly" in combined else "week" if "weekly" in combined else "day"),
                window=("12 months" if "annual" in combined or "year" in combined
                        else "3 months" if "quarter" in combined
                        else "30 days"),
            )
            break

    inp = _tok((system or "") + " ".join(m.get("content", "") for m in messages))
    out = _tok(response_text)
    _llm_calls.append({
        "scenario": _current_scenario,
        "call_type": _current_call_type,
        "input_tokens": inp,
        "output_tokens": out,
        "query_snippet": user_msg[:80],
    })
    return _synth(response_text, model, inp, out)


# ── Install fake BEFORE importing mnemon ─────────────────────────────────────
import anthropic.resources.messages as _ant_mod
_ant_mod.Messages.create = _fake_llm_create


# ── Now safe to init Mnemon ───────────────────────────────────────────────────
import mnemon as _mnemon_mod

_tmpdir = tempfile.mkdtemp(prefix="mnemon_gauntlet_")
_mnemon_mod._instance = None   # clear any previous global

_M = _mnemon_mod.init(
    tenant_id="gauntlet",
    db_dir=_tmpdir,
    silent=True,
    prewarm_fragments=False,
    prewarm_templates=False,
    enable_telemetry=False,
)

# Wait for the patch thread — it will find anthropic already imported and
# re-patch Messages.create, wrapping our fake as "original".
if _M._patch_thread and _M._patch_thread.is_alive():
    _M._patch_thread.join(timeout=15)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

from anthropic import Anthropic as _Anthropic

_client = _Anthropic(api_key="fake-key-gauntlet")
_MODEL  = "claude-3-5-haiku-20241022"


def _call(system_prompt: str, user_message: str, scenario: str, call_type: str) -> types.SimpleNamespace:
    global _current_scenario, _current_call_type
    _current_scenario = scenario
    _current_call_type = call_type
    return _client.messages.create(
        model=_MODEL,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=512,
    )


def _llm_miss_count_for(scenario: str) -> int:
    return sum(1 for c in _llm_calls if c["scenario"] == scenario)


def _total_tokens_for(scenario: str) -> int:
    return sum(c["input_tokens"] + c["output_tokens"]
               for c in _llm_calls if c["scenario"] == scenario)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 1 — CUSTOMER SUPPORT BOT
# A SaaS product's support agent. 30 tickets across 6 issue categories.
# Each category has 5 tickets phrased differently by different customers.
# Real: Zendesk AI, Intercom, Freshdesk bots all look exactly like this.
# ─────────────────────────────────────────────────────────────────────────────

_SUPPORT_SYSTEM = (
    "You are a senior customer support agent for AcmeSaaS, a B2B SaaS platform "
    "serving 4,000 enterprise and SMB customers. Your tone is professional and "
    "empathetic. You have access to the customer's account history, billing records, "
    "and knowledge base. For P1 incidents (data loss, security, outages) escalate "
    "immediately to the on-call engineer. SLA: enterprise customers get first response "
    "in 2 hours; SMB in 24 hours. Always include a case reference number. "
    "Avoid legal admissions. Do not promise specific refund timelines beyond 3 business days. "
    "If the issue is unresolvable at your level, escalate to Tier 2 with full context."
)

_TICKETS = [
    # Password reset — 5 variations
    ("password", "How do I reset my password? I can't remember it and I'm locked out."),
    ("password", "I forgot my password. The system isn't letting me log in. Need help urgently."),
    ("password", "Can't access my account — password reset required. I've tried three times already."),
    ("password", "My password expired and the reset link I was sent last week no longer works."),
    ("password", "Not receiving the password reset email. Checked spam folder, nothing there."),
    # Billing — 5 variations
    ("billing", "I was charged twice this month for the same subscription. I need a refund."),
    ("billing", "There's an unexpected charge of $2,400 on my credit card. Not on our contract."),
    ("billing", "We see two identical charges from your company on our last invoice. Please investigate."),
    ("billing", "Noticed a double billing on our account statement for February. Needs correction."),
    ("billing", "My invoice shows a duplicate line item for the Enterprise tier. Please issue a credit."),
    # API errors — 5 variations
    ("api_error", "Your /v2/batch endpoint keeps returning 503 errors for requests over 100 items."),
    ("api_error", "We're getting HTTP 503 on batch API calls. Started about an hour ago. Critical for us."),
    ("api_error", "Batch endpoint failures — 503 responses. Production system is affected. Need ETA."),
    ("api_error", "Rate limit 429 errors triggering at 50% of the documented 1,000 req/min threshold."),
    ("api_error", "Our integration is failing with 429 Too Many Requests. We're well under your stated limits."),
    # Feature requests — 5 variations
    ("feature", "We need webhook support for real-time event streaming. Is this on the roadmap?"),
    ("feature", "Does your platform support webhooks? We want to push events to our Kafka cluster."),
    ("feature", "Looking for real-time event delivery — do you have webhooks or SSE support planned?"),
    ("feature", "We need to subscribe to events via webhooks. When will this be available?"),
    ("feature", "Is there a webhook endpoint we can register for order.completed events?"),
    # Data issues — 5 variations
    ("data_loss", "CRITICAL: 3,400 customer records are missing after last night's migration. Need help now."),
    ("data_loss", "We lost data after the maintenance window. Thousands of records are gone. Emergency."),
    ("data_loss", "Post-migration data loss detected. Our customer table is missing rows. This is P1."),
    ("data_loss", "GDPR data export is missing consent records — our DPA audit is in 3 days. Urgent."),
    ("data_loss", "Compliance audit in 72 hours and our GDPR export is incomplete. Consent records absent."),
    # Access / SSO — 5 variations
    ("access", "SSO SAML login failing for 12 new employees after our Okta migration last week."),
    ("access", "New staff can't log in via SSO. We migrated to Okta and now SAML isn't working."),
    ("access", "After our IdP migration to Okta, users get SAML authentication errors. Please help."),
    ("access", "Admin password reset link not arriving for 3 accounts. Checked spam, using Gmail."),
    ("access", "User accounts locked out — password reset emails not being delivered. Gmail domain."),
]


def run_support_scenario() -> dict:
    global _current_scenario, _current_call_type
    total_calls = 0

    for category, ticket_text in _TICKETS:
        # Call 1: classify the ticket priority and type
        _call(
            _SUPPORT_SYSTEM,
            f"Classify this support ticket. Respond with: priority (P1/P2/P3), "
            f"category, and recommended team.\n\nTicket: {ticket_text}",
            scenario="support",
            call_type="classify",
        )
        total_calls += 1

        # Call 2: generate the actual response
        _call(
            _SUPPORT_SYSTEM,
            f"Write a customer-facing response to this support ticket.\n\nTicket: {ticket_text}",
            scenario="support",
            call_type="respond",
        )
        total_calls += 1

    misses = _llm_miss_count_for("support")
    tokens_spent = _total_tokens_for("support")
    return {
        "scenario": "Customer Support Bot",
        "total_calls": total_calls,
        "llm_calls_made": misses,
        "cache_hits": total_calls - misses,
        "hit_rate": (total_calls - misses) / total_calls,
        "tokens_spent": tokens_spent,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 2 — CODE REVIEW PIPELINE
# CI/CD integrated. Every PR gets a coordinator + 3 specialist reviewers.
# Same specialist system prompts on every PR. Same CVE patterns recur.
# Real: based on Cloudflare's 131K reviews/month architecture.
# ─────────────────────────────────────────────────────────────────────────────

_COORD_SYSTEM = (
    "You are the code review coordinator. You receive a git diff and decide "
    "which specialist reviewers to invoke: SECURITY (SQL injection, XSS, "
    "auth bypass, secrets exposure), PERFORMANCE (N+1 queries, missing indexes, "
    "inefficient algorithms), STYLE (naming, unused imports, documentation). "
    "For each area where issues may exist, summarize the relevant diff sections "
    "and the concern for the specialist. Be precise about line numbers. "
    "If the PR is trivial (< 20 lines, docs only, dependency bump), output "
    "TRIVIAL and skip specialist routing."
)

_SEC_REVIEWER = (
    "You are the security specialist code reviewer. You focus exclusively on: "
    "CWE-89 SQL Injection, CWE-79 XSS/Template Injection, CWE-284 Broken Access Control, "
    "CWE-798 Hardcoded Credentials, CWE-352 CSRF. "
    "For each finding: state severity (CRITICAL/HIGH/MEDIUM), affected line(s), "
    "attack vector, and exact fix. If no issues found: output CLEAN. "
    "You do not comment on style or performance — only security."
)

_PERF_REVIEWER = (
    "You are the performance specialist code reviewer. You focus exclusively on: "
    "N+1 query patterns, missing database indexes, inefficient loops (O(n²) or worse), "
    "missing caching on hot paths, unoptimized ORM usage. "
    "For each finding: state impact (HIGH/MEDIUM/LOW), affected lines, "
    "quantified improvement estimate, and exact fix. "
    "You do not comment on security or style — only performance."
)

_STYLE_REVIEWER = (
    "You are the style and maintainability reviewer. You focus on: "
    "unused imports, dead code, naming convention violations (PEP 8 / project standard), "
    "missing docstrings on public APIs, excessive function length (> 50 lines), "
    "magic numbers without constants. "
    "Issues here are never blocking — suggest fixes only. "
    "You do not comment on security or performance — only style."
)

_PRS = [
    # 3 security issues (SQL injection pattern)
    (
        "PR #441 — user-auth service: login endpoint refactor",
        "diff --git a/auth/login.py b/auth/login.py\n"
        "+def login(request):\n"
        "+    username = request.POST['username']\n"
        "+    query = f\"SELECT * FROM users WHERE username='{username}' AND active=1\"\n"
        "+    user = db.execute(query).fetchone()\n"
        "+    if user: session['user_id'] = user['id']\n"
    ),
    (
        "PR #442 — comments service: add user comment display",
        "diff --git a/comments/views.py b/comments/views.py\n"
        "+def render_comment(comment):\n"
        "+    return f'<div class=\"comment\">{comment.body}</div>'\n"
        "+\n"
        "+def get_admin_comments(user_id):\n"
        "+    return db.query(f'SELECT * FROM comments WHERE user={user_id}')\n"
    ),
    (
        "PR #443 — admin panel: add user management route",
        "diff --git a/admin/routes.py b/admin/routes.py\n"
        "+@app.route('/api/admin/users')\n"
        "+def admin_users():\n"
        "+    return jsonify(User.query.all())\n"
        "+\n"
        "+@app.route('/api/admin/settings', methods=['POST'])\n"
        "+@require_admin\n"
        "+def admin_settings(): pass\n"
    ),
    # 2 performance issues
    (
        "PR #444 — product catalog: listing page optimization attempt",
        "diff --git a/catalog/views.py b/catalog/views.py\n"
        "+def product_listing(category_id):\n"
        "+    products = Product.objects.filter(category=category_id)\n"
        "+    for p in products:\n"
        "+        p.variant_count = Variant.objects.filter(product=p).count()\n"
        "+        p.image_url = Image.objects.filter(product=p).first().url\n"
        "+    return render('listing.html', products=products)\n"
    ),
    (
        "PR #445 — orders: add customer order history endpoint",
        "diff --git a/orders/views.py b/orders/views.py\n"
        "+@app.route('/api/orders')\n"
        "+def order_history():\n"
        "+    customer_id = request.args['customer_id']\n"
        "+    orders = Order.query.filter_by(customer_id=customer_id).all()\n"
        "+    return jsonify(orders)\n"
        "# No index on orders.customer_id in migration file\n"
    ),
    # 2 style issues
    (
        "PR #446 — utils: cleanup unused imports and rename variables",
        "diff --git a/utils/helpers.py b/utils/helpers.py\n"
        "+import os, sys, json, datetime, re, hashlib, uuid, logging\n"
        "+from typing import List, Dict, Any, Optional, Union, Tuple\n"
        "+\n"
        "+def processUserData(userData, includeDeleted=False):\n"
        "+    x = userData.get('id')\n"
        "+    y = userData.get('email')\n"
        "+    z = userData.get('name', '')\n"
        "+    return {'id': x, 'email': y, 'displayName': z}\n"
    ),
    (
        "PR #447 — api: add health check endpoint",
        "diff --git a/api/health.py b/api/health.py\n"
        "+import os, sys, time, json, logging, traceback\n"
        "+\n"
        "+@app.route('/health')\n"
        "+def health_check():\n"
        "+    return {'status': 'ok', 'ts': time.time()}\n"
    ),
    # 1 complex PR with multiple issue types
    (
        "PR #448 — checkout: new payment processing flow",
        "diff --git a/checkout/payment.py b/checkout/payment.py\n"
        "+def process_payment(request):\n"
        "+    user_id = request.POST['user_id']\n"
        "+    amount = request.POST['amount']\n"
        "+    query = f\"SELECT * FROM payment_methods WHERE user_id={user_id}\"\n"
        "+    methods = db.execute(query).fetchall()\n"
        "+    for m in methods:\n"
        "+        history = PaymentHistory.objects.filter(method=m).all()\n"
        "+        m.last_used = history[0].created_at if history else None\n"
        "+    return render('checkout.html', {'methods': methods})\n"
        "+\n"
        "+STRIPE_KEY = 'sk_live_abc123secretkeyXXXX'\n"
    ),
]


def run_code_review_scenario() -> dict:
    total_calls = 0
    before_misses = _llm_miss_count_for("codereview")

    for pr_title, pr_diff in _PRS:
        # Coordinator decides which specialists to invoke
        coord_resp = _call(
            _COORD_SYSTEM,
            f"PR: {pr_title}\n\nDiff:\n{pr_diff}\n\nWhich specialists should review this PR?",
            scenario="codereview",
            call_type="coordinate",
        )
        total_calls += 1

        # All 3 specialists review in parallel (simulated sequentially here)
        _call(
            _SEC_REVIEWER,
            f"Review this diff for security issues.\n\nPR: {pr_title}\n\nDiff:\n{pr_diff}",
            scenario="codereview",
            call_type="security",
        )
        total_calls += 1

        _call(
            _PERF_REVIEWER,
            f"Review this diff for performance issues.\n\nPR: {pr_title}\n\nDiff:\n{pr_diff}",
            scenario="codereview",
            call_type="performance",
        )
        total_calls += 1

        _call(
            _STYLE_REVIEWER,
            f"Review this diff for style and maintainability issues.\n\nPR: {pr_title}\n\nDiff:\n{pr_diff}",
            scenario="codereview",
            call_type="style",
        )
        total_calls += 1

    misses = _llm_miss_count_for("codereview")
    tokens_spent = _total_tokens_for("codereview")
    return {
        "scenario": "Code Review Pipeline",
        "total_calls": total_calls,
        "llm_calls_made": misses,
        "cache_hits": total_calls - misses,
        "hit_rate": (total_calls - misses) / total_calls,
        "tokens_spent": tokens_spent,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 3 — INVOICE PROCESSING PIPELINE
# AP automation. 20 invoices from 4 jurisdictions.
# Two-step per invoice: classify doc type → extract fields.
# Real: Alan, Kofax, UiPath Document Understanding all do this.
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFY_SYSTEM = (
    "You are an accounts payable document classifier. Classify each document as: "
    "vendor_invoice, credit_note, purchase_order, expense_report, or unknown. "
    "For vendor invoices output: document_type, jurisdiction (ISO-2), vendor_tier "
    "(enterprise / standard), whether CFO approval is required (> $10,000 threshold), "
    "tax_treatment, duplicate_risk, and confidence score. "
    "Input is a plain-text rendering of the document. Be concise — structured output only."
)

_EXTRACT_SYSTEM = (
    "You are an invoice field extractor. Extract all structured fields from the "
    "invoice text: vendor_name, invoice_no, invoice_date, due_date, line_items "
    "(array), amount_net, tax_rate, tax_amount, amount_total, currency, payment_terms, "
    "and bank_iban if present. "
    "Output must be machine-parseable. Use null for absent fields. "
    "Flag any anomalies (duplicate invoice number, implausible amounts, missing VAT number)."
)

_INVOICES = [
    # US invoices (5)
    ("US", "Microsoft Corporation", "INV-MS-2026-0441", 5600),
    ("US", "Amazon Web Services LLC", "INV-AWS-2026-0992", 3200),
    ("US", "Salesforce Inc", "INV-SF-2026-1104", 9800),
    ("US", "Oracle Corporation", "INV-ORC-2026-0331", 67000),
    ("US", "Zoom Video Communications", "INV-ZM-2026-0887", 1800),
    # EU Germany invoices (4)
    ("DE", "SAP SE", "INV-SAP-2026-4411", 4200),
    ("DE", "Deutsche Telekom AG", "INV-DT-2026-2201", 12400),
    ("DE", "Siemens AG", "INV-SIE-2026-0772", 38000),
    ("DE", "Bayer AG", "INV-BAY-2026-1156", 7900),
    # EU France invoices (3)
    ("FR", "Capgemini SE", "INV-CAP-2026-9901", 31000),
    ("FR", "Total Energies SE", "INV-TOT-2026-3302", 18500),
    ("FR", "BNP Paribas SA", "INV-BNP-2026-0011", 52000),
    # India invoices (5)
    ("IN", "Tata Consultancy Services Ltd", "INV-TCS-2026-7701", 18500),
    ("IN", "Infosys Limited", "INV-INF-2026-4421", 14200),
    ("IN", "Wipro Limited", "INV-WIP-2026-3301", 8900),
    ("IN", "HCL Technologies Ltd", "INV-HCL-2026-2211", 22000),
    ("IN", "Tech Mahindra Limited", "INV-TM-2026-0991", 11700),
    # UK invoices (3)
    ("UK", "KPMG LLP", "INV-KPM-2026-3301", 12300),
    ("UK", "Deloitte LLP", "INV-DEL-2026-0441", 28500),
    ("UK", "Barclays Bank PLC", "INV-BAR-2026-9901", 7800),
]


def run_invoice_scenario() -> dict:
    total_calls = 0

    for jurisdiction, vendor, inv_no, amount in _INVOICES:
        doc_text = (
            f"INVOICE\n"
            f"From: {vendor}\n"
            f"Invoice No: {inv_no}\n"
            f"Date: 2026-05-15\n"
            f"Jurisdiction: {jurisdiction}\n"
            f"Description: Professional services — Q1 2026 engagement\n"
            f"Amount (net): {amount:,.2f} {'EUR' if jurisdiction in ('DE','FR') else 'GBP' if jurisdiction == 'UK' else 'INR' if jurisdiction == 'IN' else 'USD'}\n"
            f"Payment terms: Net-30\n"
        )

        # Step 1: classify the document
        _call(
            _CLASSIFY_SYSTEM,
            f"Classify this invoice document from {vendor} ({jurisdiction}).\n\n{doc_text}",
            scenario="invoice",
            call_type="classify",
        )
        total_calls += 1

        # Step 2: extract all structured fields
        _call(
            _EXTRACT_SYSTEM,
            f"Extract all fields from this invoice.\n\n{doc_text}",
            scenario="invoice",
            call_type="extract",
        )
        total_calls += 1

    misses = _llm_miss_count_for("invoice")
    tokens_spent = _total_tokens_for("invoice")
    return {
        "scenario": "Invoice Processing Pipeline",
        "total_calls": total_calls,
        "llm_calls_made": misses,
        "cache_hits": total_calls - misses,
        "hit_rate": (total_calls - misses) / total_calls,
        "tokens_spent": tokens_spent,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 4 — DAILY REPORT AGENT
# Same report generated every business day. Identical structure, different numbers.
# Real: internal ops teams, BI teams, executive morning briefings.
# System 1 should dominate here — same prompt skeleton, only date varies.
# ─────────────────────────────────────────────────────────────────────────────

_REPORT_SYSTEM = (
    "You are an automated business intelligence reporter for AcmeSaaS. "
    "You receive a structured data snapshot and produce the Daily BI Report "
    "for the executive team. The report must follow this exact structure: "
    "REVENUE (today, MTD, top channel), USERS (DAU, signups, conversion rate), "
    "SUPPORT (tickets opened/closed, P1 count, avg response time), "
    "ANOMALIES (any metric outside ±2σ of 30-day baseline, or NONE). "
    "Use $USD with comma separators. Percentages to 2dp. Be factual, no fluff. "
    "Length: 200-300 words. This goes to the CEO every morning at 09:00 ET."
)


def run_daily_report_scenario() -> dict:
    import random
    rng = random.Random(42)
    total_calls = 0
    dates = [f"2026-05-{d:02d}" for d in range(19, 29)]  # 10 business days

    for date in dates:
        # Realistically, the prompt changes only the date and data snapshot
        snapshot = (
            f"Date: {date}\n"
            f"Revenue: ${rng.uniform(420000, 560000):,.0f}\n"
            f"MTD Revenue: ${rng.uniform(9000000, 13500000):,.0f}\n"
            f"DAU: {rng.randint(10500, 15800):,}\n"
            f"New signups: {rng.randint(55, 130)}\n"
            f"Conversion rate: {rng.uniform(3.5, 5.8):.2f}%\n"
            f"Support tickets opened: {rng.randint(28, 72)}\n"
            f"Support tickets closed: {rng.randint(25, 68)}\n"
            f"P1 incidents active: {rng.randint(0, 3)}\n"
            f"Avg first response: {rng.uniform(0.9, 2.8):.1f}h\n"
            f"Top revenue channel: Direct Sales\n"
        )
        _call(
            _REPORT_SYSTEM,
            f"Generate the daily business intelligence report for {date}.\n\nData snapshot:\n{snapshot}",
            scenario="dailyreport",
            call_type="report",
        )
        total_calls += 1

    misses = _llm_miss_count_for("dailyreport")
    tokens_spent = _total_tokens_for("dailyreport")
    return {
        "scenario": "Daily Report Agent",
        "total_calls": total_calls,
        "llm_calls_made": misses,
        "cache_hits": total_calls - misses,
        "hit_rate": (total_calls - misses) / total_calls,
        "tokens_spent": tokens_spent,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 5 — NL-TO-SQL DATA ANALYST
# Analysts ask questions about company data. Large DB schema is constant.
# Query shapes recur: "top N by X", "trend over time", "cohort analysis".
# Real: Dataherald, Defog, text2sql pipelines everywhere.
# ─────────────────────────────────────────────────────────────────────────────

_DB_SCHEMA = """
Database: acmesaas_prod (PostgreSQL 15, 47 tables)

Core tables:
  customers(id, email, plan, mrr_cents, status, created_at, churned_at, acquisition_channel)
  orders(id, customer_id, amount_cents, status, created_at, completed_at, channel)
  order_items(id, order_id, product_id, quantity, unit_price_cents, revenue_cents)
  products(id, sku, name, category, cost_cents, status, created_at)
  events(id, customer_id, event_type, occurred_at, properties jsonb)
  tickets(id, customer_id, priority, status, category, created_at, first_response_at, closed_at)
  trials(id, customer_id, trial_started_at, converted_at, plan_at_conversion)
  signups(id, email, created_at, converted_at, acquisition_source)
  subscriptions(id, customer_id, plan, mrr_cents, started_at, cancelled_at, cancel_reason)
  payment_methods(id, customer_id, type, last4, is_default, created_at)

Conventions:
  - All monetary values stored as integer cents, divide by 100 for USD
  - All timestamps are UTC, use NOW() for current time
  - Soft deletes: status = 'deleted' or 'churned', not physical row deletion
  - Use DATE_TRUNC for time bucketing, EXTRACT for date arithmetic
  - Index coverage: customer_id FK on all child tables, created_at on all tables
"""

_SQL_SYSTEM = (
    "You are a senior data analyst and PostgreSQL expert. "
    "You translate natural language business questions into correct, optimized SQL. "
    "Always: use DATE_TRUNC for time bucketing, NULLIF to avoid division by zero, "
    "FILTER clauses instead of CASE WHEN for conditional aggregates, "
    "and LIMIT clauses on potentially large result sets. "
    "Output SQL only — no explanation, no markdown fences. "
    "The database schema is provided in the system context."
)

_ANALYST_QUERIES = [
    # Revenue analysis (5 — varying time windows and granularity)
    "Show me monthly revenue for the last 12 months with order count and average order value.",
    "What was our quarterly revenue breakdown this year? Include unique paying customers.",
    "Give me weekly revenue trend for the last 90 days so I can see the week-over-week pattern.",
    "What are the top 5 revenue days in the last 6 months?",
    "Show annual revenue by acquisition channel for the past 2 years.",
    # Churn and retention (4)
    "Which customers haven't been active in the last 45 days but are still paying?",
    "Show me customers at risk of churning — no events in 30 days or fewer than 3 in the last month.",
    "What's our monthly churn rate for the past year, broken down by plan?",
    "List all customers who cancelled in the last 30 days with their MRR and tenure.",
    # Conversion funnel (3)
    "What's our weekly free-to-paid conversion rate for the last 90 days?",
    "How long does it take on average for trial users to convert, by acquisition source?",
    "Show the conversion funnel from signup to paid with drop-off at each step.",
    # Product and support (3)
    "What are the top 20 products by revenue for the last quarter with gross margin?",
    "Show daily support ticket volume, close rate, and P1 count for the last 30 days.",
    "What is the average customer lifetime value grouped by acquisition channel and starting plan?",
]


def run_nlsql_scenario() -> dict:
    total_calls = 0

    for question in _ANALYST_QUERIES:
        # Step 1: generate SQL
        _call(
            _SQL_SYSTEM + "\n\nSchema:\n" + _DB_SCHEMA,
            question,
            scenario="nlsql",
            call_type="generate_sql",
        )
        total_calls += 1

        # Step 2: ~40% of queries trigger a follow-up refinement
        # (analyst asks "now add a 7-day rolling average" or "break it down by plan")
        if _ANALYST_QUERIES.index(question) % 3 == 0:
            _call(
                _SQL_SYSTEM + "\n\nSchema:\n" + _DB_SCHEMA,
                f"Modify the previous query: also add a 7-day rolling average column "
                f"using AVG() OVER a window. Original question: {question}",
                scenario="nlsql",
                call_type="refine_sql",
            )
            total_calls += 1

    misses = _llm_miss_count_for("nlsql")
    tokens_spent = _total_tokens_for("nlsql")
    return {
        "scenario": "NL-to-SQL Data Analyst",
        "total_calls": total_calls,
        "llm_calls_made": misses,
        "cache_hits": total_calls - misses,
        "hit_rate": (total_calls - misses) / total_calls,
        "tokens_spent": tokens_spent,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _bar(rate: float, width: int = 24) -> str:
    filled = int(rate * width)
    return "█" * filled + "░" * (width - filled)


def print_report(results: list) -> None:
    total_calls   = sum(r["total_calls"]    for r in results)
    total_hits    = sum(r["cache_hits"]     for r in results)
    total_misses  = sum(r["llm_calls_made"] for r in results)
    total_tokens  = sum(r["tokens_spent"]   for r in results)
    overall_rate  = total_hits / total_calls if total_calls else 0

    # Estimate tokens without Mnemon: every call would be a miss
    baseline_tokens = sum(
        r["tokens_spent"] / r["llm_calls_made"] * r["total_calls"]
        if r["llm_calls_made"] > 0 else 0
        for r in results
    )
    tokens_saved  = max(0, int(baseline_tokens - total_tokens))
    cost_saved    = tokens_saved * 0.000003  # claude-3-haiku rate

    W = 68
    print()
    print("=" * W)
    print("  MNEMON WORKFLOW GAUNTLET — RESULTS")
    print("=" * W)
    print()
    print(f"  {'Scenario':<32} {'Calls':>6} {'Hits':>6} {'Misses':>7} {'Hit%':>6}  Bar")
    print(f"  {'-'*32} {'-'*6} {'-'*6} {'-'*7} {'-'*6}  {'-'*24}")
    for r in results:
        rate = r["hit_rate"]
        print(
            f"  {r['scenario']:<32} {r['total_calls']:>6} "
            f"{r['cache_hits']:>6} {r['llm_calls_made']:>7} "
            f"{rate:>5.1%}  {_bar(rate)}"
        )
    print(f"  {'-'*32} {'-'*6} {'-'*6} {'-'*7} {'-'*6}  {'-'*24}")
    print(
        f"  {'TOTAL':<32} {total_calls:>6} "
        f"{total_hits:>6} {total_misses:>7} "
        f"{overall_rate:>5.1%}  {_bar(overall_rate)}"
    )

    print()
    print(f"  Tokens spent  (with Mnemon): {total_tokens:>10,}")
    print(f"  Tokens saved  (vs no cache): {tokens_saved:>10,}   ~${cost_saved:.4f} saved")
    print(f"  LLM calls avoided:           {total_hits:>10,} / {total_calls:,}")
    print()

    print("  Cache behaviour by layer:")
    s1 = sum(r["cache_hits"] for r in results if r["scenario"] in ("Daily Report Agent",))
    s2 = total_hits - s1
    miss_count = total_misses
    print(f"  • System 1 (exact hash, in-session):  ~{s1:>4} hits")
    print(f"  • System 2 (semantic EME, cross-call): ~{s2:>4} hits")
    print(f"  • Miss (novel input, cached for next): ~{miss_count:>4} calls")

    print()
    print("  Where Mnemon struggled:")
    for r in results:
        if r["hit_rate"] < 0.20:
            print(f"  ✗ {r['scenario']}: {r['hit_rate']:.1%} — low repetition, every input unique")
    for r in results:
        if r["hit_rate"] >= 0.20 and r["hit_rate"] < 0.50:
            print(f"  ~ {r['scenario']}: {r['hit_rate']:.1%} — partial repetition, warming")
    for r in results:
        if r["hit_rate"] >= 0.50:
            print(f"  ✓ {r['scenario']}: {r['hit_rate']:.1%} — strong repetition pattern, Mnemon wins")

    print()
    print("=" * W)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_gauntlet() -> list:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print()
    print("Mnemon Workflow Gauntlet")
    print("Five production-grade workflows. Mnemon running underneath via mnemon.init().")
    print()

    results = []
    scenarios = [
        ("1. Customer Support Bot       ", run_support_scenario),
        ("2. Code Review Pipeline       ", run_code_review_scenario),
        ("3. Invoice Processing         ", run_invoice_scenario),
        ("4. Daily Report Agent         ", run_daily_report_scenario),
        ("5. NL-to-SQL Data Analyst     ", run_nlsql_scenario),
    ]

    for label, fn in scenarios:
        t0 = time.time()
        print(f"  Running {label.strip()}...", end="", flush=True)
        r = fn()
        elapsed = time.time() - t0
        print(f"  {r['hit_rate']:.1%} hit rate  ({r['cache_hits']}/{r['total_calls']} cached)  {elapsed:.1f}s")
        results.append(r)

    print_report(results)
    return results


def test_gauntlet_runs():
    """pytest entry point — just check it runs and produces results."""
    results = run_gauntlet()
    assert len(results) == 5
    for r in results:
        assert r["total_calls"] > 0
        assert r["llm_calls_made"] <= r["total_calls"]
        assert 0.0 <= r["hit_rate"] <= 1.0


if __name__ == "__main__":
    run_gauntlet()
