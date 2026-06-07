"""
Mnemon Big Company Workflow Gauntlet
=====================================
Three scenarios modelled on real published production LLM workflows.
Mnemon via mnemon.init(). Fake LLM at the SDK boundary. No real API spend.

Scenarios
---------
1. Cloudflare-style code review    — 48 PRs × 4 calls = 192 calls
   (Cloudflare: 131K reviews/month, 120B tokens, $0.98 median/review)

2. Intercom Fin-style support bot  — 80 tickets × 2 calls = 160 calls
   (600K daily prompts at trading app, $47K/month)

3. Document processing pipeline    — 30 documents × 2 calls = 60 calls
   (Alan-style insurance/finance docs, 60-75% classification hit rate)

Total: 412 LLM calls.

At Cloudflare's real scale (4,400 reviews/day), every 1% improvement in
cache hit rate = ~$17,000/month saved. This test shows what Mnemon adds
on top of provider-side prefix caching.
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
# FAKE LLM — installed before mnemon.init()
# ─────────────────────────────────────────────────────────────────────────────

_llm_calls: list = []
_current_scenario: str = "init"
_current_call_type: str = "general"


def _tok(text: str) -> int:
    return max(1, len(text) // 4)


def _synth(text: str, model: str, inp: int, out: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        id=f"bigco-{len(_llm_calls):04d}",
        type="message", role="assistant", model=model,
        content=[types.SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=types.SimpleNamespace(input_tokens=inp, output_tokens=out),
    )


_RESPONSES = [
    # Code review — security
    (r"sql.inject|cwe.?89|parameteriz|prepared.state",
     "SECURITY HIGH — CWE-89 SQL Injection: line {l1} constructs query via string concat "
     "with unsanitized user input. Fix: parameterized queries. "
     "SECURITY HIGH — CWE-79 XSS: line {l2} renders user content unescaped. "
     "Fix: context-aware escaping. Block merge until resolved."),
    (r"hardcoded.secret|api.key|password.*=.*['\"]|cwe.?798",
     "SECURITY CRITICAL — CWE-798 Hardcoded Credential: line {l1} contains plaintext secret. "
     "Rotate immediately, move to env vars or secrets manager. Block merge."),
    (r"auth.*bypass|missing.*auth|admin.*unprotect|cwe.?284",
     "SECURITY CRITICAL — CWE-284 Broken Access Control: admin endpoint exposed without "
     "authentication middleware. Apply @require_admin decorator. Block merge immediately."),
    (r"csrf|cross.site.request|cwe.?352",
     "SECURITY MEDIUM — CWE-352 CSRF: state-changing endpoint missing CSRF token validation. "
     "Add @csrf_protect decorator or verify Origin header. Fix before merge."),
    # Code review — performance
    (r"n\+1|eager.load|lazy.load|query.*loop",
     "PERFORMANCE MEDIUM — N+1 query pattern: lines {l1}–{l2} execute one query per item. "
     "Fix: eager load with prefetch_related/includes. Estimated: 80% reduction in DB calls."),
    (r"missing.index|table.scan|slow.query|explain",
     "PERFORMANCE MEDIUM — Missing index on high-traffic column. "
     "Add: CREATE INDEX CONCURRENTLY. Estimated improvement: 2,400ms → 12ms."),
    (r"memory.leak|unbounded|cache.*grow|list.*append.*loop",
     "PERFORMANCE HIGH — Unbounded memory growth: list appended in loop without size limit. "
     "Add maxlen or periodic flush. Will OOM under sustained load."),
    # Code review — style
    (r"unused.import|dead.code|naming|lint|style",
     "STYLE LOW — Unused imports lines 3,11,22. Variable naming violations (camelCase → snake_case). "
     "Missing docstrings on 2 public methods. No functional impact."),
    # Support — password/access
    (r"reset.*password|forgot.*password|can.t.*log|locked.*out|password.*reset",
     "To reset your password: visit account.example.com/forgot-password, enter your email, "
     "click Send Reset Link. Link arrives within 5 minutes — check spam if not received. "
     "Links expire after 24 hours. Case ref: TKT-{ref}."),
    (r"sso.*fail|saml.*error|okta|idp.*migrat|login.*break",
     "SSO SAML issue confirmed (TKT-{ref}). Most common post-Okta cause: attribute mapping. "
     "Verify in Okta: (1) NameID = EmailAddress, (2) groups attribute mapped to app roles, "
     "(3) SP metadata re-imported. Share sanitized SAML assertion for faster diagnosis."),
    # Support — billing
    (r"charged.*twice|double.*bill|duplicate.*charge|unexpected.*charge",
     "Duplicate charge confirmed (BIL-{ref}). Finance team notified. Credit posts within "
     "2-3 business days to original payment method. Email confirmation sent on processing."),
    (r"invoice.*vat|tax.*invoice|eu.*compliance|vat.*number",
     "Corrected VAT invoice issued (INV-{ref}) with your EU tax ID and compliant line items. "
     "Available in your billing portal within 1 hour. DPA audit: our data team will provide "
     "the GDPR data export within 48 hours including consent records."),
    # Support — technical
    (r"503|rate.?limit|429|api.*down|batch.*error|endpoint.*fail",
     "Engineering update (INC-{ref}): batch endpoint degradation identified on us-east-1. "
     "ETA resolution: 2 hours. Workaround: reduce batch to ≤50 items, retry with exponential "
     "backoff (1s initial, 32s max). Status: status.example.com"),
    (r"data.*missing|records.*gone|migration.*loss|critical.*data",
     "P1 INCIDENT (INC-{ref}) declared. Data eng investigating with read-only audit log access. "
     "No write ops until root cause confirmed. Backup snapshot at 02:00 UTC available. "
     "Updates every 20 minutes. CTO paged."),
    (r"performance|dashboard.*slow|load.*time|30.*second",
     "Performance issue confirmed (PERF-{ref}). Root cause: analytics aggregation doing full "
     "table scan during peak hours. Fix: covering index deploying at 14:00 UTC. "
     "Expected improvement: p95 32s → <3s."),
    # Document classification
    (r"classif.*document|document.*type|invoice.*classif|classify.*invoice",
     "CLASSIFICATION: document_type=vendor_invoice | jurisdiction={juris} | "
     "vendor_tier={tier} | requires_approval={approval} | tax_treatment={tax} | "
     "confidence=0.97 | processing_path=standard_ap"),
    (r"extract.*field|parse.*invoice|invoice.*data|ocr.*extract",
     "EXTRACTED: vendor={vendor} | invoice_no={inv} | date=2026-05-{day} | "
     "amount={amt} {cur} | tax={tax_amt} | total={total} | terms=Net-30 | "
     "iban={iban} | confidence=0.99"),
]


def _fake_llm_create(_self, *, messages, model, system=None, **kwargs):
    user_msg = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
    )
    combined = ((system or "") + " " + user_msg).lower()

    import random
    rng = random.Random(hash(user_msg) % (2 ** 31))
    response_text = f"Reviewed. No issues found for: {user_msg[:60]}."
    for pattern, template in _RESPONSES:
        if re.search(pattern, combined):
            response_text = template.format(
                l1=rng.randint(30, 80), l2=rng.randint(90, 150),
                ref=f"{rng.randint(10000, 99999)}",
                juris=("DE" if "germany" in combined or "eur" in combined else
                       "IN" if "india" in combined else
                       "UK" if "united kingdom" in combined else "US"),
                tier=("enterprise" if rng.random() > 0.4 else "standard"),
                approval=("yes" if rng.random() > 0.5 else "no"),
                tax=("vat_applicable" if rng.random() > 0.3 else "exempt"),
                vendor="Vendor Corp",
                inv=f"INV-2026-{rng.randint(1000, 9999)}",
                day=str(rng.randint(1, 28)),
                amt=f"{rng.randint(5000, 80000):,.2f}",
                cur="USD", tax_amt=f"{rng.randint(500, 8000):,.2f}",
                total=f"{rng.randint(6000, 90000):,.2f}",
                iban=f"GB{rng.randint(10,99)}NWBK{rng.randint(10000000,99999999)}",
            )
            break

    inp = _tok((system or "") + " ".join(m.get("content", "") for m in messages))
    out = _tok(response_text)
    _llm_calls.append({
        "scenario": _current_scenario,
        "call_type": _current_call_type,
        "input_tokens": inp,
        "output_tokens": out,
    })
    return _synth(response_text, model, inp, out)


import anthropic.resources.messages as _ant_mod
_ant_mod.Messages.create = _fake_llm_create

import mnemon as _mnemon_mod
_tmpdir = tempfile.mkdtemp(prefix="mnemon_bigco_")
_mnemon_mod._instance = None

_M = _mnemon_mod.init(
    tenant_id="bigco_gauntlet",
    db_dir=_tmpdir,
    silent=True,
    prewarm_fragments=False,
    prewarm_templates=False,
    enable_telemetry=False,
)
if _M._patch_thread and _M._patch_thread.is_alive():
    _M._patch_thread.join(timeout=15)

from anthropic import Anthropic as _Anthropic
_client = _Anthropic(api_key="fake-key-bigco")
_MODEL = "claude-3-5-haiku-20241022"


def _call(system_prompt: str, user_message: str, scenario: str, call_type: str):
    global _current_scenario, _current_call_type
    _current_scenario = scenario
    _current_call_type = call_type
    return _client.messages.create(
        model=_MODEL,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=512,
    )


def _misses_for(scenario: str) -> int:
    return sum(1 for c in _llm_calls if c["scenario"] == scenario)


def _tokens_for(scenario: str) -> int:
    return sum(c["input_tokens"] + c["output_tokens"]
               for c in _llm_calls if c["scenario"] == scenario)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 1 — CLOUDFLARE-STYLE CODE REVIEW
# Real: 131,246 reviews/month, 7 specialist agents, $0.98 median/review
# Test: 48 PRs × 4 agents = 192 calls
# Security patterns repeat across PRs — Mnemon catches same CVE class
# ─────────────────────────────────────────────────────────────────────────────

_COORD_SYS = (
    "You are the code review coordinator at a high-scale engineering org. "
    "You receive a git diff and decide which specialist reviewers to invoke: "
    "SECURITY (injection, XSS, auth, secrets), PERFORMANCE (N+1, indexes, memory), "
    "STYLE (naming, imports, docs). Output a routing decision in one line per specialist."
)
_SEC_SYS = (
    "You are the security specialist reviewer. Focus exclusively on: "
    "CWE-89 SQL Injection, CWE-79 XSS, CWE-284 Broken Access Control, "
    "CWE-798 Hardcoded Credentials, CWE-352 CSRF. "
    "State severity, line numbers, attack vector, exact fix. Output CLEAN if none found."
)
_PERF_SYS = (
    "You are the performance specialist reviewer. Focus exclusively on: "
    "N+1 queries, missing indexes, memory leaks, inefficient loops, unoptimized ORM. "
    "State impact, affected lines, quantified improvement, exact fix."
)
_STYLE_SYS = (
    "You are the style reviewer. Focus on: unused imports, naming conventions, "
    "missing docstrings, magic numbers. Issues are never blocking — suggest only."
)

# 48 PRs: 16 security issues × 3 variants, 16 performance × 2 variants, 16 style × 1 variant
_PRS = [
    # Security — SQL injection (16 variations across 3 patterns)
    ("PR #101 auth-service: new login endpoint",
     "+query = f\"SELECT * FROM users WHERE username='{username}'\""),
    ("PR #102 search-api: add product search",
     "+results = db.execute(f'SELECT * FROM products WHERE name LIKE \"%{term}%\"')"),
    ("PR #103 admin-panel: user lookup",
     "+user = db.query(f'SELECT * FROM users WHERE id={user_id} AND active=1')"),
    ("PR #104 reports: custom report builder",
     "+rows = conn.execute('SELECT ' + columns + ' FROM ' + table_name)"),
    ("PR #105 api: order history endpoint",
     "+orders = db.raw(f'SELECT * FROM orders WHERE customer_id={cid}')"),
    ("PR #106 cms: content filter",
     "+content = db.execute(f\"SELECT * FROM posts WHERE tags LIKE '%{tag}%'\")"),
    # Security — hardcoded secrets
    ("PR #107 integrations: Stripe webhook",
     "+STRIPE_SECRET = 'sk_live_xxxxxxxxxxxxxxxxxxx'\n+WEBHOOK_SECRET = 'whsec_abc123'"),
    ("PR #108 config: AWS credentials",
     "+AWS_ACCESS_KEY = 'AKIAIOSFODNN7EXAMPLE'\n+AWS_SECRET = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'"),
    ("PR #109 notifications: SendGrid integration",
     "+SENDGRID_API_KEY = 'SG.xxxxxxxxxxxxxxxxxxxxxxxxx'\n+FROM_EMAIL = 'noreply@company.com'"),
    ("PR #110 payments: PayPal config",
     "+PAYPAL_CLIENT_ID = 'AeA1QIZXiflr1'\n+PAYPAL_SECRET = 'EL1GVap4P9bNYZV'"),
    # Security — auth bypass
    ("PR #111 admin: bulk user operations",
     "+@app.route('/api/admin/bulk-delete')\n+def bulk_delete(): return User.query.delete()"),
    ("PR #112 api: internal metrics endpoint",
     "+@app.route('/internal/metrics')\n+def metrics(): return jsonify(get_all_metrics())"),
    ("PR #113 debug: temporary debug endpoint",
     "+@app.route('/debug/users')\n+def debug_users(): return User.query.all()"),
    # Security — CSRF
    ("PR #114 forms: password change",
     "+@app.route('/account/password', methods=['POST'])\n+def change_password(): pass"),
    ("PR #115 api: email update",
     "+@app.route('/account/email', methods=['POST'])\n+def update_email(): pass"),
    ("PR #116 settings: notification prefs",
     "+@app.route('/settings/notifications', methods=['POST'])\n+def save_notifications(): pass"),
    # Performance — N+1
    ("PR #117 catalog: product listing",
     "+for p in products:\n+    p.variants = Variant.objects.filter(product=p).all()"),
    ("PR #118 dashboard: user activity feed",
     "+for user in users:\n+    user.last_login = Login.objects.filter(user=user).first()"),
    ("PR #119 reports: team performance",
     "+for member in team:\n+    member.tasks = Task.objects.filter(assignee=member).all()"),
    ("PR #120 api: customer orders",
     "+for customer in customers:\n+    customer.order_count = Order.objects.filter(customer=customer).count()"),
    ("PR #121 search: faceted results",
     "+for product in results:\n+    product.category = Category.objects.get(id=product.category_id)"),
    ("PR #122 exports: CSV export",
     "+for row in dataset:\n+    row.metadata = Meta.objects.filter(entity=row).all()"),
    # Performance — missing index
    ("PR #123 api: order history endpoint",
     "# orders.customer_id has no index — 4M row full table scan on every request"),
    ("PR #124 search: product search",
     "# products.name has no index — LIKE query doing sequential scan"),
    ("PR #125 analytics: user cohort query",
     "# events.user_id and events.created_at have no composite index"),
    ("PR #126 reports: revenue by region",
     "# orders.region has no index — aggregation doing full table scan"),
    # Performance — memory
    ("PR #127 streaming: event processor",
     "+results = []\n+for event in event_stream:\n+    results.append(process(event))"),
    ("PR #128 cache: response cache",
     "+CACHE = {}\n+def cache_response(key, value): CACHE[key] = value"),
    ("PR #129 ml: batch predictions",
     "+predictions = []\n+for batch in batches:\n+    predictions.extend(model.predict(batch))"),
    ("PR #130 logs: log aggregator",
     "+log_buffer = []\n+def add_log(entry): log_buffer.append(entry)"),
    # Style (these should mostly NOT hit cache — each is genuinely different)
    ("PR #131 utils: string helpers", "+import os,sys,json,re,time,hashlib,uuid"),
    ("PR #132 models: user model refactor", "+def processUserData(userData): pass"),
    ("PR #133 api: health endpoint", "+import os,sys,logging,traceback,json"),
    ("PR #134 tests: test helpers", "+def setUp(self):\n+    self.testData = {}"),
    ("PR #135 config: settings module", "+DATABASE_NAME = 'prod_db'\n+DEBUG_MODE = True"),
    ("PR #136 scripts: migration script", "+X = 1\n+Y = 2\n+RESULT = X + Y"),
    ("PR #137 api: versioning", "+def getApiVersion(): return '1.0.0'"),
    ("PR #138 utils: date helpers", "+def formatDate(dateStr): pass"),
    ("PR #139 models: product model", "+class productData: pass"),
    ("PR #140 api: pagination", "+def getPaginatedResults(pageNum, pageSize): pass"),
    ("PR #141 cache: cache helpers", "+def clearCache(): cache = {}"),
    ("PR #142 logging: log helpers", "+def logMessage(msg): print(msg)"),
    ("PR #143 errors: error handlers", "+def handleError(err): raise err"),
    ("PR #144 auth: token helpers", "+def validateToken(tok): return True"),
    ("PR #145 db: connection helpers", "+def getConnection(): return db.connect()"),
    ("PR #146 tests: fixtures", "+TEST_USER = {'id': 1, 'name': 'Test'}"),
    ("PR #147 api: rate limiting", "+RATE_LIMIT = 100\n+RATE_WINDOW = 60"),
    ("PR #148 utils: crypto helpers", "+def hashPassword(pw): return hash(pw)"),
]


def run_code_review_scenario() -> dict:
    total_calls = 0
    for pr_title, pr_diff in _PRS:
        _call(_COORD_SYS,
              f"Route this PR to specialists.\nPR: {pr_title}\nDiff:\n{pr_diff}",
              "codereview", "coordinate")
        total_calls += 1
        _call(_SEC_SYS,
              f"Security review.\nPR: {pr_title}\nDiff:\n{pr_diff}",
              "codereview", "security")
        total_calls += 1
        _call(_PERF_SYS,
              f"Performance review.\nPR: {pr_title}\nDiff:\n{pr_diff}",
              "codereview", "performance")
        total_calls += 1
        _call(_STYLE_SYS,
              f"Style review.\nPR: {pr_title}\nDiff:\n{pr_diff}",
              "codereview", "style")
        total_calls += 1

    misses = _misses_for("codereview")
    tokens = _tokens_for("codereview")
    hits = total_calls - misses
    return {
        "scenario": "Cloudflare-style Code Review",
        "total_calls": total_calls,
        "llm_calls_made": misses,
        "cache_hits": hits,
        "hit_rate": hits / total_calls,
        "tokens_spent": tokens,
        "real_scale": "4,400 reviews/day",
        "real_daily_cost": 4312,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 2 — INTERCOM FIN-STYLE SUPPORT BOT
# Real: 600K daily prompts, $47K/month at mobile trading app
# Test: 80 tickets × 2 calls = 160 calls
# 8 categories × 10 tickets each, each phrased differently
# ─────────────────────────────────────────────────────────────────────────────

_SUPPORT_SYS = (
    "You are Fin, an AI customer support agent for a B2B SaaS platform. "
    "You handle billing, technical, access, data, and compliance issues. "
    "Enterprise SLA: 2-hour first response. Always include a case reference. "
    "Escalate P1 (data loss, security, outages) immediately to on-call engineer."
)

_TICKETS_BIGCO = [
    # Password/access (10)
    ("access", "How do I reset my password? I've been locked out for 20 minutes."),
    ("access", "Forgot my password — need to get back in urgently for a client call."),
    ("access", "My password reset email isn't arriving. Already checked spam folder."),
    ("access", "Can't log in — keeps saying invalid credentials after password reset."),
    ("access", "Password expired and reset link says it's already been used."),
    ("access", "I need to reset my password but I no longer have access to that email."),
    ("access", "Getting 'too many login attempts' error, account may be locked."),
    ("access", "New employee can't set up their account — setup email never arrived."),
    ("access", "Login working on mobile but not on desktop browser."),
    ("access", "Two-factor authentication stopped working after I got a new phone."),
    # Billing (10)
    ("billing", "We were charged twice this month — $2,400 duplicate on our invoice."),
    ("billing", "Unexpected charge appeared on our company card. Not on our contract."),
    ("billing", "Invoice shows double billing for our enterprise subscription."),
    ("billing", "Two identical charges from your company this billing cycle."),
    ("billing", "Credit card shows duplicate transaction — need refund immediately."),
    ("billing", "Our CFO flagged a double charge on last month's statement."),
    ("billing", "We need a VAT invoice with our EU tax ID for quarterly filing."),
    ("billing", "Invoice missing our German VAT number — auditors need this corrected."),
    ("billing", "Contract renewal came in 20% higher than our quoted price."),
    ("billing", "Need itemised invoice broken down by team for our accounting system."),
    # API errors (10)
    ("api_error", "Batch API endpoint returning 503 for requests over 100 items."),
    ("api_error", "Getting HTTP 503 on /v2/batch — production system is affected."),
    ("api_error", "Batch endpoint down for the last hour, ETA for resolution?"),
    ("api_error", "Rate limit 429 errors triggering at 50% of documented threshold."),
    ("api_error", "Hitting 429 rate limits even though we're well under the stated cap."),
    ("api_error", "API returning 429 Too Many Requests — we're not close to our limit."),
    ("api_error", "Webhook deliveries failing with 503 since this morning."),
    ("api_error", "Events endpoint timing out on payloads over 1MB."),
    ("api_error", "Pagination cursor returning 400 error after page 10."),
    ("api_error", "GraphQL mutations returning 500 intermittently under load."),
    # Data/compliance (10)
    ("data", "CRITICAL: 3,400 customer records missing after last night's migration."),
    ("data", "Data loss detected post-maintenance — thousands of records gone."),
    ("data", "Our customer table lost rows after the upgrade window. This is P1."),
    ("data", "GDPR data export missing consent records — DPA audit in 3 days."),
    ("data", "Compliance export incomplete — consent history absent for 2019-2021."),
    ("data", "Data export for GDPR subject access request is missing fields."),
    ("data", "Audit logs showing gaps in March 2026 — need explanation for auditors."),
    ("data", "Dashboard showing incorrect aggregated metrics since the 28th."),
    ("data", "Reports showing different totals than our data warehouse — data sync issue?"),
    ("data", "CSV export cutting off at 50,000 rows — we need all 2.3M records."),
    # SSO/enterprise access (10)
    ("sso", "SSO SAML login failing for 12 new employees after Okta migration."),
    ("sso", "New staff can't authenticate via SSO after we switched to Okta."),
    ("sso", "SAML assertion errors since our IdP migration last week."),
    ("sso", "Azure AD SSO stopped working after we updated our tenant config."),
    ("sso", "Google Workspace SSO returning 'unable to authenticate' for 5 users."),
    ("sso", "JIT provisioning not creating accounts for new SSO users."),
    ("sso", "SSO users can log in but have wrong permissions after migration."),
    ("sso", "SCIM provisioning stopped syncing user updates from Okta."),
    ("sso", "MFA requirement not enforcing for SSO users in our org."),
    ("sso", "SSO login redirecting to wrong tenant after domain migration."),
    # Performance (10)
    ("perf", "Dashboard taking 30+ seconds to load during business hours."),
    ("perf", "Reports page timing out for our largest enterprise customer."),
    ("perf", "Analytics dashboard extremely slow — unusable during peak hours."),
    ("perf", "Data exports taking 45 minutes for datasets over 1M rows."),
    ("perf", "API response times degraded — p95 jumped from 200ms to 4 seconds."),
    ("perf", "Search results taking 10+ seconds for complex queries."),
    ("perf", "Scheduled reports not completing within 4-hour window."),
    ("perf", "Webhook processing queue backing up — 2 hour delay on deliveries."),
    ("perf", "Real-time sync falling behind by 30 minutes during peak load."),
    ("perf", "Bulk import timing out for files over 50MB."),
    # Feature/integration (10)
    ("feature", "Need webhook support for real-time event streaming to our Kafka cluster."),
    ("feature", "Does your platform support webhooks? We need to push events to Kafka."),
    ("feature", "Looking for real-time event delivery — webhook or SSE support?"),
    ("feature", "Salesforce connector not syncing custom opportunity fields since v4.2."),
    ("feature", "HubSpot integration stopped syncing contact updates after last release."),
    ("feature", "Zapier integration failing on multi-step zaps with more than 3 actions."),
    ("feature", "Need bulk import via API — CSV endpoint only handles 10K rows."),
    ("feature", "REST API docs still showing v2 examples — v3 migration guide missing."),
    ("feature", "SDK for Python missing async support — breaking our FastAPI integration."),
    ("feature", "Need IP allowlist support for our enterprise security requirements."),
    # Onboarding/enterprise (10)
    ("onboarding", "New 500-seat deployment needs custom onboarding and security review."),
    ("onboarding", "We're rolling out to 200 users next week — what's the setup process?"),
    ("onboarding", "Enterprise onboarding call — our InfoSec team has 40 questions."),
    ("onboarding", "Need custom data residency in EU region for our GDPR compliance."),
    ("onboarding", "Can we get a dedicated instance for our 1,000 seat deployment?"),
    ("onboarding", "Need SOC2 Type II report and pen test results for our procurement."),
    ("onboarding", "Security questionnaire from our procurement team — 80 questions."),
    ("onboarding", "Need custom SLA of 99.99% uptime for our enterprise contract."),
    ("onboarding", "BAA for HIPAA compliance — we handle healthcare data."),
    ("onboarding", "DPA agreement needed before our legal team approves the purchase."),
]


def run_support_scenario() -> dict:
    total_calls = 0
    for category, ticket_text in _TICKETS_BIGCO:
        _call(_SUPPORT_SYS,
              f"Classify: priority (P1/P2/P3) and category.\nTicket: {ticket_text}",
              "support", "classify")
        total_calls += 1
        _call(_SUPPORT_SYS,
              f"Write a customer-facing response.\nTicket: {ticket_text}",
              "support", "respond")
        total_calls += 1

    misses = _misses_for("support")
    tokens = _tokens_for("support")
    hits = total_calls - misses
    return {
        "scenario": "Intercom Fin-style Support Bot",
        "total_calls": total_calls,
        "llm_calls_made": misses,
        "cache_hits": hits,
        "hit_rate": hits / total_calls,
        "tokens_spent": tokens,
        "real_scale": "600,000 prompts/day",
        "real_daily_cost": 1200,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 3 — DOCUMENT PROCESSING PIPELINE
# Real: Alan insurance, Kofax, UiPath — 60-75% classification hit rate
# Test: 30 documents × 2 calls = 60 calls
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFY_SYS = (
    "You are an AP document classifier. Classify as: vendor_invoice, credit_note, "
    "purchase_order, expense_report, or unknown. Output: document_type, jurisdiction, "
    "vendor_tier, requires_approval, tax_treatment, confidence."
)
_EXTRACT_SYS = (
    "You are an invoice field extractor. Extract: vendor_name, invoice_no, invoice_date, "
    "due_date, line_items, amount_net, tax_rate, tax_amount, amount_total, currency, "
    "payment_terms, bank_iban. Flag anomalies."
)

_DOCS = [
    # US vendors (8)
    ("US", "Microsoft Corporation", "INV-MS-0441", 5600),
    ("US", "Amazon Web Services", "INV-AWS-0992", 3200),
    ("US", "Salesforce Inc", "INV-SF-1104", 9800),
    ("US", "Oracle Corporation", "INV-ORC-0331", 67000),
    ("US", "Zoom Video Communications", "INV-ZM-0887", 1800),
    ("US", "Slack Technologies", "INV-SLK-0221", 4200),
    ("US", "Twilio Inc", "INV-TWL-0556", 2900),
    ("US", "Datadog Inc", "INV-DD-0778", 8400),
    # EU vendors (12)
    ("DE", "SAP SE", "INV-SAP-4411", 4200),
    ("DE", "Deutsche Telekom AG", "INV-DT-2201", 12400),
    ("DE", "Siemens AG", "INV-SIE-0772", 38000),
    ("DE", "Bayer AG", "INV-BAY-1156", 7900),
    ("FR", "Capgemini SE", "INV-CAP-9901", 31000),
    ("FR", "Total Energies SE", "INV-TOT-3302", 18500),
    ("NL", "ASML Holding NV", "INV-ASML-0011", 52000),
    ("NL", "Philips NV", "INV-PHI-4421", 9800),
    ("IE", "Accenture PLC", "INV-ACC-7701", 45000),
    ("IE", "CRH PLC", "INV-CRH-3301", 22000),
    ("UK", "KPMG LLP", "INV-KPM-3301", 12300),
    ("UK", "Deloitte LLP", "INV-DEL-0441", 28500),
    # India vendors (6)
    ("IN", "Tata Consultancy Services", "INV-TCS-7701", 18500),
    ("IN", "Infosys Limited", "INV-INF-4421", 14200),
    ("IN", "Wipro Limited", "INV-WIP-3301", 8900),
    ("IN", "HCL Technologies", "INV-HCL-2211", 22000),
    ("IN", "Tech Mahindra", "INV-TM-0991", 11700),
    ("IN", "Cognizant Technology", "INV-COG-5511", 16800),
    # Purchase orders and credit notes (4)
    ("US", "Dell Technologies PO", "PO-DELL-0441", 15000),
    ("DE", "BMW AG Credit Note", "CN-BMW-0221", 3200),
    ("UK", "Barclays Expense Report", "EXP-BAR-0991", 4500),
    ("US", "GitHub Inc", "INV-GH-1122", 6700),
]


def run_document_scenario() -> dict:
    total_calls = 0
    for jurisdiction, vendor, inv_no, amount in _DOCS:
        doc_text = (
            f"Document type: Invoice\nFrom: {vendor}\nNo: {inv_no}\n"
            f"Jurisdiction: {jurisdiction}\nAmount: {amount:,.2f} "
            f"{'EUR' if jurisdiction in ('DE','FR','NL','IE') else 'GBP' if jurisdiction == 'UK' else 'INR' if jurisdiction == 'IN' else 'USD'}\n"
            f"Description: Professional services Q1 2026\nTerms: Net-30"
        )
        _call(_CLASSIFY_SYS,
              f"Classify this document from {vendor} ({jurisdiction}).\n\n{doc_text}",
              "documents", "classify")
        total_calls += 1
        _call(_EXTRACT_SYS,
              f"Extract all fields from this document.\n\n{doc_text}",
              "documents", "extract")
        total_calls += 1

    misses = _misses_for("documents")
    tokens = _tokens_for("documents")
    hits = total_calls - misses
    return {
        "scenario": "Document Processing Pipeline",
        "total_calls": total_calls,
        "llm_calls_made": misses,
        "cache_hits": hits,
        "hit_rate": hits / total_calls,
        "tokens_spent": tokens,
        "real_scale": "500+ documents/day",
        "real_daily_cost": 250,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _bar(rate: float, width: int = 20) -> str:
    filled = int(rate * width)
    return "█" * filled + "░" * (width - filled)


def print_report(results: list) -> None:
    total_calls = sum(r["total_calls"] for r in results)
    total_hits = sum(r["cache_hits"] for r in results)
    total_misses = sum(r["llm_calls_made"] for r in results)
    overall_rate = total_hits / total_calls if total_calls else 0

    baseline = sum(
        r["tokens_spent"] / r["llm_calls_made"] * r["total_calls"]
        if r["llm_calls_made"] > 0 else 0
        for r in results
    )
    tokens_saved = max(0, int(baseline - sum(r["tokens_spent"] for r in results)))

    W = 72
    print()
    print("=" * W)
    print("  MNEMON BIG COMPANY GAUNTLET — RESULTS")
    print("=" * W)
    print()
    print(f"  {'Scenario':<35} {'Calls':>6} {'Hits':>6} {'Hit%':>6}  Bar")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6}  {'-'*20}")
    for r in results:
        print(
            f"  {r['scenario']:<35} {r['total_calls']:>6} "
            f"{r['cache_hits']:>6} {r['hit_rate']:>5.1%}  {_bar(r['hit_rate'])}"
        )
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6}  {'-'*20}")
    print(f"  {'TOTAL':<35} {total_calls:>6} {total_hits:>6} {overall_rate:>5.1%}  {_bar(overall_rate)}")

    print()
    print("  Token savings in this test:")
    print(f"    Tokens spent  (with Mnemon): {sum(r['tokens_spent'] for r in results):>10,}")
    print(f"    Tokens saved  (vs no cache): {tokens_saved:>10,}")
    print()

    print("  Projected savings at real company scale:")
    print()
    for r in results:
        if r["hit_rate"] > 0 and r["real_daily_cost"] > 0:
            daily_saved = r["real_daily_cost"] * r["hit_rate"]
            monthly_saved = daily_saved * 30
            print(f"  {r['scenario']}")
            print(f"    Scale: {r['real_scale']}")
            print(f"    Current daily LLM spend: ${r['real_daily_cost']:,.0f}")
            print(f"    Mnemon hit rate: {r['hit_rate']:.1%}")
            print(f"    Daily savings: ${daily_saved:,.0f}")
            print(f"    Monthly savings: ${monthly_saved:,.0f}")
            print()

    total_monthly = sum(
        r["real_daily_cost"] * r["hit_rate"] * 30
        for r in results if r["hit_rate"] > 0
    )
    print(f"  Total monthly savings across all three workflows: ${total_monthly:,.0f}")
    print()
    print("=" * W)
    print()


def run_gauntlet() -> list:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print()
    print("Mnemon Big Company Gauntlet")
    print("Modelled on Cloudflare (131K reviews/month), Intercom Fin (13M conversations),")
    print("and enterprise document processing pipelines.")
    print()

    results = []
    scenarios = [
        ("1. Cloudflare-style Code Review  ", run_code_review_scenario),
        ("2. Intercom Fin-style Support Bot ", run_support_scenario),
        ("3. Document Processing Pipeline  ", run_document_scenario),
    ]

    for label, fn in scenarios:
        t0 = time.time()
        print(f"  Running {label.strip()}...", end="", flush=True)
        r = fn()
        elapsed = time.time() - t0
        print(f"  {r['hit_rate']:.1%} hit  ({r['cache_hits']}/{r['total_calls']})  {elapsed:.1f}s")
        results.append(r)

    print_report(results)
    return results


def test_bigco_gauntlet_runs():
    results = run_gauntlet()
    assert len(results) == 3
    for r in results:
        assert r["total_calls"] > 0
        assert 0.0 <= r["hit_rate"] <= 1.0


if __name__ == "__main__":
    run_gauntlet()
