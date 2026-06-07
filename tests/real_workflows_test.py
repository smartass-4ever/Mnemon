"""
Real business workflow tests — neutral code a developer would write,
Mnemon added as two lines. Tests System 1, System 2, Fragment Assembly
and UX (messages, friction, first-run experience) across 4 high-priority
enterprise workflows.

Workflows:
  1. Invoice data extraction      (document processing — Tier 1)
  2. Support ticket triage        (customer support — Tier 1)
  3. Weekly business report       (recurring report — compound value)
  4. NL-to-SQL query generation   (analytics — fragment reuse)

Run as: python tests/real_workflows_test.py
Each workflow uses its own tenant — isolated, no cross-contamination.
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

DIVIDER = "=" * 62


# ── Helpers ───────────────────────────────────────────────────────────────────

def fresh(tenant: str) -> mnemon.MnemonSync:
    mnemon._instance = None
    return mnemon.init(tenant_id=tenant, db_dir=".")


def call(m, label: str, goal: str, inputs: dict, caps: list,
         mock_fn, mock_latency=1.5):
    t0 = time.time()

    def gen(g, inp, ctx, c, con):
        time.sleep(mock_latency)
        return mock_fn(inp)

    result = m.run(goal=goal, inputs=inputs, generation_fn=gen,
                   capabilities=caps)
    ms = (time.time() - t0) * 1000
    level = result.get("cache_level", "miss")
    tag = {"system1": "S1 HIT ", "system2": "S2 HIT ", "miss": "MISS   "}.get(level, level)
    saved = result.get("tokens_saved", 0)
    print(f"  [{tag}] {ms:6.0f}ms  {'+' + str(saved) + 'tk':>8}  {label}")
    return level


def wait_fragments(m, timeout=25):
    from mnemon.fragments.library import FRAGMENT_COUNT
    deadline = time.time() + timeout
    while time.time() < deadline:
        if m.get_stats().get("db", {}).get("fragments", 0) >= FRAGMENT_COUNT:
            return
        time.sleep(0.4)


# ── Workflow 1: Invoice Data Extraction ───────────────────────────────────────

def workflow_invoice():
    print(f"\n{DIVIDER}")
    print("WORKFLOW 1 — Invoice Data Extraction")
    print("Extracts structured fields from vendor invoices for accounting.")
    print(DIVIDER)

    m = fresh("real_invoices")
    wait_fragments(m)

    def extract(inp):
        return {
            "vendor": inp.get("vendor", "Unknown"),
            "invoice_number": f"INV-{inp.get('invoice_id', '0000')}",
            "total": inp.get("amount", 0),
            "due_date": inp.get("due", "unknown"),
            "line_items": [{"desc": "Services", "amount": inp.get("amount", 0)}],
            "currency": "USD",
        }

    caps = [
        "parse vendor and invoice number from document",
        "extract line items and amounts",
        "identify payment terms and due date",
        "output structured JSON for accounting system",
    ]

    invoices = [
        ("Acme Cloud invoice May",
         "Extract invoice data for accounting records",
         {"vendor": "Acme Cloud", "invoice_id": "2026-0481", "amount": 299.00, "due": "2026-06-27"}),
        ("DataPipeline Pro invoice",
         "Parse invoice document and extract billing fields",
         {"vendor": "DataPipeline Pro", "invoice_id": "2026-1134", "amount": 245.00, "due": "2026-06-30"}),
        ("Acme Cloud invoice June",
         "Extract structured data from vendor invoice for finance",
         {"vendor": "Acme Cloud", "invoice_id": "2026-0522", "amount": 299.00, "due": "2026-07-27"}),
        ("Jane Dev Consulting",
         "Pull invoice fields for accounts payable processing",
         {"vendor": "Jane Dev Consulting", "invoice_id": "JD-0042", "amount": 3000.00, "due": "2026-07-01"}),
        ("Acme Cloud invoice July",
         "Extract invoice data for accounting records",
         {"vendor": "Acme Cloud", "invoice_id": "2026-0601", "amount": 299.00, "due": "2026-08-27"}),
    ]

    results = []
    for label, goal, inputs in invoices:
        r = call(m, label, goal, inputs, caps, extract, mock_latency=0.8)
        results.append(r)

    hits = sum(1 for r in results if r != "miss")
    print(f"\n  Hits: {hits}/{len(results)} — S1={results.count('system1')} S2={results.count('system2')}")


# ── Workflow 2: Support Ticket Triage ─────────────────────────────────────────

def workflow_support():
    print(f"\n{DIVIDER}")
    print("WORKFLOW 2 — Customer Support Ticket Triage")
    print("Classifies tickets and drafts responses. Runs 100s/day.")
    print(DIVIDER)

    m = fresh("real_support")
    wait_fragments(m)

    def triage(inp):
        msg = inp.get("message", "").lower()
        if "charge" in msg or "refund" in msg or "bill" in msg:
            cat, pri = "billing", "high"
        elif "slow" in msg or "down" in msg or "error" in msg:
            cat, pri = "technical", "urgent"
        elif "cancel" in msg:
            cat, pri = "account", "high"
        else:
            cat, pri = "general", "normal"
        return (
            f"Category: {cat} | Priority: {pri}\n"
            f"Response: Thank you for reaching out. We've reviewed your case "
            f"and our team will follow up within 24 hours. Ref: TKT-{inp.get('id', '0000')}"
        )

    caps = [
        "classify ticket category and priority",
        "identify customer intent and sentiment",
        "draft appropriate response based on category",
        "route to correct team and set SLA",
    ]

    tickets = [
        ("billing dispute",
         "Triage customer support ticket and draft response",
         {"id": "001", "message": "I was charged twice this month, please refund the duplicate"}),
        ("service outage report",
         "Handle customer complaint and write reply",
         {"id": "002", "message": "Your service has been down for 2 hours and we are losing money"}),
        ("cancellation request",
         "Process customer support request and respond",
         {"id": "003", "message": "I want to cancel my subscription immediately"}),
        ("billing question variant",
         "Respond to customer billing issue",
         {"id": "004", "message": "There is an unauthorized charge on my account I did not approve"}),
        ("another outage",
         "Classify and reply to customer support ticket",
         {"id": "005", "message": "The API keeps returning 503 errors for the past hour"}),
        ("refund request",
         "Triage support ticket and draft customer response",
         {"id": "006", "message": "Please refund my last payment, I was billed incorrectly"}),
    ]

    results = []
    for label, goal, inputs in tickets:
        r = call(m, label, goal, inputs, caps, triage, mock_latency=1.2)
        results.append(r)

    hits = sum(1 for r in results if r != "miss")
    print(f"\n  Hits: {hits}/{len(results)} — S1={results.count('system1')} S2={results.count('system2')}")


# ── Workflow 3: Weekly Business Report ────────────────────────────────────────

def workflow_weekly_report():
    print(f"\n{DIVIDER}")
    print("WORKFLOW 3 — Weekly Business Report Generation")
    print("Runs every Monday. Same pipeline, different data each week.")
    print(DIVIDER)

    m = fresh("real_reports")
    wait_fragments(m)

    def generate_report(inp):
        return (
            f"Weekly Report — {inp.get('team', 'Engineering')} — {inp.get('week', 'W00')}\n\n"
            f"Revenue: ${inp.get('revenue', 0):,.0f}\n"
            f"Active users: {inp.get('users', 0):,}\n"
            f"Key wins: Shipped {inp.get('features', 0)} features\n"
            f"Blockers: None critical\n"
            f"Next week: Continue Q2 roadmap execution"
        )

    caps = [
        "aggregate weekly metrics from data sources",
        "identify trends and compare to previous week",
        "highlight key wins and blockers",
        "generate executive summary narrative",
        "format report for distribution",
    ]

    weeks = [
        ("W21 Engineering",
         "Generate weekly engineering team report from metrics",
         {"team": "Engineering", "week": "2026-W21", "revenue": 284000, "users": 4820, "features": 3}),
        ("W22 Engineering",
         "Create weekly team performance report and summary",
         {"team": "Engineering", "week": "2026-W22", "revenue": 291000, "users": 4950, "features": 2}),
        ("W21 Product",
         "Write weekly product team status report",
         {"team": "Product", "week": "2026-W21", "revenue": 284000, "users": 4820, "features": 5}),
        ("W23 Engineering",
         "Generate weekly business report for engineering leadership",
         {"team": "Engineering", "week": "2026-W23", "revenue": 305000, "users": 5100, "features": 4}),
        ("W22 Product",
         "Produce weekly summary report for product and design teams",
         {"team": "Product", "week": "2026-W22", "revenue": 291000, "users": 4950, "features": 3}),
    ]

    results = []
    for label, goal, inputs in weeks:
        r = call(m, label, goal, inputs, caps, generate_report, mock_latency=2.0)
        results.append(r)

    hits = sum(1 for r in results if r != "miss")
    print(f"\n  Hits: {hits}/{len(results)} — S1={results.count('system1')} S2={results.count('system2')}")


# ── Workflow 4: NL-to-SQL ─────────────────────────────────────────────────────

def workflow_nl_to_sql():
    print(f"\n{DIVIDER}")
    print("WORKFLOW 4 — Natural Language to SQL")
    print("Analysts query the data warehouse in plain English. Runs 50-500x/day.")
    print(DIVIDER)

    m = fresh("real_nlsql")
    wait_fragments(m)

    def generate_sql(inp):
        q = inp.get("question", "").lower()
        table = inp.get("table", "orders")
        if "revenue" in q or "sales" in q:
            return f"SELECT region, SUM(amount) as revenue FROM {table} WHERE date >= '2026-01-01' GROUP BY region ORDER BY revenue DESC"
        if "user" in q or "customer" in q:
            return f"SELECT COUNT(DISTINCT user_id) as users, DATE_TRUNC('week', created_at) as week FROM {table} GROUP BY week ORDER BY week"
        if "top" in q or "best" in q:
            return f"SELECT product_name, SUM(quantity) as total FROM {table} GROUP BY product_name ORDER BY total DESC LIMIT 10"
        return f"SELECT * FROM {table} LIMIT 100"

    caps = [
        "parse natural language question intent",
        "map entities to database schema tables and columns",
        "construct SQL query with correct joins and aggregations",
        "validate query against schema constraints",
    ]

    queries = [
        ("revenue by region",
         "Convert natural language question to SQL query",
         {"question": "What is total revenue by region this year", "table": "orders", "db": "analytics"}),
        ("weekly active users",
         "Translate business question to database query",
         {"question": "Show me weekly active users for the last quarter", "table": "sessions", "db": "analytics"}),
        ("top products",
         "Generate SQL from natural language request",
         {"question": "Which are the top 10 best selling products", "table": "order_items", "db": "analytics"}),
        ("sales by region",
         "Write SQL query for business intelligence question",
         {"question": "Give me sales figures broken down by geographic region", "table": "sales", "db": "analytics"}),
        ("monthly users",
         "Convert analyst question to SQL for data warehouse",
         {"question": "How many unique customers do we have per month", "table": "events", "db": "analytics"}),
    ]

    results = []
    for label, goal, inputs in queries:
        r = call(m, label, goal, inputs, caps, generate_sql, mock_latency=1.0)
        results.append(r)

    hits = sum(1 for r in results if r != "miss")
    print(f"\n  Hits: {hits}/{len(results)} — S1={results.count('system1')} S2={results.count('system2')}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(DIVIDER)
    print("REAL BUSINESS WORKFLOW TEST")
    print("Mnemon v" + mnemon.__version__ + " — neutral production workflows")
    print("Observing: cache behaviour + UX messages + friction points")
    print(DIVIDER)

    t0 = time.time()
    workflow_invoice()
    workflow_support()
    workflow_weekly_report()
    workflow_nl_to_sql()

    print(f"\n{DIVIDER}")
    print(f"Total time: {time.time()-t0:.1f}s")
    print(DIVIDER)


if __name__ == "__main__":
    main()
