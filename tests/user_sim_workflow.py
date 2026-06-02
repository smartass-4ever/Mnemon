"""
User simulation workflow test.

Simulates 5 genuinely different production workflows a real developer
would already have — not designed for Mnemon. Mnemon is added as a user
would: two lines at the top, nothing else changed.

Goals:
  1. Observe the real friction / service experience (messages, errors, confusion)
  2. Push System 2 — each workflow runs 3x with semantically similar but
     lexically different goals to test semantic cache hits.

Run:
    pip install mnemon-ai
    python tests/user_sim_workflow.py

No API key required — uses a mock LLM that returns realistic outputs.
Set REAL_LLM=1 to use your actual Anthropic key instead.
"""

import os
import sys
import time
import textwrap

# ── Mnemon: two lines a user would add ────────────────────────────────────────
import mnemon
# ─────────────────────────────────────────────────────────────────────────────

REAL_LLM = os.environ.get("REAL_LLM", "0") == "1"
DIVIDER = "-" * 60


# ── Mock LLM (realistic latency + outputs) ────────────────────────────────────

def mock_llm(prompt: str, latency_ms: int = 2200) -> str:
    """Simulates an LLM call with realistic latency and plausible output."""
    time.sleep(latency_ms / 1000)
    # Return something vaguely relevant based on keywords in the prompt
    p = prompt.lower()
    if "support" in p or "ticket" in p or "complaint" in p or "refund" in p:
        return (
            "Category: Billing | Priority: High\n"
            "Response: Thank you for reaching out. I've reviewed your account and "
            "can confirm the charge. A refund will be processed within 3-5 business days. "
            "Reference: TKT-{:.0f}".format(time.time() % 10000)
        )
    if "standup" in p or "pr" in p or "pull request" in p or "sprint" in p:
        return (
            "Yesterday: Merged auth refactor PR, reviewed 2 PRs.\n"
            "Today: Starting on rate limiting feature.\n"
            "Blockers: Waiting on API spec from backend team."
        )
    if "meeting" in p or "action" in p or "transcript" in p or "summary" in p:
        return (
            "Summary: Discussed Q3 roadmap and resource allocation.\n"
            "Action items:\n"
            "  - @alice: Finalize design doc by Friday\n"
            "  - @bob: Schedule follow-up with data team\n"
            "  - @carol: Update Jira board with new estimates"
        )
    if "code" in p or "review" in p or "diff" in p or "bug" in p or "security" in p:
        return (
            "Issues found: 2 medium, 1 low\n"
            "- Line 47: SQL query uses string concatenation (injection risk)\n"
            "- Line 83: No input validation on user_id parameter\n"
            "- Line 102: Unused import (minor)\n"
            "Overall: Approve with changes."
        )
    if "invoice" in p or "extract" in p or "vendor" in p or "amount" in p:
        return (
            "vendor: Acme Supplies Ltd\n"
            "invoice_number: INV-2026-04821\n"
            "date: 2026-05-28\n"
            "line_items: [{item: 'Cloud storage', qty: 1, unit_price: 299.00}]\n"
            "total: 299.00\n"
            "currency: USD\n"
            "due_date: 2026-06-27"
        )
    return "Processed successfully."


def call_llm(prompt: str) -> str:
    if REAL_LLM:
        from anthropic import Anthropic
        client = Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    return mock_llm(prompt)


# ── Workflow definitions ───────────────────────────────────────────────────────

def run_workflow(label: str, goal: str, inputs: dict, prompt: str, run_num: int, m):
    """Run one workflow call through Mnemon and print what the user sees."""
    print(f"\n  [{run_num}] {label}")
    print(f"      Goal: {goal[:70]}")

    t0 = time.time()

    def generation_fn(g, inp, ctx, caps, constraints):
        return call_llm(prompt)

    result = m.run(goal=goal, inputs=inputs, generation_fn=generation_fn)
    elapsed = (time.time() - t0) * 1000

    level = result.get("cache_level", "miss")
    tokens_saved = result.get("tokens_saved", 0)
    latency_saved = result.get("latency_saved_ms", 0)

    level_label = {
        "system1": "CACHE HIT (exact)",
        "system2": "CACHE HIT (semantic)",
        "miss":    "cache miss",
    }.get(level, level)

    print(f"      Result: {level_label}  |  {elapsed:.0f}ms elapsed", end="")
    if tokens_saved:
        print(f"  |  {tokens_saved} tokens saved", end="")
    print()
    return result


# ── Five genuinely different workflows ────────────────────────────────────────

def _fresh_mnemon(tenant_id: str):
    """Each workflow is a separate 'user project' with its own tenant."""
    mnemon._instance = None
    return mnemon.init(tenant_id=tenant_id, db_dir=".")


def workflow_support_triage():
    print(f"\n{DIVIDER}")
    print("WORKFLOW 1: Customer Support Triage")
    print("(A support bot that classifies tickets and drafts replies)")
    print(DIVIDER)

    m = _fresh_mnemon("sim_support")
    scenarios = [
        (
            "billing dispute — first contact",
            "Triage customer support ticket and draft response",
            {"ticket_id": "TKT-001", "channel": "email", "tier": "free"},
            "Customer email: 'I was charged twice for my subscription last month. "
            "Please refund the duplicate charge immediately.' Classify and draft response.",
        ),
        (
            "different customer, shipping complaint",
            "Handle customer complaint and write reply",
            {"ticket_id": "TKT-002", "channel": "chat", "tier": "pro"},
            "Customer message: 'My order shipped 10 days ago and still hasn't arrived. "
            "Tracking says in transit. Need resolution.' Classify and draft response.",
        ),
        (
            "billing again, different wording",
            "Respond to billing issue raised by customer",
            {"ticket_id": "TKT-003", "channel": "email", "tier": "free"},
            "Customer email: 'I see an unauthorized charge on my credit card from your service. "
            "I did not authorize this. Please reverse it.' Classify and draft response.",
        ),
    ]

    for i, (label, goal, inputs, prompt) in enumerate(scenarios, 1):
        run_workflow(label, goal, inputs, prompt, i, m)


def workflow_standup_generator():
    print(f"\n{DIVIDER}")
    print("WORKFLOW 2: Daily Standup Generator")
    print("(Dev tool: generates standup summaries from PR/issue activity)")
    print(DIVIDER)

    m = _fresh_mnemon("sim_standup")
    scenarios = [
        (
            "backend team, Monday",
            "Generate daily standup summary from PR activity",
            {"team": "backend", "date": "2026-06-02", "sprint": "Sprint 14"},
            "PRs merged: auth-refactor (#341), fix null pointer in user service (#338). "
            "PRs open: rate-limiting feature (#344, in review). "
            "Issues closed: login timeout bug. Generate standup.",
        ),
        (
            "frontend team, same day",
            "Create standup update based on team activity",
            {"team": "frontend", "date": "2026-06-02", "sprint": "Sprint 14"},
            "PRs merged: dashboard redesign (#201). PRs in review: dark mode toggle (#205). "
            "Blocked on: API spec for new notifications endpoint. Generate standup.",
        ),
        (
            "backend team, Tuesday — similar week",
            "Write up daily standup from yesterday's engineering activity",
            {"team": "backend", "date": "2026-06-03", "sprint": "Sprint 14"},
            "PRs merged: rate-limiting (#344). PRs open: DB migration script (#347, needs review). "
            "Issues: memory leak in worker process (investigating). Generate standup.",
        ),
    ]

    for i, (label, goal, inputs, prompt) in enumerate(scenarios, 1):
        run_workflow(label, goal, inputs, prompt, i, m)


def workflow_meeting_notes():
    print(f"\n{DIVIDER}")
    print("WORKFLOW 3: Meeting Notes & Action Item Extractor")
    print("(Processes meeting transcripts into structured summaries)")
    print(DIVIDER)

    m = _fresh_mnemon("sim_meetings")
    scenarios = [
        (
            "product roadmap meeting",
            "Summarize meeting transcript and extract action items",
            {"meeting_type": "roadmap", "attendees": 6, "duration_min": 45},
            "Meeting: Q3 product roadmap. Discussed feature prioritization for next quarter. "
            "Alice to own design spec. Bob handles data pipeline. Ship date: end of August. "
            "Extract action items and write summary.",
        ),
        (
            "incident retrospective",
            "Extract action items and key decisions from meeting notes",
            {"meeting_type": "retro", "attendees": 4, "duration_min": 30},
            "Incident retro for the June 1 outage. Root cause: DB connection pool exhausted. "
            "Fix: increase pool size, add alerting. Carol owns runbook update. "
            "Dave owns monitoring dashboard. Extract action items.",
        ),
        (
            "weekly team sync — similar to roadmap",
            "Summarize team meeting and list follow-ups",
            {"meeting_type": "sync", "attendees": 5, "duration_min": 30},
            "Weekly sync: reviewed sprint progress, discussed upcoming launch. "
            "Eve to coordinate with marketing on launch date. Frank to finish QA by Thursday. "
            "Write summary and action items.",
        ),
    ]

    for i, (label, goal, inputs, prompt) in enumerate(scenarios, 1):
        run_workflow(label, goal, inputs, prompt, i, m)


def workflow_code_review():
    print(f"\n{DIVIDER}")
    print("WORKFLOW 4: Automated Code Review")
    print("(Reviews PRs for bugs, security issues, style)")
    print(DIVIDER)

    m = _fresh_mnemon("sim_codereview")
    scenarios = [
        (
            "Python backend PR — auth changes",
            "Review code changes for bugs and security issues",
            {"pr_number": 341, "language": "python", "repo": "api-server"},
            "Diff: modified user_auth.py — added JWT validation. "
            "Line 47: token = query_params['token'] + user_id (string concat). "
            "Line 83: if user_id > 0: (no type check). Review for issues.",
        ),
        (
            "JavaScript frontend PR — form handling",
            "Check pull request for security vulnerabilities and code quality",
            {"pr_number": 205, "language": "javascript", "repo": "web-app"},
            "Diff: modified LoginForm.jsx — updated form submission. "
            "Line 23: fetch('/api/login?token=' + userInput) (XSS risk). "
            "Line 41: eval(response.data) (dangerous). Review for issues.",
        ),
        (
            "Python PR again — different service",
            "Audit code diff for bugs and potential security problems",
            {"pr_number": 347, "language": "python", "repo": "worker-service"},
            "Diff: modified task_processor.py — added DB migration logic. "
            "Line 12: cursor.execute('DELETE FROM tasks WHERE id=' + task_id). "
            "Line 58: password stored in plaintext log. Review for issues.",
        ),
    ]

    for i, (label, goal, inputs, prompt) in enumerate(scenarios, 1):
        run_workflow(label, goal, inputs, prompt, i, m)


def workflow_invoice_extraction():
    print(f"\n{DIVIDER}")
    print("WORKFLOW 5: Invoice Data Extraction")
    print("(Pulls structured fields from vendor invoices for accounting)")
    print(DIVIDER)

    m = _fresh_mnemon("sim_invoices")
    scenarios = [
        (
            "SaaS vendor invoice",
            "Extract structured data from vendor invoice for accounting",
            {"vendor_type": "saas", "currency": "USD"},
            "Invoice text: Acme Cloud Services. Invoice #INV-2026-04821. "
            "Date: May 28 2026. Item: Cloud storage 1TB — $299/month. "
            "Total due: $299.00. Due June 27. Extract all fields as JSON.",
        ),
        (
            "contractor invoice",
            "Parse invoice document and output structured fields",
            {"vendor_type": "contractor", "currency": "USD"},
            "Invoice: Jane Dev Consulting. Invoice #JD-0042. Date: June 1 2026. "
            "Services: Backend API development, 20hrs @ $150/hr = $3,000. "
            "Total: $3,000. Net 30. Extract all fields as JSON.",
        ),
        (
            "another SaaS invoice — different vendor",
            "Extract billing information from invoice for our records",
            {"vendor_type": "saas", "currency": "USD"},
            "Invoice from DataPipeline Pro. Invoice #DP-2026-1134. May 31 2026. "
            "Seat licenses x 5 @ $49/seat = $245. Total: $245.00. Due June 30. "
            "Extract all fields as JSON.",
        ),
    ]

    for i, (label, goal, inputs, prompt) in enumerate(scenarios, 1):
        run_workflow(label, goal, inputs, prompt, i, m)


# ── Main runner ───────────────────────────────────────────────────────────────

def main():
    # Pre-load sentence-transformers in the main thread so concurrent prewarm
    # threads don't race on _ST_MODEL_CACHE and cause "meta tensor" errors.
    print("Loading embedder...", end=" ", flush=True)
    from mnemon.core.embedder import _try_load_sentence_transformers
    _try_load_sentence_transformers()
    print("ready.")

    print("=" * 60)
    print("MNEMON USER SIMULATION — 5 REAL WORKFLOWS")
    print(f"Mode: {'REAL LLM (Anthropic)' if REAL_LLM else 'mock LLM'}")
    print(f"Mnemon version: {mnemon.__version__}")
    print("=" * 60)
    print("\nWhat you're watching:")
    print("  - First run of each scenario = cache miss (LLM called)")
    print("  - Later runs = hopefully cache hits (System 1 or System 2)")
    print("  - Pay attention to what Mnemon prints on each run")
    print()

    t_start = time.time()

    workflow_support_triage()
    workflow_standup_generator()
    workflow_meeting_notes()
    workflow_code_review()
    workflow_invoice_extraction()

    total_s = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"All workflows complete in {total_s:.1f}s")
    print("=" * 60)
    print()
    print("Check above for:")
    print("  - Which runs hit System 2 (semantic cache)")
    print("  - What messages Mnemon printed to stderr")
    print("  - Any confusion points a real user would hit")


if __name__ == "__main__":
    main()
