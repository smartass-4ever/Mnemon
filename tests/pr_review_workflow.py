"""
PR Review Agent — System 2 test using Path 2 (m.run()).

Simulates a real PR review pipeline run as separate process invocations.
Each run is a fresh Python process — tests real cross-session caching.

Usage:
    python tests/pr_review_workflow.py --repo "acme/api" --pr 101 --lang python
    python tests/pr_review_workflow.py --repo "acme/api" --pr 102 --lang python
    python tests/pr_review_workflow.py --repo "acme/web" --pr 55  --lang javascript
"""

import argparse
import os
import sys
import time

# Force local dev package over site-packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Pre-load sentence-transformers in main thread before Mnemon init
# so prewarm threads don't race on the model cache
from mnemon.core.embedder import _try_load_sentence_transformers
_try_load_sentence_transformers()

import mnemon
m = mnemon.init(tenant_id="pr_review_agent", db_dir=".")


# ── Mock review logic (simulates real LLM calls) ──────────────────────────────

MOCK_DIFFS = {
    "python": (
        "diff --git a/auth/views.py\n"
        "+    query = 'SELECT * FROM users WHERE id=' + user_id\n"
        "+    def validate_token(self, token):\n"
    ),
    "javascript": (
        "diff --git a/src/LoginForm.jsx\n"
        "+    const res = await fetch('/api/login?token=' + userInput);\n"
        "+    eval(await res.text());\n"
    ),
    "go": (
        "diff --git a/internal/handler/user.go\n"
        "+    row := db.QueryRow('SELECT * FROM users WHERE id = ' + id)\n"
        "+    log.Printf('Password: %s', user.Password)\n"
    ),
}


def do_review(repo: str, pr_number: int, language: str) -> str:
    """Full PR review — 3 sequential LLM calls (fetch, review, summarise)."""
    diff = MOCK_DIFFS.get(language, MOCK_DIFFS["python"])

    # Node 1: fetch + static scan (fast)
    time.sleep(0.2)
    issues = []
    if "SELECT" in diff and "+" in diff:
        issues.append("CRITICAL: SQL injection — use parameterised queries")
    if "eval(" in diff:
        issues.append("CRITICAL: eval() on untrusted input — RCE risk")
    if "token=" in diff:
        issues.append("HIGH: auth token in query string — use Authorization header")
    if "Password" in diff and "log" in diff.lower():
        issues.append("HIGH: password logged in plaintext")

    # Node 2: LLM deep review (slow)
    time.sleep(1.8)
    verdict = "REQUEST CHANGES" if any("CRITICAL" in i for i in issues) else "APPROVE"

    # Node 3: summarise
    time.sleep(0.3)
    lines = [f"PR #{pr_number} -- {repo}  [{language}]", f"Verdict: {verdict}", ""]
    lines += issues or ["No issues found."]
    return "\n".join(lines)


# ── Mnemon-wrapped run ────────────────────────────────────────────────────────

GOAL_VARIANTS = [
    "Review pull request for security vulnerabilities and code quality issues",
    "Analyse PR for bugs, security problems and style issues",
    "Check pull request code for vulnerabilities and code review feedback",
    "Perform automated code review on pull request for security and quality",
]


def run(repo: str, pr_number: int, language: str, goal_variant: int = 0) -> dict:
    goal = GOAL_VARIANTS[goal_variant % len(GOAL_VARIANTS)]

    def generation_fn(g, inputs, ctx, caps, constraints):
        return do_review(inputs["repo"], inputs["pr_number"], inputs["language"])

    return m.run(
        goal=goal,
        inputs={"repo": repo, "pr_number": pr_number, "language": language},
        generation_fn=generation_fn,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo",    default="acme/api-server")
    parser.add_argument("--pr",      type=int, default=101)
    parser.add_argument("--lang",    default="python")
    parser.add_argument("--variant", type=int, default=0,
                        help="Goal phrasing variant 0-3")
    args = parser.parse_args()

    t0 = time.time()
    result = run(args.repo, args.pr, args.lang, args.variant)
    elapsed = (time.time() - t0) * 1000

    level = result.get("cache_level", "miss")
    tokens = result.get("tokens_saved", 0)
    label = {
        "system1": "SYSTEM 1 HIT (exact)",
        "system2": "SYSTEM 2 HIT (semantic)",
        "miss":    "cache miss",
    }.get(level, level)

    print(f"\nPR #{args.pr} | {args.repo} | {args.lang} | goal variant {args.variant}")
    print(f"Result : {label}")
    print(f"Elapsed: {elapsed:.0f}ms", end="")
    if tokens:
        print(f"  |  {tokens} tokens saved", end="")
    print()
    if result.get("output") and level == "miss":
        print(f"\n{result['output']}")


if __name__ == "__main__":
    main()
