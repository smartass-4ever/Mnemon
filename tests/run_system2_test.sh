#!/usr/bin/env bash
# System 2 test — separate process per run (real user behaviour)
set -e
cd "$(dirname "$0")/.."

echo "========================================================"
echo "SYSTEM 2 TEST — PR Review (separate processes)"
echo "========================================================"

rm -f mnemon_tenant_pr_review_agent.db \
       mnemon_system_pr_review_agent.db \
       mnemon_signal_pr_review_agent.db \
       mnemon_stats_pr_review_agent.json \
       mnemon_bus_pr_review_agent.json 2>/dev/null || true

echo ""
echo "--- Run 1: cold first run (miss expected) ---"
python tests/pr_review_workflow.py --repo "acme/api" --pr 101 --lang python --variant 0

echo ""
echo "--- Run 2: same pipeline, next PR, same goal (system1 or system2 expected) ---"
python tests/pr_review_workflow.py --repo "acme/api" --pr 102 --lang python --variant 0

echo ""
echo "--- Run 3: same pipeline, different goal phrasing (system2 expected) ---"
python tests/pr_review_workflow.py --repo "acme/api" --pr 103 --lang python --variant 1

echo ""
echo "--- Run 4: different repo+lang, rephrased goal (system2 expected) ---"
python tests/pr_review_workflow.py --repo "acme/web" --pr 55 --lang javascript --variant 2

echo ""
echo "--- Run 5: exact repeat of Run 2 (system1 expected) ---"
python tests/pr_review_workflow.py --repo "acme/api" --pr 102 --lang python --variant 0

echo ""
echo "========================================================"
echo "Expected: miss | system1/2 | system2 | system2 | system1"
echo "========================================================"
