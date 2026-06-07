"""
Mnemon Real Workflow Gauntlet
==============================
Uses ACTUAL system prompts from real open-source production agents.
Not synthetic — these are the real prompts copied verbatim.

Sources:
  SWE-agent:    github.com/SWE-agent/SWE-agent  (NeurIPS 2024, 79% SWE-bench)
  ReAct agent:  github.com/langchain-ai/react-agent

Scenarios
---------
1. SWE-agent debugging loop   — 12 issues × 6 steps = 72 calls
   Real: 5-20 LLM calls per GitHub issue. Same patterns repeat across issues
   of the same error class (AttributeError, KeyError, ImportError, etc.)

2. ReAct research agent       — 5 topics × 3 phrasings × 3 steps = 45 calls
   Real: researcher tool-use loop, 2-5 calls per query.
   Same topic asked in different ways by different team members.

3. Code assistant loop        — 8 tasks × 4 calls = 32 calls
   Real: Copilot/Cursor-style agent called repeatedly for similar coding tasks.
   Explain code, generate tests, refactor, add types — patterns repeat heavily.

Total: 149 calls. Every hit = tokens not sent to OpenAI/Anthropic.
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
# ─────────────────────────────────────────────────────────────────────────────

_llm_calls: list = []
_current_scenario: str = "init"
_current_call_type: str = "general"


def _tok(t: str) -> int:
    return max(1, len(t) // 4)


def _synth(text: str, model: str, inp: int, out: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        id=f"real-{len(_llm_calls):04d}", type="message", role="assistant", model=model,
        content=[types.SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=types.SimpleNamespace(input_tokens=inp, output_tokens=out),
    )


_RESPONSES = [
    # SWE-agent: find relevant file
    (r"find.*relevant|locate.*file|search.*code|which.*file|where.*implement",
     "I'll start by searching for the relevant code.\n"
     "```bash\nfind . -name '*.py' | xargs grep -l '{keyword}' | head -20\n```"),
    # SWE-agent: reproduce error
    (r"reproduc|script.*error|python.*error|run.*test|confirm.*error",
     "Let me create a reproduction script.\n"
     "```python\nimport sys\nsys.path.insert(0, '.')\n# Reproduce the issue\ntry:\n"
     "    from module import function\n    result = function(test_input)\nexcept Exception as exc:\n"
     "    print('Error:', exc)\n```\n"
     "OBSERVATION: AttributeError: 'NoneType' object has no attribute 'value'"),
    # SWE-agent: analyze error
    (r"analyz|root.cause|understand.*error|traceback|stack.trace|why.*fail",
     "The error occurs because the object is None before attribute access. "
     "Looking at the code flow:\n1. `parse()` can return None when input is empty\n"
     "2. The caller doesn't check for None before accessing `.value`\n"
     "The fix: add a None check before attribute access."),
    # SWE-agent: apply fix
    (r"fix|edit|patch|modif|change.*code|implement.*solution|apply",
     "Applying the minimal fix:\n"
     "```python\n# Before:\nresult = obj.value\n\n# After:\nresult = obj.value if obj is not None else default\n```\n"
     "This handles the edge case without breaking existing behaviour."),
    # SWE-agent: verify fix
    (r"verif|confirm.*fix|test.*pass|run.*again|check.*work|does.*work",
     "Running the reproduction script again.\n"
     "```bash\npython reproduce.py\n```\n"
     "OBSERVATION: Script ran successfully and produced expected output. "
     "No AttributeError raised. Fix confirmed."),
    # SWE-agent: edge cases
    (r"edge.case|corner.case|handle.*case|what.*if|additional|other.*scenario",
     "Edge cases to consider:\n"
     "1. Empty input → returns default value ✓\n"
     "2. None input → handled by None check ✓\n"
     "3. List input → type check needed, added\n"
     "4. Unicode input → no change needed, str handles it\n"
     "All edge cases handled. Minimal change, no test modifications."),
    # ReAct: research/think
    (r"market.size|market.*ai|ai.*market|industry.*size|how.big|revenue.*ai|billion.*ai",
     "Thought: I need to find recent market data for AI developer tools.\n"
     "Action: search\n"
     "Action Input: 'AI coding assistant market size 2025 2026'\n"
     "Observation: The AI coding tools market was valued at $4.8B in 2024, "
     "projected to reach $12.6B by 2027 (CAGR 37%). GitHub Copilot has 1.8M paying "
     "subscribers. Key players: GitHub, Cursor, Tabnine, Amazon CodeWhisperer."),
    (r"competi|competitor|versus|compare|vs\.|alternative|other.*product",
     "Thought: I need competitive landscape information.\n"
     "Action: search\n"
     "Action Input: 'AI code assistant competitors comparison 2026'\n"
     "Observation: GitHub Copilot leads with 1.8M subscribers at $19/month. "
     "Cursor growing fastest (500K users, $20/month). Tabnine enterprise-focused. "
     "Amazon CodeWhisperer free tier popular. Key differentiator: context window size."),
    (r"customer|user|who.*use|target.*market|adopt|developer.*use",
     "Thought: I should identify the target customer profile.\n"
     "Action: search\n"
     "Action Input: 'who uses AI coding assistants enterprise survey 2026'\n"
     "Observation: 72% of enterprise developers now use AI coding tools (Stack Overflow 2026). "
     "Adoption highest in fintech (84%) and cloud-native (89%). Main blockers: "
     "security concerns (41%), code quality trust (38%), cost (21%)."),
    (r"price|pricing|cost|monetiz|revenue|business.model|subscription",
     "Thought: Pricing strategy analysis needed.\n"
     "Action: search\n"
     "Action Input: 'AI developer tools pricing models SaaS 2026'\n"
     "Observation: Most AI dev tools use per-seat pricing ($10-30/user/month). "
     "Enterprise deals: $50-150/seat with SSO, audit logs, private models. "
     "Usage-based pricing growing — Cursor moved to hybrid model Q1 2026."),
    (r"trend|future|predict|forecast|growth|next.*year|roadmap",
     "Thought: Future trends in AI coding tools.\n"
     "Action: search\n"
     "Action Input: 'AI coding tools trends 2026 2027 future'\n"
     "Observation: Top trends: (1) multi-file context becoming standard, "
     "(2) agent-based code execution replacing suggestion-only, "
     "(3) private model fine-tuning for enterprise, "
     "(4) integration with CI/CD pipelines, (5) voice-to-code gaining traction."),
    # Code assistant: explain
    (r"explain.*code|what.*does|how.*work|describe.*function|understand.*code",
     "This function implements a retry mechanism with exponential backoff. "
     "It takes a callable, max_attempts (default 3), and base_delay (default 1s). "
     "On each failure it waits base_delay * 2^attempt seconds before retrying. "
     "Raises the last exception if all attempts fail. Thread-safe."),
    # Code assistant: generate tests
    (r"generat.*test|write.*test|unit.*test|test.*case|pytest|unittest",
     "```python\nimport pytest\nfrom unittest.mock import patch, MagicMock\n\n"
     "def test_happy_path():\n    result = function_under_test(valid_input)\n"
     "    assert result == expected_output\n\n"
     "def test_edge_case_empty():\n    result = function_under_test(None)\n"
     "    assert result is None\n\n"
     "def test_raises_on_invalid():\n    with pytest.raises(ValueError):\n"
     "        function_under_test(invalid_input)\n```"),
    # Code assistant: refactor
    (r"refactor|clean.*up|improve.*code|simplif|restructur|better.*way",
     "Refactored version:\n"
     "1. Extracted magic numbers to named constants\n"
     "2. Split 80-line function into 3 focused functions\n"
     "3. Replaced nested ifs with early returns\n"
     "4. Added type hints throughout\n"
     "Functionality unchanged. All existing tests pass."),
    # Code assistant: add types
    (r"type.hint|typing|annotate|mypy|strict.*type|add.*type",
     "```python\nfrom typing import Optional, List, Dict, Union\n\n"
     "def process(\n    items: List[Dict[str, str]],\n    config: Optional[Dict] = None,\n"
     "    max_items: int = 100,\n) -> List[str]:\n    ...\n```\n"
     "All parameters and return types annotated. mypy passes with strict mode."),
]


def _fake_llm_create(_self, *, messages, model, system=None, **kwargs):
    user_msg = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
    )
    combined = ((system or "") + " " + user_msg).lower()

    response_text = f"Processing: {user_msg[:80]}. Step complete."
    for pattern, template in _RESPONSES:
        if re.search(pattern, combined):
            response_text = template.replace("{keyword}", user_msg[:20])
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
_tmpdir = tempfile.mkdtemp(prefix="mnemon_real_")
_mnemon_mod._instance = None

_M = _mnemon_mod.init(
    tenant_id="real_workflows",
    db_dir=_tmpdir,
    silent=True,
    prewarm_fragments=False,
    prewarm_templates=False,
    enable_telemetry=False,
)
if _M._patch_thread and _M._patch_thread.is_alive():
    _M._patch_thread.join(timeout=15)

from anthropic import Anthropic as _Anthropic
_client = _Anthropic(api_key="fake-key-real")
_MODEL = "claude-3-5-sonnet-20241022"


def _call(system_prompt: str, user_message: str, scenario: str, call_type: str):
    global _current_scenario, _current_call_type
    _current_scenario = scenario
    _current_call_type = call_type
    return _client.messages.create(
        model=_MODEL,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=1024,
    )


def _misses(scenario: str) -> int:
    return sum(1 for c in _llm_calls if c["scenario"] == scenario)


def _tokens(scenario: str) -> int:
    return sum(c["input_tokens"] + c["output_tokens"]
               for c in _llm_calls if c["scenario"] == scenario)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 1 — SWE-AGENT DEBUGGING LOOP
# Real system prompt from github.com/SWE-agent/SWE-agent (verbatim)
# 12 GitHub issues × 6 steps = 72 calls
# Issues cluster into 4 error types × 3 repos — patterns repeat
# ─────────────────────────────────────────────────────────────────────────────

# VERBATIM from SWE-agent config/default.yaml
_SWE_SYSTEM = "You are a helpful assistant that can interact with a computer to solve tasks."

# 12 issues: 4 error types × 3 different repos/contexts
_ISSUES = [
    # AttributeError (3 issues in different repos)
    ("django/django", "AttributeError: 'NoneType' object has no attribute 'name' in QuerySet.filter()",
     "django", "models.py"),
    ("pallets/flask", "AttributeError: 'NoneType' object has no attribute 'headers' in request handler",
     "flask", "wrappers.py"),
    ("psf/requests", "AttributeError: 'NoneType' object has no attribute 'encode' in PreparedRequest",
     "requests", "models.py"),
    # KeyError (3 issues)
    ("huggingface/transformers", "KeyError: 'attention_mask' in tokenizer output when padding=False",
     "transformers", "tokenization_utils.py"),
    ("pandas-dev/pandas", "KeyError: column not found after merge when using suffixes",
     "pandas", "merge.py"),
    ("scikit-learn/scikit-learn", "KeyError: 'feature_names_in_' on model fitted before sklearn 1.0",
     "sklearn", "base.py"),
    # ImportError (3 issues)
    ("numpy/numpy", "ImportError: cannot import name 'bool8' from numpy — deprecated in 1.24",
     "numpy", "__init__.py"),
    ("pytorch/pytorch", "ImportError: DLL load failed while importing _C — CUDA version mismatch",
     "torch", "__init__.py"),
    ("langchain-ai/langchain", "ImportError: langchain.llms is deprecated, use langchain_community",
     "langchain", "llms/__init__.py"),
    # TypeError (3 issues)
    ("tiangolo/fastapi", "TypeError: Object of type UUID is not JSON serializable in response model",
     "fastapi", "encoders.py"),
    ("pydantic/pydantic", "TypeError: 'NoneType' is not a valid field type in v2 migration",
     "pydantic", "fields.py"),
    ("sqlalchemy/sqlalchemy", "TypeError: Argument 'table' is not an instance of Table in 2.0 style",
     "sqlalchemy", "schema.py"),
]

_SWE_STEPS = [
    ("find relevant", "Find the relevant code files for this issue: {issue}. Working dir: {repo}"),
    ("reproduce error", "Create a script to reproduce this error: {issue}. Target file: {file}"),
    ("analyze error", "Analyze the root cause of this error based on the traceback: {issue}"),
    ("apply fix", "Apply the minimal fix to resolve: {issue} in {file}"),
    ("verify fix", "Verify the fix works — run the reproduction script again for: {issue}"),
    ("edge cases", "Check edge cases for the fix applied to: {issue} in {repo}"),
]


def run_sweagent_scenario() -> dict:
    total_calls = 0
    for repo, issue, pkg, file in _ISSUES:
        for step_name, step_template in _SWE_STEPS:
            user_msg = step_template.format(issue=issue, repo=repo, file=file, pkg=pkg)
            _call(_SWE_SYSTEM, user_msg, "sweagent", step_name.replace(" ", "_"))
            total_calls += 1

    misses = _misses("sweagent")
    tokens = _tokens("sweagent")
    hits = total_calls - misses
    return {
        "scenario": "SWE-agent Debugging Loop (real prompts)",
        "total_calls": total_calls,
        "llm_calls_made": misses,
        "cache_hits": hits,
        "hit_rate": hits / total_calls,
        "tokens_spent": tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 2 — REACT RESEARCH AGENT
# Real system prompt from github.com/langchain-ai/react-agent (verbatim)
# 5 topics × 3 phrasings × 3 reasoning steps = 45 calls
# Same question asked differently by different team members
# ─────────────────────────────────────────────────────────────────────────────

# VERBATIM from react-agent prompts.py (system_time injected at runtime)
_REACT_SYSTEM = "You are a helpful AI assistant.\n\nSystem time: 2026-06-01T10:00:00Z"

# 5 topics, each asked 3 different ways (realistic: different team members, different days)
_RESEARCH_QUERIES = [
    # Market size — 3 phrasings
    ("market_size", "What is the current market size of AI coding assistant tools?"),
    ("market_size", "How big is the AI code completion and developer tools market in 2026?"),
    ("market_size", "What's the total addressable market for AI-powered developer productivity tools?"),
    # Competition — 3 phrasings
    ("competition", "Who are the main competitors in the AI coding assistant space?"),
    ("competition", "Compare GitHub Copilot vs Cursor vs other AI coding tools"),
    ("competition", "What companies are competing in AI developer tools and what's their market share?"),
    # Customer profile — 3 phrasings
    ("customers", "Who are the primary users of AI coding assistants in enterprise?"),
    ("customers", "What types of developers and companies are adopting AI coding tools?"),
    ("customers", "What is the customer profile for AI developer productivity tools?"),
    # Pricing — 3 phrasings
    ("pricing", "How are AI coding assistant tools priced? What's the typical subscription model?"),
    ("pricing", "What pricing strategies do AI developer tools companies use?"),
    ("pricing", "How much do companies charge for AI coding tools — per seat, usage-based, or flat?"),
    # Trends — 3 phrasings
    ("trends", "What are the emerging trends in AI coding tools for 2026-2027?"),
    ("trends", "Where is the AI developer tools market heading in the next 2 years?"),
    ("trends", "What new capabilities are AI coding assistants adding in 2026?"),
]

_REACT_STEPS = [
    "Think and search: {query}",
    "Synthesize findings for: {query}",
    "Write final answer for: {query}",
]


def run_react_scenario() -> dict:
    total_calls = 0
    for topic, query in _RESEARCH_QUERIES:
        for step_template in _REACT_STEPS:
            user_msg = step_template.format(query=query)
            _call(_REACT_SYSTEM, user_msg, "react", topic)
            total_calls += 1

    misses = _misses("react")
    tokens = _tokens("react")
    hits = total_calls - misses
    return {
        "scenario": "ReAct Research Agent (real prompts)",
        "total_calls": total_calls,
        "llm_calls_made": misses,
        "cache_hits": hits,
        "hit_rate": hits / total_calls,
        "tokens_spent": tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 3 — CODE ASSISTANT LOOP
# Modelled on Copilot/Cursor usage patterns
# 8 coding tasks × 4 call types = 32 calls
# Same tasks repeat constantly across a dev team
# ─────────────────────────────────────────────────────────────────────────────

_CODE_ASSISTANT_SYSTEM = (
    "You are an expert software engineer and coding assistant. "
    "You help developers understand, test, refactor, and improve code. "
    "Be concise and precise. Output code in fenced blocks."
)

# 8 different functions developers ask about — realistic team usage
_CODE_TASKS = [
    ("retry_logic",
     "def retry(fn, max_attempts=3, delay=1):\n    for i in range(max_attempts):\n        try: return fn()\n        except: time.sleep(delay * 2**i)\n    raise"),
    ("rate_limiter",
     "class RateLimiter:\n    def __init__(self, max_calls, period):\n        self.calls = []\n    def __call__(self, fn): pass"),
    ("connection_pool",
     "class Pool:\n    def __init__(self, factory, size=10):\n        self._pool = [factory() for _ in range(size)]\n    def get(self): return self._pool.pop()"),
    ("cache_decorator",
     "def ttl_cache(ttl=300):\n    cache = {}\n    def decorator(fn):\n        def wrapper(*args):\n            if args in cache and time.time() < cache[args][1]: return cache[args][0]\n            result = fn(*args)\n            cache[args] = (result, time.time() + ttl)\n            return result\n        return wrapper\n    return decorator"),
    ("event_emitter",
     "class EventEmitter:\n    def __init__(self): self._handlers = {}\n    def on(self, event, handler): self._handlers.setdefault(event, []).append(handler)\n    def emit(self, event, *args): [h(*args) for h in self._handlers.get(event, [])]"),
    ("circuit_breaker",
     "class CircuitBreaker:\n    CLOSED, OPEN, HALF_OPEN = 'closed', 'open', 'half_open'\n    def __init__(self, threshold=5, timeout=60): self.state = self.CLOSED"),
    ("pagination_helper",
     "def paginate(query, page=1, per_page=20):\n    total = query.count()\n    items = query.offset((page-1)*per_page).limit(per_page).all()\n    return {'items': items, 'total': total, 'pages': ceil(total/per_page)}"),
    ("webhook_handler",
     "def handle_webhook(payload, secret):\n    sig = hmac.new(secret.encode(), payload, sha256).hexdigest()\n    if not hmac.compare_digest(sig, request.headers.get('X-Signature', '')): abort(403)"),
]

_CODE_CALL_TYPES = [
    ("explain", "Explain what this code does and how it works:\n\n```python\n{code}\n```"),
    ("test", "Generate comprehensive pytest unit tests for:\n\n```python\n{code}\n```"),
    ("refactor", "Refactor this code for clarity and best practices:\n\n```python\n{code}\n```"),
    ("types", "Add complete type hints to this code:\n\n```python\n{code}\n```"),
]


def run_code_assistant_scenario() -> dict:
    total_calls = 0
    for task_name, code in _CODE_TASKS:
        for call_name, call_template in _CODE_CALL_TYPES:
            user_msg = call_template.format(code=code)
            _call(_CODE_ASSISTANT_SYSTEM, user_msg, "codeassist", call_name)
            total_calls += 1

    misses = _misses("codeassist")
    tokens = _tokens("codeassist")
    hits = total_calls - misses
    return {
        "scenario": "Code Assistant Loop (Copilot/Cursor style)",
        "total_calls": total_calls,
        "llm_calls_made": misses,
        "cache_hits": hits,
        "hit_rate": hits / total_calls,
        "tokens_spent": tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _bar(rate: float, width: int = 22) -> str:
    filled = int(rate * width)
    return "█" * filled + "░" * (width - filled)


def print_report(results: list) -> None:
    total_calls = sum(r["total_calls"] for r in results)
    total_hits = sum(r["cache_hits"] for r in results)
    overall_rate = total_hits / total_calls if total_calls else 0
    total_tokens = sum(r["tokens_spent"] for r in results)
    baseline = sum(
        r["tokens_spent"] / r["llm_calls_made"] * r["total_calls"]
        if r["llm_calls_made"] > 0 else 0
        for r in results
    )
    tokens_saved = max(0, int(baseline - total_tokens))

    W = 70
    print()
    print("=" * W)
    print("  MNEMON REAL WORKFLOW GAUNTLET — RESULTS")
    print("  (Real system prompts from SWE-agent + LangChain ReAct)")
    print("=" * W)
    print()
    print(f"  {'Scenario':<38} {'Calls':>6} {'Hits':>5} {'Hit%':>6}  Bar")
    print(f"  {'-'*38} {'-'*6} {'-'*5} {'-'*6}  {'-'*22}")
    for r in results:
        print(
            f"  {r['scenario']:<38} {r['total_calls']:>6} "
            f"{r['cache_hits']:>5} {r['hit_rate']:>5.1%}  {_bar(r['hit_rate'])}"
        )
    print(f"  {'-'*38} {'-'*6} {'-'*5} {'-'*6}  {'-'*22}")
    print(f"  {'TOTAL':<38} {total_calls:>6} {total_hits:>5} {overall_rate:>5.1%}  {_bar(overall_rate)}")

    print()
    print(f"  Tokens spent  (with Mnemon): {total_tokens:>10,}")
    print(f"  Tokens saved  (vs no cache): {tokens_saved:>10,}")
    print()

    print("  What these hit rates mean in production:")
    print()
    print("  SWE-agent at 1,000 issues/month ($2/issue = $2,000/month):")
    sweagent_rate = next(r["hit_rate"] for r in results if "SWE" in r["scenario"])
    print(f"    {sweagent_rate:.1%} hit rate → ${2000 * sweagent_rate:,.0f}/month saved")
    print()
    print("  Research agent at 10,000 queries/month ($0.05/query = $500/month):")
    react_rate = next(r["hit_rate"] for r in results if "ReAct" in r["scenario"])
    print(f"    {react_rate:.1%} hit rate → ${500 * react_rate:,.0f}/month saved")
    print()
    print("  Code assistant at 100,000 calls/month ($0.002/call = $200/month):")
    code_rate = next(r["hit_rate"] for r in results if "Code" in r["scenario"])
    print(f"    {code_rate:.1%} hit rate → ${200 * code_rate:,.0f}/month saved")
    print()
    print("=" * W)
    print()


def run_gauntlet() -> list:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print()
    print("Mnemon Real Workflow Gauntlet")
    print("System prompts from SWE-agent (NeurIPS 2024) and LangChain ReAct — verbatim.")
    print()

    results = []
    scenarios = [
        ("1. SWE-agent Debugging Loop  ", run_sweagent_scenario),
        ("2. ReAct Research Agent      ", run_react_scenario),
        ("3. Code Assistant Loop       ", run_code_assistant_scenario),
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


def test_real_workflows_run():
    results = run_gauntlet()
    assert len(results) == 3
    for r in results:
        assert r["total_calls"] > 0
        assert 0.0 <= r["hit_rate"] <= 1.0


if __name__ == "__main__":
    run_gauntlet()
