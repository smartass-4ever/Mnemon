"""
benchmark_longmemeval.py
========================
Evaluates Mnemon's protein bond memory retrieval on the real LongMemEval
benchmark (ICLR 2025) — 100% external data, nothing made by us.

Dataset: xiaowu0162/LongMemEval  (HuggingFace / ICLR 2025)
Paper:   https://arxiv.org/abs/2410.10813

What this tests:
  Given 30-40 sessions of real conversation history ingested into Mnemon,
  can protein bond retrieval surface the right memory when asked a question
  whose answer is buried in those sessions?

Question types (from the paper):
  single-session-user        — fact stated explicitly by the user
  single-session-assistant   — fact from an assistant response
  single-session-preference  — user preference that must be remembered
  temporal-reasoning         — what changed between two sessions
  knowledge-update           — fact that was corrected/updated later
  multi-session              — requires connecting info across 2+ sessions

Scoring (mirrors the paper's LLM-as-judge methodology, keyword variant):
  YES  if >= JUDGE_THRESHOLD fraction of ground-truth answer keywords
       appear in the text Mnemon retrieved
  NO   otherwise

Baseline comparison:
  We also score a naive "retrieve nothing" baseline and a "retrieve random"
  baseline so the Mnemon numbers have honest context.

Usage:
  python benchmark_longmemeval.py              # auto-downloads data, runs all questions
  python benchmark_longmemeval.py --n 100      # first 100 questions only
  python benchmark_longmemeval.py --variant s  # _s variant (default, ~40 sessions each)
  python benchmark_longmemeval.py --no-download  # skip download, use cached data
"""

import argparse
import asyncio
import hashlib
import io
import json
import logging
import os
import random
import re
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# UTF-8 on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

# Silence Mnemon internals
logging.disable(logging.WARNING)

from mnemon.core.memory import CognitiveMemorySystem, SimpleEmbedder
from mnemon.core.models import ExperienceSignal, MemoryLayer, SignalType
from mnemon.core.persistence import EROSDatabase, InvertedIndex
from mnemon.llm.client import MockLLMClient

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_THRESHOLD = 0.60          # >= 60% keywords must match → YES
DATA_DIR        = Path("data/longmemeval")
REPORT_DIR      = Path("reports")

# HuggingFace URLs — cleaned dataset (original is deprecated)
_HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
_VARIANTS = {
    "s":      f"{_HF_BASE}/longmemeval_s_cleaned.json",     # 277 MB, ~40 sessions each (includes filler)
    "oracle": f"{_HF_BASE}/longmemeval_oracle.json",        #  15 MB, evidence sessions only (easier)
}

BOLD  = "\033[1m"
GREEN = "\033[92m"
BLUE  = "\033[94m"
YELLOW= "\033[93m"
RED   = "\033[91m"
DIM   = "\033[2m"
RESET = "\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_dataset(variant: str = "s") -> Path:
    """
    Download LongMemEval from HuggingFace if not already cached.
    Returns path to the JSON file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "oracle" if variant == "oracle" else f"{variant}_cleaned"
    dest = DATA_DIR / f"longmemeval_{suffix}.json"

    if dest.exists():
        print(f"  {DIM}Dataset cached: {dest}{RESET}")
        return dest

    url = _VARIANTS[variant]
    print(f"  Downloading LongMemEval_{variant.upper()} from HuggingFace...")
    print(f"  {DIM}{url}{RESET}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "mnemon-benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 65536
            buf = b""
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                buf += chunk
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    mb  = downloaded / 1_048_576
                    print(f"\r  {mb:.1f} MB / {total/1_048_576:.1f} MB  ({pct:.0f}%)", end="", flush=True)
        print()
        dest.write_bytes(buf)
        print(f"  Saved to {dest}")
        return dest
    except Exception as e:
        print(f"\n  {RED}Download failed: {e}{RESET}")
        print(f"  Manual download: {url}")
        print(f"  Save to: {dest}")
        sys.exit(1)


def load_dataset(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # The dataset may be a list directly or wrapped in a dict
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return list(data.values()) if isinstance(data, dict) else []


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, length >= 3, stop-words removed."""
    _STOP = {
        "the","and","for","that","this","with","are","was","were","have",
        "has","had","not","but","from","they","what","when","where","who",
        "will","would","could","should","which","been","also","its","their",
        "there","then","than","into","onto","upon","about","after","before",
        "during","while","since","until","between","through","within",
    }
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 3 and t not in _STOP]


def keyword_judge(retrieved_text: str, ground_truth) -> Tuple[bool, float]:
    """
    Returns (YES/NO, coverage_fraction).
    YES = fraction of ground-truth keywords found in retrieved text >= JUDGE_THRESHOLD.
    """
    gt_tokens   = set(_tokenise(str(ground_truth)))
    ret_tokens  = set(_tokenise(retrieved_text))
    if not gt_tokens:
        return True, 1.0
    overlap = gt_tokens & ret_tokens
    coverage = len(overlap) / len(gt_tokens)
    return coverage >= JUDGE_THRESHOLD, coverage


# ─────────────────────────────────────────────────────────────────────────────
# MNEMON RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

TENANT   = "longmemeval_bench"
EMBEDDER: Optional[SimpleEmbedder] = None   # shared across questions


def _get_embedder() -> SimpleEmbedder:
    global EMBEDDER
    if EMBEDDER is None:
        EMBEDDER = SimpleEmbedder()
    return EMBEDDER


async def ingest_and_retrieve(
    item: Dict,
    db_dir: str,
    llm_client,
) -> Tuple[str, float]:
    """
    For one LongMemEval item:
      1. Ingest all haystack sessions into a fresh CognitiveMemorySystem.
      2. Retrieve using the question as the task signal.
      3. Return (retrieved_text, retrieval_latency_ms).
    """
    embedder = _get_embedder()
    question_id = item.get("question_id", "unknown")

    db      = EROSDatabase(tenant_id=TENANT, db_dir=db_dir)
    index   = InvertedIndex()
    cms     = CognitiveMemorySystem(
        tenant_id=TENANT,
        db=db,
        index=index,
        embedder=embedder,
        llm_client=llm_client,
    )
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    await db.connect()
    await cms.start()

    # ── Ingest ──────────────────────────────────────────────────────────────
    sessions     = item.get("haystack_sessions", [])
    session_dates = item.get("haystack_dates", [])

    for s_idx, session in enumerate(sessions):
        session_id  = f"{question_id}_s{s_idx}"
        date_str    = session_dates[s_idx] if s_idx < len(session_dates) else ""

        # Combine all turns in this session into one narrative memory
        # (preserves context better than per-turn fragmentation)
        turn_texts = []
        for turn in session:
            role    = turn.get("role", "unknown")
            content = turn.get("content", "").strip()
            if content:
                turn_texts.append(f"{role}: {content}")

        if not turn_texts:
            continue

        session_text = "\n".join(turn_texts)

        # Determine layer heuristically:
        #  - user turns dominate → EPISODIC (things that happened)
        #  - assistant-heavy → SEMANTIC (facts learned)
        user_turns  = sum(1 for t in session if t.get("role") == "user")
        asst_turns  = sum(1 for t in session if t.get("role") == "assistant")
        layer = MemoryLayer.EPISODIC if user_turns >= asst_turns else MemoryLayer.SEMANTIC

        sig = ExperienceSignal(
            signal_id   = f"{question_id}_s{s_idx}",
            tenant_id   = TENANT,
            session_id  = session_id,
            timestamp   = time.time() - (len(sessions) - s_idx) * 86400,  # older sessions get older timestamps
            signal_type = SignalType.CONTEXT_UPDATE,
            layer       = layer,
            content     = {"text": session_text, "session_date": date_str, "session_index": s_idx},
            importance  = 0.7,
        )
        await cms.write(sig)

    # ── Retrieve ─────────────────────────────────────────────────────────────
    # We use a two-stage retrieval:
    #   Stage 1: Mnemon's standard protein bond retrieval (RESONANCE_FLOOR=0.70,
    #            calibrated for enterprise agent tasks)
    #   Stage 2: If Stage 1 returns nothing, fall back to direct cosine scan
    #            with a relaxed threshold (0.30) — gives Mnemon a fair shot on
    #            personal-conversation data that doesn't hit enterprise tags.
    question = item.get("question", "")
    t0 = time.time()
    result = await cms.retrieve(
        task_signal = question,
        session_id  = f"{question_id}_query",
        task_goal   = question,
        top_k       = 8,
    )
    memories = result.get("memories", [])

    # Stage 2 fallback: direct cosine scan when protein bond finds nothing.
    # All stored memories carry the "episodic" tag — use it to enumerate
    # the full candidate set, then rank purely by cosine similarity.
    if not memories:
        from mnemon.core.memory import SimpleEmbedder as _SE
        FALLBACK_THRESHOLD = 0.25
        q_vec = embedder.embed(question)
        # "episodic" is added to every memory in extract_tags → reliable enumeration
        all_ids = await index.intersect(TENANT, {"episodic"})
        all_mems = await db.fetch_memories(TENANT, list(all_ids)) if all_ids else []
        scored = []
        for m in all_mems:
            if m.activation_signature:
                score = _SE.cosine_similarity(q_vec, m.activation_signature)
                if score >= FALLBACK_THRESHOLD:
                    scored.append((m, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        memories = []
        for m, _ in scored[:8]:
            c = m.content
            memories.append(c if isinstance(c, dict) else {"text": str(c)})

    latency_ms = (time.time() - t0) * 1000

    # Flatten to text blob for scoring
    retrieved_parts = []
    for mem in memories:
        if isinstance(mem, dict):
            retrieved_parts.append(mem.get("text", str(mem)))
        else:
            retrieved_parts.append(str(mem))
    retrieved_text = " ".join(retrieved_parts)

    await cms.stop()
    return retrieved_text, latency_ms


# ─────────────────────────────────────────────────────────────────────────────
# BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def baseline_null(_item: Dict) -> str:
    """Retrieve nothing — pure zero baseline."""
    return ""


def baseline_random(item: Dict) -> str:
    """Pick a random session turn and return it as context."""
    sessions = item.get("haystack_sessions", [])
    if not sessions:
        return ""
    session = random.choice(sessions)
    turns   = [t.get("content", "") for t in session if t.get("content")]
    return " ".join(random.sample(turns, min(3, len(turns))))


def baseline_first_session(item: Dict) -> str:
    """Return the first session verbatim — naive recency heuristic."""
    sessions = item.get("haystack_sessions", [])
    if not sessions:
        return ""
    return " ".join(t.get("content", "") for t in sessions[0])


# ─────────────────────────────────────────────────────────────────────────────
# PER-QUESTION RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    qid:             str
    qtype:           str
    mnemon_yes:      bool
    mnemon_coverage: float
    null_yes:        bool
    random_yes:      bool
    first_yes:       bool
    latency_ms:      float
    n_sessions:      int


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BENCHMARK LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def run_benchmark(items: List[Dict], n: int) -> List[QuestionResult]:
    items   = items[:n]
    results = []
    llm_client = MockLLMClient()  # no LLM calls — pure embedding retrieval

    w = 72
    print()
    print("━" * w)
    print(f"  {'QID':<28}  {'TYPE':<28}  {'MNEMON':>6}  {'RAND':>5}  {'ms':>6}")
    print(f"  {'─'*28}  {'─'*28}  {'─'*6}  {'─'*5}  {'─'*6}")

    for i, item in enumerate(items, 1):
        qid   = item.get("question_id", f"q{i}")
        qtype = item.get("question_type", "unknown")
        answer = item.get("answer", "")
        n_sess = len(item.get("haystack_sessions", []))

        # Fresh temp DB per question (isolation — no cross-contamination)
        db_dir = str(Path(os.environ.get("TEMP", "/tmp")) / f"mnemon_lme_{qid[:20]}")
        if Path(db_dir).exists():
            shutil.rmtree(db_dir, ignore_errors=True)

        # Mnemon retrieval
        try:
            retrieved, lat = await ingest_and_retrieve(item, db_dir, llm_client)
        except Exception as e:
            retrieved, lat = "", 0.0

        # Score all systems
        mnemon_yes, mnemon_cov = keyword_judge(retrieved, answer)
        null_yes,   _          = keyword_judge(baseline_null(item), answer)
        rand_yes,   _          = keyword_judge(baseline_random(item), answer)
        first_yes,  _          = keyword_judge(baseline_first_session(item), answer)

        result = QuestionResult(
            qid=qid, qtype=qtype,
            mnemon_yes=mnemon_yes, mnemon_coverage=mnemon_cov,
            null_yes=null_yes, random_yes=rand_yes, first_yes=first_yes,
            latency_ms=lat, n_sessions=n_sess,
        )
        results.append(result)

        # Clean up
        shutil.rmtree(db_dir, ignore_errors=True)

        # Print row
        m_col   = f"{GREEN}YES{RESET}" if mnemon_yes else f"{RED} NO{RESET}"
        r_col   = f"{GREEN}Y{RESET}"   if rand_yes   else f"{DIM}n{RESET}"
        qid_str = (qid[:26] + "..") if len(qid) > 28 else qid
        qtyp_str= (qtype[:26] + "..") if len(qtype) > 28 else qtype
        print(f"  {DIM}#{i:>3}{RESET} {qid_str:<24}  {qtyp_str:<28}  {m_col}  {r_col}    {lat:>5.0f}")

    print("━" * w)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: List[QuestionResult], embedder_name: str) -> Dict:
    n = len(results)
    if n == 0:
        print("No results.")
        return {}

    mnemon_acc = sum(r.mnemon_yes for r in results) / n
    null_acc   = sum(r.null_yes   for r in results) / n
    rand_acc   = sum(r.random_yes for r in results) / n
    first_acc  = sum(r.first_yes  for r in results) / n
    avg_cov    = sum(r.mnemon_coverage for r in results) / n
    avg_lat    = sum(r.latency_ms for r in results) / n

    # Per question-type breakdown
    by_type: Dict[str, List[QuestionResult]] = {}
    for r in results:
        by_type.setdefault(r.qtype, []).append(r)

    w = 72
    print()
    print("━" * w)
    print(f"  RESULTS  ({n} questions · {embedder_name})")
    print("━" * w)
    print(f"  {'System':<30}  {'Accuracy':>10}  {'vs Mnemon':>10}")
    print(f"  {'─'*30}  {'─'*10}  {'─'*10}")
    print(f"  {'Mnemon (protein bond retrieval)':<30}  {mnemon_acc*100:>9.1f}%  {'—':>10}")
    print(f"  {'Baseline: retrieve nothing':<30}  {null_acc*100:>9.1f}%  {(null_acc-mnemon_acc)*100:>+9.1f}%")
    print(f"  {'Baseline: random session':<30}  {rand_acc*100:>9.1f}%  {(rand_acc-mnemon_acc)*100:>+9.1f}%")
    print(f"  {'Baseline: first session':<30}  {first_acc*100:>9.1f}%  {(first_acc-mnemon_acc)*100:>+9.1f}%")
    print()
    print(f"  Avg keyword coverage (Mnemon): {avg_cov*100:.1f}%")
    print(f"  Avg retrieval latency:         {avg_lat:.0f}ms")
    print()

    print(f"  {'Question Type':<35}  {'N':>4}  {'Mnemon':>8}  {'Random':>8}")
    print(f"  {'─'*35}  {'─'*4}  {'─'*8}  {'─'*8}")
    for qtype, rs in sorted(by_type.items()):
        nt = len(rs)
        m  = sum(r.mnemon_yes for r in rs) / nt * 100
        r_ = sum(r.random_yes for r in rs) / nt * 100
        print(f"  {qtype:<35}  {nt:>4}  {m:>7.1f}%  {r_:>7.1f}%")

    print("━" * w)
    print()
    print(f"  {DIM}Scoring: keyword overlap >= {JUDGE_THRESHOLD*100:.0f}% of answer keywords → YES{RESET}")
    print(f"  {DIM}Source:  LongMemEval (ICLR 2025) — 100% external data{RESET}")
    print(f"  {DIM}Ref:     https://arxiv.org/abs/2410.10813{RESET}")
    print()

    return {
        "n_questions": n,
        "embedder": embedder_name,
        "judge_threshold": JUDGE_THRESHOLD,
        "mnemon_accuracy": round(mnemon_acc, 4),
        "null_baseline_accuracy": round(null_acc, 4),
        "random_baseline_accuracy": round(rand_acc, 4),
        "first_session_baseline_accuracy": round(first_acc, 4),
        "avg_keyword_coverage": round(avg_cov, 4),
        "avg_retrieval_latency_ms": round(avg_lat, 1),
        "by_question_type": {
            qtype: {
                "n": len(rs),
                "mnemon_accuracy": round(sum(r.mnemon_yes for r in rs) / len(rs), 4),
                "random_accuracy": round(sum(r.random_yes for r in rs) / len(rs), 4),
            }
            for qtype, rs in by_type.items()
        },
    }


def save_report(summary: Dict, results: List[QuestionResult], variant: str):
    REPORT_DIR.mkdir(exist_ok=True)
    report = {
        "benchmark":     "LongMemEval",
        "paper":         "https://arxiv.org/abs/2410.10813",
        "dataset":       f"xiaowu0162/longmemeval (variant: {variant})",
        "generated_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary":       summary,
        "per_question":  [
            {
                "qid":             r.qid,
                "qtype":           r.qtype,
                "mnemon_yes":      r.mnemon_yes,
                "mnemon_coverage": round(r.mnemon_coverage, 4),
                "null_yes":        r.null_yes,
                "random_yes":      r.random_yes,
                "first_yes":       r.first_yes,
                "latency_ms":      round(r.latency_ms),
                "n_sessions":      r.n_sessions,
            }
            for r in results
        ],
    }
    path = REPORT_DIR / "benchmark_longmemeval_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Mnemon × LongMemEval Benchmark")
    parser.add_argument("--n",           type=int, default=500,   help="Number of questions to evaluate (default: all)")
    parser.add_argument("--variant",     default="s",             help="Dataset variant: s or oracle (default: s)")
    parser.add_argument("--no-download", action="store_true",     help="Skip download, use cached data only")
    parser.add_argument("--seed",        type=int, default=42,    help="Random seed for baseline (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    w = 72
    print()
    print("━" * w)
    print(f"  MNEMON × LONGMEMEVAL BENCHMARK")
    print(f"  External dataset (ICLR 2025) — no synthetic data")
    print(f"  Variant: LongMemEval_{args.variant.upper()}")
    print("━" * w)

    # Detect embedder quality upfront
    embedder = _get_embedder()
    emb_name = embedder.backend_name
    emb_note = "(production quality)" if "sentence" in emb_name else "(fallback — pip install sentence-transformers for better results)"
    print(f"\n  Embedder: {emb_name} {DIM}{emb_note}{RESET}")

    # Download / load
    if not args.no_download:
        data_path = download_dataset(args.variant)
    else:
        suffix = "oracle" if args.variant == "oracle" else f"{args.variant}_cleaned"
        data_path = DATA_DIR / f"longmemeval_{suffix}.json"
        if not data_path.exists():
            print(f"  {RED}Data not found: {data_path}{RESET}")
            print(f"  Remove --no-download to fetch it.")
            sys.exit(1)

    print(f"\n  Loading dataset...")
    items = load_dataset(data_path)
    print(f"  {len(items)} questions loaded")

    n = min(args.n, len(items))
    if n < len(items):
        print(f"  Running on first {n} questions (--n {n})")

    # Run
    results = await run_benchmark(items, n)

    # Summary
    summary = print_summary(results, emb_name)

    # Save
    path = save_report(summary, results, args.variant)
    print(f"  Report saved → {path}\n")


if __name__ == "__main__":
    asyncio.run(main())
