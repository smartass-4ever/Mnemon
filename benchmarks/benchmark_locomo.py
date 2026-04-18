"""
benchmark_locomo.py
===================
Evaluates Mnemon's protein bond memory retrieval on the real LoCoMo benchmark
(Stanford, EMNLP 2024) — 100% external data, nothing made by us.

Dataset: KimmoZZZ/locomo on HuggingFace (public mirror of snap-stanford/locomo)
Paper:   https://arxiv.org/abs/2309.11998

What this tests:
  Given up to 35 sessions of real conversational history ingested into Mnemon,
  can protein bond retrieval surface the right memory when asked a question?

  The 10 long-form conversations in LoCoMo are between two real-sounding people
  (Caroline & Melanie, etc.) across months of chat, with QA pairs annotated
  with evidence turns.

Question categories (from the paper):
  1 = single_hop   — direct factual recall from one turn
  2 = temporal     — when / time-based question
  3 = multi_hop    — requires connecting 2+ turns
  4 = open_domain  — broad open-ended question
  5 = adversarial  — trick question / answer not clearly in the conversation

Scoring (mirrors the paper's LLM-as-judge methodology, keyword variant):
  YES  if >= JUDGE_THRESHOLD fraction of ground-truth answer keywords
       appear in the text Mnemon retrieved
  NO   otherwise

Baseline comparison:
  null     — retrieve nothing
  random   — pick a random session turn
  last     — return the most recent session (recency heuristic)

Usage:
  python benchmark_locomo.py               # all 10 conversations, all QA
  python benchmark_locomo.py --n 200       # first 200 questions
  python benchmark_locomo.py --skip-cat 5  # skip adversarial questions
  python benchmark_locomo.py --no-download # use cached data
"""

import argparse
import asyncio
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

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
logging.disable(logging.WARNING)

from mnemon.core.memory import CognitiveMemorySystem, SimpleEmbedder
from mnemon.core.models import ExperienceSignal, MemoryLayer, SignalType
from mnemon.core.persistence import EROSDatabase, InvertedIndex
from mnemon.llm.client import MockLLMClient

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_THRESHOLD = 0.60
DATA_DIR        = Path("data/locomo")
REPORT_DIR      = Path("reports")
TENANT          = "locomo_bench"

# Public mirror of snap-stanford/locomo (same data, no auth required)
_HF_URL = "https://huggingface.co/datasets/KimmoZZZ/locomo/resolve/main/locomo10.json"

CATEGORY_NAMES = {
    1: "single_hop",
    2: "temporal",
    3: "multi_hop",
    4: "open_domain",
    5: "adversarial",
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

def download_dataset() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / "locomo10.json"

    if dest.exists():
        print(f"  {DIM}Dataset cached: {dest}{RESET}")
        return dest

    print(f"  Downloading LoCoMo from HuggingFace (public mirror)...")
    print(f"  {DIM}{_HF_URL}{RESET}")

    try:
        req = urllib.request.Request(_HF_URL, headers={"User-Agent": "mnemon-benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            buf = b""
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                buf += chunk
                downloaded += len(chunk)
                if total:
                    print(f"\r  {downloaded/1_048_576:.1f} MB / {total/1_048_576:.1f} MB  ({downloaded/total*100:.0f}%)",
                          end="", flush=True)
        print()
        dest.write_bytes(buf)
        print(f"  Saved to {dest}")
        return dest
    except Exception as e:
        print(f"\n  {RED}Download failed: {e}{RESET}")
        sys.exit(1)


def load_dataset(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PARSING
# ─────────────────────────────────────────────────────────────────────────────

def iter_sessions(conv: Dict) -> List[Tuple[int, str, List[Dict]]]:
    """
    Yields (session_number, date_time_str, turns_list) for each session
    present in a conversation dict.
    """
    sessions = []
    for i in range(1, 40):
        key      = f"session_{i}"
        date_key = f"session_{i}_date_time"
        turns    = conv.get(key)
        if not isinstance(turns, list):
            continue
        date_str = conv.get(date_key, "")
        sessions.append((i, date_str, turns))
    return sessions


def build_flat_qa(item: Dict) -> List[Dict]:
    """
    Flatten QA pairs, attaching conversation context.
    Returns list of dicts with: question, answer, category, evidence, conv_id
    """
    conv_id = item.get("sample_id", "unknown")
    qa_list = []
    for q in item.get("qa", []):
        qa_list.append({
            "question":  q.get("question", ""),
            "answer":    str(q.get("answer", "")),
            "category":  int(q.get("category", 0)),
            "evidence":  q.get("evidence", []),
            "conv_id":   conv_id,
        })
    return qa_list


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────

_STOP = {
    "the","and","for","that","this","with","are","was","were","have",
    "has","had","not","but","from","they","what","when","where","who",
    "will","would","could","should","which","been","also","its","their",
    "there","then","than","into","onto","upon","about","after","before",
    "during","while","since","until","between","through","within",
}

def _tokenise(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 3 and t not in _STOP]


def keyword_judge(retrieved: str, ground_truth: str) -> Tuple[bool, float]:
    gt_tokens  = set(_tokenise(ground_truth))
    ret_tokens = set(_tokenise(retrieved))
    if not gt_tokens:
        return True, 1.0
    coverage = len(gt_tokens & ret_tokens) / len(gt_tokens)
    return coverage >= JUDGE_THRESHOLD, coverage


# ─────────────────────────────────────────────────────────────────────────────
# BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def _all_turns_text(item: Dict) -> List[str]:
    """All turn texts across all sessions for a conversation item."""
    all_texts = []
    for _, _, turns in iter_sessions(item["conversation"]):
        for t in turns:
            txt = t.get("text", "").strip()
            if txt:
                all_texts.append(txt)
    return all_texts


def baseline_null(_item: Dict, _qa: Dict) -> str:
    return ""


def baseline_random(item: Dict, _qa: Dict) -> str:
    texts = _all_turns_text(item)
    if not texts:
        return ""
    return " ".join(random.sample(texts, min(5, len(texts))))


def baseline_last_session(item: Dict, _qa: Dict) -> str:
    """Most recent session in full — recency heuristic."""
    sessions = iter_sessions(item["conversation"])
    if not sessions:
        return ""
    _, _, turns = sessions[-1]
    return " ".join(t.get("text", "") for t in turns)


# ─────────────────────────────────────────────────────────────────────────────
# MNEMON INGESTION + RETRIEVAL (per conversation)
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDER: Optional[SimpleEmbedder] = None

def _get_embedder() -> SimpleEmbedder:
    global EMBEDDER
    if EMBEDDER is None:
        EMBEDDER = SimpleEmbedder()
    return EMBEDDER


async def build_cms_for_conversation(item: Dict, db_dir: str) -> CognitiveMemorySystem:
    """
    Ingest all sessions of one LoCoMo conversation into a fresh CognitiveMemorySystem.

    Two memories are written per session:

    1. EPISODIC — raw turns prefixed with the session date.
       "[Date: 8 May 2023]\nCaroline: I went to a LGBTQ support group yesterday..."
       The date prefix ensures temporal questions can match month/year tokens.

    2. SEMANTIC — the pre-computed session_summary from the dataset (when present).
       "Caroline and Melanie had a conversation on 8 May 2023. Caroline attended an
        LGBTQ support group and found the transgender stories inspiring..."
       Summaries pack distilled facts + explicit dates into a single dense block,
       which retrieval can match more reliably than noisy raw dialogue.

    Both memories carry the same session timestamp so recency sorting is consistent.
    """
    embedder   = _get_embedder()
    conv       = item["conversation"]
    conv_id    = item.get("sample_id", "unknown")
    speaker_a  = conv.get("speaker_a", "A")
    speaker_b  = conv.get("speaker_b", "B")
    summaries  = item.get("session_summary", {})

    db    = EROSDatabase(tenant_id=TENANT, db_dir=db_dir)
    index = InvertedIndex()
    cms   = CognitiveMemorySystem(
        tenant_id=TENANT,
        db=db,
        index=index,
        embedder=embedder,
        llm_client=MockLLMClient(),
    )
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    await db.connect()
    await cms.start()

    sessions   = iter_sessions(conv)
    n_sessions = len(sessions)

    for sess_num, date_str, turns in sessions:
        ts = time.time() - (n_sessions - sess_num) * 86400

        # ── Memory 1: EPISODIC — raw turns with date header ──────────────────
        turn_texts = []
        for t in turns:
            spk  = t.get("speaker", "?")
            text = t.get("text", "").strip()
            if text:
                turn_texts.append(f"{spk}: {text}")

        if turn_texts:
            # Prepend date so temporal keyword matching can find month/year tokens
            date_header  = f"[Date: {date_str}]\n" if date_str else ""
            episodic_text = date_header + "\n".join(turn_texts)

            await cms.write(ExperienceSignal(
                signal_id   = f"{conv_id}_s{sess_num}_ep",
                tenant_id   = TENANT,
                session_id  = f"{conv_id}_s{sess_num}",
                timestamp   = ts,
                signal_type = SignalType.CONTEXT_UPDATE,
                layer       = MemoryLayer.EPISODIC,
                content     = {
                    "text":          episodic_text,
                    "session_date":  date_str,
                    "session_index": sess_num,
                    "speaker_a":     speaker_a,
                    "speaker_b":     speaker_b,
                    "conv_id":       conv_id,
                },
                importance = 0.7,
            ))

        # ── Memory 2: SEMANTIC — pre-computed session summary ─────────────────
        summary_text = summaries.get(f"session_{sess_num}_summary", "").strip()
        if summary_text:
            await cms.write(ExperienceSignal(
                signal_id   = f"{conv_id}_s{sess_num}_sm",
                tenant_id   = TENANT,
                session_id  = f"{conv_id}_s{sess_num}",
                timestamp   = ts,
                signal_type = SignalType.CONTEXT_UPDATE,
                layer       = MemoryLayer.SEMANTIC,
                content     = {
                    "text":          summary_text,
                    "session_date":  date_str,
                    "session_index": sess_num,
                    "conv_id":       conv_id,
                },
                importance = 0.8,   # summaries are higher-quality signal
            ))

    return cms, db


def _mem_to_text(mem) -> str:
    """
    Extract text from a retrieved memory object.
    The content dict now always has 'text' baked in (date header + turns, or summary),
    so mem.get("text") is sufficient — no separate date extraction needed.
    """
    if isinstance(mem, dict):
        return mem.get("text", str(mem))
    return str(mem)


async def retrieve_for_question(cms: CognitiveMemorySystem, question: str, conv_id: str) -> Tuple[str, float]:
    t0 = time.time()
    result   = await cms.retrieve(
        task_signal = question,
        session_id  = f"{conv_id}_query",
        task_goal   = question,
        top_k       = 12,      # wider net: captures both EPISODIC + SEMANTIC memories
    )
    memories = result.get("memories", [])

    # Fallback: direct cosine scan across all memories when protein bond finds nothing
    if not memories:
        FALLBACK_THRESHOLD = 0.20   # relaxed — conversational data has lower cosine scores
        q_vec = _get_embedder().embed(question)
        # Scan both episodic and semantic tags
        for tag in ("episodic", "semantic"):
            tag_ids = await cms._index.intersect(TENANT, {tag})
            if not tag_ids:
                continue
            all_mems = await cms._db.fetch_memories(TENANT, list(tag_ids))
            scored = []
            for m in all_mems:
                if m.activation_signature:
                    score = SimpleEmbedder.cosine_similarity(q_vec, m.activation_signature)
                    if score >= FALLBACK_THRESHOLD:
                        scored.append((m, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            for m, _ in scored[:12]:
                c = m.content
                memories.append(c if isinstance(c, dict) else {"text": str(c)})
        # Deduplicate by text prefix
        seen, deduped = set(), []
        for mem in memories:
            key = _mem_to_text(mem)[:80]
            if key not in seen:
                seen.add(key)
                deduped.append(mem)
        memories = deduped

    latency_ms = (time.time() - t0) * 1000
    retrieved_text = " ".join(_mem_to_text(m) for m in memories)
    return retrieved_text, latency_ms


# ─────────────────────────────────────────────────────────────────────────────
# PER-QUESTION RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    conv_id:         str
    category:        int
    question:        str
    answer:          str
    mnemon_yes:      bool
    mnemon_coverage: float
    null_yes:        bool
    random_yes:      bool
    last_yes:        bool
    latency_ms:      float


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BENCHMARK LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def run_benchmark(data: List[Dict], n: int, skip_cats: set) -> List[QuestionResult]:
    results: List[QuestionResult] = []

    w = 80
    print()
    print("━" * w)
    print(f"  {'CONV':<10}  {'CAT':<12}  {'QUESTION':<32}  {'MNEMON':>6}  {'RAND':>5}  {'ms':>6}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*32}  {'─'*6}  {'─'*5}  {'─'*6}")

    total_q = 0

    for item_idx, item in enumerate(data):
        conv_id  = item.get("sample_id", f"conv{item_idx}")
        qa_pairs = build_flat_qa(item)

        if skip_cats:
            qa_pairs = [q for q in qa_pairs if q["category"] not in skip_cats]
        if not qa_pairs:
            continue

        # Ingest this conversation once, answer all its QA pairs
        db_dir = str(Path(os.environ.get("TEMP", "/tmp")) / f"mnemon_locomo_{conv_id}")
        if Path(db_dir).exists():
            shutil.rmtree(db_dir, ignore_errors=True)

        try:
            cms, db = await build_cms_for_conversation(item, db_dir)
        except Exception as e:
            print(f"  {RED}Ingest failed for {conv_id}: {e}{RESET}")
            continue

        for qa in qa_pairs:
            if total_q >= n:
                break

            question = qa["question"]
            answer   = qa["answer"]
            cat      = qa["category"]
            cat_name = CATEGORY_NAMES.get(cat, f"cat_{cat}")

            try:
                retrieved, lat = await retrieve_for_question(cms, question, conv_id)
            except Exception:
                retrieved, lat = "", 0.0

            mnemon_yes, mnemon_cov = keyword_judge(retrieved, answer)
            null_yes,   _          = keyword_judge(baseline_null(item, qa), answer)
            rand_yes,   _          = keyword_judge(baseline_random(item, qa), answer)
            last_yes,   _          = keyword_judge(baseline_last_session(item, qa), answer)

            result = QuestionResult(
                conv_id=conv_id, category=cat,
                question=question, answer=answer,
                mnemon_yes=mnemon_yes, mnemon_coverage=mnemon_cov,
                null_yes=null_yes, random_yes=rand_yes, last_yes=last_yes,
                latency_ms=lat,
            )
            results.append(result)
            total_q += 1

            m_col  = f"{GREEN}YES{RESET}" if mnemon_yes else f"{RED} NO{RESET}"
            r_col  = f"{GREEN}Y{RESET}"   if rand_yes   else f"{DIM}n{RESET}"
            q_str  = (question[:30] + "..") if len(question) > 32 else question
            print(f"  {DIM}{conv_id:<10}{RESET}  {cat_name:<12}  {q_str:<32}  {m_col}  {r_col}    {lat:>5.0f}")

        await cms.stop()
        shutil.rmtree(db_dir, ignore_errors=True)

        if total_q >= n:
            break

    print("━" * w)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: List[QuestionResult], embedder_name: str) -> Dict:
    n = len(results)
    if n == 0:
        print("No results.")
        return {}

    mnemon_acc = sum(r.mnemon_yes for r in results) / n
    null_acc   = sum(r.null_yes   for r in results) / n
    rand_acc   = sum(r.random_yes for r in results) / n
    last_acc   = sum(r.last_yes   for r in results) / n
    avg_cov    = sum(r.mnemon_coverage for r in results) / n
    avg_lat    = sum(r.latency_ms for r in results) / n

    by_cat: Dict[int, List[QuestionResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    w = 80
    print()
    print("━" * w)
    print(f"  RESULTS  ({n} questions · {embedder_name})")
    print("━" * w)
    print(f"  {'System':<35}  {'Accuracy':>10}  {'vs Mnemon':>10}")
    print(f"  {'─'*35}  {'─'*10}  {'─'*10}")
    print(f"  {'Mnemon (protein bond retrieval)':<35}  {mnemon_acc*100:>9.1f}%  {'—':>10}")
    print(f"  {'Baseline: retrieve nothing':<35}  {null_acc*100:>9.1f}%  {(null_acc-mnemon_acc)*100:>+9.1f}%")
    print(f"  {'Baseline: random turn':<35}  {rand_acc*100:>9.1f}%  {(rand_acc-mnemon_acc)*100:>+9.1f}%")
    print(f"  {'Baseline: last session':<35}  {last_acc*100:>9.1f}%  {(last_acc-mnemon_acc)*100:>+9.1f}%")
    print()
    print(f"  Avg keyword coverage (Mnemon): {avg_cov*100:.1f}%")
    print(f"  Avg retrieval latency:         {avg_lat:.0f}ms")
    print()

    print(f"  {'Category':<20}  {'N':>4}  {'Mnemon':>8}  {'Random':>8}  {'Last':>8}")
    print(f"  {'─'*20}  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}")
    for cat, rs in sorted(by_cat.items()):
        nt    = len(rs)
        m_acc = sum(r.mnemon_yes for r in rs) / nt * 100
        r_acc = sum(r.random_yes for r in rs) / nt * 100
        l_acc = sum(r.last_yes   for r in rs) / nt * 100
        name  = CATEGORY_NAMES.get(cat, f"cat_{cat}")
        print(f"  {name:<20}  {nt:>4}  {m_acc:>7.1f}%  {r_acc:>7.1f}%  {l_acc:>7.1f}%")

    print("━" * w)
    print()
    print(f"  {DIM}Scoring: keyword overlap >= {JUDGE_THRESHOLD*100:.0f}% of answer keywords → YES{RESET}")
    print(f"  {DIM}Source:  LoCoMo (Stanford, EMNLP 2024) — 100% external data{RESET}")
    print(f"  {DIM}Ref:     https://arxiv.org/abs/2309.11998{RESET}")
    print()

    return {
        "n_questions":                n,
        "embedder":                   embedder_name,
        "judge_threshold":            JUDGE_THRESHOLD,
        "mnemon_accuracy":            round(mnemon_acc, 4),
        "null_baseline_accuracy":     round(null_acc, 4),
        "random_baseline_accuracy":   round(rand_acc, 4),
        "last_session_accuracy":      round(last_acc, 4),
        "avg_keyword_coverage":       round(avg_cov, 4),
        "avg_retrieval_latency_ms":   round(avg_lat, 1),
        "by_category": {
            CATEGORY_NAMES.get(cat, f"cat_{cat}"): {
                "n":               len(rs),
                "mnemon_accuracy": round(sum(r.mnemon_yes for r in rs) / len(rs), 4),
                "random_accuracy": round(sum(r.random_yes for r in rs) / len(rs), 4),
            }
            for cat, rs in by_cat.items()
        },
    }


def save_report(summary: Dict, results: List[QuestionResult]) -> Path:
    REPORT_DIR.mkdir(exist_ok=True)
    report = {
        "benchmark":    "LoCoMo",
        "paper":        "https://arxiv.org/abs/2309.11998",
        "dataset":      "KimmoZZZ/locomo (public mirror of snap-stanford/locomo)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary":      summary,
        "per_question": [
            {
                "conv_id":         r.conv_id,
                "category":        CATEGORY_NAMES.get(r.category, f"cat_{r.category}"),
                "question":        r.question,
                "answer":          r.answer,
                "mnemon_yes":      r.mnemon_yes,
                "mnemon_coverage": round(r.mnemon_coverage, 4),
                "null_yes":        r.null_yes,
                "random_yes":      r.random_yes,
                "last_yes":        r.last_yes,
                "latency_ms":      round(r.latency_ms),
            }
            for r in results
        ],
    }
    path = REPORT_DIR / "benchmark_locomo_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Mnemon × LoCoMo Benchmark")
    parser.add_argument("--n",           type=int,   default=99999, help="Max questions to evaluate (default: all ~2000)")
    parser.add_argument("--skip-cat",    type=int,   nargs="+",     help="Skip question categories (e.g. --skip-cat 5)")
    parser.add_argument("--no-download", action="store_true",       help="Use cached data, skip download")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    skip_cats = set(args.skip_cat) if args.skip_cat else set()

    w = 80
    print()
    print("━" * w)
    print(f"  MNEMON × LOCOMO BENCHMARK")
    print(f"  Stanford EMNLP 2024 dataset — 10 long conversations, ~2000 QA pairs")
    print(f"  Real external data — no synthetic questions")
    print("━" * w)

    embedder  = _get_embedder()
    emb_name  = embedder.backend_name
    emb_note  = "(production)" if "sentence" in emb_name else "(fallback — pip install sentence-transformers)"
    print(f"\n  Embedder: {emb_name} {DIM}{emb_note}{RESET}")

    if not args.no_download:
        data_path = download_dataset()
    else:
        data_path = DATA_DIR / "locomo10.json"
        if not data_path.exists():
            print(f"  {RED}Data not found: {data_path}{RESET}")
            print(f"  Remove --no-download to fetch it.")
            sys.exit(1)

    print(f"\n  Loading dataset...")
    data = load_dataset(data_path)
    total_qa = sum(len(item.get("qa", [])) for item in data)
    print(f"  {len(data)} conversations, {total_qa} QA pairs")
    if skip_cats:
        names = [CATEGORY_NAMES.get(c, str(c)) for c in skip_cats]
        print(f"  Skipping categories: {', '.join(names)}")

    n = args.n
    if n < total_qa:
        print(f"  Running on first {n} questions (--n {n})")

    results = await run_benchmark(data, n, skip_cats)
    summary = print_summary(results, emb_name)
    path    = save_report(summary, results)
    print(f"  Report saved → {path}\n")


if __name__ == "__main__":
    asyncio.run(main())
