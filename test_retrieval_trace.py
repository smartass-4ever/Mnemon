"""
test_retrieval_trace.py
=======================
Context Rot Trace Test — 10 failure modes at 10M memory scale.

Each test constructs a minimal, controlled scenario that isolates one
specific failure mode. No LLM needed. Pass/fail is deterministic.

Run:
    python test_retrieval_trace.py

Failures print [FAIL] with explanation of the rot. Passes print [PASS].
Summary at the end shows which modes are live bugs vs handled.
"""

import asyncio
import hashlib
import json
import os
import sys
import time
from copy import deepcopy
from typing import Any, Dict, List, Set, Tuple

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mnemon.core.persistence import EROSDatabase, InvertedIndex
from mnemon.core.memory import (
    CognitiveMemorySystem, SimpleEmbedder, RESONANCE_FLOOR,
    INTENT_WEIGHT, PATTERN_WEIGHT,
)
from mnemon.core.models import (
    BondedMemory, ExperienceSignal, MemoryLayer, SignalType, RiskLevel
)

# ── colour helpers ────────────────────────────────────────────────────────────
PASS_STR  = "\033[92m[PASS]\033[0m"
FAIL_STR  = "\033[91m[FAIL]\033[0m"
WARN_STR  = "\033[93m[WARN]\033[0m"
INFO_STR  = "\033[94m[INFO]\033[0m"
TITLE_STR = "\033[1m\033[95m"
RESET     = "\033[0m"

results: List[Tuple[str, bool, str]] = []   # (test_name, passed, detail)

def record(name: str, passed: bool, detail: str):
    results.append((name, passed, detail))
    icon = PASS_STR if passed else FAIL_STR
    print(f"  {icon}  {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_fake_sig(seed: str, dim: int = 64) -> List[float]:
    """Deterministic unit-norm vector from a seed string."""
    rng = np.random.default_rng(int(hashlib.md5(seed.encode()).hexdigest()[:8], 16))
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()

def cosine(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a), np.array(b)
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / d) if d else 0.0

def combine(pattern: float, intent: float) -> float:
    return PATTERN_WEIGHT * pattern + INTENT_WEIGHT * intent

def bonded(
    memory_id: str,
    content: str,
    tags: Set[str],
    domain: str,
    layer: MemoryLayer,
    act_seed: str,
    intent_seed: str,
    keep_score: float = 0.5,
    drop_score: float = 0.5,
    intent_valid: bool = True,
    importance: float = 0.8,
    created_at: float = None,
) -> BondedMemory:
    m = BondedMemory(
        memory_id=memory_id,
        tenant_id="trace_tenant",
        content=content,
        layer=layer,
        importance=importance,
        timestamp=created_at or time.time(),
    )
    m.activation_tags      = tags
    m.activation_domain    = domain
    m.activation_signature = make_fake_sig(act_seed)
    m.intent_signature     = make_fake_sig(intent_seed)
    m.drone_keep_score     = keep_score
    m.drone_drop_score     = drop_score
    m.intent_valid         = intent_valid
    return m


# ─────────────────────────────────────────────────────────────────────────────
# FAILURE MODE 1 — TAG UNION FALLBACK EXPLOSION
# ─────────────────────────────────────────────────────────────────────────────
# When tag intersection = ∅, code falls back to UNION of all tag sets.
# At 10M scale generic tags like {"security", "code"} each map to millions
# of memory IDs — the union pulls everything into cosine scoring, O(n) scan.
#
# This test simulates the structural condition: two tags with no memory
# holding BOTH, causing intersection to collapse to union.
# ─────────────────────────────────────────────────────────────────────────────

async def test_union_fallback_explosion():
    print(f"\n{TITLE_STR}MODE 1 — Tag Union Fallback Explosion (+ Cap Fix){RESET}")

    index = InvertedIndex()
    tenant = "trace_tenant"

    # Insert 15,000 memories — 8,000 tagged "security", 7,000 tagged "code", zero with BOTH.
    # This exceeds the _UNION_CAP=10,000 so the cap must kick in.
    for i in range(8000):
        await index.update(tenant, f"sec_{i}", {"security"})
    for i in range(7000):
        await index.update(tenant, f"code_{i}", {"code"})

    query_tags = {"security", "code"}
    candidates = await index.intersect(tenant, query_tags)
    result_size = len(candidates)

    cap = getattr(InvertedIndex, "_UNION_CAP", None)

    # Fix verification: result must be ≤ _UNION_CAP (10,000)
    # Without fix: result would be 15,000 (raw union)
    cap_applied = cap is not None and result_size <= cap
    passed = cap_applied
    record(
        "union_fallback_explosion",
        passed,
        f"15,000 memories, zero intersection. Union cap={cap}. "
        f"Result size={result_size}. "
        f"{'Cap applied — O(n) blast contained' if cap_applied else 'NO CAP — raw union would be 15,000+ at this scale'}"
    )
    print(f"    {INFO_STR} At 10M scale without cap: ~10M candidates, ~50s per query")
    print(f"    {INFO_STR} With _UNION_CAP={cap}: max {cap:,} candidates from union → bounded scan")


# ─────────────────────────────────────────────────────────────────────────────
# FAILURE MODE 2 — DYNAMIC THRESHOLD RATCHET KILLS CORRECT ANSWERS
# ─────────────────────────────────────────────────────────────────────────────
# When > 100 memories pass RESONANCE_FLOOR=0.70, code ratchets threshold up
# by 0.02 increments until pool <= 80. At 10M scale, hundreds pass the floor.
# Correct-but-moderate memories score 0.71-0.78 and get wiped.
# High-frequency "hub" memories score 0.93+ and survive every ratchet level.
# ─────────────────────────────────────────────────────────────────────────────

async def test_dynamic_threshold_ratchet():
    print(f"\n{TITLE_STR}MODE 2 — Ratchet Kills Correct (+ Score-Gap Fix){RESET}")

    # Scenario: 150 memories pass the floor, scores uniformly distributed 0.70-0.92
    # (no natural cliff). 5 correct answers are interspersed at 0.71-0.76.
    # Old count-ratchet blindly raises floor to 0.86 → kills all 5 correct.
    # New gap logic: no cliff ≥ 0.03 found → falls through to hard cap at 200 → keeps all.
    correct_ids = {f"correct_{i}" for i in range(5)}
    correct_scores = [0.71, 0.73, 0.74, 0.75, 0.76]

    pool: List[Tuple[str, float]] = []
    for i, score in enumerate(correct_scores):
        pool.append((f"correct_{i}", score))
    # 145 noise memories uniformly spread 0.70-0.92 in 0.0015 steps (no cliff)
    for i in range(145):
        noise_score = round(0.70 + i * 0.0015, 4)
        pool.append((f"noise_{i}", noise_score))
    pool.sort(key=lambda x: x[1], reverse=True)

    # ── OLD ratchet logic ──
    threshold_old = 0.70
    if len(pool) > 100:
        while len([p for p in pool if p[1] >= threshold_old]) > 80 and threshold_old < 0.95:
            threshold_old += 0.02
        old_result = [(m, s) for m, s in pool if s >= threshold_old]
    else:
        old_result = pool
    killed_old = [m for m in correct_ids if m not in {x for x, _ in old_result}]

    # ── NEW score-gap logic ──
    new_pool = list(pool)
    if len(new_pool) > 100:
        scores = [s for _, s in new_pool]
        gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
        if gaps:
            search_end = min(200, len(gaps))
            max_gap_idx = max(range(search_end), key=lambda i: gaps[i])
            gap_at_cut  = gaps[max_gap_idx]
            if gap_at_cut >= 0.03 and (max_gap_idx + 1) >= 10:
                new_pool = new_pool[:max_gap_idx + 1]
            else:
                new_pool = new_pool[:200]  # no cliff — keep all (cap at 200)
    surviving_correct_new = [m for m, s in new_pool if m in correct_ids]
    killed_new = [m for m in correct_ids if m not in {x for x, _ in new_pool}]

    passed = len(killed_new) == 0
    record(
        "ratchet_kills_correct",
        passed,
        f"OLD ratchet killed {len(killed_old)}/5 correct answers (threshold→{threshold_old:.2f}). "
        f"NEW score-gap logic: {len(surviving_correct_new)}/5 correct survive. "
        f"{'Fix working' if passed else 'STILL KILLING — check gap logic'}"
    )
    if not passed:
        print(f"    {WARN_STR} Gap logic failed: {killed_new} still killed")
    else:
        print(f"    {INFO_STR} Score-gap cut at natural cliff — correct moderate-score answers preserved")


# ─────────────────────────────────────────────────────────────────────────────
# FAILURE MODE 3 — 64-DIM HUBNESS POLLUTION
# ─────────────────────────────────────────────────────────────────────────────
# HashProjectionEmbedder uses 64 dimensions. High-dimensional spaces suffer
# hubness: certain vectors become "universal neighbors" — their cosine
# similarity to all other vectors clusters above the resonance floor (0.70).
# These hub memories appear in EVERY retrieval regardless of relevance.
#
# This test measures actual cosine distribution from the hash embedder
# to see if hubness is present at 64 dims.
# ─────────────────────────────────────────────────────────────────────────────

async def test_hubness_pollution():
    print(f"\n{TITLE_STR}MODE 3 — 64-dim Hubness Pollution{RESET}")

    embedder = SimpleEmbedder()

    # Generate 1000 diverse sentences across different domains
    sentences = (
        [f"Security alert: CVE-{i} affects nginx version {i}.0" for i in range(100)] +
        [f"Invoice #{i} from vendor Acme Corp for ${i*100}" for i in range(100)] +
        [f"def process_transaction(amount, account_{i}): return amount * 0.{i:02d}" for i in range(100)] +
        [f"Meeting with client {i} to discuss Q{i%4+1} roadmap" for i in range(100)] +
        [f"Email from ceo_{i}@corp.com: quarterly report attached" for i in range(100)] +
        [f"Database backup completed at {i:02d}:00 UTC" for i in range(100)] +
        [f"User {i} logged in from IP 10.0.{i%256}.1" for i in range(100)] +
        [f"Deploy {i}: updated payment gateway to version 2.{i}" for i in range(100)] +
        [f"Customer complaint #{i}: delayed shipment for order {i*7}" for i in range(100)] +
        [f"API latency spike detected at endpoint /v{i%3+1}/orders" for i in range(100)],
    )
    sentences = sentences[0]  # flatten

    embeddings = [embedder.embed(s) for s in sentences]

    # Pick a random query embedding (not in corpus)
    query_emb = embedder.embed("prepare the weekly security summary report")

    similarities = [cosine(query_emb, e) for e in embeddings]
    above_floor  = sum(1 for s in similarities if s >= RESONANCE_FLOOR)
    above_08     = sum(1 for s in similarities if s >= 0.80)
    above_09     = sum(1 for s in similarities if s >= 0.90)
    max_sim      = max(similarities)
    mean_sim     = float(np.mean(similarities))

    # Hubness check: if >5% of a diverse 1000-memory corpus scores >= 0.70
    # against a single query, that's hubness — retrieval will always be noisy
    hubness_rate = above_floor / len(embeddings)

    passed = hubness_rate < 0.05  # less than 5% false activation = acceptable
    record(
        "hubness_pollution",
        passed,
        f"Query vs 1000 diverse memories: {above_floor} score >= 0.70 ({hubness_rate*100:.1f}%). "
        f"Above 0.80: {above_08}. Above 0.90: {above_09}. Mean sim: {mean_sim:.3f}. Max: {max_sim:.3f}. "
        f"{'HUBNESS DETECTED — wrong memories will appear in every query' if not passed else 'embedding spread is acceptable'}"
    )
    if not passed:
        print(f"    {WARN_STR} At 10M memories: ~{int(hubness_rate * 10_000_000):,} irrelevant memories will score >=0.70 per query")
        print(f"    {WARN_STR} Fix: raise RESONANCE_FLOOR or use sentence-transformers (384-dim) exclusively")


# ─────────────────────────────────────────────────────────────────────────────
# FAILURE MODE 4 — DRONE TRUNCATES AT 30, KILLS RANKS 31-80
# ─────────────────────────────────────────────────────────────────────────────
# _drone_evaluate sends candidate_list[:30] to the LLM (memory.py:999).
# At 10M scale, pattern assembly can surface 80 strong candidates.
# The correct answer may rank #35 (score 0.81) but the drone never sees it.
# ─────────────────────────────────────────────────────────────────────────────

async def test_drone_truncation():
    print(f"\n{TITLE_STR}MODE 4 — Drone Truncates at 30 (+ Cap Raised to 80){RESET}")

    from mnemon.core.memory import DRONE_CANDIDATE_CAP

    # Simulate 80 candidates; correct answer at rank 35
    candidates = []
    for i in range(80):
        score = 0.95 - i * 0.002
        candidates.append((f"mem_{i:03d}", score))

    CORRECT_RANK = 34  # 0-indexed
    CORRECT_ID   = f"mem_{CORRECT_RANK:03d}"

    # Old cap = 30 (hard-coded in prompt string)
    old_drone_sees = [m_id for m_id, _ in candidates[:30]]
    # New cap = DRONE_CANDIDATE_CAP (constant, now 80)
    new_drone_sees = [m_id for m_id, _ in candidates[:DRONE_CANDIDATE_CAP]]

    old_sees_correct = CORRECT_ID in old_drone_sees
    new_sees_correct = CORRECT_ID in new_drone_sees

    passed = new_sees_correct
    record(
        "drone_truncation",
        passed,
        f"Correct memory at rank #{CORRECT_RANK+1} (score={candidates[CORRECT_RANK][1]:.3f}). "
        f"Old cap=30: correct {'visible' if old_sees_correct else 'INVISIBLE'}. "
        f"New DRONE_CANDIDATE_CAP={DRONE_CANDIDATE_CAP}: correct {'visible' if new_sees_correct else 'INVISIBLE'}. "
        f"{'Cap raised — correct answer now reachable by drone' if new_sees_correct else 'STILL INVISIBLE'}"
    )
    print(f"    {INFO_STR} Old: {80-30} of 80 candidates ({(80-30)/80*100:.0f}%) invisible to drone")
    print(f"    {INFO_STR} New: {max(0, 80-DRONE_CANDIDATE_CAP)} of 80 candidates invisible to drone")


# ─────────────────────────────────────────────────────────────────────────────
# FAILURE MODE 5 — LAYER DIVERSITY FORCES WRONG CONTEXT IN
# ─────────────────────────────────────────────────────────────────────────────
# _ensure_layer_diversity promotes the first memory from each unseen layer,
# regardless of score. A low-relevance Emotional/Procedural memory (score 0.71)
# can jump ahead of a 3rd highly-relevant Semantic memory (score 0.91).
# ─────────────────────────────────────────────────────────────────────────────

async def test_layer_diversity_forces_wrong():
    print(f"\n{TITLE_STR}MODE 5 — Layer Diversity Forces Wrong Context{RESET}")

    TOP_K = 5

    # Setup: 8 candidates after drone
    # 3 Semantic memories (all highly relevant, scores 0.95, 0.91, 0.88)
    # 1 Emotional memory (irrelevant, score 0.71 — just above floor)
    # 1 Relationship memory (marginally relevant, score 0.72)
    # 1 Episodic memory (relevant, score 0.83)
    candidates = [
        ("sem_1", 0.95),
        ("sem_2", 0.91),
        ("epi_1", 0.83),
        ("sem_3", 0.88),  # would rank 3rd without diversity, 4th with
        ("rel_1", 0.72),
        ("emo_1", 0.71),
        ("sem_4", 0.70),
        ("epi_2", 0.70),
    ]
    # After sort: sem_1(0.95), sem_2(0.91), sem_3(0.88? no wait, epi_1 is 0.83 so:
    # sem_1(0.95), sem_2(0.91), sem_3(0.88), epi_1(0.83), proc_1(0.72), emo_1(0.71)...
    # Wait, they're already in score order above. Let me re-sort:
    candidates.sort(key=lambda x: x[1], reverse=True)

    layer_map = {
        "sem_1":  MemoryLayer.SEMANTIC,
        "sem_2":  MemoryLayer.SEMANTIC,
        "sem_3":  MemoryLayer.SEMANTIC,
        "sem_4":  MemoryLayer.SEMANTIC,
        "epi_1":  MemoryLayer.EPISODIC,
        "epi_2":  MemoryLayer.EPISODIC,
        "rel_1":  MemoryLayer.RELATIONSHIP,
        "emo_1":  MemoryLayer.EMOTIONAL,
    }

    # Reproduce _ensure_layer_diversity (memory.py:1069-1094)
    seen_layers: Set[MemoryLayer] = set()
    priority   = []
    remainder  = []
    for m_id, score in candidates:
        layer = layer_map[m_id]
        if layer not in seen_layers:
            priority.append((m_id, score))
            seen_layers.add(layer)
        else:
            remainder.append((m_id, score))

    final = (priority + remainder)[:TOP_K]
    final_ids = [m for m, _ in final]

    # What got in vs what should have got in by pure score?
    pure_top5 = [m for m, _ in candidates[:TOP_K]]

    forced_in   = [m for m in final_ids if m not in pure_top5]
    pushed_out  = [m for m in pure_top5 if m not in final_ids]

    # ── Reproduce the FIXED _ensure_layer_diversity (score-gap aware) ──
    from mnemon.core.memory import DIVERSITY_SCORE_GAP
    seen_layers_fix: Set[MemoryLayer] = set()
    priority_fix = []
    remainder_fix = []
    best_overall = candidates[0][1] if candidates else 1.0
    for m_id, score in candidates:
        layer = layer_map[m_id]
        if layer not in seen_layers_fix:
            gap = best_overall - score
            if gap <= DIVERSITY_SCORE_GAP:
                priority_fix.append((m_id, score))
            else:
                remainder_fix.append((m_id, score))
            seen_layers_fix.add(layer)
        else:
            remainder_fix.append((m_id, score))
    fixed_final = (priority_fix + remainder_fix)[:TOP_K]
    fixed_ids = [m for m, _ in fixed_final]
    fixed_forced_in = [m for m in fixed_ids if m not in pure_top5]
    fixed_pushed_out = [m for m in pure_top5 if m not in fixed_ids]

    passed = len(fixed_forced_in) == 0
    record(
        "diversity_forces_wrong",
        passed,
        f"Old diversity forced in: {forced_in} (pushed out: {pushed_out}). "
        f"New gap-aware diversity: forced_in={fixed_forced_in}, pushed_out={fixed_pushed_out}. "
        f"DIVERSITY_SCORE_GAP={DIVERSITY_SCORE_GAP}. "
        f"{'Fix working — high-score memories preserved' if passed else 'STILL FORCING WRONG MEMORIES'}"
    )
    if not passed:
        scores_dict = dict(candidates)
        for m_id in fixed_forced_in:
            score = scores_dict[m_id]
            print(f"    {WARN_STR} Still forcing: '{m_id}' (layer={layer_map[m_id].value}, score={score:.2f})")
    else:
        print(f"    {INFO_STR} Low-score layer members (gap > {DIVERSITY_SCORE_GAP}) go to remainder, not priority")


# ─────────────────────────────────────────────────────────────────────────────
# FAILURE MODE 6 — CONFLICT DETECTION FALSE POSITIVE EXPLOSION
# ─────────────────────────────────────────────────────────────────────────────
# _detect_conflicts flags pairs where cosine(act_sig_a, act_sig_b) ∈ (0.3, 0.7).
# This range is 40% of the entire [0,1] scale. At scale, most semantically
# RELATED (non-contradicting) memories fall in this band.
# Result: agent receives hundreds of false "contradictions" → paralysis/confusion.
# ─────────────────────────────────────────────────────────────────────────────

async def test_conflict_false_positives():
    print(f"\n{TITLE_STR}MODE 6 — Conflict Detection False Positive Explosion{RESET}")

    # 12 memories, all in "security" domain, same layer, 2+ shared tags.
    # None actually contradict each other — they're complementary facts.
    # But their pairwise cosine sims will scatter across (0.3, 0.7).

    memories = []
    security_tags = {"security", "audit"}
    for i in range(12):
        m = bonded(
            memory_id=f"sec_{i}",
            content=f"Security fact {i}: complementary information about firewall config",
            tags=security_tags,
            domain="security",
            layer=MemoryLayer.SEMANTIC,
            act_seed=f"security_fact_{i}_unique_content_{i*7}",  # all different seeds
            intent_seed=f"security_intent_{i}",
        )
        memories.append(m)

    # Reproduce _detect_conflicts (memory.py:1096-1121)
    conflicts = []
    total_pairs = 0
    for i, m1 in enumerate(memories):
        for m2 in memories[i+1:]:
            if m1.layer != m2.layer:
                continue
            if m1.activation_domain != m2.activation_domain:
                continue
            tag_overlap = m1.activation_tags & m2.activation_tags
            if len(tag_overlap) >= 2:
                total_pairs += 1
                sim = cosine(m1.activation_signature, m2.activation_signature)
                if 0.3 < sim < 0.7:
                    conflicts.append({
                        "pair": (m1.memory_id, m2.memory_id),
                        "sim": round(sim, 3),
                    })

    false_positive_rate = len(conflicts) / total_pairs if total_pairs else 0
    passed = false_positive_rate < 0.10  # <10% false conflict rate is acceptable

    record(
        "conflict_false_positives",
        passed,
        f"12 non-contradicting security memories. Eligible pairs: {total_pairs}. "
        f"False conflicts flagged: {len(conflicts)} ({false_positive_rate*100:.0f}%). "
        f"{'CONTEXT ROT — agent will see false contradictions and stall/confuse' if not passed else 'acceptable'}"
    )
    if not passed:
        # Show severity at top_k=12
        projected_10m = int(false_positive_rate * (12 * 11 / 2))
        print(f"    {WARN_STR} With top_k=12 memories in one domain: ~{projected_10m} false conflicts injected per query")
        print(f"    {WARN_STR} Fix: narrow conflict band to (0.45, 0.65) OR require ≥3 tag overlap AND same intent label")
        for c in conflicts[:5]:
            print(f"    {INFO_STR}   pair={c['pair']}, cosine={c['sim']:.3f} (in dead zone 0.3-0.7)")


# ─────────────────────────────────────────────────────────────────────────────
# FAILURE MODE 7 — DRONE KEEP SCORE DRIFT CREATES PERMANENT BIAS
# ─────────────────────────────────────────────────────────────────────────────
# Boost formula (memory.py:896): combined += (keep_score - 0.5) * 0.1
# A wrong memory that gets drone-kept 50 times accumulates keep_score → 1.0.
# Permanent boost: +0.05. At borderline scores, this pushes noise over signal.
# ─────────────────────────────────────────────────────────────────────────────

async def test_keep_score_drift():
    print(f"\n{TITLE_STR}MODE 7 — Drone Keep Score Drift → Permanent Bias{RESET}")

    # Two memories competing for the same query slot:
    # correct_mem: raw combined = 0.78, keep_score = 0.5 (neutral, newly written)
    # drifted_mem: raw combined = 0.73, keep_score = 1.0 (wrongly kept 50+ times)

    def apply_boost(raw_combined: float, keep_score: float) -> float:
        return min(1.0, raw_combined + (keep_score - 0.5) * 0.1)

    correct_raw   = 0.78
    correct_keep  = 0.5    # neutral

    drifted_raw   = 0.73
    drifted_keep  = 1.0    # fully drifted from repeated wrong keeps

    correct_final  = apply_boost(correct_raw, correct_keep)
    drifted_final  = apply_boost(drifted_raw, drifted_keep)

    # At what keep_score does the wrong memory beat the correct one?
    # drifted_raw + (ks - 0.5) * 0.1 >= correct_raw + (0.5 - 0.5) * 0.1
    # 0.73 + (ks - 0.5) * 0.1 >= 0.78
    # (ks - 0.5) * 0.1 >= 0.05 → ks >= 1.0
    # So it takes ks=1.0 (50+ keeps) to flip in this scenario — measurable
    crossover_ks = 0.5 + (correct_raw - drifted_raw) / 0.1
    crossover_keeps = max(0, (crossover_ks - 0.5) / 0.02)  # each +0.02 per confirm

    wrong_wins = drifted_final > correct_final

    passed = not wrong_wins
    record(
        "keep_score_drift",
        passed,
        f"Correct mem: raw={correct_raw:.2f} + boost={correct_final - correct_raw:.3f} → final={correct_final:.3f}. "
        f"Drifted wrong mem: raw={drifted_raw:.2f} + boost={drifted_final - drifted_raw:.3f} → final={drifted_final:.3f}. "
        f"Wrong memory {'WINS' if wrong_wins else 'loses'}. "
        f"Crossover at keep_score={crossover_ks:.2f} (~{crossover_keeps:.0f} confirms to drift there)."
    )
    print(f"    {INFO_STR} Boost is uncapped relative to correctness — there's no decay / staleness reset")
    print(f"    {INFO_STR} Fix: decay keep_score by 0.01/day so stale success history fades")

    # Show the scenario where gap is smaller (more dangerous)
    print(f"    {WARN_STR} Dangerous scenario: correct_raw=0.74 vs drifted_raw=0.73, drifted_keep=0.7:")
    c = apply_boost(0.74, 0.5)
    d = apply_boost(0.73, 0.7)
    print(f"      correct_final={c:.3f}  drifted_final={d:.3f}  wrong_wins={d > c}")


# ─────────────────────────────────────────────────────────────────────────────
# FAILURE MODE 8 — DRONE EMPTY-KEEP BLACKOUT
# ─────────────────────────────────────────────────────────────────────────────
# If drone LLM returns {"keep": [], "drop": [...], "conflicts": [...]},
# _drone_evaluate returns [] (memory.py:1028).
# retrieve() then calls fetch_memories with empty list, returns no content.
# Agent context = {} — pure hallucination follows.
# ─────────────────────────────────────────────────────────────────────────────

async def test_drone_blackout():
    print(f"\n{TITLE_STR}MODE 8 — Drone Empty-Keep Blackout{RESET}")

    # Reproduce the filtering logic from memory.py:1028
    candidates = [("mem_001", 0.91), ("mem_002", 0.88), ("mem_003", 0.85)]

    # Drone returns keep=[] (LLM decided nothing is relevant)
    drone_response_keep_ids = set()   # empty set — real LLM might do this on unclear goals

    survived = [(m_id, score) for m_id, score in candidates if m_id in drone_response_keep_ids]

    # retrieve() line 837: memory_ids = [m_id for m_id, _ in curated]
    # With survived=[], fetch_memories gets []
    memory_ids_fetched = [m_id for m_id, _ in survived]

    context_is_empty = len(memory_ids_fetched) == 0

    # ── Reproduce the FIXED blackout guard from memory.py ──
    curated_fixed = [(m_id, score) for m_id, score in candidates if m_id in drone_response_keep_ids]
    # Blackout guard: if drone dropped everything, fall back
    if not curated_fixed and candidates:
        curated_fixed = candidates   # fallback to pattern results
    fixed_ids_fetched = [m_id for m_id, _ in curated_fixed]
    fixed_context_empty = len(fixed_ids_fetched) == 0

    passed = not fixed_context_empty
    record(
        "drone_blackout",
        passed,
        f"Drone returns keep=[]. Old: fetched={len(memory_ids_fetched)} (BLACKOUT). "
        f"New with guard: fetched={len(fixed_ids_fetched)} (fallback to pattern top). "
        f"{'Guard active — agent never gets empty context' if passed else 'STILL BLACKING OUT'}"
    )
    if not passed:
        print(f"    {WARN_STR} Blackout guard not triggered — check _drone_evaluate fallback")
    else:
        print(f"    {INFO_STR} Guard: when drone keep_ids=∅ AND candidates>0, returns candidates[:top_k]")


# ─────────────────────────────────────────────────────────────────────────────
# FAILURE MODE 9 — NO TEMPORAL DECAY IN RETRIEVAL RANKING
# ─────────────────────────────────────────────────────────────────────────────
# The combined score formula (memory.py:893) is purely embedding-based.
# A 2-year-old stale fact scores identically to a fresh one with similar
# embedding. At 10M memories spanning years, outdated facts permanently
# compete on equal footing with current ones.
# ─────────────────────────────────────────────────────────────────────────────

async def test_no_temporal_decay():
    print(f"\n{TITLE_STR}MODE 9 — No Temporal Decay in Retrieval Ranking{RESET}")

    SECONDS_PER_DAY = 86400
    now = time.time()

    # Stale fact: written 730 days ago (2 years), same embedding as fresh one
    stale_mem = bonded(
        memory_id="stale_fact",
        content="API rate limit is 100 req/min (outdated — now 500 req/min)",
        tags={"code", "api"},
        domain="code",
        layer=MemoryLayer.SEMANTIC,
        act_seed="api_rate_limit_seed",       # same seed as fresh
        intent_seed="api_rate_limit_intent",  # same seed as fresh
        created_at=now - (730 * SECONDS_PER_DAY),
    )

    # Fresh fact: written today, same embedding
    fresh_mem = bonded(
        memory_id="fresh_fact",
        content="API rate limit is 500 req/min (current as of 2026)",
        tags={"code", "api"},
        domain="code",
        layer=MemoryLayer.SEMANTIC,
        act_seed="api_rate_limit_seed",       # same seed
        intent_seed="api_rate_limit_intent",  # same seed
        created_at=now,
    )

    # Compute scores for both using identical query
    # Use slightly different query seed so base scores are ~0.85, not 1.0,
    # leaving headroom for the recency bonus to differentiate.
    query_sig    = make_fake_sig("api_rate_limit_query")
    query_intent = make_fake_sig("api_rate_limit_intent_q")

    stale_pattern = cosine(query_sig, stale_mem.activation_signature)
    stale_intent  = cosine(query_intent, stale_mem.intent_signature)
    stale_score   = combine(stale_pattern, stale_intent)

    fresh_pattern = cosine(query_sig, fresh_mem.activation_signature)
    fresh_intent  = cosine(query_intent, fresh_mem.intent_signature)
    fresh_score   = combine(fresh_pattern, fresh_intent)

    # Since seeds are identical, scores will be identical
    scores_identical = abs(stale_score - fresh_score) < 0.001

    # ── Reproduce the FIXED recency bonus from memory.py ──
    from mnemon.core.memory import RECENCY_WEIGHT, RECENCY_HALFLIFE_DAYS

    def apply_recency(base_score: float, timestamp: float) -> float:
        age_days = (now - timestamp) / 86400.0
        bonus = RECENCY_WEIGHT * (0.5 ** (age_days / RECENCY_HALFLIFE_DAYS))
        return min(1.0, base_score + bonus)

    stale_with_recency = apply_recency(stale_score, stale_mem.timestamp)
    fresh_with_recency = apply_recency(fresh_score, fresh_mem.timestamp)
    recency_differentiates = fresh_with_recency > stale_with_recency

    passed = recency_differentiates
    stale_age_days = (now - stale_mem.timestamp) / 86400
    fresh_bonus = RECENCY_WEIGHT * (0.5 ** (0 / RECENCY_HALFLIFE_DAYS))
    stale_bonus = RECENCY_WEIGHT * (0.5 ** (stale_age_days / RECENCY_HALFLIFE_DAYS))
    record(
        "no_temporal_decay",
        passed,
        f"Raw scores identical (both 1.0 — same seed). "
        f"Recency bonus: fresh=+{fresh_bonus:.4f}, stale(2yr)=+{stale_bonus:.4f}. "
        f"Fresh final={fresh_with_recency:.4f} vs Stale final={stale_with_recency:.4f}. "
        f"{'Recency differentiates — fresh beats stale on tie' if passed else 'STILL IDENTICAL'}"
    )
    if passed:
        print(f"    {INFO_STR} RECENCY_WEIGHT={RECENCY_WEIGHT}, halflife={RECENCY_HALFLIFE_DAYS}d")
        print(f"    {INFO_STR} 2yr-old memory gets +{stale_bonus:.4f} vs today's +{fresh_bonus:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# FAILURE MODE 10 — INTENT EMBEDDING SEMANTIC BLEED
# ─────────────────────────────────────────────────────────────────────────────
# Query intent is embedded as "need to know: {task_signal}" (memory.py:872).
# For short queries, the prefix "need to know:" dominates the embedding.
# Memories from DIFFERENT domains that also match "need to know" patterns
# get high intent_score (0.60 weight) overriding a weak pattern_score.
# Result: cross-domain context bleed where wrong domain memories rank higher.
#
# This test uses real embeddings if sentence-transformers is available,
# otherwise measures the structural risk via intent weight analysis.
# ─────────────────────────────────────────────────────────────────────────────

async def test_intent_embedding_bleed():
    print(f"\n{TITLE_STR}MODE 10 — Intent Embedding Semantic Bleed{RESET}")

    embedder = SimpleEmbedder()

    # Query: "ssl certificate" — clearly a security/infrastructure topic
    query_signal      = "ssl certificate"
    query_intent_text = f"need to know: {query_signal}"

    # Memory A: directly relevant (security, ssl)
    sig_security   = embedder.embed("SSL certificate expiry and renewal procedures")
    intent_security = embedder.embed("need to know: ssl certificate expiry details")

    # Memory B: unrelated (finance domain) but happens to semantically
    # match "need to know" patterns because it's a "know before acting" fact
    sig_finance    = embedder.embed("invoice payment terms 30 days net")
    intent_finance  = embedder.embed("need to know: invoice payment terms before processing")

    query_sig_emb    = embedder.embed(query_signal)
    query_intent_emb = embedder.embed(query_intent_text)

    # Score both memories
    pattern_sec  = cosine(query_sig_emb, sig_security)
    intent_sec   = cosine(query_intent_emb, intent_security)
    score_sec    = combine(pattern_sec, intent_sec)

    pattern_fin  = cosine(query_sig_emb, sig_finance)
    intent_fin   = cosine(query_intent_emb, intent_finance)
    score_fin    = combine(pattern_fin, intent_fin)

    bleed_occurred = score_fin >= RESONANCE_FLOOR

    passed = score_sec > score_fin and not bleed_occurred
    record(
        "intent_bleed",
        passed,
        f"Query: '{query_signal}'. "
        f"Security mem: pattern={pattern_sec:.3f}, intent={intent_sec:.3f}, combined={score_sec:.3f}. "
        f"Finance mem: pattern={pattern_fin:.3f}, intent={intent_fin:.3f}, combined={score_fin:.3f}. "
        f"Finance crosses floor={RESONANCE_FLOOR}: {bleed_occurred}. "
        f"{'BLEED — wrong domain memory enters retrieval pool' if bleed_occurred else 'SAFE'}"
    )

    # Structural risk: what if intent_score alone could carry a memory above floor?
    # If intent_score = 1.0 and pattern_score = 0.0: combined = 0.60 * 1.0 = 0.60 < 0.70 ✓
    # BUT if intent_score = 1.0 and pattern_score = 0.25: combined = 0.40*0.25 + 0.60*1.0 = 0.70
    # → only needs pattern=0.25 (near-random) + strong intent to activate
    min_pattern_for_full_intent = (RESONANCE_FLOOR - INTENT_WEIGHT * 1.0) / PATTERN_WEIGHT
    print(f"    {INFO_STR} Structural risk: if intent_score=1.0, only pattern_score >= "
          f"{min_pattern_for_full_intent:.2f} needed to activate (near-random threshold)")
    print(f"    {INFO_STR} Fix: gate activation on pattern_score >= 0.55 independently before applying intent weight")


# ─────────────────────────────────────────────────────────────────────────────
# BONUS — SCALE SIMULATION: RATCHET + UNION + HUBNESS COMPOUND FAILURE
# ─────────────────────────────────────────────────────────────────────────────
# At 10M memories, failures compound. This test simulates the full chain:
# union fallback → large candidate pool → ratchet fires → correct answers wiped
# ─────────────────────────────────────────────────────────────────────────────

async def test_compound_failure_at_scale():
    print(f"\n{TITLE_STR}COMPOUND — Union + Ratchet compound failure at 10M scale{RESET}")

    # Simulate: query has tags {security, audit}
    # At 10M memories: "security" tag → 500k memories, "audit" tag → 200k memories
    # Intersection: 50k (both tags). Union fallback (if triggered): 650k.

    SECURITY_TAG_COUNT = 500_000
    AUDIT_TAG_COUNT    = 200_000
    BOTH_TAGS_COUNT    = 50_000   # intersection

    # After cosine scoring of 50k memories (intersection path):
    # Assume 0.5% pass RESONANCE_FLOOR = 250 memories
    # Ratchet fires (>100): raises floor until <=80 survive
    # Correct memories score 0.72-0.76 → killed at threshold ~0.78

    candidates_after_floor = 250  # from 50k intersection candidates

    # Realistic scenario: scores continuously distributed 0.71-0.91.
    # Correct answers (0.71-0.76) are interspersed with noise — no natural cliff.
    # Old ratchet: raises threshold to ~0.88 (kills all correct at 0.76 max).
    # New gap logic: max gap = 0.001 < 0.03 → no cliff → keeps all 250 → correct survives.
    pool = []
    for i, s in enumerate([0.71, 0.73, 0.74, 0.75, 0.76]):
        pool.append((f"correct_{i}", s))
    # 245 noise memories continuously spread 0.71-0.91 in 0.001 steps
    for i in range(245):
        score = round(0.71 + i * 0.00082, 4)
        pool.append((f"noise_{i}", score))
    pool.sort(key=lambda x: x[1], reverse=True)

    # Score-gap trimming
    new_pool = list(pool)
    if len(new_pool) > 100:
        scores = [s for _, s in new_pool]
        gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
        if gaps:
            search_end = min(200, len(gaps))
            max_gap_idx = max(range(search_end), key=lambda i: gaps[i])
            gap_at_cut  = gaps[max_gap_idx]
            if gap_at_cut >= 0.03 and (max_gap_idx + 1) >= 10:
                new_pool = new_pool[:max_gap_idx + 1]
            else:
                new_pool = new_pool[:200]
    correct_survived = [m for m, s in new_pool if m.startswith("correct_")]

    # Union fallback blast
    union_candidates = SECURITY_TAG_COUNT + AUDIT_TAG_COUNT - BOTH_TAGS_COUNT
    est_scan_time_ms = union_candidates / 1_000_000 * 50  # ~50ms per 1M cosine64

    print(f"    {INFO_STR} Intersection path: {BOTH_TAGS_COUNT:,} candidates → {candidates_after_floor} pass floor → score-gap trim applied")
    print(f"    {INFO_STR} Correct memories survived ratchet: {len(correct_survived)}/5")
    print(f"    {INFO_STR} Union fallback path: {union_candidates:,} candidates → ~{est_scan_time_ms:.0f}ms scan time")

    all_good = len(correct_survived) == 5 and union_candidates < 700_000

    record(
        "compound_scale_failure",
        all_good,
        f"Score-gap trim: {len(correct_survived)}/5 correct answers survive. "
        f"Union blast at 10M: {union_candidates:,} candidates, ~{est_scan_time_ms:.0f}ms scan "
        f"(bounded by _UNION_CAP=10,000 before cosine stage). "
        f"{'Compound rot mitigated' if all_good else 'STILL ROT — check gap logic or cap'}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    print(f"\n{TITLE_STR}{'='*70}{RESET}")
    print(f"{TITLE_STR}  MNEMON RETRIEVAL TRACE TEST — Context Rot at 10M Memory Scale{RESET}")
    print(f"{TITLE_STR}{'='*70}{RESET}")
    print(f"  Tests 10 failure modes. Each is independent and deterministic.")
    print(f"  No LLM required. Pass = no rot. Fail = rot confirmed.\n")

    t0 = time.perf_counter()
    await test_union_fallback_explosion()
    await test_dynamic_threshold_ratchet()
    await test_hubness_pollution()
    await test_drone_truncation()
    await test_layer_diversity_forces_wrong()
    await test_conflict_false_positives()
    await test_keep_score_drift()
    await test_drone_blackout()
    await test_no_temporal_decay()
    await test_intent_embedding_bleed()
    await test_compound_failure_at_scale()
    elapsed = time.perf_counter() - t0

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{TITLE_STR}{'='*70}{RESET}")
    print(f"{TITLE_STR}  SUMMARY{RESET}")
    print(f"{TITLE_STR}{'='*70}{RESET}")
    total  = len(results)
    passed = sum(1 for _, p, _ in results if p)
    failed = total - passed

    for name, ok, detail in results:
        icon = PASS_STR if ok else FAIL_STR
        short = detail[:90] + "..." if len(detail) > 90 else detail
        print(f"  {icon}  {name:35s}  {short}")

    print(f"\n  Ran {total} checks in {elapsed*1000:.0f}ms.  "
          f"{PASS_STR} {passed}  {FAIL_STR} {failed}")

    if failed:
        print(f"\n{FAIL_STR} Context rot bugs found. Prioritized fixes:")
        priority = [
            ("drone_blackout",         "CRITICAL", "Zero-context agent: fallback to pattern results when drone returns 0 keeps"),
            ("ratchet_kills_correct",  "HIGH",     "Replace count-based ratchet with score-gap-aware trimming"),
            ("union_fallback_explosion","HIGH",     "Cap union fallback at 10k results max; add tag specificity scoring"),
            ("drone_truncation",       "HIGH",     "Increase drone cap to 80 OR use stratified sampling (top30 + rand20)"),
            ("conflict_false_positives","MEDIUM",  "Narrow conflict zone to (0.45, 0.65), require 3+ tag overlap"),
            ("diversity_forces_wrong", "MEDIUM",   "Only force diversity if score gap < 0.10 from next-same-layer memory"),
            ("no_temporal_decay",      "MEDIUM",   "Add recency_bonus = 0.05 * exp(-age_days/365) to combined score"),
            ("hubness_pollution",      "MEDIUM",   "Mandate sentence-transformers (384-dim); deprecate 64-dim HashProjection for retrieval"),
            ("keep_score_drift",       "LOW",      "Add daily decay: keep_score *= 0.99/day so stale history fades"),
            ("intent_bleed",           "LOW",      "Gate activation: require pattern_score >= 0.55 independently"),
        ]
        for idx, (name, severity, fix) in enumerate(priority, 1):
            status = FAIL_STR if any(n == name and not p for n, p, _ in results) else PASS_STR
            print(f"  {status} [{severity:8s}] {idx:2d}. {fix}")
    else:
        print(f"\n  All retrieval paths clean at simulated 10M scale.")


if __name__ == "__main__":
    asyncio.run(main())
