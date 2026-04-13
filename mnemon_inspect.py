"""
mnemon_inspect.py — builder's view of Claude's live memory.

Shows exactly what's stored, what tags were assigned, what scores
come back on recall, and what the inverted index looks like.

Run: python mnemon_inspect.py
"""

import asyncio
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from mnemon import Mnemon
from mnemon.core.models import MemoryLayer
from mnemon.core.memory import (
    RuleClassifier, SimpleEmbedder,
    PATTERN_WEIGHT, INTENT_WEIGHT
)

TENANT = "claude_global"
DB_DIR = r"C:\Users\MAHIKA\.claude"
FLOOR  = 0.40


async def main():
    async with Mnemon(
        tenant_id=TENANT, agent_id="claude", db_dir=DB_DIR,
        memory_enabled=True, eme_enabled=False, bus_enabled=False,
        resonance_floor=FLOOR, enable_telemetry=False,
    ) as m:

        embedder = m._embedder
        index    = m._memory.index
        db       = m._db

        # ── 1. All stored memories ────────────────────────────────
        print("=" * 58)
        print("  STORED MEMORIES")
        print("=" * 58)
        # fetch all memory IDs first via the index, then fetch full records
        shard = index._shards.get(TENANT, {})
        all_ids = set()
        for ids in shard.values():
            all_ids |= ids
        all_mems = await db.fetch_memories(TENANT, list(all_ids)) if all_ids else []
        for mem in all_mems:
            text = mem.content.get("text", "")
            tags = mem.activation_tags
            sig_len = len(mem.activation_signature) if mem.activation_signature else 0
            print(f"\n  [{mem.layer.value}] imp={mem.importance:.2f}")
            print(f"  text: {text[:90]}")
            print(f"  tags: {tags}")
            print(f"  embedding: {sig_len}-dim  id: {mem.memory_id}")

        # ── 2. Inverted index state ───────────────────────────────
        print("\n" + "=" * 58)
        print("  INVERTED INDEX  (tag -> memory_ids)")
        print("=" * 58)
        for tag, ids in sorted(shard.items()):
            print(f"  {tag:<25} -> {len(ids)} memor{'y' if len(ids)==1 else 'ies'}")

        # ── 3. Live recall with scores ────────────────────────────
        print("\n" + "=" * 58)
        print("  LIVE RECALL  (with raw scores)")
        print("=" * 58)

        queries = [
            "what did we fix in the resonance floor bug",
            "how does EME caching work",
            "what are the user preferences",
        ]

        for query in queries:
            print(f"\n  Q: '{query}'")
            q_sig    = embedder.embed(query)
            q_intent = embedder.embed(f"need to know: {query}")
            q_tags   = RuleClassifier.extract_tags(query, MemoryLayer.EPISODIC)
            print(f"  query tags: {q_tags}")

            # tag intersection
            candidates = await index.intersect(TENANT, q_tags)
            print(f"  candidates from index: {len(candidates)}")

            # score each candidate
            scored = []
            for mem in all_mems:
                if mem.memory_id not in candidates:
                    continue
                if not mem.activation_signature:
                    continue
                p = SimpleEmbedder.cosine_similarity(q_sig, mem.activation_signature)
                i = SimpleEmbedder.cosine_similarity(q_intent, mem.intent_signature) if mem.intent_signature else 0.0
                combined = PATTERN_WEIGHT * p + INTENT_WEIGHT * i
                scored.append((combined, p, i, mem))

            scored.sort(reverse=True)
            for combined, p, i, mem in scored:
                status = "PASS" if combined >= FLOOR else "FAIL"
                text = mem.content.get("text", "")[:60]
                print(f"  [{status}] combined={combined:.3f} p={p:.3f} i={i:.3f} | {text}...")

            # actual recall result
            res  = await m.recall(query, top_k=5)
            mems = res.get("memories", [])
            print(f"  => returned {len(mems)} memor{'y' if len(mems)==1 else 'ies'}")
            for mem_content in mems:
                print(f"     '{mem_content.get('text','')[:70]}...'")

        # ── 4. Facts vault ────────────────────────────────────────
        print("\n" + "=" * 58)
        print("  FACTS VAULT")
        print("=" * 58)
        for key in ["project:Mnemon", "user_stack", "user_preferences", "recent_session"]:
            val = await m.recall_fact(key)
            print(f"\n  {key}:")
            print(f"    {val}")

    print("\n" + "=" * 58)


if __name__ == "__main__":
    asyncio.run(main())
