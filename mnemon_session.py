"""
mnemon_session.py — Claude's live memory interface.

Usage:
  python mnemon_session.py recall "query"
  python mnemon_session.py store "memory text" [episodic|semantic] [importance 0-1]
  python mnemon_session.py fact-set key "value"
  python mnemon_session.py fact-get key
  python mnemon_session.py stats
  python mnemon_session.py dump          # show all stored memories
"""

import asyncio
import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from mnemon import Mnemon
from mnemon.core.models import MemoryLayer

TENANT   = "claude_code"
AGENT    = "claude"
DB_DIR   = os.path.dirname(os.path.abspath(__file__))
FLOOR    = 0.40   # tuned for code knowledge base


async def run(cmd, args):
    async with Mnemon(
        tenant_id=TENANT, agent_id=AGENT, db_dir=DB_DIR,
        memory_enabled=True, eme_enabled=False, bus_enabled=False,
        resonance_floor=FLOOR,
    ) as m:

        if cmd == "recall":
            query = " ".join(args) if args else ""
            if not query:
                print("usage: recall <query>"); return
            res = await m.recall(query, top_k=5)
            mems = res.get("memories", [])
            print(f"[{len(mems)} memories for: '{query}']")
            for mem in mems:
                text = mem.get("text", str(mem))
                print(f"  - {text}")

        elif cmd == "store":
            if not args:
                print("usage: store <text> [episodic|semantic] [0.0-1.0]"); return
            text  = args[0]
            layer = MemoryLayer.EPISODIC
            imp   = 0.80
            for a in args[1:]:
                if a == "semantic":   layer = MemoryLayer.SEMANTIC
                elif a == "episodic": layer = MemoryLayer.EPISODIC
                else:
                    try: imp = float(a)
                    except ValueError: pass
            mid = await m.remember(text, layer=layer, importance=imp)
            print(f"stored [{layer.value}] importance={imp}: {text[:80]}")

        elif cmd == "fact-set":
            if len(args) < 2:
                print("usage: fact-set <key> <value>"); return
            await m.learn_fact(args[0], " ".join(args[1:]))
            print(f"fact set: {args[0]} = {' '.join(args[1:])}")

        elif cmd == "fact-get":
            if not args:
                print("usage: fact-get <key>"); return
            val = await m.recall_fact(args[0])
            print(f"{args[0]}: {val}")

        elif cmd == "stats":
            stats = m.get_stats()
            db = stats.get("db", {})
            print(f"tenant:    {TENANT}")
            print(f"memories:  {db.get('memories', 0)}")
            print(f"facts:     {db.get('facts', 0)}")
            print(f"embedder:  {m._embedder.backend_name}")
            print(f"floor:     {FLOOR}")

        elif cmd == "dump":
            # fetch all memories directly from DB
            mems = await m._db.fetch_all_memories(TENANT)
            print(f"[{len(mems)} total memories stored]")
            for mem in mems:
                text = mem.content.get("text", "")[:100]
                print(f"  [{mem.layer.value}] imp={mem.importance:.2f} | {text}")

        else:
            print(__doc__)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    cmd  = sys.argv[1]
    args = sys.argv[2:]
    asyncio.run(run(cmd, args))
