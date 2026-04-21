"""
Real-world demo: LangChain + Groq agent with Mnemon memory.

This is what a user actually writes. No Mnemon internals exposed.
Run it twice — second run shows memory recalled from prior session.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import mnemon

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── 1. One-line Mnemon setup ──────────────────────────────────────────────────
m = mnemon.init(tenant_id="groq_demo")
# moth (auto-instrumentation) activates automatically via init()

# ── 2. Normal LangChain agent — nothing Mnemon-specific below this line ───────
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant. Answer concisely."),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()

# ── 3. Multi-turn session ─────────────────────────────────────────────────────
questions = [
    "What is the capital of France?",
    "What is 15 multiplied by 7?",
    "What is the capital of France?",   # repeat — should hit cache
    "What is 15 multiplied by 7?",      # repeat — should hit cache
    "Tell me something interesting about Paris.",
]

print("\n=== Mnemon + LangChain + Groq Demo ===\n")

for i, q in enumerate(questions, 1):
    print(f"[Q{i}] {q}")
    answer = chain.invoke({"question": q})
    print(f"     -> {answer.strip()}\n")

# ── 4. Stats ──────────────────────────────────────────────────────────────────
stats = m.stats
print("=== Mnemon Stats ===")
print(f"  Cache hits      : {stats.get('cache_hits', 0)}")
print(f"  Tokens saved    : {stats.get('tokens_saved_est', 0)}")
print(f"  Injections      : {stats.get('memory_injections', 0)}")
print(f"  Protein bond gates: {stats.get('protein_bond_gates', 0)}")
print()

# ── 5. What does memory look like now? ───────────────────────────────────────
recalled = m.recall("Paris France capital")
mems = recalled.get("memories", [])
ids   = recalled.get("memory_ids", [])
scores = recalled.get("memory_scores", {})
print(f"=== Memory recall: 'Paris France capital' ({len(mems)} results) ===")
for mem, mid in zip(mems[:5], ids[:5]):
    text  = mem.get("text", "").encode("ascii", errors="replace").decode("ascii")
    score = scores.get(mid, 0.0)
    print(f"  [{score:.2f}] {text[:120]}")

m.close()
