# Mnemon: Execution Memory as the Experience Component of Adaptive Intelligence

**Mahika Jadhav**  
Independent Research  
mahikajadhav22@gmail.com

---

## Abstract

We argue that the path to adaptive artificial intelligence requires a system capable of producing *decisions* — not retrievals, not predictions — from the combination of context, accumulated experience, explicit goals, and core priors. Current approaches (larger models, wider context windows) scale one component while neglecting the others, particularly experience. We present Mnemon, a production system that implements the experience component: a dual-system execution memory engine that accumulates agent plan executions across sessions, enables semantic recall without retraining, and learns from failure through automated reputation management. In evaluation over 45 runs across three enterprise workflow types, Mnemon reduces token consumption by 93.3% (651 vs. 9,786 tokens), eliminates 93% of LLM API calls (3 vs. 45), and achieves cache hit latency of 2.66ms compared to ~20,000ms for fresh generation — a 7,500× speedup. Under 50-agent concurrent load, wall-clock time drops from ~1,000s to 0.18s. The system requires no model retraining, no additional infrastructure, and activates with a single import. We discuss how Mnemon fits into a broader architecture (EROS) designed to assemble all components of the decision formula.

---

## 1. Introduction

There are two directions current AI development is taking.

The first is scaling: larger models trained on more data, with wider context windows. Tokens become cheaper. Attention spans grow longer. The implicit assumption is that sufficient context and sufficient parameters will produce general intelligence.

This path has a measurable ceiling. Training improvements are subject to diminishing returns as data availability and model architecture constrain further gains [CITE]. Context window scaling introduces its own failure mode: as window size grows, fine-grained details — edge cases, low-frequency patterns, historically relevant anomalies — are consistently underweighted against high-volume signals. A model with a 1M-token context window does not remember better; it attends to more while retaining proportionally less of what matters.

The second path starts from a different question: what does a system need in order to *decide*?

We define adaptability as the capacity to handle novel situations using context, environment, and accumulated experience. Adaptability is the clearest behavioral indicator of intelligence — a system that is adaptable in the general sense is, by definition, an intelligent one. To produce adaptable behavior, a system must generate decisions. A decision, formally, is a function of:

```
D = f(context, experience, goal, priors)
```

Where:
- **context**: the current environmental state
- **experience**: accumulated knowledge from prior interactions
- **goal**: the objective being pursued
- **priors**: fixed beliefs or instinctive constraints (equivalent to survival instincts or core values)

Current LLM systems address context and goal through prompt construction. They do not accumulate experience — every call begins from the same parametric state. They have priors only insofar as training data encodes them, which means those priors cannot be updated through interaction.

This paper addresses the experience component. We present **Mnemon**: a production execution memory system that gives agents the ability to accumulate, recall, and learn from prior executions without retraining. We describe its architecture in full, present empirical results, and situate it within a broader vision for assembling the complete decision formula.

---

## 2. Background and Related Work

**Prompt caching** (Anthropic, 2024; OpenAI, 2024) caches the KV state of a fixed prompt prefix. This reduces cost for repeated prompts but does not persist across sessions, does not generalize to semantically similar inputs, and does not learn from outcomes.

**RAG (Retrieval-Augmented Generation)** [CITE] enables retrieval from external knowledge stores. RAG retrieves facts; it does not cache computations. An agent that retrieves relevant documents still generates its plan from scratch on every call.

**Semantic caching** (GPTCache, Redis SemanticCache) caches LLM responses by embedding similarity. These systems treat each query independently, do not decompose plans into reusable segments, and provide no failure learning loop.

**Memory systems** (Mem0, MemGPT, LangMem) maintain persistent user or session memory for conversational context. They do not cache execution plans, do not decompose agent workflows into reusable fragments, and do not track outcome-based reputation.

**The NeurIPS 2025 Stanford result** (arXiv:2506.14852) validated plan caching as an approach, reporting 50.31% cost reduction. Mnemon is an independent production implementation of this class of idea, built from first principles with a complete learning loop absent from that work.

Mnemon differs from all prior work in three ways: (1) it caches structured execution plans at the segment level, enabling partial reuse; (2) it learns from failure outcomes through automated quarantine; (3) it integrates transparently with all major agent frameworks without code changes.

---

## 3. System Architecture

Mnemon consists of four components: the Execution Memory Engine (EME), the Retrospector, the Persistence Layer (EROS Database), and the MOTH auto-instrumentation layer.

### 3.1 Execution Memory Engine

The EME implements two caching systems operating in cascade.

**System 1** is an exact fingerprint match. It maintains an in-memory hash table mapping 32-byte fingerprints to template IDs. Sub-millisecond lookup; zero LLM calls.

**System 2** is a semantic similarity match. It uses goal embeddings and multi-component scoring to identify structurally similar cached plans, decomposes them at the segment level, and reuses matched segments while routing unmatched segments to a three-tier fragment library.

The cascade is: System 1 → System 2 → Full Generation. The first level that produces a valid result wins. Successful generations are always cached for future retrieval.

### 3.2 Fingerprinting (System 1)

Each computation is fingerprinted with five components (Equation 1):

```
h_goal       = MD5(goal)[:16]
h_schema     = MD5(JSON(input_schema))[:16]
h_context    = MD5(JSON(context))[:16]
h_capability = MD5(JSON(sorted(capabilities)))[:16]
h_constraint = MD5(JSON(constraints))[:16]

full_hash = SHA256(h_goal | h_schema | h_context | h_capability | h_constraint)[:32]
```

The five-component design ensures that same-goal queries with different schemas, contexts, or capability sets produce distinct fingerprints. This prevents false cache hits across structurally different tasks while enabling exact reuse for truly identical computations.

### 3.3 Semantic Similarity Matching (System 2)

System 2 activates when System 1 produces no hit. The pipeline is:

**Step 1: Candidate shortlisting.** Goal embedding is compared against an in-memory TemplateIndex — a numpy matrix of all cached template embeddings, updated incrementally on each cache write. Top-20 candidates are retrieved via vectorized cosine product (`mat @ qvec`), replacing a full table scan. Under 100 concurrent agents, this eliminates 100 full DB scans per second.

**Step 2: Goal similarity gate.** Each candidate must pass `cosine_similarity(goal_embedding, template_embedding) >= 0.60` (MIN_GOAL_SIMILARITY). This gates templates that are structurally indexed near the query but semantically distant.

**Step 3: Multi-component scoring.** The best-scoring template is selected by:

```
score = 0.30 × goal_sim
      + 0.25 × schema_sim
      + 0.25 × context_sim
      + 0.20 × capability_sim
```

Where schema\_sim = 1.0 if hashes match, else 0.3; context\_sim = 1.0 if hashes match, else 0.4; capability\_sim = Jaccard overlap / max cardinality. The template is accepted if `score >= 0.70`.

**Step 4: Collective boost.** Pre-warmed templates with confirmed cross-tenant success receive a learned boost from the SignalDatabase, applied before threshold comparison: `score = min(1.0, score + boost)`.

### 3.4 Segment-Level Intent Matching with Spreading Activation

Accepted templates are decomposed into segments. Each segment carries an intent embedding — a compact semantic representation of what the step does, embedded in the same vector space as the goal.

**Base scores** are computed as `cosine_similarity(goal_embedding, segment.signature)`.

**Spreading activation** propagates confidence through the dependency graph. Segments declare dependencies (edges in a DAG); Mnemon builds a bidirectional adjacency from these edges and runs BFS:

```
for each source_id with base_score >= 0.35:
  queue = [(source_id, base_score)]
  while queue:
    current_id, score = queue.pop()
    for neighbor in adj[current_id]:
      spread = score × 0.85          # SPREADING_DECAY
      if spread > activated[neighbor]:
        activated[neighbor] = spread
        queue.append((neighbor, spread))
```

This allows a high-confidence match on one step to boost adjacent steps — critical when the goal varies but the plan structure is largely reusable.

**Classification** is threshold-based:

| Score range | Classification | Action |
|---|---|---|
| ≥ 0.72 | Clear match | Reuse from cache |
| 0.35 – 0.72 | Ambiguous | LLM drone verify (YES/NO prompt) |
| < 0.35 | Clear miss | Gap fill |

The drone covers approximately 3–8% of segments in practice. If no drone function is configured, ambiguous segments default to gap fill (conservative).

### 3.5 Three-Tier Gap Fill

Unmatched segments route through a three-tier fragment library lookup. Each tier uses a combined score:

```
adjusted = cosine × 0.60 + reputation × 0.30 + edge_strength × 0.10
```

Where `reputation` is the framework-specific success rate from the SignalDatabase (default 0.5 neutral), and `edge_strength` is the Hebbian synaptic weight from the previous fragment in the sequence (learned from co-occurrence in successful plans, default 0.5).

Quarantined fragments (identified by the Retrospector) are skipped before scoring.

| Tier | Threshold | Cost |
|---|---|---|
| 1: Exact | adjusted ≥ 0.98 | Zero |
| 2: Similar | adjusted ≥ 0.80 | Zero |
| 3: Miss | No match | LLM gap-fill |

Tier 3 produces a `GapFillRequest` — a structured directive containing a position, a context window of neighboring segments, and a hint. These are collected and passed to `_guided_generation`.

### 3.6 Guided Generation

When pending gaps exist, Mnemon constructs a capsule brief. Matched segments are represented as `{position, intent, outputs}` — compact summaries, not full content. The user's LLM receives:

```json
{
  "pre_filled": [
    {"position": 0, "intent": "authenticate user", "outputs": ["session_token"]},
    {"position": 2, "intent": "run analysis", "outputs": ["report_data"]}
  ],
  "gaps_to_fill": [
    {"position": 1, "receives": ["session_token"], "hint": "fetch_data using data_api"}
  ],
  "instruction": "Generate ONLY the gaps. Return JSON: {position: content}"
}
```

Full cached content is never sent to the user's LLM — only capsule summaries. The LLM generates only what is missing, and the full plan is reconstituted by Mnemon.

Gap-fill outputs are embedded, added to the fragment library, and cached as a new template. The next call with the same variation hits System 1 immediately.

A three-tier parser handles LLM output variability: strict JSON parse → embedded JSON scan → positional decomposition. All-tier failure falls back to full generation, ensuring correctness is never compromised.

### 3.7 Retrospector: Failure Learning

The Retrospector analyzes every failed execution. Its failure diagnosis pipeline is:

1. **Cascade root** (confidence 0.95) — if the Experience Bus has confirmed a root cause fragment
2. **Failed step** (0.90) — if the framework reported a specific failure position
3. **Failure class inference** (0.60–0.65) — from error type signals (tool_error, schema, max_iter)
4. **Last fragment fallback** (0.40) — uncertain, soft flag only

**Quarantine TTLs by failure class:**

| Class | TTL |
|---|---|
| exception | 168h (7 days) |
| wrong_plan | 96h |
| tool_error | 72h (3 days) |
| validation / schema | 48h |
| max_iter | 24h |
| retry | 12h |

Quarantine is applied only at confidence ≥ 0.70, preventing over-aggressive eviction from uncertain diagnoses. Pattern confirmation requires ≥ 2 failures of the same fragment in the same tenant before full quarantine is applied.

On success, Retrospector strengthens Hebbian edge weights for all fragment co-occurrences in the plan. On failure, it weakens the edge leading to the failed fragment. This creates a self-improving fragment graph that routes future gap fills toward proven sequences.

### 3.8 Persistence Layer

Mnemon uses SQLite with WAL mode for all persistence:

```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=-65536;   -- 64 MB page cache
PRAGMA busy_timeout=5000;   -- 5s retry on SQLITE_BUSY
```

WAL mode allows concurrent readers with a single writer, suitable for multi-agent deployments. Per-tenant isolation is enforced at the query level (`WHERE tenant_id=?`). A 10ms write-behind queue (`WriteBehindQueue`) batches fragment writes during burst load, preventing serialization under 100+ concurrent agents.

### 3.9 Embedding Pipeline

The embedder selects the best available backend automatically:

| Priority | Backend | Model | Dimensions | System 2 Active |
|---|---|---|---|---|
| 1 | sentence-transformers | all-MiniLM-L6-v2 | 384 | Yes (~85% recall) |
| 2 | OpenAI | text-embedding-3-small | 1536 | Yes (~90% recall) |
| 3 | Hash Projection | — | 64 (activation) | No (System 1 only) |

The hash projection fallback uses MD5-indexed token frequency vectors with L2 normalization. It enables System 1 exact caching for users without embedder access, while clearly indicating that System 2 semantic recall is inactive.

An LRU embedding cache (2048 slots) prevents redundant embedding calls for repeated goal strings — critical in agent swarms where many instances share goal prefixes.

### 3.10 MOTH Auto-Instrumentation

MOTH (Memory Orchestration Through Hooks) patches installed agent frameworks at import time:

| Framework | Patch Point |
|---|---|
| Anthropic SDK | `client.messages.create` |
| OpenAI SDK | `client.chat.completions.create` |
| LangChain | `BaseChatModel.invoke` / `ainvoke` |
| LangGraph | `CompiledGraph.invoke` / `ainvoke` |
| AutoGen | `ConversableAgent.generate_reply` |
| CrewAI | Event bus (`AgentExecutionStarted/Completed`) |

Patches are applied only to already-imported modules (reducing cold-import overhead). All patches are fail-safe: if patching fails, the original framework behavior is preserved.

---

## 4. Evaluation

### 4.1 Setup

- **Model:** claude-sonnet-4-6 ($3.00/M input · $15.00/M output)
- **Workflow types:** security audits, invoice processing, weekly reports
- **Total runs:** 45 (3 types × 15 runs each)
- **Embedder:** sentence-transformers (all-MiniLM-L6-v2)
- **Storage:** local SQLite

### 4.2 Token and Cost Reduction

| | Without Mnemon | With Mnemon | Reduction |
|---|---|---|---|
| Total tokens | 9,786 | 651 | **93.3%** |
| LLM API calls | 45 | 3 | **93%** |
| Total cost | $0.005774 | $0.000384 | **93.3%** |

Cache miss breakdown: 3 cold-start runs (first run per workflow type). System 2 hits: 12 runs. System 1 hits: 30 runs.

### 4.3 Latency

| | System 1 Hit | Fresh LLM |
|---|---|---|
| p50 | 1ms | ~18,000ms |
| p95 | 7ms | ~25,000ms |
| p99 | 12ms | ~30,000ms |
| **Mean** | **2.66ms** | **~20,000ms** |

**Speedup: 7,500×**

### 4.4 Concurrency (50-Agent Burst)

| | Without | With |
|---|---|---|
| API calls | 50 | 0 |
| Tokens | 62,500 | 0 |
| Cost | $0.9375 | $0.00 |
| Wall-clock | ~1,000s | **0.18s** |

### 4.5 System 2 Segment-Level Savings

Partial cache hits save proportional tokens based on matched segment count:

| Segments matched | Tokens saved | Cost saved |
|---|---|---|
| 5/5 | 1,250 | $0.019 |
| 4/5 | 1,000 | $0.015 |
| 3/5 | 750 | $0.011 |
| 2/5 | 500 | $0.008 |
| 1/5 | 250 | $0.004 |

### 4.6 Projected Monthly Savings (80% S1, 15% S2 avg 3 matched, 5% miss)

| Daily runs | Monthly tokens saved | Monthly cost saved |
|---|---|---|
| 100 | 3,750,000 | $56 |
| 1,000 | 33,562,500 | $503 |
| 10,000 | 335,625,000 | $5,034 |
| 100,000 | 3,356,250,000 | $50,344 |

---

## 5. The Decision Formula and EROS

Mnemon implements one component of the decision formula:

```
D = f(context, experience, goal, priors)
```

**Experience** is what Mnemon addresses: accumulated execution memory, outcome-based learning, semantic recall across sessions.

The remaining components require separate systems:

- **Context** — real-time environmental state, tool outputs, world model updates
- **Goal** — objective alignment, goal decomposition, subgoal generation
- **Priors** — fixed belief constraints, safety boundaries, core values

Assembling these four components into a unified decision-making system is the objective of **EROS** — the broader architecture this work belongs to.

When complete, EROS is designed to function as a human-machine interface: a system that accumulates experience from every interaction, maintains context across sessions, pursues explicit goals with stable priors, and produces decisions that adapt to novel situations without retraining.

This is the second path. Not a larger model. Not a wider context window. A system that learns from its environment and grows.

Mnemon is the first component. EROS is the full vision.

---

## 6. Conclusion

We have presented Mnemon, a production execution memory system implementing the experience component of a general decision formula. The system achieves 93.3% token reduction and 7,500× latency improvement in evaluation, operates without infrastructure changes, and integrates transparently with all major agent frameworks.

The broader claim is architectural: adaptive intelligence requires more than scale. It requires a system that accumulates experience, operates on explicit goals, maintains stable priors, and integrates real-time context — all to produce *decisions*, not predictions. Mnemon demonstrates that the experience component is buildable, measurable, and deployable today.

The path to AGI is not a larger model. It is a more complete system.

---

## References

[1] arXiv:2506.14852 — *Agentic Plan Caching: Test-Time Memory for Fast and Cost-Efficient LLM Agents*, Stanford, NeurIPS 2025.

[2] Lewis et al. — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020.

[3] Anthropic — Prompt Caching documentation, 2024.

[4] OpenAI — Prompt Caching documentation, 2024.

---

*Code: https://github.com/smartass-4ever/Mnemon*  
*Package: pip install mnemon-ai*
