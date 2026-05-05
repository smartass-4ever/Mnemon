# Mnemon — How It Works

## The Problem

Every agent framework is **stateless by default**. Your agent runs the same task Monday, Tuesday, Wednesday — and pays the LLM full price every single time.

```mermaid
flowchart LR
    classDef run fill:#1e1e1e,color:#ccc,stroke:#555
    classDef llm fill:#c0392b,color:#fff,stroke:#96281b
    classDef result fill:#27ae60,color:#fff,stroke:#1e8449
    classDef total fill:#7f1d1d,color:#fff,stroke:#991b1b

    R1["🗓 Monday\nweekly security report\nfor Acme Corp"]:::run
    R2["🗓 Tuesday\nweekly security report\nfor Acme Corp"]:::run
    R3["🗓 Wednesday\nweekly security report\nfor Acme Corp"]:::run

    L1["LLM\n1,250 tokens · 20s · $0.004"]:::llm
    L2["LLM\n1,250 tokens · 20s · $0.004"]:::llm
    L3["LLM\n1,250 tokens · 20s · $0.004"]:::llm

    O1["Same report"]:::result
    O2["Same report"]:::result
    O3["Same report"]:::result

    R1 --> L1 --> O1
    R2 --> L2 --> O2
    R3 --> L3 --> O3

    O1 & O2 & O3 --> WASTE["3× cost · 0 learning · same result every time\n100k plans/day = $50,000/month wasted"]:::total
```

> **You built a smart agent. You got an amnesiac that invoices you twice.**

---

## The Fix — One Line

```python
import mnemon
m = mnemon.init()      # ← this is the entire integration
```

---

## How Mnemon Works

```mermaid
flowchart TB
    classDef init fill:#1a1a2e,color:#e0e0e0,stroke:#4a4a8a
    classDef moth fill:#16213e,color:#e0e0e0,stroke:#0f3460
    classDef eme fill:#0f3460,color:#fff,stroke:#533483
    classDef s1 fill:#16a085,color:#fff,stroke:#1abc9c
    classDef s2 fill:#2980b9,color:#fff,stroke:#3498db
    classDef miss fill:#8e44ad,color:#fff,stroke:#9b59b6
    classDef bus fill:#d35400,color:#fff,stroke:#e67e22
    classDef retro fill:#c0392b,color:#fff,stroke:#e74c3c
    classDef output fill:#27ae60,color:#fff,stroke:#2ecc71

    INIT["m = mnemon.init()"]:::init

    subgraph MOTH_BOX["MOTH — Zero-Code Auto-Instrumentation"]
        MOTH["Patches installed frameworks at startup\nAnthropic SDK · OpenAI SDK · LangChain\nLangGraph · CrewAI · AutoGen\n\nNo changes to your existing agent code"]:::moth
    end

    INIT -->|"one line"| MOTH_BOX

    REQUEST["Your agent makes any LLM call\ne.g. llm.invoke('weekly security report for Acme')"]:::output

    MOTH -->|intercepts every call| EME_BOX

    subgraph EME_BOX["EME — Execution Memory Engine"]
        CHECK{"Seen this\nbefore?"}:::eme
        S1["⚡ System 1\nExact fingerprint match\n2.66ms · 0 tokens · $0.00\nLLM never called"]:::s1
        S2["🔀 System 2\nSemantic match\nReuse matching segments\nLLM fills only the delta"]:::s2
        MISS["📥 Cache miss\nLLM called normally\nResult fingerprinted\n& stored for next time"]:::miss
        CHECK -->|"exact match"| S1
        CHECK -->|"semantically similar\n(≥70% overlap)"| S2
        CHECK -->|"new goal"| MISS
    end

    REQUEST --> MOTH_BOX

    subgraph PREWARMED["Pre-warmed on Day 1"]
        FRAGS["49 fragments from\nreal enterprise runs\nRAG · reasoning\nmulti-agent · tool use\nerror handling · streaming"]
    end

    FRAGS -.->|"cache starts warm\nbefore first real run"| CHECK

    S1 & S2 & MISS --> BUS_BOX

    subgraph BUS_BOX["Experience Bus — Always-On Learning Loop"]
        BUS["Observes every outcome\nlatency · tokens · success · failure\nPattern detection in background"]:::bus
        SIGNALS["Signals fired automatically:\nDEGRADATION — latency spike vs baseline\nPATTERN_FOUND — task type failing >30%\nANOMALY — failure after success streak\nRECOVERY — stable again after failures"]
        BUS --> SIGNALS
    end

    RETRO["Retrospector\nQuarantines bad patterns\nStrengthens winning ones\nFeeds intelligence back to EME"]:::retro

    SIGNALS --> RETRO
    RETRO -->|"cache gets smarter\nhit rate improves\nevery run"| CHECK
```

---

## The Numbers

| | First run | Every repeat |
|---|---|---|
| Latency | ~20,000ms | **2.66ms** |
| Tokens | 1,250 | **0** |
| Cost | $0.004 | **$0.00** |
| Speedup | — | **7,500×** |

### At scale (80% System 1 + 15% System 2 hit rate)

| Daily plans | Monthly cost saved |
|---|---|
| 100 | $56 |
| 1,000 | $503 |
| 10,000 | $5,034 |
| 100,000 | **$50,344** |

---

## What Makes This Different

| | Mnemon | Mem0 | LangMem | Roll your own |
|---|:---:|:---:|:---:|:---:|
| Skip LLM entirely on repeated work | ✅ | ❌ | ❌ | ❌ |
| System learning loop | ✅ | ❌ | ❌ | ❌ |
| Zero-code auto-instrumentation | ✅ | ❌ | ❌ | ❌ |
| Fully local — no cloud, no API | ✅ | ❌ | ❌ | ✅ |
| Drift detection | ✅ | ❌ | ❌ | ❌ |
| One-line setup | ✅ | ❌ | ❌ | ❌ |

Every other library makes your prompt slightly better.  
**Mnemon eliminates the LLM call on repeated work and makes the next run cheaper than the last.**
