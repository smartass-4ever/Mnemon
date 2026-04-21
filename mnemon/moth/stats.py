"""
Persistent statistics for the Mnemon moth.
Tracks cache hits, memory injections, and protein bond gates per integration.
Stats survive process restarts — loaded from JSON on init, saved on every write.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional


_COST_PER_CALL_USD = 0.006  # conservative mid-tier model average


@dataclass
class RecallTrace:
    source: str
    query: str
    injected: bool
    preview: str
    ts: float

    def __repr__(self) -> str:
        status = "injected" if self.injected else "gated"
        age = f"{time.time() - self.ts:.0f}s ago"
        preview = f"\n  └─ {self.preview!r}" if self.injected and self.preview else ""
        return (
            f"RecallTrace(source={self.source!r}, {status}, {age}, "
            f"query={self.query[:60]!r}{preview})"
        )


class MothStats:
    _EST_TOKENS_PER_HIT = 500

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self._persist_path       = persist_path
        self._hits:              Dict[str, int] = defaultdict(int)
        self._injections:        Dict[str, int] = defaultdict(int)
        self._gates:             Dict[str, int] = defaultdict(int)
        self._tokens:            int = 0
        self._tokens_known_hits: int = 0
        self._history:           deque = deque(maxlen=20)
        # query_log: hash → {preview, count, first_seen, last_seen}
        self._query_log:         Dict[str, dict] = {}
        if persist_path:
            self._load()

    def _load(self) -> None:
        try:
            with open(self._persist_path) as f:
                data = json.load(f)
            self._hits              = defaultdict(int, data.get("hits", {}))
            self._injections        = defaultdict(int, data.get("injections", {}))
            self._gates             = defaultdict(int, data.get("gates", {}))
            self._tokens            = data.get("tokens", 0)
            self._tokens_known_hits = data.get("tokens_known_hits", 0)
            self._query_log         = data.get("query_log", {})
            for h in data.get("history", []):
                try:
                    self._history.append(RecallTrace(**h))
                except Exception:
                    pass
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

    def _save(self) -> None:
        if not self._persist_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._persist_path)), exist_ok=True)
            data = {
                "hits":               dict(self._hits),
                "injections":         dict(self._injections),
                "gates":              dict(self._gates),
                "tokens":             self._tokens,
                "tokens_known_hits":  self._tokens_known_hits,
                "query_log":          self._query_log,
                "history": [
                    {"source": h.source, "query": h.query, "injected": h.injected,
                     "preview": h.preview, "ts": h.ts}
                    for h in self._history
                ],
            }
            with open(self._persist_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def record_query(self, query: str) -> None:
        """Track every LLM call query. Repeated queries across sessions = wasted money."""
        if not query or len(query) < 5:
            return
        try:
            qhash = hashlib.md5(query[:150].lower().strip().encode()).hexdigest()[:16]
            now = time.time()
            if qhash in self._query_log:
                self._query_log[qhash]["count"] += 1
                self._query_log[qhash]["last_seen"] = now
            else:
                self._query_log[qhash] = {
                    "preview":    query[:100],
                    "count":      1,
                    "first_seen": now,
                    "last_seen":  now,
                }
            if len(self._query_log) > 2000:
                # Evict least-recently-seen when log grows large
                oldest = sorted(self._query_log.items(), key=lambda x: x[1]["last_seen"])
                for k, _ in oldest[:200]:
                    del self._query_log[k]
            self._save()
        except Exception:
            pass

    def record_hit(self, source: str, tokens: Optional[int] = None) -> None:
        self._hits[source] += 1
        if tokens is not None:
            self._tokens += tokens
            self._tokens_known_hits += 1
        self._save()

    def record_injection(self, source: str, query: str, context: str) -> None:
        self._injections[source] += 1
        self._history.append(
            RecallTrace(source=source, query=query, injected=True,
                        preview=context[:150], ts=time.time())
        )
        self._save()

    def record_gate(self, source: str, query: str) -> None:
        self._gates[source] += 1
        self._history.append(
            RecallTrace(source=source, query=query, injected=False,
                        preview="", ts=time.time())
        )
        self._save()

    @property
    def total_hits(self) -> int:
        return sum(self._hits.values())

    @property
    def tokens_saved_est(self) -> int:
        unknown_hits = self.total_hits - self._tokens_known_hits
        return self._tokens + (unknown_hits * self._EST_TOKENS_PER_HIT)

    @property
    def summary(self) -> dict:
        all_sources = sorted(
            set(self._hits) | set(self._injections) | set(self._gates)
        )
        return {
            "cache_hits":         self.total_hits,
            "llm_calls_saved":    self.total_hits,
            "tokens_saved_est":   self.tokens_saved_est,
            "memory_injections":  sum(self._injections.values()),
            "protein_bond_gates": sum(self._gates.values()),
            "by_integration": {
                src: {
                    "hits":       self._hits.get(src, 0),
                    "injections": self._injections.get(src, 0),
                    "gates":      self._gates.get(src, 0),
                }
                for src in all_sources
            },
        }

    @property
    def last_recall(self) -> Optional[RecallTrace]:
        return self._history[-1] if self._history else None

    @property
    def recall_history(self) -> List[RecallTrace]:
        return list(self._history)

    def waste_report(self, cost_per_call: float = _COST_PER_CALL_USD) -> str:
        """
        Personal waste summary — shows repeated LLM calls across sessions
        and their estimated dollar cost. Designed to be visceral and specific.
        """
        repeated = {
            qhash: entry
            for qhash, entry in self._query_log.items()
            if entry["count"] > 1
        }
        if not repeated:
            hits = self.total_hits
            if hits == 0:
                return (
                    "Mnemon waste report — no data yet.\n"
                    "Run your agent a few times to see what you're paying for twice."
                )
            return (
                f"Mnemon waste report — {hits} LLM call(s) served from cache.\n"
                f"No repeated queries detected yet across sessions. Good signal.\n"
                f"Estimated saved: ~${hits * cost_per_call:.3f}"
            )

        total_repeated_calls = sum(e["count"] - 1 for e in repeated.values())
        total_wasted_usd     = total_repeated_calls * cost_per_call
        days_tracked = 0
        if repeated:
            earliest = min(e["first_seen"] for e in repeated.values())
            days_tracked = max(1, int((time.time() - earliest) / 86400))

        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "  Mnemon Waste Report",
            f"  {days_tracked} day(s) of history · {len(repeated)} repeated query pattern(s)",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
        ]

        sorted_repeated = sorted(repeated.items(), key=lambda x: x[1]["count"], reverse=True)
        for _, entry in sorted_repeated[:10]:
            count     = entry["count"]
            wasted    = (count - 1) * cost_per_call
            preview   = entry["preview"]
            if len(preview) > 72:
                preview = preview[:69] + "..."
            lines.append(f'  "{preview}"')
            lines.append(
                f"    → asked {count}x · {count - 1} redundant call(s) · "
                f"wasted ~${wasted:.3f}"
            )
            lines.append("")

        if len(sorted_repeated) > 10:
            lines.append(f"  ... and {len(sorted_repeated) - 10} more repeated patterns")
            lines.append("")

        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Total redundant calls: {total_repeated_calls}",
            f"  Estimated wasted:      ~${total_wasted_usd:.3f}",
            f"  Mnemon saved:          ~${self.total_hits * cost_per_call:.3f} (cache hits)",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        s = self.summary
        lines = [
            "Mnemon moth — session stats",
            f"  Cache hits:          {s['cache_hits']}  "
            f"({s['llm_calls_saved']} LLM calls saved, "
            f"~{s['tokens_saved_est']:,} tokens)",
            f"  Memory injections:   {s['memory_injections']}",
            f"  Protein bond gates:  {s['protein_bond_gates']}  "
            f"(context rot prevented)",
        ]
        if s["by_integration"]:
            lines.append("")
            lines.append("  By integration:")
            for src, c in s["by_integration"].items():
                parts = []
                if c["hits"]:
                    parts.append(f"{c['hits']} hit{'s' if c['hits'] != 1 else ''}")
                if c["injections"]:
                    parts.append(
                        f"{c['injections']} injection"
                        f"{'s' if c['injections'] != 1 else ''}"
                    )
                if c["gates"]:
                    parts.append(
                        f"{c['gates']} gate{'s' if c['gates'] != 1 else ''}"
                    )
                lines.append(
                    f"    {src:<24} {' · '.join(parts) or 'no activity'}"
                )
        return "\n".join(lines)
