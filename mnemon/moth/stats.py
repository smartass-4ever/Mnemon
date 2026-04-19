"""
Persistent statistics for the Mnemon moth.
Tracks cache hits, memory injections, and protein bond gates per integration.
Stats survive process restarts — loaded from JSON on init, saved on every write.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional


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
