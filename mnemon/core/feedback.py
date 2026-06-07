"""
Feedback loop tracker — per-tenant, JSON-backed, cross-session.

Drives three user-facing feedback mechanisms:
  1. Zero-hit diagnostic  — printed once after 20 runs with no cache hits
  2. Weekly nudge         — prints this-week summary once per calendar week
  3. PostHog session_end  — fires aggregate session data for Mahika's dashboard
"""

import json
import os
from datetime import date
from typing import Optional


class FeedbackTracker:
    """
    Reads/writes ~/.mnemon/mnemon_feedback_{tenant_id}.json.
    Thread-safe enough for single-user CLI use (no explicit lock needed).
    """

    _ZERO_HIT_THRESHOLD = 20  # runs before we warn

    def __init__(self, tenant_id: str, db_dir: str = "."):
        self._tenant = tenant_id
        home = os.path.join(os.path.expanduser("~"), ".mnemon")
        try:
            os.makedirs(home, exist_ok=True)
        except OSError:
            home = db_dir
        self._path = os.path.join(home, f"mnemon_feedback_{tenant_id}.json")
        self._data = self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        try:
            with open(self._path) as f:
                return json.load(f)
        except Exception:
            return {
                "total_runs": 0,
                "total_hits": 0,
                "zero_hit_warned": False,
                "week_key": "",
                "week_runs": 0,
                "week_hits": 0,
                "week_tokens": 0,
                "last_nudge_week": "",
            }

    def _save(self) -> None:
        try:
            with open(self._path, "w") as f:
                json.dump(self._data, f)
        except Exception:
            pass

    # ── write path ───────────────────────────────────────────────────────────

    def record_run(self, hit: bool, tokens_saved: int = 0) -> None:
        """Call once per LLM call — before or after the call, hit=True if served from cache."""
        current_week = self._week_key()
        if self._data.get("week_key") != current_week:
            self._data["week_key"]   = current_week
            self._data["week_runs"]  = 0
            self._data["week_hits"]  = 0
            self._data["week_tokens"] = 0

        self._data["total_runs"]  = self._data.get("total_runs", 0) + 1
        self._data["week_runs"]   = self._data.get("week_runs", 0) + 1

        if hit:
            self._data["total_hits"]   = self._data.get("total_hits", 0) + 1
            self._data["week_hits"]    = self._data.get("week_hits", 0) + 1
            self._data["week_tokens"]  = self._data.get("week_tokens", 0) + tokens_saved

        self._save()

    # ── read path ────────────────────────────────────────────────────────────

    def should_warn_zero_hits(self) -> bool:
        return (
            self._data.get("total_runs", 0) >= self._ZERO_HIT_THRESHOLD
            and self._data.get("total_hits", 0) == 0
            and not self._data.get("zero_hit_warned", False)
        )

    def mark_zero_hit_warned(self) -> None:
        self._data["zero_hit_warned"] = True
        self._save()

    def get_weekly_nudge(self) -> Optional[dict]:
        """
        Returns this-week summary dict if we haven't nudged yet this week and
        there's at least one run. Returns None otherwise.
        """
        current_week = self._week_key()
        if self._data.get("last_nudge_week") == current_week:
            return None
        if self._data.get("week_key") != current_week:
            return None
        runs = self._data.get("week_runs", 0)
        if runs == 0:
            return None
        return {
            "week":   current_week,
            "runs":   runs,
            "hits":   self._data.get("week_hits", 0),
            "tokens": self._data.get("week_tokens", 0),
        }

    def mark_nudge_sent(self) -> None:
        self._data["last_nudge_week"] = self._week_key()
        self._save()

    def lifetime_summary(self) -> dict:
        return {
            "total_runs": self._data.get("total_runs", 0),
            "total_hits": self._data.get("total_hits", 0),
        }

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _week_key() -> str:
        year, week, _ = date.today().isocalendar()
        return f"{year}-W{week:02d}"
