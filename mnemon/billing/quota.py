"""
Mnemon quota enforcement — free vs. Pro tier.

Free tier: FREE_TIER_DAILY_HITS cache hits per day.
Pro tier:  unlimited (valid Lemon Squeezy license key).

When the free tier limit is hit, cache lookups are bypassed and the
generation_fn is called normally. The agent keeps working — it just
loses the token savings for the rest of the day.
"""

import json
import logging
import urllib.request
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

FREE_TIER_DAILY_HITS = 100
LEMON_SQUEEZY_VALIDATE_URL = "https://api.lemonsqueezy.com/v1/licenses/validate"


class QuotaEnforcer:
    def __init__(self, db, tenant_id: str, license_key: Optional[str] = None):
        self._db = db
        self._tenant_id = tenant_id
        self._license_key = license_key
        self._is_pro = False
        self._daily_hits = 0
        self._today: Optional[str] = None

    async def start(self) -> None:
        self._today = date.today().isoformat()
        self._daily_hits = await self._db.get_daily_hits(self._tenant_id, self._today)
        if self._license_key:
            self._is_pro = await self._validate_license(self._license_key)
            if self._is_pro:
                logger.debug(f"Mnemon: Pro tier active for tenant {self._tenant_id}")
            else:
                logger.warning(
                    "Mnemon: license key is invalid or could not be verified — "
                    "running on free tier"
                )

    async def can_serve_cache_hit(self) -> bool:
        if self._is_pro:
            return True
        # Refresh date in case process runs past midnight
        today = date.today().isoformat()
        if today != self._today:
            self._today = today
            self._daily_hits = await self._db.get_daily_hits(self._tenant_id, self._today)
        allowed = self._daily_hits < FREE_TIER_DAILY_HITS
        if not allowed:
            try:
                from mnemon.core import ph_telemetry
                ph_telemetry._fire("quota_exhausted", {"hits_today": self._daily_hits})
            except Exception:
                pass
        return allowed

    async def record_hit(self) -> None:
        self._daily_hits += 1
        await self._db.record_daily_hit(self._tenant_id, self._today)
        await self._check_milestone()

    async def _check_milestone(self) -> None:
        """Print a one-time message when total cache hits reaches 10.

        At 10 hits the fragment library has seen enough variety to start
        compounding. This is the moment savings become visible.
        """
        try:
            import os as _os, sys as _sys
            db_dir = getattr(self._db, "_db_dir", ".")
            flag = _os.path.join(db_dir, f".mnemon_milestone10_{self._tenant_id}")
            if _os.path.exists(flag):
                return
            total = await self._db.get_total_hits(self._tenant_id)
            if total >= 10:
                try:
                    _os.makedirs(db_dir, exist_ok=True)
                    open(flag, "w").close()
                except OSError:
                    pass
                print(
                    "\nMnemon: 10 cache hits — the library is starting to take shape.\n"
                    "\n"
                    "  This is the part where it starts to get interesting. Mnemon has now\n"
                    "  seen enough of your agent's patterns to begin reusing plan pieces\n"
                    "  across runs. The hit rate will climb from here — not linearly, but\n"
                    "  in jumps as each new workflow pattern gets absorbed.\n"
                    "\n"
                    "  Keep running it. Week two is where people usually notice the difference.\n",
                    file=_sys.stderr, flush=True,
                )
        except Exception:
            pass

    @property
    def is_pro(self) -> bool:
        return self._is_pro

    @property
    def hits_today(self) -> int:
        return self._daily_hits

    @property
    def hits_remaining(self) -> Optional[int]:
        if self._is_pro:
            return None
        return max(0, FREE_TIER_DAILY_HITS - self._daily_hits)

    async def _validate_license(self, key: str) -> bool:
        try:
            payload = json.dumps({"license_key": key}).encode()
            req = urllib.request.Request(
                LEMON_SQUEEZY_VALIDATE_URL,
                data=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read())
                return bool(data.get("valid", False))
        except Exception as e:
            # Fail open — never block a paying user because of a network blip
            logger.debug(f"Mnemon: license validation failed ({e}) — assuming valid")
            return True
