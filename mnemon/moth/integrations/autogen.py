"""
Mnemon moth — AutoGen integration.

Supports both AutoGen v0.2 (autogen) and v0.4+ (autogen_agentchat).

AutoGen exposes register_reply() — a first-class hook for injecting
custom reply functions. This integration uses it to:

  1. Cache agent replies at the task level (System 2 EME per agent turn)
  2. Zero user code changes required.

For v0.2:  patches ConversableAgent.generate_reply directly
For v0.4+: patches BaseChatAgent.on_messages
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Dict, List, Optional, Union

from mnemon.moth import MnemonIntegration
from ._utils import prompt_hash, track_cache_hit
from ._cache import BoundedTTLCache

logger = logging.getLogger(__name__)

_reply_cache = BoundedTTLCache(maxsize=500, ttl=3600)


class AutoGenIntegration(MnemonIntegration):
    name = "autogen"

    def __init__(self) -> None:
        self._original_generate_reply: Optional[Any] = None
        self._patched_v2 = False
        self._mnemon: Optional[Any] = None

    def is_available(self) -> bool:
        return (
            importlib.util.find_spec("autogen_agentchat") is not None
            or importlib.util.find_spec("autogen") is not None
        )

    def patch(self, mnemon: Any) -> None:
        self._mnemon = mnemon
        if importlib.util.find_spec("autogen_agentchat") is not None:
            self._patch_v4(mnemon)
        else:
            self._patch_v2(mnemon)

    def _patch_v4(self, mnemon: Any) -> None:
        try:
            from autogen_agentchat.agents._base_chat_agent import BaseChatAgent
            self._original_generate_reply = BaseChatAgent.on_messages
            m    = mnemon
            orig = self._original_generate_reply

            async def patched_on_messages(
                _self: Any, messages: Any, cancellation_token: Any = None
            ) -> Any:
                cache_key = _autogen_cache_key(_self, messages)
                if cache_key in _reply_cache:
                    track_cache_hit(m, "autogen")
                    logger.debug("Mnemon: AutoGen v4 cache hit")
                    return _reply_cache[cache_key]

                result = await orig(_self, messages, cancellation_token)
                _reply_cache[cache_key] = result
                return result

            BaseChatAgent.on_messages = patched_on_messages
            self._patched_v2 = False

        except Exception as e:
            logger.debug(f"Mnemon: AutoGen v4 patch failed — {e}")

    def _patch_v2(self, mnemon: Any) -> None:
        try:
            from autogen import ConversableAgent
            self._original_generate_reply = ConversableAgent.generate_reply
            m    = mnemon
            orig = self._original_generate_reply

            def patched_generate_reply(
                _self: Any,
                messages: Optional[List[Dict]] = None,
                sender: Optional[Any] = None,
                **kwargs: Any,
            ) -> Union[str, Dict, None]:
                cache_key = _autogen_cache_key(_self, messages or [])
                if cache_key in _reply_cache:
                    track_cache_hit(m, "autogen")
                    logger.debug("Mnemon: AutoGen v2 cache hit")
                    return _reply_cache[cache_key]

                result = orig(_self, messages=messages, sender=sender, **kwargs)
                if result is not None:
                    _reply_cache[cache_key] = result
                return result

            ConversableAgent.generate_reply = patched_generate_reply
            self._patched_v2 = True

        except Exception as e:
            logger.debug(f"Mnemon: AutoGen v2 patch failed — {e}")

    def unpatch(self) -> None:
        if self._original_generate_reply is None:
            return
        try:
            if self._patched_v2:
                from autogen import ConversableAgent
                ConversableAgent.generate_reply = self._original_generate_reply
            else:
                from autogen_agentchat.agents._base_chat_agent import BaseChatAgent
                BaseChatAgent.on_messages = self._original_generate_reply
        except Exception as e:
            logger.debug(f"Mnemon: AutoGen unpatch failed — {e}")
        finally:
            self._original_generate_reply = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _autogen_cache_key(agent: Any, messages: Any) -> str:
    agent_name = getattr(agent, "name", type(agent).__name__)
    msg_hash   = prompt_hash(
        [{"role": "user", "content": str(messages)}], None, agent_name
    )
    return f"{agent_name}:{msg_hash}"
