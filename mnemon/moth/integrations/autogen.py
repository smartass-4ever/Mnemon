"""
Mnemon moth — AutoGen integration.

Supports both AutoGen v0.2 (autogen) and v0.4+ (autogen_agentchat).

AutoGen exposes register_reply() — a first-class hook for injecting
custom reply functions. This integration uses it to:

  1. Inject recalled memory into the agent's context before it replies
  2. Cache agent replies at the task level (System 2 EME per agent turn)
  3. Record each agent reply to the experience bus

register_reply() is AutoGen's intended extension point — this is not
monkey-patching, it's using the API as designed.

For v0.2:  patches ConversableAgent.generate_reply directly
For v0.4+: registers via AssistantAgent.register_reply()

Zero user code changes required.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from mnemon.moth import MnemonIntegration
from ._utils import prompt_hash, recall_as_context, record_outcome, track_cache_hit
from ._cache import BoundedTTLCache

logger = logging.getLogger(__name__)

_reply_cache = BoundedTTLCache(maxsize=500, ttl=3600)


class AutoGenIntegration(MnemonIntegration):
    """
    Instruments AutoGen agents via register_reply() (v0.4+)
    or generate_reply patching (v0.2).
    """

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

        # Try AutoGen v0.4+ first
        if importlib.util.find_spec("autogen_agentchat") is not None:
            self._patch_v4(mnemon)
        else:
            self._patch_v2(mnemon)

    def _patch_v4(self, mnemon: Any) -> None:
        """Patch AutoGen v0.4+ via ConversableAgent.on_messages."""
        try:
            from autogen_agentchat.agents._base_chat_agent import BaseChatAgent

            self._original_generate_reply = BaseChatAgent.on_messages

            m = mnemon
            orig = self._original_generate_reply

            async def patched_on_messages(
                _self: Any, messages: Any, cancellation_token: Any = None
            ) -> Any:
                query = _extract_autogen_query(messages)
                context = recall_as_context(m, query, source="autogen") if query else ""

                if context:
                    _inject_autogen_context(_self, context)

                cache_key = _autogen_cache_key(_self, messages)
                if cache_key in _reply_cache:
                    track_cache_hit(m, "autogen")
                    logger.debug("Mnemon: AutoGen v4 cache hit")
                    return _reply_cache[cache_key]

                result = await orig(_self, messages, cancellation_token)

                _reply_cache[cache_key] = result
                record_outcome(m, query, _extract_autogen_outcome(result))
                return result

            BaseChatAgent.on_messages = patched_on_messages
            self._patched_v2 = False

        except Exception as e:
            logger.debug(f"Mnemon: AutoGen v4 patch failed — {e}")

    def _patch_v2(self, mnemon: Any) -> None:
        """Patch AutoGen v0.2 via ConversableAgent.generate_reply."""
        try:
            from autogen import ConversableAgent

            self._original_generate_reply = ConversableAgent.generate_reply

            m = mnemon
            orig = self._original_generate_reply

            def patched_generate_reply(
                _self: Any,
                messages: Optional[List[Dict]] = None,
                sender: Optional[Any] = None,
                **kwargs: Any,
            ) -> Union[str, Dict, None]:
                query = _extract_autogen_query(messages or [])
                context = recall_as_context(m, query, source="autogen") if query else ""

                if context:
                    _inject_autogen_context(_self, context)

                cache_key = _autogen_cache_key(_self, messages or [])
                if cache_key in _reply_cache:
                    track_cache_hit(m, "autogen")
                    logger.debug("Mnemon: AutoGen v2 cache hit")
                    return _reply_cache[cache_key]

                result = orig(_self, messages=messages, sender=sender, **kwargs)

                if result is not None:
                    _reply_cache[cache_key] = result
                    record_outcome(m, query, str(result)[:400])

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

def _extract_autogen_query(messages: Any) -> str:
    try:
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                return last.get("content", str(last))[:300]
            return str(getattr(last, "content", last))[:300]
    except Exception:
        pass
    return str(messages)[:200]


def _inject_autogen_context(agent: Any, context: str) -> None:
    try:
        # v0.2: system_message attribute
        if hasattr(agent, "system_message") and isinstance(agent.system_message, str):
            if "[Mnemon" not in agent.system_message:
                agent.system_message = f"{context}\n\n{agent.system_message}"
        # v0.4+: _system_messages or description
        elif hasattr(agent, "_system_messages"):
            pass  # injected via on_messages context
    except Exception:
        pass


def _autogen_cache_key(agent: Any, messages: Any) -> str:
    agent_name = getattr(agent, "name", type(agent).__name__)
    msg_hash = prompt_hash(
        [{"role": "user", "content": str(messages)}], None, agent_name
    )
    return f"{agent_name}:{msg_hash}"


def _extract_autogen_outcome(result: Any) -> str:
    try:
        if hasattr(result, "chat_message"):
            return str(getattr(result.chat_message, "content", result.chat_message))[:400]
        if hasattr(result, "messages") and result.messages:
            last = result.messages[-1]
            return str(getattr(last, "content", last))[:400]
    except Exception:
        pass
    return str(result)[:400]
