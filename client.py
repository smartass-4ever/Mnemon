"""
Mnemon LLM Client Interface
Abstract base class and concrete implementations.

Mnemon uses LLMs in exactly four places:
1. Memory router  — classify layer + generate tags + intent label
2. Tag verifier   — sanity check tags async after write
3. Intent drone   — curate memory candidates (conditional on pool size)
4. EME gap fill   — generate missing template segments

All calls go through this interface.
Swap implementations without changing any core code.

Shipped implementations:
- AnthropicClient  — uses claude-haiku / claude-sonnet
- OpenAIClient     — uses gpt-4o-mini / gpt-4o
- MockClient       — for testing, returns deterministic responses

AUTO-DETECTION:
    from mnemon.llm.client import auto_client
    llm = auto_client()   # picks up ANTHROPIC_API_KEY or OPENAI_API_KEY from env
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# BASE INTERFACE
# ─────────────────────────────────────────────

class LLMClient(ABC):
    """
    Abstract LLM client for Mnemon internal calls.
    All methods are async. All return plain strings.
    """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 200,
        temperature: float = 0.1,
    ) -> str:
        ...

    async def complete_json(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 200,
    ) -> Dict[str, Any]:
        text = await self.complete(prompt, model, max_tokens)
        try:
            clean = text.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            return json.loads(clean)
        except json.JSONDecodeError as e:
            logger.warning(f"LLM JSON parse failed: {e} — response: {text[:200]}")
            return {}


class LLMError(Exception):
    pass


# ─────────────────────────────────────────────
# ANTHROPIC CLIENT
# ─────────────────────────────────────────────

class AnthropicClient(LLMClient):
    """
    Anthropic API client for Mnemon.

    Usage:
        from mnemon.llm.client import AnthropicClient
        llm = AnthropicClient()                        # reads ANTHROPIC_API_KEY from env
        llm = AnthropicClient(api_key="sk-ant-...")    # explicit key
        mnemon = Mnemon(tenant_id="x", llm_client=llm)
    """

    # Model tiers used by Mnemon internally
    ROUTER_MODEL   = "claude-haiku-4-5-20251001"   # cheap — classification + tags
    DRONE_MODEL    = "claude-haiku-4-5-20251001"   # cheap — intent curation
    GAP_FILL_MODEL = "claude-sonnet-4-6"           # smart — segment generation

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client  = None
        self._call_count   = 0
        self._total_tokens = 0

        if not self._api_key:
            logger.warning(
                "AnthropicClient: no API key found. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key= explicitly."
            )

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            except ImportError:
                raise LLMError(
                    "anthropic package not installed. "
                    "Run: pip install mnemon-ai[full]"
                )
        return self._client

    async def complete(
        self,
        prompt: str,
        model: str = ROUTER_MODEL,
        max_tokens: int = 200,
        temperature: float = 0.1,
    ) -> str:
        client = self._get_client()
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            self._call_count += 1
            if response.usage:
                self._total_tokens += response.usage.input_tokens + response.usage.output_tokens
            return response.content[0].text
        except Exception as e:
            logger.warning(f"Anthropic API call failed: {e}")
            raise LLMError(str(e))

    def get_stats(self) -> Dict:
        return {
            "provider":     "anthropic",
            "call_count":   self._call_count,
            "total_tokens": self._total_tokens,
        }


# ─────────────────────────────────────────────
# OPENAI CLIENT
# ─────────────────────────────────────────────

class OpenAIClient(LLMClient):
    """
    OpenAI API client for Mnemon.

    Usage:
        from mnemon.llm.client import OpenAIClient
        llm = OpenAIClient()                       # reads OPENAI_API_KEY from env
        llm = OpenAIClient(api_key="sk-...")       # explicit key
        mnemon = Mnemon(tenant_id="x", llm_client=llm,
                        router_model="gpt-4o-mini",
                        gap_fill_model="gpt-4o")
    """

    ROUTER_MODEL   = "gpt-4o-mini"
    DRONE_MODEL    = "gpt-4o-mini"
    GAP_FILL_MODEL = "gpt-4o"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client  = None
        self._call_count = 0

        if not self._api_key:
            logger.warning(
                "OpenAIClient: no API key found. "
                "Set OPENAI_API_KEY environment variable or pass api_key= explicitly."
            )

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self._api_key)
            except ImportError:
                raise LLMError(
                    "openai package not installed. "
                    "Run: pip install mnemon-ai[full]"
                )
        return self._client

    async def complete(
        self,
        prompt: str,
        model: str = ROUTER_MODEL,
        max_tokens: int = 200,
        temperature: float = 0.1,
    ) -> str:
        client = self._get_client()
        try:
            response = await client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            self._call_count += 1
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI API call failed: {e}")
            raise LLMError(str(e))


# ─────────────────────────────────────────────
# AUTO-DETECTION
# ─────────────────────────────────────────────

def auto_client() -> Optional[LLMClient]:
    """
    Automatically detect and return the best available LLM client.

    Priority:
        1. ANTHROPIC_API_KEY in environment → AnthropicClient
        2. OPENAI_API_KEY in environment    → OpenAIClient
        3. Neither found                    → None (rule-based mode)

    Usage:
        from mnemon.llm.client import auto_client
        mnemon = Mnemon(tenant_id="x", llm_client=auto_client())
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("Mnemon LLM: auto-detected ANTHROPIC_API_KEY → AnthropicClient")
        return AnthropicClient()

    if os.environ.get("OPENAI_API_KEY"):
        logger.info("Mnemon LLM: auto-detected OPENAI_API_KEY → OpenAIClient")
        return OpenAIClient()

    logger.info(
        "Mnemon LLM: no API key found — running in rule-based mode. "
        "Set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable full intelligence."
    )
    return None


# ─────────────────────────────────────────────
# MOCK CLIENT (for testing)
# ─────────────────────────────────────────────

class MockLLMClient(LLMClient):
    """
    Deterministic mock client for testing.
    Returns sensible responses without any API calls.
    """

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self._responses = responses or {}
        self._call_log: List[Dict] = []

    async def complete(
        self,
        prompt: str,
        model: str = "mock",
        max_tokens: int = 200,
        temperature: float = 0.1,
    ) -> str:
        self._call_log.append({"prompt": prompt[:100], "model": model, "time": time.time()})

        for key, response in self._responses.items():
            if key.lower() in prompt.lower():
                return response

        prompt_lower = prompt.lower()

        if "classify this memory" in prompt_lower or ("classify" in prompt_lower and "layer" in prompt_lower):
            if any(w in prompt_lower for w in ["emotional", "tense", "happy", "frustrated", "anxious"]):
                return '{"layer": "emotional", "tags": ["emotion", "sentiment"], "intent": "use when setting tone"}'
            if any(w in prompt_lower for w in ["prefer", "always", "usually", "style"]):
                return '{"layer": "semantic", "tags": ["preference", "fact"], "intent": "use when recalling preferences"}'
            if any(w in prompt_lower for w in ["last week", "yesterday", "found", "discovered"]):
                return '{"layer": "episodic", "tags": ["event", "discovery"], "intent": "use when recalling past events"}'
            return '{"layer": "episodic", "tags": ["general"], "intent": "use when relevant to current task"}'

        if "check if these tags" in prompt_lower or "tags accurately" in prompt_lower:
            return '{"verdict": "YES", "amended_tags": null}'

        if "memory curator" in prompt_lower or "review these candidate" in prompt_lower:
            try:
                import re
                ids = re.findall(r'"id":\s*"([^"]+)"', prompt)
                if ids:
                    keep = ids[:max(1, len(ids) // 2)]
                    drop = ids[len(keep):]
                    return json.dumps({"keep": keep, "drop": drop, "conflicts": []})
            except Exception:
                pass
            return '{"keep": [], "drop": [], "conflicts": []}'

        if "execution step" in prompt_lower or "generate one" in prompt_lower:
            return '{"step_id": "mock_step", "action": "process", "params": {"mock": true}}'

        if "adapt this execution step" in prompt_lower:
            return '{"content": {"action": "adapted_process", "params": {"adapted": true}}}'

        return '{"result": "mock_response", "status": "ok"}'

    @property
    def call_count(self) -> int:
        return len(self._call_log)

    def get_call_log(self) -> List[Dict]:
        return self._call_log


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# BASE INTERFACE
# ─────────────────────────────────────────────

class LLMClient(ABC):
    """
    Abstract LLM client for EROS internal calls.
    All methods are async. All return plain strings.
    Implementations handle retries, rate limits, and error normalisation.
    """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 200,
        temperature: float = 0.1,
    ) -> str:
        """
        Single completion call. Returns response text.
        Raises LLMError on unrecoverable failure.
        """
        ...

    async def complete_json(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 200,
    ) -> Dict[str, Any]:
        """
        Completion that parses JSON response.
        Strips markdown fences. Returns parsed dict.
        Falls back to empty dict on parse failure.
        """
        text = await self.complete(prompt, model, max_tokens)
        try:
            clean = text.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            return json.loads(clean)
        except json.JSONDecodeError as e:
            logger.warning(f"LLM JSON parse failed: {e} — response: {text[:200]}")
            return {}


class LLMError(Exception):
    pass


# ─────────────────────────────────────────────
# ANTHROPIC CLIENT
# ─────────────────────────────────────────────

class AnthropicClient(LLMClient):
    """
    Anthropic API client for EROS.
    Uses the anthropic Python SDK.

    Usage:
        from mnemon.llm.client import AnthropicClient
        llm = AnthropicClient(api_key="sk-ant-...")
        mnemon = Mnemon(tenant_id="x", llm_client=llm)
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
        self._client  = None
        self._call_count = 0
        self._total_tokens = 0

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            except ImportError:
                raise LLMError(
                    "anthropic package not installed. "
                    "Run: pip install anthropic"
                )
        return self._client

    async def complete(
        self,
        prompt: str,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 200,
        temperature: float = 0.1,
    ) -> str:
        client = self._get_client()
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            self._call_count += 1
            if response.usage:
                self._total_tokens += response.usage.input_tokens + response.usage.output_tokens
            return response.content[0].text
        except Exception as e:
            logger.warning(f"Anthropic API call failed: {e}")
            raise LLMError(str(e))

    def get_stats(self) -> Dict:
        return {
            "provider":     "anthropic",
            "call_count":   self._call_count,
            "total_tokens": self._total_tokens,
        }


# ─────────────────────────────────────────────
# OPENAI CLIENT
# ─────────────────────────────────────────────

class OpenAIClient(LLMClient):
    """
    OpenAI API client for EROS.

    Usage:
        from mnemon.llm.client import OpenAIClient
        llm = OpenAIClient(api_key="sk-...")
        mnemon = Mnemon(tenant_id="x", llm_client=llm, router_model="gpt-4o-mini")
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
        self._client  = None
        self._call_count = 0

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self._api_key)
            except ImportError:
                raise LLMError(
                    "openai package not installed. "
                    "Run: pip install openai"
                )
        return self._client

    async def complete(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 200,
        temperature: float = 0.1,
    ) -> str:
        client = self._get_client()
        try:
            response = await client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            self._call_count += 1
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI API call failed: {e}")
            raise LLMError(str(e))


# ─────────────────────────────────────────────
# MOCK CLIENT (for testing)
# ─────────────────────────────────────────────

class MockLLMClient(LLMClient):
    """
    Deterministic mock client for testing.
    Returns sensible responses without any API calls.
    Recognises prompt patterns and returns appropriate JSON.
    """

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self._responses = responses or {}
        self._call_log: List[Dict] = []

    async def complete(
        self,
        prompt: str,
        model: str = "mock",
        max_tokens: int = 200,
        temperature: float = 0.1,
    ) -> str:
        self._call_log.append({"prompt": prompt[:100], "model": model, "time": time.time()})

        # Custom responses first
        for key, response in self._responses.items():
            if key.lower() in prompt.lower():
                return response

        # Pattern matching for common EROS prompts
        prompt_lower = prompt.lower()

        # Memory router
        if "classify this memory" in prompt_lower or "classify" in prompt_lower and "layer" in prompt_lower:
            if "emotional" in prompt_lower or "tense" in prompt_lower or "happy" in prompt_lower:
                return '{"layer": "emotional", "tags": ["emotion", "sentiment"], "intent": "use when setting tone"}'
            if "prefer" in prompt_lower or "always" in prompt_lower:
                return '{"layer": "semantic", "tags": ["preference", "fact"], "intent": "use when recalling preferences"}'
            if "last week" in prompt_lower or "yesterday" in prompt_lower or "found" in prompt_lower:
                return '{"layer": "episodic", "tags": ["event", "discovery"], "intent": "use when recalling past events"}'
            return '{"layer": "episodic", "tags": ["general"], "intent": "use when relevant to current task"}'

        # Tag verification
        if "check if these tags" in prompt_lower or "tags accurately" in prompt_lower:
            return '{"verdict": "YES", "amended_tags": null}'

        # Intent drone
        if "memory curator" in prompt_lower or "review these candidate" in prompt_lower:
            # Extract IDs from the candidates JSON if present
            try:
                lines = prompt.split("\n")
                for line in lines:
                    if '"id"' in line:
                        import re
                        ids = re.findall(r'"id":\s*"([^"]+)"', prompt)
                        if ids:
                            keep = ids[:max(1, len(ids)//2)]
                            drop = ids[len(keep):]
                            return json.dumps({"keep": keep, "drop": drop, "conflicts": []})
            except Exception:
                pass
            return '{"keep": [], "drop": [], "conflicts": []}'

        # EME gap fill
        if "execution step" in prompt_lower or "generate one" in prompt_lower:
            return '{"step_id": "mock_step", "action": "process", "params": {"mock": true}}'

        # EME fragment adaptation
        if "adapt this execution step" in prompt_lower:
            return '{"content": {"action": "adapted_process", "params": {"adapted": true}}}'

        # Default
        return '{"result": "mock_response", "status": "ok"}'

    @property
    def call_count(self) -> int:
        return len(self._call_log)

    def get_call_log(self) -> List[Dict]:
        return self._call_log
