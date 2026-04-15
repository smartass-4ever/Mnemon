"""
Mnemon LLM Client Interface
Abstract base class and concrete implementations.

Mnemon uses LLMs in exactly four places:
1. Memory router  — classify layer + generate tags + intent label
2. Tag verifier   — sanity check tags async after write
3. Intent drone   — curate memory candidates (conditional on pool size)
4. EME gap fill   — generate missing template segments

Supported providers (set ONE environment variable, Mnemon does the rest):

    ANTHROPIC_API_KEY  → AnthropicClient  (claude-haiku / claude-sonnet)
    OPENAI_API_KEY     → OpenAIClient     (gpt-4o-mini / gpt-4o)
    GOOGLE_API_KEY     → GoogleClient     (gemini-1.5-flash / gemini-1.5-pro)
    GROQ_API_KEY       → GroqClient       (llama-3.1-8b / llama-3.1-70b) — free tier available

Priority order when multiple keys are set:
    ANTHROPIC → OPENAI → GOOGLE → GROQ

Or pass explicitly:
    from mnemon.llm.client import AnthropicClient
    m = Mnemon(tenant_id="x", llm_client=AnthropicClient(api_key="sk-ant-..."))
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
# ANTHROPIC — claude-haiku / claude-sonnet
# ─────────────────────────────────────────────

class AnthropicClient(LLMClient):
    """
    Anthropic API — Claude models.

    Set env var:  export ANTHROPIC_API_KEY=sk-ant-...
    Or pass:      AnthropicClient(api_key="sk-ant-...")

    Models used internally:
      Router + Drone: claude-haiku-4-5-20251001   (fast, cheap)
      Gap fill:       claude-sonnet-4-6            (smart)
    """

    ROUTER_MODEL   = "claude-haiku-4-5-20251001"
    DRONE_MODEL    = "claude-haiku-4-5-20251001"
    GAP_FILL_MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client  = None
        self._call_count   = 0
        self._total_tokens = 0
        if not self._api_key:
            logger.warning("AnthropicClient: no API key — set ANTHROPIC_API_KEY")

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            except ImportError:
                raise LLMError("Run: pip install anthropic")
        return self._client

    async def complete(self, prompt, model=None, max_tokens=200, temperature=0.1):
        model = model or self.ROUTER_MODEL
        client = self._get_client()
        try:
            response = await client.messages.create(
                model=model, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            self._call_count += 1
            if response.usage:
                self._total_tokens += response.usage.input_tokens + response.usage.output_tokens
            return response.content[0].text
        except Exception as e:
            logger.warning(f"Anthropic call failed: {e}")
            raise LLMError(str(e))

    def get_stats(self):
        return {"provider": "anthropic", "calls": self._call_count, "tokens": self._total_tokens}


# ─────────────────────────────────────────────
# OPENAI — gpt-4o-mini / gpt-4o
# ─────────────────────────────────────────────

class OpenAIClient(LLMClient):
    """
    OpenAI API — GPT models.

    Set env var:  export OPENAI_API_KEY=sk-...
    Or pass:      OpenAIClient(api_key="sk-...")

    Models used internally:
      Router + Drone: gpt-4o-mini   (fast, cheap)
      Gap fill:       gpt-4o        (smart)
    """

    ROUTER_MODEL   = "gpt-4o-mini"
    DRONE_MODEL    = "gpt-4o-mini"
    GAP_FILL_MODEL = "gpt-4o"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client  = None
        self._call_count = 0
        if not self._api_key:
            logger.warning("OpenAIClient: no API key — set OPENAI_API_KEY")

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self._api_key)
            except ImportError:
                raise LLMError("Run: pip install openai")
        return self._client

    async def complete(self, prompt, model=None, max_tokens=200, temperature=0.1):
        model = model or self.ROUTER_MODEL
        client = self._get_client()
        try:
            response = await client.chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            self._call_count += 1
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI call failed: {e}")
            raise LLMError(str(e))

    def get_stats(self):
        return {"provider": "openai", "calls": self._call_count}


# ─────────────────────────────────────────────
# GOOGLE — gemini-1.5-flash / gemini-1.5-pro
# ─────────────────────────────────────────────

class GoogleClient(LLMClient):
    """
    Google Gemini API.

    Set env var:  export GOOGLE_API_KEY=AIza...
    Or pass:      GoogleClient(api_key="AIza...")

    Models used internally:
      Router + Drone: gemini-1.5-flash   (fast, cheap)
      Gap fill:       gemini-1.5-pro     (smart)

    Install:  pip install google-generativeai
    """

    ROUTER_MODEL   = "gemini-1.5-flash"
    DRONE_MODEL    = "gemini-1.5-flash"
    GAP_FILL_MODEL = "gemini-1.5-pro"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._configured = False
        self._call_count = 0
        if not self._api_key:
            logger.warning("GoogleClient: no API key — set GOOGLE_API_KEY")

    def _configure(self):
        if not self._configured:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self._api_key)
                self._configured = True
                return genai
            except ImportError:
                raise LLMError("Run: pip install google-generativeai")

    async def complete(self, prompt, model=None, max_tokens=200, temperature=0.1):
        model = model or self.ROUTER_MODEL
        genai = self._configure()
        try:
            import asyncio
            import google.generativeai as genai
            gen_model = genai.GenerativeModel(model)
            # Gemini sync → run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: gen_model.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": max_tokens, "temperature": temperature}
                )
            )
            self._call_count += 1
            return response.text
        except Exception as e:
            logger.warning(f"Google call failed: {e}")
            raise LLMError(str(e))

    def get_stats(self):
        return {"provider": "google", "calls": self._call_count}


# ─────────────────────────────────────────────
# GROQ — llama / mixtral (free tier available)
# ─────────────────────────────────────────────

class GroqClient(LLMClient):
    """
    Groq API — ultra-fast inference, free tier available.
    Great for early users who don't want to pay for API access.

    Set env var:  export GROQ_API_KEY=gsk_...
    Or pass:      GroqClient(api_key="gsk_...")

    Free tier: groq.com — no credit card needed to start.

    Models used internally:
      Router + Drone: llama-3.1-8b-instant    (extremely fast, free tier)
      Gap fill:       llama-3.1-70b-versatile (smart, free tier)

    Install:  pip install groq
    """

    ROUTER_MODEL   = "llama-3.1-8b-instant"
    DRONE_MODEL    = "llama-3.1-8b-instant"
    GAP_FILL_MODEL = "llama-3.1-70b-versatile"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        self._client  = None
        self._call_count = 0
        if not self._api_key:
            logger.warning("GroqClient: no API key — set GROQ_API_KEY (free at groq.com)")

    def _get_client(self):
        if self._client is None:
            try:
                from groq import AsyncGroq
                self._client = AsyncGroq(api_key=self._api_key)
            except ImportError:
                raise LLMError("Run: pip install groq")
        return self._client

    async def complete(self, prompt, model=None, max_tokens=200, temperature=0.1):
        model = model or self.ROUTER_MODEL
        client = self._get_client()
        try:
            response = await client.chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            self._call_count += 1
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq call failed: {e}")
            raise LLMError(str(e))

    def get_stats(self):
        return {"provider": "groq", "calls": self._call_count}


# ─────────────────────────────────────────────
# AUTO-DETECTION
# ─────────────────────────────────────────────

def auto_client() -> Optional[LLMClient]:
    """
    Automatically detect and return the best available LLM client.

    Checks environment variables in priority order:
        1. ANTHROPIC_API_KEY → AnthropicClient  (claude-haiku / sonnet)
        2. OPENAI_API_KEY    → OpenAIClient     (gpt-4o-mini / gpt-4o)
        3. GOOGLE_API_KEY    → GoogleClient     (gemini-flash / pro)
        4. GROQ_API_KEY      → GroqClient       (llama3 — free tier)
        5. None found        → rule-based mode  (still functional)

    Usage:
        from mnemon.llm.client import auto_client
        m = Mnemon(tenant_id="x", llm_client=auto_client())

    Or just don't pass llm_client at all — Mnemon calls auto_client() automatically.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("Mnemon LLM: detected ANTHROPIC_API_KEY → AnthropicClient (claude)")
        return AnthropicClient()

    if os.environ.get("OPENAI_API_KEY"):
        logger.info("Mnemon LLM: detected OPENAI_API_KEY → OpenAIClient (gpt-4o)")
        return OpenAIClient()

    if os.environ.get("GOOGLE_API_KEY"):
        logger.info("Mnemon LLM: detected GOOGLE_API_KEY → GoogleClient (gemini)")
        return GoogleClient()

    if os.environ.get("GROQ_API_KEY"):
        logger.info("Mnemon LLM: detected GROQ_API_KEY → GroqClient (llama3 — free)")
        return GroqClient()

    logger.info(
        "Mnemon LLM: no API key found — running in rule-based mode.\n"
        "Set any of these to enable full intelligence:\n"
        "  ANTHROPIC_API_KEY  (claude)  — anthropic.com\n"
        "  OPENAI_API_KEY     (gpt-4o)  — platform.openai.com\n"
        "  GOOGLE_API_KEY     (gemini)  — aistudio.google.com\n"
        "  GROQ_API_KEY       (llama3)  — groq.com (free tier available)"
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

    async def complete(self, prompt, model="mock", max_tokens=200, temperature=0.1):
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
                    # Keep all candidates — mock has no semantic context to curate with
                    return json.dumps({"keep": ids, "drop": [], "conflicts": []})
            except Exception:
                pass
            return '{"keep": [], "drop": [], "conflicts": []}'

        if "execution step" in prompt_lower or "generate one" in prompt_lower:
            return '{"step_id": "mock_step", "action": "process", "params": {"mock": true}}'

        if "adapt this execution step" in prompt_lower:
            return '{"content": {"action": "adapted_process", "params": {"adapted": true}}}'

        return '{"result": "mock_response", "status": "ok"}'

    @property
    def call_count(self):
        return len(self._call_log)

    def get_call_log(self):
        return self._call_log
