"""
mnemon.moth
-----------
Auto-instrumentation engine.

On mnemon.init(), the moth wakes up, scans the environment for installed
agent frameworks, and patches each one at the right execution boundary —
without the user writing a single line of integration code.

Frameworks patched automatically:
  anthropic     → messages.create (memory inject + System 1 cache + bus)
  openai        → chat.completions.create (same)
  langchain     → chain execution + callback system (memory + bus)
  langgraph     → CompiledGraph.invoke (per-node EME + memory + bus)
  autogen       → generate_reply (memory inject + EME + bus)

Each integration is fail-safe: if patching fails for any reason, the
original framework behavior is preserved and Mnemon logs a debug message.

Usage:
    The moth is activated automatically by mnemon.init().
    Users never interact with it directly.

    To inspect what was activated:
        m = mnemon.init()
        print(m.active_integrations)
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Registry of all integration classes to attempt on init
_INTEGRATION_CLASSES = [
    ("mnemon.moth.integrations.anthropic",  "AnthropicIntegration"),
    ("mnemon.moth.integrations.openai_sdk", "OpenAIIntegration"),
    ("mnemon.moth.integrations.langchain",  "LangChainIntegration"),
    ("mnemon.moth.integrations.langgraph",  "LangGraphIntegration"),
    ("mnemon.moth.integrations.autogen",    "AutoGenIntegration"),
]


class MnemonIntegration(ABC):
    """
    Base class for all framework integrations.

    Each subclass knows:
      - Whether its target framework is installed (is_available)
      - How to patch it to give Mnemon access (patch)
      - How to cleanly restore original behavior (unpatch)

    Implementations must be fail-safe: patch() should never raise.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short framework identifier e.g. 'crewai', 'langchain'."""
        ...

    def is_available(self) -> bool:
        """Return True if the target framework is installed."""
        return importlib.util.find_spec(self.name) is not None

    @abstractmethod
    def patch(self, mnemon: Any) -> None:
        """Apply patches. Store originals so unpatch() can restore them."""
        ...

    @abstractmethod
    def unpatch(self) -> None:
        """Restore all patched methods to their originals."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(active={self.name in _active_names()})"


def _active_names() -> List[str]:
    return []  # placeholder, real tracking is in Moth


def _framework_version(name: str) -> Optional[str]:
    """Return installed version of a framework, or None."""
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return None


class Moth:
    """
    Scans the environment on init and activates all available integrations.

    Activated integrations patch the framework at the right execution
    boundary so Mnemon provides full value without user code changes.
    """

    def __init__(self) -> None:
        self._registered: List[MnemonIntegration] = []
        self._active: List[MnemonIntegration] = []
        self._load_integrations()

    def _load_integrations(self) -> None:
        for module_path, class_name in _INTEGRATION_CLASSES:
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                self._registered.append(cls())
            except Exception as e:
                logger.debug(f"Moth: could not load {class_name}: {e}")

    def activate(self, mnemon: Any) -> List[str]:
        """
        Auto-detect installed frameworks and activate their integrations.
        Returns list of activated integration names.
        """
        activated: List[str] = []
        for integration in self._registered:
            if not integration.is_available():
                continue
            version = _framework_version(integration.name)
            try:
                integration.patch(mnemon)
                self._active.append(integration)
                activated.append(integration.name)
                logger.info(
                    f"Mnemon moth: {integration.name}"
                    + (f" {version}" if version else "")
                    + " activated"
                )
            except Exception as e:
                logger.warning(
                    f"Mnemon moth: {integration.name}"
                    + (f" {version}" if version else "")
                    + f" could not be instrumented — {e}. "
                    f"Original behavior preserved. "
                    f"Check github.com/smartass-4ever/Mnemon for version compatibility."
                )
        return activated

    def deactivate(self) -> None:
        """Restore all patched frameworks to original behavior."""
        for integration in self._active:
            try:
                integration.unpatch()
            except Exception as e:
                logger.debug(f"Mnemon moth: unpatch {integration.name} failed — {e}")
        self._active.clear()

    @property
    def active(self) -> List[str]:
        return [i.name for i in self._active]
