# Adapters package.
# CrewAI adapter (v2) lives in mnemon._future.adapters.crewai and is loaded
# lazily by mnemon.init() via _detect_adapter(). No top-level imports here to
# avoid crashing when the optional framework SDKs are not installed.
