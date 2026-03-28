"""
Mnemon Core Types
Shared enums, dataclasses and constants used across all modules.
Based on architecture designed by Mahika Jadhav (smartass-4ever).
"""

import time
import hashlib
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum


# ─────────────────────────────────────────────
# MEMORY LAYERS
# ─────────────────────────────────────────────

class MemoryLayer(Enum):
    WORKING      = "working"       # ephemeral scratchpad — flushes at task end
    EPISODIC     = "episodic"      # chronological experiences — importance scored
    SEMANTIC     = "semantic"      # stable facts — key/value vault
    RELATIONSHIP = "relationship"  # per-user interaction patterns
    EMOTIONAL    = "emotional"     # emotional context — decays over time


# ─────────────────────────────────────────────
# SIGNAL TYPES
# ─────────────────────────────────────────────

class SignalType(Enum):
    # Tier 2 — agent intelligence signals
    ERROR_RESOLVED    = "error_resolved"
    WORKAROUND_FOUND  = "workaround_found"
    OPTIMIZATION      = "optimization"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    CONTEXT_UPDATE    = "context_update"
    PAD_ALERT         = "pad_alert"

    # Tier 1 — system learning signals
    SUCCESS           = "success"
    FAILURE           = "failure"
    DEGRADATION       = "degradation"
    RECOVERY          = "recovery"
    PATTERN_FOUND     = "pattern_found"
    ANOMALY           = "anomaly"


# ─────────────────────────────────────────────
# RISK LEVELS
# ─────────────────────────────────────────────

class RiskLevel(Enum):
    LOW      = "low"       # informational, fully reversible
    MEDIUM   = "medium"    # external comms, data writes
    HIGH     = "high"      # financial, deletions, prod deploys
    CRITICAL = "critical"  # irreversible, major consequences


# ─────────────────────────────────────────────
# PAD SEVERITY
# ─────────────────────────────────────────────

class PADSeverity(Enum):
    HEALTHY  = "healthy"
    WARNING  = "warning"   # nudge injected into agent context
    CRITICAL = "critical"  # orchestrator alerted
    FLATLINE = "flatline"  # SIG_KILL + respawn


# ─────────────────────────────────────────────
# EXPERIENCE SIGNAL
# ─────────────────────────────────────────────

@dataclass
class ExperienceSignal:
    """
    Stateless signal representing any learning or health event.
    Agents broadcast signals — never write directly to memory.
    The bus sequences and commits atomically. No race conditions.
    """
    signal_id:      str
    tenant_id:      str
    session_id:     str
    timestamp:      float
    signal_type:    SignalType
    layer:          Optional[MemoryLayer]

    content:        Dict[str, Any]
    importance:     float = 0.5          # 0.0–1.0
    risk_level:     RiskLevel = RiskLevel.LOW
    context:        Dict[str, Any] = field(default_factory=dict)

    # Propagation tracking (Tier 2)
    agent_id:       Optional[str] = None
    propagated_to:  Set[str] = field(default_factory=set)
    mitigation:     Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "signal_id":     self.signal_id,
            "tenant_id":     self.tenant_id,
            "session_id":    self.session_id,
            "timestamp":     self.timestamp,
            "signal_type":   self.signal_type.value,
            "layer":         self.layer.value if self.layer else None,
            "content":       self.content,
            "importance":    self.importance,
            "risk_level":    self.risk_level.value,
            "context":       self.context,
            "agent_id":      self.agent_id,
            "propagated_to": list(self.propagated_to),
            "mitigation":    self.mitigation,
        }


# ─────────────────────────────────────────────
# BONDED MEMORY
# ─────────────────────────────────────────────

@dataclass
class BondedMemory:
    """
    A memory with its activation signature — the 'protein half'
    that snaps to matching signals. Stores both content and the
    conditions that should wake it up.
    """
    # Identity
    memory_id:          str
    tenant_id:          str
    schema_version:     int = 1

    # Content
    layer:              MemoryLayer = MemoryLayer.EPISODIC
    content:            Any = None
    session_id:         str = ""

    # Pattern matching (protein bond)
    activation_tags:    Set[str] = field(default_factory=set)
    activation_domain:  str = "general"
    activation_signature: List[float] = field(default_factory=list)  # 64-dim

    # Intent matching
    intent_label:       str = ""       # WHY this memory is useful
    intent_valid:       bool = True    # False = superseded
    intent_signature:   List[float] = field(default_factory=list)  # 64-dim
    superseded_at:      Optional[float] = None
    superseded_by:      Optional[str] = None

    # Scoring
    importance:         float = 0.5
    confidence:         float = 1.0
    risk_level:         RiskLevel = RiskLevel.LOW

    # Timestamps
    timestamp:          float = field(default_factory=time.time)
    last_accessed:      float = field(default_factory=time.time)
    access_count:       int = 0

    # Cross-layer references
    cross_layer_refs:   List[str] = field(default_factory=list)  # other layer memory_ids

    # Drone feedback
    drone_keep_score:   float = 0.5   # learned from feedback loop
    drone_drop_score:   float = 0.5

    def access(self):
        self.access_count += 1
        self.last_accessed = time.time()

    def supersede(self, new_memory_id: str):
        self.intent_valid = False
        self.superseded_at = time.time()
        self.superseded_by = new_memory_id


# ─────────────────────────────────────────────
# SEMANTIC FACT (versioned)
# ─────────────────────────────────────────────

@dataclass
class FactVersion:
    value:          Any
    confidence:     float
    recorded_at:    float
    source_session: str
    superseded_at:  Optional[float] = None


@dataclass
class SemanticFact:
    """Stable fact with full version history."""
    fact_id:        str
    tenant_id:      str
    key:            str
    current_value:  Any
    confidence:     float
    history:        List[FactVersion] = field(default_factory=list)
    last_updated:   float = field(default_factory=time.time)
    access_count:   int = 0

    def update(self, new_value: Any, confidence: float, source_session: str):
        old = FactVersion(
            value=self.current_value,
            confidence=self.confidence,
            recorded_at=self.last_updated,
            source_session=source_session,
            superseded_at=time.time()
        )
        self.history.append(old)
        self.current_value = new_value
        self.confidence = confidence
        self.last_updated = time.time()


# ─────────────────────────────────────────────
# PAD TELEMETRY
# ─────────────────────────────────────────────

@dataclass
class PADVector:
    """
    Three-dimensional agent health metric.
    Pleasure = goal alignment (semantic similarity).
    Arousal  = loop/error detection (heuristic).
    Dominance = confident vs uncertain language (heuristic).
    """
    pleasure:   float  # 0.0–1.0  low = task divergence
    arousal:    float  # 0.0–1.0  high = looping
    dominance:  float  # 0.0–1.0  low = paralysed

    agent_id:   str = ""
    tenant_id:  str = ""
    timestamp:  float = field(default_factory=time.time)

    def severity(self) -> PADSeverity:
        if self.pleasure < 0.1 and self.arousal > 0.9:
            return PADSeverity.FLATLINE
        if self.pleasure < 0.2 or self.arousal > 0.8 or self.dominance < 0.3:
            return PADSeverity.CRITICAL
        if self.pleasure < 0.4 or self.arousal > 0.6 or self.dominance < 0.45:
            return PADSeverity.WARNING
        return PADSeverity.HEALTHY

    def is_healthy(self) -> bool:
        return self.severity() == PADSeverity.HEALTHY


# ─────────────────────────────────────────────
# COMPUTATION FINGERPRINT (EME)
# ─────────────────────────────────────────────

@dataclass
class ComputationFingerprint:
    """
    Five-component fingerprint for System 1 exact matching.
    Tool version hashing prevents stale cache hits after API changes.
    """
    goal_hash:         str
    input_schema_hash: str
    context_hash:      str
    capability_hash:   str
    constraint_hash:   str

    @property
    def full_hash(self) -> str:
        combined = "|".join([
            self.goal_hash,
            self.input_schema_hash,
            self.context_hash,
            self.capability_hash,
            self.constraint_hash,
        ])
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    @classmethod
    def build(
        cls,
        goal: str,
        input_schema: Dict,
        context: Dict,
        capabilities: List[str],
        constraints: Dict,
    ) -> "ComputationFingerprint":
        def _h(obj) -> str:
            import json
            s = json.dumps(obj, sort_keys=True) if isinstance(obj, (dict, list)) else str(obj)
            return hashlib.md5(s.encode()).hexdigest()[:16]

        return cls(
            goal_hash=_h(goal),
            input_schema_hash=_h(input_schema),
            context_hash=_h(context),
            capability_hash=_h(sorted(capabilities)),
            constraint_hash=_h(constraints),
        )


# ─────────────────────────────────────────────
# TEMPLATE SEGMENT (EME)
# ─────────────────────────────────────────────

@dataclass
class TemplateSegment:
    """
    One logical unit of a cached execution template.
    Segments can be individually matched and gap-filled.
    """
    segment_id:            str
    tenant_id:             str
    content:               Any
    fingerprint:           str
    signature:             List[float] = field(default_factory=list)  # 64-dim
    dependencies:          List[str] = field(default_factory=list)    # segment_ids needed as input
    outputs:               List[str] = field(default_factory=list)    # what this segment produces
    domain_tags:           Set[str] = field(default_factory=set)
    success_rate:          float = 1.0
    use_count:             int = 0
    confidence:            float = 1.0   # set by LLM when gap-filling
    is_generated:          bool = False  # True = LLM gap-fill, False = cached


@dataclass
class ExecutionTemplate:
    """
    A full cached execution plan made of segments.
    Carries dependency manifest for version tracking.
    """
    template_id:       str
    tenant_id:         str
    intent:            str
    fingerprint:       ComputationFingerprint
    segments:          List[TemplateSegment]

    success_count:     int = 0
    failure_count:     int = 0
    created_at:        float = field(default_factory=time.time)
    last_used_at:      float = field(default_factory=time.time)
    embedding:         List[float] = field(default_factory=list)  # 384-dim full semantic

    # Dependency manifest for version tracking
    tool_versions:     Dict[str, str] = field(default_factory=dict)   # tool → version hash
    api_schemas:       Dict[str, str] = field(default_factory=dict)   # endpoint → schema hash
    needs_reverification: bool = False

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 1.0

    @property
    def should_evict(self) -> bool:
        total = self.success_count + self.failure_count
        if total < 3:
            return False
        return (self.failure_count / total) > 0.5


# ─────────────────────────────────────────────
# AUDIT LOG ENTRY
# ─────────────────────────────────────────────

@dataclass
class AuditEntry:
    """Immutable audit trail entry for every EROS-influenced action."""
    tenant_id:      str
    task_id:        str
    timestamp:      float
    action:         str
    mnemon_version:   str
    template_id:    Optional[str]
    memory_ids:     List[str]
    risk_level:     RiskLevel
    human_approved: Optional[bool]
    outcome:        str   # "success" | "failure" | "pending"
    component:      str   # "eme" | "memory" | "bus"


# ─────────────────────────────────────────────
# COST TRACKING
# ─────────────────────────────────────────────

@dataclass
class LLMCallLog:
    """Cost tracking for every LLM call EROS makes."""
    tenant_id:     str
    component:     str   # "router" | "drone" | "gap_fill" | "tag_verify"
    model:         str
    tokens_input:  int
    tokens_output: int
    cost_usd:      float
    timestamp:     float
    task_id:       str


# ─────────────────────────────────────────────
# DECISION TRACE (Retrospector)
# ─────────────────────────────────────────────

@dataclass
class DecisionTrace:
    """
    Immutable record of one EME decision.
    Stored in SystemDatabase (never EROSDatabase).
    goal_hash is a fingerprint only — raw goal text is never persisted.
    """
    trace_id:             str
    tenant_id:            str
    task_id:              str
    goal_hash:            str                        # fingerprint only, never goal text
    fragment_ids_used:    List[str]                  # fragment IDs retrieved in gap fill
    memory_ids_retrieved: List[str]                  # memory IDs from CognitiveMemorySystem
    segments_generated:   List[str]                  # segment IDs created by LLM gap fill
    tools_called:         List[str]                  # capability list passed to EME
    step_outcomes:        Dict[str, str]             # step_id → "ok"|"fail"|"timeout"
    overall_outcome:      str                        # EMEResult.status
    latency_ms:           float
    timestamp:            float


MNEMON_VERSION = "1.0.2"
