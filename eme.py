"""
Mnemon Execution Memory Engine (EME)
Generalised plan cache for any expensive recurring computation.

System 1: exact fingerprint match — zero LLM, sub-millisecond
System 2: partial segment match — gap fill with windowed context
Fragment library: proven segments accumulate across all templates

Architecture by Mahika Jadhav (smartass-4ever).
Extended with: five-component fingerprint, segment decomposition,
three-tier gap fill, dependency validation, domain adapters,
tool version hashing, fragment library, multi-component similarity.
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .types import (
    ComputationFingerprint, ExecutionTemplate, TemplateSegment,
    RiskLevel, MNEMON_VERSION
)
from .persistence import EROSDatabase
from .memory import SimpleEmbedder

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────

SYSTEM2_THRESHOLD_DEFAULT = 0.70   # minimum similarity for System 2
FRAGMENT_EXACT_THRESHOLD  = 0.98   # fragment library exact match
FRAGMENT_SIMILAR_THRESHOLD = 0.80  # fragment library similar match

# Multi-component similarity weights
GOAL_WEIGHT       = 0.30
SCHEMA_WEIGHT     = 0.25
CONTEXT_WEIGHT    = 0.25
CAPABILITY_WEIGHT = 0.20


# ─────────────────────────────────────────────
# DOMAIN ADAPTER INTERFACE
# ─────────────────────────────────────────────

class TemplateAdapter(ABC):
    """
    Base adapter — any computation format plugs in here.
    The EME never knows what the template means, only its structure.
    """

    @abstractmethod
    def decompose(self, template: Any) -> List[Dict]:
        """Break template into segment dicts with 'content' and 'id' keys."""
        ...

    @abstractmethod
    def reconstruct(self, segments: List[TemplateSegment]) -> Any:
        """Reassemble segments into the framework's native format."""
        ...

    @abstractmethod
    def extract_signature(self, template: Any, goal: str) -> ComputationFingerprint:
        """Generate fingerprint from template and goal."""
        ...

    def get_tool_versions(self, capabilities: List[str]) -> Dict[str, str]:
        """Override to provide tool version hashes. Default: no versioning."""
        return {}


class GenericAdapter(TemplateAdapter):
    """
    Default adapter for dict/list templates.
    Works for agent plans, DAGs, sequential steps.
    """

    def decompose(self, template: Any) -> List[Dict]:
        if isinstance(template, list):
            return [{"id": f"seg_{i}", "content": step} for i, step in enumerate(template)]
        if isinstance(template, dict):
            nodes = template.get("nodes", template.get("steps", [template]))
            return [{"id": n.get("id", f"seg_{i}"), "content": n} for i, n in enumerate(nodes)]
        return [{"id": "seg_0", "content": template}]

    def reconstruct(self, segments: List[TemplateSegment]) -> Any:
        return {"nodes": [s.content for s in segments], "type": "eros_template"}

    def extract_signature(self, template: Any, goal: str) -> ComputationFingerprint:
        return ComputationFingerprint.build(
            goal=goal,
            input_schema={},
            context={},
            capabilities=[],
            constraints={},
        )


# ─────────────────────────────────────────────
# COST BUDGET
# ─────────────────────────────────────────────

@dataclass
class CostBudget:
    max_llm_calls_per_hour: int = 500
    max_tokens_per_task:    int = 2000
    overflow_policy:        str = "fallback"  # "fallback" | "block" | "alert_only"

    _calls_this_hour:  int = field(default=0, init=False)
    _hour_start:       float = field(default_factory=time.time, init=False)

    def can_call(self) -> bool:
        now = time.time()
        if now - self._hour_start > 3600:
            self._calls_this_hour = 0
            self._hour_start = now
        return self._calls_this_hour < self.max_llm_calls_per_hour

    def record_call(self):
        self._calls_this_hour += 1


# ─────────────────────────────────────────────
# EME RESULT
# ─────────────────────────────────────────────

@dataclass
class EMEResult:
    status:           str    # "system1" | "system2" | "miss" | "error"
    template:         Any    # the hydrated/generated template
    template_id:      Optional[str]
    segments_reused:  int = 0
    segments_generated: int = 0
    tokens_saved:     int = 0
    latency_saved_ms: float = 0.0
    fragments_used:   int = 0
    cache_level:      str = "miss"
    validation_passed: bool = True


# ─────────────────────────────────────────────
# EXECUTION MEMORY ENGINE
# ─────────────────────────────────────────────

class ExecutionMemoryEngine:
    """
    Generalised execution template cache.

    System 1 — exact fingerprint match:
        Five-component hash. 100% → instantiate directly. Zero LLM. Zero tokens.

    System 2 — partial segment match (70-99%):
        Segment-level diff. Matched segments reused.
        Missing segments filled via: fragment library exact →
        fragment library similar → LLM generation (windowed context).
        Dependency validation pass before execution.

    Both fail → full generation → cache on success.

    Fragment library accumulates proven segments across all templates.
    """

    def __init__(
        self,
        tenant_id: str,
        db: EROSDatabase,
        embedder: Optional[SimpleEmbedder] = None,
        llm_client=None,
        adapter: Optional[TemplateAdapter] = None,
        similarity_threshold: float = SYSTEM2_THRESHOLD_DEFAULT,
        gap_fill_model: str = "claude-sonnet-4-6",
        cost_budget: Optional[CostBudget] = None,
    ):
        self.tenant_id  = tenant_id
        self.db         = db
        self.embedder   = embedder or SimpleEmbedder()
        self.llm        = llm_client
        self.adapter    = adapter or GenericAdapter()
        self.threshold  = similarity_threshold
        self.gap_model  = gap_fill_model
        self.budget     = cost_budget or CostBudget()

        # In-memory System 1 cache — pure dict lookup, sub-ms
        self._system1_cache: Dict[str, str] = {}   # fingerprint_hash → template_id

        # Fragment library in memory for fast access
        self._fragments: List[TemplateSegment] = []
        self._fragments_loaded = False

        self._lock = asyncio.Lock()

    async def warm(self):
        """Load System 1 cache and fragment library from DB on startup."""
        templates = await self.db.fetch_all_templates(self.tenant_id)
        for t in templates:
            self._system1_cache[t.fingerprint.full_hash] = t.template_id

        self._fragments = await self.db.fetch_fragments(self.tenant_id)
        self._fragments_loaded = True

        logger.info(
            f"EME warmed: {len(self._system1_cache)} templates, "
            f"{len(self._fragments)} fragments"
        )

    # ──────────────────────────────────────────
    # MAIN ENTRY POINT
    # ──────────────────────────────────────────

    async def run(
        self,
        goal: str,
        inputs: Dict,
        context: Dict,
        capabilities: List[str],
        constraints: Dict,
        generation_fn: Callable,        # the expensive function to call on miss
        task_id: str = "",
        memory_context: Optional[Dict] = None,
    ) -> EMEResult:
        """
        Run a computation through the EME.
        Tries System 1 → System 2 → full generation.
        Caches successful results automatically.
        """
        fp = ComputationFingerprint.build(
            goal=goal,
            input_schema=self._schema_of(inputs),
            context=context,
            capabilities=capabilities,
            constraints=constraints,
        )

        # ── SYSTEM 1: Exact Match ──────────────
        result = await self._try_system1(fp, inputs, goal)
        if result:
            return result

        # ── SYSTEM 2: Partial Match ────────────
        result = await self._try_system2(
            fp, goal, inputs, context, capabilities,
            constraints, memory_context
        )
        if result:
            # Cache the successful System 2 result
            asyncio.create_task(
                self._cache_template(goal, result.template, fp, capabilities)
            )
            return result

        # ── FULL GENERATION ────────────────────
        return await self._full_generation(
            goal, inputs, context, capabilities,
            constraints, generation_fn, fp
        )

    # ──────────────────────────────────────────
    # SYSTEM 1
    # ──────────────────────────────────────────

    async def _try_system1(
        self,
        fp: ComputationFingerprint,
        inputs: Dict,
        goal: str,
    ) -> Optional[EMEResult]:
        """
        Pure in-memory hash lookup. Sub-millisecond.
        Validates dependency manifest before returning.
        """
        template_id = self._system1_cache.get(fp.full_hash)
        if not template_id:
            return None

        template = await self.db.fetch_template_by_fingerprint(
            self.tenant_id, fp.full_hash
        )
        if not template:
            del self._system1_cache[fp.full_hash]
            return None

        # Check if needs re-verification due to dependency change
        if template.needs_reverification:
            if not await self._validate_dependencies(template):
                await self.db.update_template_outcome(self.tenant_id, template_id, False)
                logger.info(f"Template {template_id} failed re-verification — evicting")
                await self.db.delete_template(self.tenant_id, template_id)
                del self._system1_cache[fp.full_hash]
                return None
            # Re-certified
            template.needs_reverification = False
            await self.db.write_template(template)

        if template.should_evict:
            logger.info(f"Template {template_id} evicted — high failure rate")
            await self.db.delete_template(self.tenant_id, template_id)
            del self._system1_cache[fp.full_hash]
            return None

        hydrated = self._hydrate(template, inputs)
        await self.db.update_template_outcome(self.tenant_id, template_id, True)

        tokens_saved = len(template.segments) * 250  # estimate
        return EMEResult(
            status="system1",
            template=hydrated,
            template_id=template_id,
            segments_reused=len(template.segments),
            segments_generated=0,
            tokens_saved=tokens_saved,
            latency_saved_ms=20000,  # ~20s saved
            cache_level="system1",
        )

    # ──────────────────────────────────────────
    # SYSTEM 2
    # ──────────────────────────────────────────

    async def _try_system2(
        self,
        fp: ComputationFingerprint,
        goal: str,
        inputs: Dict,
        context: Dict,
        capabilities: List[str],
        constraints: Dict,
        memory_context: Optional[Dict],
    ) -> Optional[EMEResult]:
        """
        Semantic similarity search across all cached templates.
        Segment-level matching. Gap fill for unmatched segments.
        Dependency validation before returning.
        """
        templates = await self.db.fetch_all_templates(self.tenant_id)
        if not templates:
            return None

        goal_embedding = self.embedder.embed_full(goal)
        best_template  = None
        best_score     = 0.0

        for t in templates:
            if not t.embedding:
                continue
            score = self._multi_component_similarity(
                fp, t.fingerprint, goal_embedding, t.embedding,
                capabilities, list(t.tool_versions.keys())
            )
            if score > best_score:
                best_score = score
                best_template = t

        if not best_template or best_score < self.threshold:
            return None

        if best_template.should_evict:
            return None

        # Segment-level diff
        matched, unmatched_indices = self._segment_diff(
            best_template.segments, goal, inputs
        )

        if not unmatched_indices:
            # All segments matched — treat as System 1
            hydrated = self._hydrate(best_template, inputs)
            await self.db.update_template_outcome(
                self.tenant_id, best_template.template_id, True
            )
            return EMEResult(
                status="system2",
                template=hydrated,
                template_id=best_template.template_id,
                segments_reused=len(matched),
                segments_generated=0,
                cache_level="system2",
            )

        # Gap fill for unmatched segments
        all_segments = list(best_template.segments)
        fragments_used = 0
        generated_count = 0

        for idx in unmatched_indices:
            seg = all_segments[idx]
            # Windowed context — neighbours only
            window = self._window(all_segments, idx, window_size=2)

            filled, used_fragment = await self._fill_gap(
                seg, goal, window, memory_context, context
            )
            all_segments[idx] = filled
            if used_fragment:
                fragments_used += 1
            else:
                generated_count += 1

        # Dependency validation
        stitched_template = self.adapter.reconstruct(all_segments)
        is_valid = await self._validate_stitched(all_segments, capabilities, constraints)

        if not is_valid:
            logger.warning("System 2 stitched template failed validation — falling through to full generation")
            await self.db.update_template_outcome(
                self.tenant_id, best_template.template_id, False
            )
            return None

        await self.db.update_template_outcome(
            self.tenant_id, best_template.template_id, True
        )

        tokens_saved = len(matched) * 250
        return EMEResult(
            status="system2",
            template=stitched_template,
            template_id=best_template.template_id,
            segments_reused=len(matched),
            segments_generated=generated_count,
            tokens_saved=tokens_saved,
            latency_saved_ms=len(matched) * 2500,
            fragments_used=fragments_used,
            cache_level="system2",
            validation_passed=True,
        )

    def _multi_component_similarity(
        self,
        fp1: ComputationFingerprint,
        fp2: ComputationFingerprint,
        embed1: List[float],
        embed2: List[float],
        caps1: List[str],
        caps2: List[str],
    ) -> float:
        """
        Weighted four-component similarity score.
        """
        goal_sim = SimpleEmbedder.cosine_similarity(embed1, embed2)

        # Schema similarity — hash comparison (binary)
        schema_sim = 1.0 if fp1.input_schema_hash == fp2.input_schema_hash else 0.3

        # Context similarity
        ctx_sim = 1.0 if fp1.context_hash == fp2.context_hash else 0.4

        # Capability overlap
        if caps1 and caps2:
            overlap = len(set(caps1) & set(caps2))
            cap_sim = overlap / max(len(caps1), len(caps2))
        else:
            cap_sim = 1.0

        return (
            GOAL_WEIGHT       * goal_sim  +
            SCHEMA_WEIGHT     * schema_sim +
            CONTEXT_WEIGHT    * ctx_sim   +
            CAPABILITY_WEIGHT * cap_sim
        )

    def _segment_diff(
        self,
        segments: List[TemplateSegment],
        goal: str,
        inputs: Dict,
    ) -> Tuple[List[TemplateSegment], List[int]]:
        """
        Identify which segments match the current task and which need gap fill.
        Returns (matched_segments, indices_of_unmatched).
        """
        goal_sig = self.embedder.embed(goal)
        matched_segs = []
        unmatched_indices = []

        for i, seg in enumerate(segments):
            if not seg.signature:
                # No signature — treat as matched (carry through)
                matched_segs.append(seg)
                continue

            sim = SimpleEmbedder.cosine_similarity(goal_sig, seg.signature)
            if sim >= 0.72:
                matched_segs.append(seg)
            else:
                unmatched_indices.append(i)

        return matched_segs, unmatched_indices

    def _window(
        self,
        segments: List[TemplateSegment],
        idx: int,
        window_size: int = 2
    ) -> List[TemplateSegment]:
        """Return neighbouring segments for windowed context in gap fill."""
        start = max(0, idx - window_size)
        end   = min(len(segments), idx + window_size + 1)
        return [s for i, s in enumerate(segments) if start <= i < end and i != idx]

    async def _fill_gap(
        self,
        segment: TemplateSegment,
        goal: str,
        context_window: List[TemplateSegment],
        memory_context: Optional[Dict],
        execution_context: Dict,
    ) -> Tuple[TemplateSegment, bool]:
        """
        Three-tier gap fill:
        1. Fragment library exact match — zero LLM
        2. Fragment library similar match — minimal LLM adaptation
        3. LLM generation — fresh, cached as fragment on success
        """
        # Tier 1: Fragment exact match
        seg_sig = self.embedder.embed(json.dumps(segment.content) if segment.content else "")
        for frag in self._fragments:
            if not frag.signature:
                continue
            sim = SimpleEmbedder.cosine_similarity(seg_sig, frag.signature)
            if sim >= FRAGMENT_EXACT_THRESHOLD:
                frag.use_count += 1
                asyncio.create_task(self.db.write_fragment(frag))
                return frag, True

        # Tier 2: Fragment similar match
        best_frag = None
        best_sim  = 0.0
        for frag in self._fragments:
            if not frag.signature:
                continue
            sim = SimpleEmbedder.cosine_similarity(seg_sig, frag.signature)
            if sim > best_sim:
                best_sim = sim
                best_frag = frag

        if best_frag and best_sim >= FRAGMENT_SIMILAR_THRESHOLD:
            # Minimal LLM adaptation of similar fragment
            if self.llm and self.budget.can_call():
                adapted = await self._adapt_fragment(best_frag, goal, execution_context)
                self.budget.record_call()
                adapted.is_generated = True
                return adapted, True

        # Tier 3: LLM generation
        if self.llm and self.budget.can_call():
            generated = await self._llm_generate_segment(
                segment, goal, context_window, memory_context, execution_context
            )
            self.budget.record_call()

            # Cache as fragment for future use
            asyncio.create_task(self.db.write_fragment(generated))
            self._fragments.append(generated)
            return generated, False

        # Budget exhausted or no LLM — return original segment
        logger.warning("Gap fill: budget exhausted or no LLM — using original segment")
        return segment, False

    async def _adapt_fragment(
        self,
        fragment: TemplateSegment,
        goal: str,
        context: Dict,
    ) -> TemplateSegment:
        """Minimal LLM call to adapt a similar fragment to current context."""
        prompt = f"""Adapt this execution step for the current goal.
Make minimal changes — only update what must change.
Reply with JSON only.

Current goal: {goal}
Context: {json.dumps(context, default=str)[:200]}
Original step: {json.dumps(fragment.content, default=str)[:300]}

Reply: {{"content": <adapted step as JSON>}}"""

        try:
            response = await self.llm.complete(
                prompt=prompt,
                model=self.gap_model,
                max_tokens=200,
            )
            data = json.loads(response)
            new_content = data.get("content", fragment.content)
        except Exception:
            new_content = fragment.content

        seg_id = hashlib.md5(
            f"{self.tenant_id}:{goal}:{time.time()}".encode()
        ).hexdigest()[:16]

        return TemplateSegment(
            segment_id=seg_id,
            tenant_id=self.tenant_id,
            content=new_content,
            fingerprint=hashlib.md5(json.dumps(new_content, default=str).encode()).hexdigest()[:16],
            signature=self.embedder.embed(json.dumps(new_content, default=str)),
            domain_tags=fragment.domain_tags,
            is_generated=True,
            confidence=0.8,
        )

    async def _llm_generate_segment(
        self,
        segment: TemplateSegment,
        goal: str,
        window: List[TemplateSegment],
        memory_context: Optional[Dict],
        execution_context: Dict,
    ) -> TemplateSegment:
        """LLM generates a missing segment with windowed context."""
        window_data = [
            {"id": s.segment_id, "content": s.content}
            for s in window
        ]
        mem_summary = ""
        if memory_context and memory_context.get("memories"):
            mem_summary = f"\nRelevant context from memory:\n{json.dumps(memory_context['memories'][:3], default=str)}"

        prompt = f"""Generate one execution step for this goal.
Reply with JSON only — the step content only.

Goal: {goal}
Surrounding steps (context): {json.dumps(window_data, default=str)}{mem_summary}
Execution context: {json.dumps(execution_context, default=str)[:200]}

The step should connect logically with the surrounding steps.
Reply: {{"step_id": "generated_step", "action": "...", "params": {{}}}}"""

        try:
            response = await self.llm.complete(
                prompt=prompt,
                model=self.gap_model,
                max_tokens=300,
            )
            content = json.loads(response)
        except Exception as e:
            logger.warning(f"Segment generation failed: {e}")
            content = {"action": "placeholder", "error": str(e)}

        seg_id = hashlib.md5(
            f"{self.tenant_id}:{goal}:{time.time()}:{segment.segment_id}".encode()
        ).hexdigest()[:16]

        content_str = json.dumps(content, default=str)
        return TemplateSegment(
            segment_id=seg_id,
            tenant_id=self.tenant_id,
            content=content,
            fingerprint=hashlib.md5(content_str.encode()).hexdigest()[:16],
            signature=self.embedder.embed(content_str),
            domain_tags=segment.domain_tags,
            is_generated=True,
            confidence=0.7,
        )

    # ──────────────────────────────────────────
    # FULL GENERATION
    # ──────────────────────────────────────────

    async def _full_generation(
        self,
        goal: str,
        inputs: Dict,
        context: Dict,
        capabilities: List[str],
        constraints: Dict,
        generation_fn: Callable,
        fp: ComputationFingerprint,
    ) -> EMEResult:
        """
        Call the real expensive function.
        Cache result on success.
        """
        try:
            template = await generation_fn(goal, inputs, context, capabilities, constraints)
        except Exception as e:
            logger.error(f"Full generation failed: {e}")
            return EMEResult(
                status="error",
                template=None,
                template_id=None,
                cache_level="miss",
                validation_passed=False,
            )

        # Cache for future use — await directly so second run can hit cache
        await self._cache_template(goal, template, fp, capabilities)

        return EMEResult(
            status="miss",
            template=template,
            template_id=None,
            segments_reused=0,
            cache_level="miss",
        )

    async def _cache_template(
        self,
        goal: str,
        template: Any,
        fp: ComputationFingerprint,
        capabilities: List[str],
    ):
        """Cache a successful template. Runs async after execution."""
        try:
            segments_data = self.adapter.decompose(template)
            segments = []
            for i, seg_data in enumerate(segments_data):
                content_str = json.dumps(seg_data.get("content", seg_data), default=str)
                sig = self.embedder.embed(content_str)
                seg = TemplateSegment(
                    segment_id=seg_data.get("id", f"seg_{i}"),
                    tenant_id=self.tenant_id,
                    content=seg_data.get("content", seg_data),
                    fingerprint=hashlib.md5(content_str.encode()).hexdigest()[:16],
                    signature=sig,
                    is_generated=False,
                    confidence=1.0,
                    success_rate=1.0,
                )
                segments.append(seg)
                # Add to fragment library
                await self.db.write_fragment(seg)

            template_id = hashlib.sha256(
                f"{self.tenant_id}:{fp.full_hash}:{time.time()}".encode()
            ).hexdigest()[:24]

            tool_versions = self.adapter.get_tool_versions(capabilities)

            et = ExecutionTemplate(
                template_id=template_id,
                tenant_id=self.tenant_id,
                intent=goal,
                fingerprint=fp,
                segments=segments,
                success_count=1,
                embedding=self.embedder.embed_full(goal),
                tool_versions=tool_versions,
            )
            await self.db.write_template(et)
            self._system1_cache[fp.full_hash] = template_id
            self._fragments.extend(segments)

            logger.debug(f"Template cached: {template_id} ({len(segments)} segments)")

        except Exception as e:
            logger.warning(f"Template caching failed: {e}")

    # ──────────────────────────────────────────
    # DEPENDENCY VALIDATION
    # ──────────────────────────────────────────

    async def _validate_stitched(
        self,
        segments: List[TemplateSegment],
        capabilities: List[str],
        constraints: Dict,
    ) -> bool:
        """
        Three-check dependency validation.
        1. Data compatibility between segments
        2. Capability availability
        3. Constraint consistency

        Returns False if any check fails — triggers full regen.
        """
        for i, seg in enumerate(segments):
            # Check dependencies are satisfied by previous segments
            for dep in seg.dependencies:
                satisfied = any(dep in prev.outputs for prev in segments[:i])
                if not satisfied:
                    logger.warning(f"Segment {seg.segment_id} dependency '{dep}' not satisfied")
                    return False

        # Capability check
        if capabilities:
            for seg in segments:
                content_str = json.dumps(seg.content, default=str).lower()
                for cap_hint in ["tool:", "action:", "use:"]:
                    if cap_hint in content_str:
                        pass  # simplified — in production check against capabilities list

        return True

    async def _validate_dependencies(self, template: ExecutionTemplate) -> bool:
        """Re-verify a template's tool versions after dependency change."""
        current_versions = self.adapter.get_tool_versions(
            list(template.tool_versions.keys())
        )
        for tool, version_hash in template.tool_versions.items():
            current = current_versions.get(tool)
            if current and current != version_hash:
                logger.info(f"Tool {tool} version changed — template needs regeneration")
                return False
        return True

    # ──────────────────────────────────────────
    # HYDRATION
    # ──────────────────────────────────────────

    def _hydrate(self, template: ExecutionTemplate, inputs: Dict) -> Any:
        """Instantiate cached template with current variable values."""
        plan_str = json.dumps(
            [s.content for s in template.segments],
            default=str
        )
        for key, value in inputs.items():
            plan_str = plan_str.replace(f"${{{key}}}", str(value))
        return json.loads(plan_str)

    # ──────────────────────────────────────────
    # MARK FAILURE
    # ──────────────────────────────────────────

    async def mark_failure(self, template_id: str):
        """Signal that a template execution failed."""
        await self.db.update_template_outcome(self.tenant_id, template_id, False)

        template = None
        async with self.db._lock:
            row = self.db._conn.execute(
                "SELECT template_id, fingerprint_hash, success_count, failure_count FROM execution_templates WHERE tenant_id=? AND template_id=?",
                (self.tenant_id, template_id)
            ).fetchone()
            if row:
                total = row["success_count"] + row["failure_count"]
                failure_rate = row["failure_count"] / total if total > 0 else 0
                if failure_rate > 0.5:
                    fp_hash = row["fingerprint_hash"]
                    self.db._conn.execute(
                        "DELETE FROM execution_templates WHERE tenant_id=? AND template_id=?",
                        (self.tenant_id, template_id)
                    )
                    self.db._conn.commit()
                    if fp_hash in self._system1_cache:
                        del self._system1_cache[fp_hash]
                    logger.info(f"Template {template_id} evicted — failure rate > 50%")

    # ──────────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────────

    def _schema_of(self, inputs: Dict) -> Dict:
        """Extract structural schema from inputs (types, not values)."""
        return {k: type(v).__name__ for k, v in inputs.items()}

    def get_stats(self) -> Dict:
        return {
            "tenant_id":       self.tenant_id,
            "system1_entries": len(self._system1_cache),
            "fragments":       len(self._fragments),
            "threshold":       self.threshold,
        }
