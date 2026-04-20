"""
Mnemon Pre-Warmed Fragment Library
100 curated execution fragments for common LLM agent task patterns.
Ships with Mnemon — provides immediate EME value on day one.

100 fragments across 10 categories (grounded in real LangChain/LangGraph/CrewAI usage):
- RAG & retrieval (15)
- Reasoning & planning (12)
- Multi-agent orchestration (10)
- Tool use & function calling (10)
- Memory & context management (10)
- Structured output & extraction (10)
- Error handling & resilience (8)
- Streaming & async (7)
- Human-in-the-loop (8)
- Optimization & efficiency (10)

Hand-curated from production usage patterns. EME semantic matching means
approximate matches still benefit from these — exact match not required.
"""

import hashlib
import json
import time
from typing import List

from ..core.models import TemplateSegment


def _seg(domain: str, action: str, params: dict, outputs: List[str]) -> dict:
    """Helper to build a fragment dict."""
    content = {"action": action, "params": params, "outputs": outputs}
    content_str = json.dumps(content, sort_keys=True)
    return {
        "domain":      domain,
        "action":      action,
        "content":     content,
        "fingerprint": hashlib.md5(content_str.encode()).hexdigest()[:16],
        "domain_tags": [domain, action.split("_")[0]],
        "outputs":     outputs,
        "success_rate": 1.0,
        "confidence":   0.95,
    }


# ─────────────────────────────────────────────
# CURATED FRAGMENTS
# ─────────────────────────────────────────────

CURATED_FRAGMENTS = [

    # ── RAG & Retrieval (15) ──────────────────────────────────────────────────

    _seg("rag", "retrieve_and_generate",
         {"top_k": 5, "similarity_threshold": 0.7, "rerank": False},
         ["answer", "source_docs", "citations"]),

    _seg("rag", "chunk_and_embed_document",
         {"chunk_size": 512, "overlap": 64, "embedding_model": "${embed_model}"},
         ["chunks", "embeddings", "chunk_count"]),

    _seg("rag", "hybrid_search",
         {"vector_weight": 0.7, "keyword_weight": 0.3, "top_k": 10},
         ["results", "scores", "result_count"]),

    _seg("rag", "rerank_retrieved_docs",
         {"reranker": "cross-encoder", "top_n": 3, "score_threshold": 0.5},
         ["reranked_docs", "scores"]),

    _seg("rag", "multi_query_retrieval",
         {"num_queries": 3, "dedup": True, "top_k_per_query": 5},
         ["all_docs", "queries_used", "total_retrieved"]),

    _seg("rag", "contextual_compression",
         {"max_tokens": 2000, "filter_irrelevant": True},
         ["compressed_context", "original_count", "kept_count"]),

    _seg("rag", "parent_document_retrieval",
         {"child_chunk_size": 200, "parent_chunk_size": 1000},
         ["parent_docs", "child_matches"]),

    _seg("rag", "generate_with_citations",
         {"cite_inline": True, "max_sources": 5, "format": "markdown"},
         ["answer", "citations", "source_urls"]),

    _seg("rag", "check_retrieval_relevance",
         {"relevance_threshold": 0.6, "fallback_to_llm": True},
         ["relevant_docs", "filtered_count", "used_fallback"]),

    _seg("rag", "semantic_query_expansion",
         {"num_variants": 3, "use_llm": True},
         ["expanded_queries", "original_query"]),

    _seg("rag", "long_context_rag",
         {"max_context_tokens": 8000, "hierarchical": True, "summary_first": True},
         ["answer", "context_used", "chunks_summarized"]),

    _seg("rag", "knowledge_graph_retrieval",
         {"hop_depth": 2, "entity_types": "${entity_types}", "limit": 50},
         ["subgraph", "entities", "relations"]),

    _seg("rag", "fact_grounded_generation",
         {"verify_claims": True, "source_required": True},
         ["answer", "verified_facts", "unverified_claims"]),

    _seg("rag", "update_vector_store",
         {"upsert": True, "batch_size": 100, "dedup_by": "content_hash"},
         ["upserted_count", "skipped_count"]),

    _seg("rag", "cross_domain_fusion",
         {"domains": "${domains}", "merge_strategy": "weighted", "top_k": 5},
         ["fused_results", "domain_scores"]),

    # ── Reasoning & Planning (12) ─────────────────────────────────────────────

    _seg("reasoning", "react_loop",
         {"max_iterations": 10, "stop_on_final_answer": True},
         ["final_answer", "steps_taken", "tools_used"]),

    _seg("reasoning", "chain_of_thought",
         {"show_steps": True, "verify_conclusion": True},
         ["reasoning_steps", "conclusion", "confidence"]),

    _seg("reasoning", "decompose_goal_to_tasks",
         {"max_subtasks": 8, "include_dependencies": True, "estimate_effort": True},
         ["subtasks", "dependency_graph", "total_effort"]),

    _seg("reasoning", "tree_of_thought",
         {"branching_factor": 3, "max_depth": 4, "pruning": True},
         ["best_path", "explored_paths", "final_answer"]),

    _seg("reasoning", "create_execution_plan",
         {"ordered": True, "parallel_steps": True, "include_rollback": True},
         ["plan", "step_count", "estimated_duration"]),

    _seg("reasoning", "dynamic_replan",
         {"trigger": "step_failure", "preserve_completed": True},
         ["updated_plan", "replanned_steps", "reason"]),

    _seg("reasoning", "hypothesis_verification",
         {"max_hypotheses": 5, "verify_with_tools": True},
         ["verified_hypothesis", "evidence", "confidence"]),

    _seg("reasoning", "conditional_routing",
         {"conditions": "${routing_rules}", "default_route": "fallback"},
         ["selected_route", "matched_condition", "route_reason"]),

    _seg("reasoning", "self_critique",
         {"critique_dimensions": ["accuracy", "completeness", "clarity"],
          "min_score": 0.8},
         ["critique", "score", "needs_revision"]),

    _seg("reasoning", "constraint_satisfaction",
         {"constraints": "${constraints}", "hard_fail": False},
         ["solution", "satisfied_constraints", "violated_constraints"]),

    _seg("reasoning", "evaluate_options",
         {"criteria": "${criteria}", "weights": "${weights}", "top_n": 1},
         ["ranked_options", "scores", "recommendation"]),

    _seg("reasoning", "root_cause_analysis",
         {"depth": 5, "include_evidence": True, "format": "fishbone"},
         ["root_causes", "contributing_factors", "evidence"]),

    # ── Multi-Agent Orchestration (10) ────────────────────────────────────────

    _seg("multiagent", "supervisor_delegate",
         {"worker_agents": "${agents}", "max_rounds": 10, "require_consensus": False},
         ["final_result", "agent_outputs", "rounds_taken"]),

    _seg("multiagent", "agent_handoff",
         {"target_agent": "${agent_name}", "pass_context": True, "pass_memory": True},
         ["handoff_complete", "context_transferred"]),

    _seg("multiagent", "parallel_agent_execution",
         {"agents": "${agents}", "timeout_seconds": 120, "aggregate": "merge"},
         ["results", "agent_statuses", "failed_agents"]),

    _seg("multiagent", "swarm_coordination",
         {"coordination": "peer_to_peer", "max_hops": 5, "shared_memory": True},
         ["final_state", "agent_path", "messages_passed"]),

    _seg("multiagent", "agent_consensus",
         {"min_agreement": 0.7, "voting_method": "majority"},
         ["consensus_result", "agreement_score", "dissenting_agents"]),

    _seg("multiagent", "skill_based_routing",
         {"skill_registry": "${skills}", "fallback_agent": "generalist"},
         ["assigned_agent", "skill_match_score", "result"]),

    _seg("multiagent", "subagent_as_tool",
         {"subagent": "${agent_name}", "timeout": 60, "retry_on_fail": True},
         ["subagent_result", "execution_time", "success"]),

    _seg("multiagent", "aggregate_agent_results",
         {"strategy": "best_of", "dedup": True, "rank_by": "confidence"},
         ["aggregated_result", "source_agents", "confidence"]),

    _seg("multiagent", "fallback_agent_chain",
         {"agents_in_order": "${agent_list}", "escalate_on_low_confidence": True},
         ["result", "agent_used", "fallback_triggered"]),

    _seg("multiagent", "broadcast_task",
         {"agents": "${agents}", "collect_all": True, "timeout": 30},
         ["all_results", "responded_agents", "timed_out_agents"]),

    # ── Tool Use & Function Calling (10) ──────────────────────────────────────

    _seg("tool", "select_and_call_tool",
         {"available_tools": "${tools}", "validate_args": True, "timeout": 30},
         ["tool_result", "tool_used", "args_used"]),

    _seg("tool", "multi_tool_sequence",
         {"tools": "${tool_chain}", "pass_outputs": True, "stop_on_error": False},
         ["final_result", "intermediate_results", "tools_called"]),

    _seg("tool", "parallel_tool_calls",
         {"tools": "${tools}", "max_concurrent": 5, "merge_results": True},
         ["results", "execution_times", "failed_tools"]),

    _seg("tool", "tool_error_recovery",
         {"max_retries": 3, "fallback_tool": "${fallback}", "log_failure": True},
         ["result", "retries_used", "used_fallback"]),

    _seg("tool", "validate_tool_output",
         {"schema": "${output_schema}", "required_fields": "${fields}",
          "coerce": True},
         ["validated_output", "validation_errors", "is_valid"]),

    _seg("tool", "tool_result_cache",
         {"ttl_seconds": 300, "key_by": "args_hash"},
         ["result", "cache_hit", "cache_key"]),

    _seg("tool", "nl_to_sql_query",
         {"schema": "${db_schema}", "dialect": "postgres", "validate": True},
         ["sql_query", "tables_used", "estimated_rows"]),

    _seg("tool", "web_search_and_extract",
         {"query": "${query}", "top_results": 5, "extract_text": True},
         ["results", "urls", "extracted_text"]),

    _seg("tool", "code_execution_sandbox",
         {"language": "python", "timeout": 30, "capture_output": True},
         ["stdout", "stderr", "exit_code", "success"]),

    _seg("tool", "dynamic_tool_loading",
         {"tool_registry": "${registry}", "filter_by_task": True},
         ["loaded_tools", "tool_descriptions"]),

    # ── Memory & Context Management (10) ─────────────────────────────────────

    _seg("memory", "summarize_conversation",
         {"max_tokens": 500, "keep_last_n": 5, "preserve_entities": True},
         ["summary", "kept_messages", "summarized_count"]),

    _seg("memory", "inject_relevant_memories",
         {"top_k": 5, "similarity_threshold": 0.65, "max_tokens": 1000},
         ["injected_context", "memories_used", "tokens_used"]),

    _seg("memory", "update_entity_memory",
         {"entity": "${entity_name}", "merge_strategy": "latest_wins"},
         ["updated_entity", "previous_value", "changed_fields"]),

    _seg("memory", "checkpoint_state",
         {"thread_id": "${thread_id}", "include_messages": True},
         ["checkpoint_id", "state_snapshot", "timestamp"]),

    _seg("memory", "restore_from_checkpoint",
         {"checkpoint_id": "${checkpoint_id}", "validate": True},
         ["restored_state", "messages_restored", "success"]),

    _seg("memory", "compress_context_window",
         {"target_tokens": 4000, "strategy": "hierarchical_summary"},
         ["compressed_context", "original_tokens", "compressed_tokens"]),

    _seg("memory", "prune_stale_memories",
         {"max_age_days": 30, "min_access_count": 2, "keep_high_importance": True},
         ["pruned_count", "kept_count", "freed_bytes"]),

    _seg("memory", "thread_isolated_state",
         {"thread_id": "${thread_id}", "isolate_from": "global"},
         ["thread_state", "isolation_confirmed"]),

    _seg("memory", "cross_session_context",
         {"session_id": "${session_id}", "merge_with_current": True},
         ["merged_context", "sessions_merged", "conflicts_resolved"]),

    _seg("memory", "track_conversation_entities",
         {"entity_types": ["person", "place", "concept", "decision"],
          "update_on_mention": True},
         ["entity_map", "new_entities", "updated_entities"]),

    # ── Structured Output & Extraction (10) ──────────────────────────────────

    _seg("extract", "json_schema_extraction",
         {"schema": "${json_schema}", "strict": True, "retry_on_invalid": True},
         ["extracted_data", "validation_errors", "is_valid"]),

    _seg("extract", "pydantic_model_parse",
         {"model": "${model_class}", "coerce_types": True},
         ["parsed_object", "field_errors", "success"]),

    _seg("extract", "named_entity_recognition",
         {"entity_types": ["PERSON", "ORG", "DATE", "LOC", "MONEY"],
          "include_context": True},
         ["entities", "entity_count", "entity_map"]),

    _seg("extract", "extract_action_items",
         {"include_owner": True, "include_deadline": True, "priority": True},
         ["action_items", "owners", "deadlines"]),

    _seg("extract", "extract_key_facts",
         {"max_facts": 10, "include_confidence": True, "deduplicate": True},
         ["facts", "confidence_scores", "sources"]),

    _seg("extract", "table_extraction",
         {"from_format": "text", "headers": "${headers}", "normalize": True},
         ["rows", "headers", "row_count"]),

    _seg("extract", "requirements_extraction",
         {"types": ["functional", "non_functional"], "format": "structured"},
         ["requirements", "priorities", "ambiguities"]),

    _seg("extract", "sentiment_and_intent",
         {"granularity": "sentence", "include_confidence": True},
         ["sentiment", "intent", "confidence", "key_phrases"]),

    _seg("extract", "relationship_extraction",
         {"entity_pairs": True, "relation_types": "${relation_types}"},
         ["relationships", "entity_pairs", "relation_count"]),

    _seg("extract", "document_classification",
         {"categories": "${categories}", "multi_label": False,
          "include_confidence": True},
         ["category", "confidence", "runner_up"]),

    # ── Error Handling & Resilience (8) ──────────────────────────────────────

    _seg("error", "retry_with_exponential_backoff",
         {"max_attempts": 5, "base_seconds": 2, "max_seconds": 60,
          "jitter": True},
         ["result", "attempts_used", "total_wait_seconds"]),

    _seg("error", "fallback_to_alternate_model",
         {"primary": "${primary_model}", "fallbacks": "${fallback_models}",
          "on_errors": ["rate_limit", "timeout", "overload"]},
         ["result", "model_used", "fallback_triggered"]),

    _seg("error", "self_correction_loop",
         {"max_corrections": 3, "correction_prompt": "${prompt}",
          "validation_fn": "${validator}"},
         ["corrected_output", "correction_count", "final_valid"]),

    _seg("error", "circuit_breaker",
         {"failure_threshold": 5, "recovery_seconds": 60,
          "half_open_test": True},
         ["allowed", "circuit_state", "failure_count"]),

    _seg("error", "graceful_degradation",
         {"degraded_response": "${fallback_response}",
          "log_degradation": True},
         ["response", "is_degraded", "reason"]),

    _seg("error", "validate_with_guardrails",
         {"safety_checks": ["toxicity", "pii", "hallucination"],
          "block_on_fail": True},
         ["passed", "violations", "safe_output"]),

    _seg("error", "timeout_with_partial_result",
         {"timeout_seconds": 30, "return_partial": True},
         ["result", "is_partial", "completed_fraction"]),

    _seg("error", "classify_and_route_error",
         {"error_taxonomy": "${taxonomy}", "auto_retry_transient": True},
         ["error_class", "action_taken", "resolved"]),

    # ── Streaming & Async (7) ─────────────────────────────────────────────────

    _seg("stream", "token_streaming",
         {"buffer_size": 1, "on_token": "${callback}", "collect_full": True},
         ["full_response", "token_count", "stream_complete"]),

    _seg("stream", "chunk_streaming",
         {"chunk_by": "sentence", "emit_partial": True},
         ["chunks", "chunk_count", "full_text"]),

    _seg("stream", "async_parallel_tasks",
         {"tasks": "${task_list}", "max_concurrent": 10, "gather_errors": True},
         ["results", "errors", "completed_count"]),

    _seg("stream", "event_driven_step",
         {"event_type": "${event}", "handler": "${handler}",
          "ack_on_success": True},
         ["handled", "event_id", "response"]),

    _seg("stream", "batch_llm_calls",
         {"batch_size": 20, "max_tokens_per_call": 2000, "dedup_inputs": True},
         ["batch_results", "batches_sent", "total_processed"]),

    _seg("stream", "long_running_task_poll",
         {"poll_interval_seconds": 5, "timeout_seconds": 300,
          "status_field": "status"},
         ["final_status", "result", "poll_count"]),

    _seg("stream", "streaming_aggregation",
         {"aggregate_fn": "concat", "flush_on_complete": True},
         ["aggregated", "item_count", "duration_seconds"]),

    # ── Human-in-the-Loop (8) ────────────────────────────────────────────────

    _seg("hitl", "pause_for_approval",
         {"prompt_human": "${approval_message}", "timeout_seconds": 3600,
          "default_on_timeout": "reject"},
         ["approved", "human_response", "timed_out"]),

    _seg("hitl", "confidence_based_escalation",
         {"confidence_threshold": 0.75, "escalate_to": "${human_queue}"},
         ["escalated", "confidence_score", "agent_answer"]),

    _seg("hitl", "human_feedback_incorporation",
         {"feedback": "${feedback}", "update_memory": True,
          "rerun_with_feedback": True},
         ["revised_output", "changes_made", "memory_updated"]),

    _seg("hitl", "interactive_refinement",
         {"max_rounds": 5, "accept_keywords": ["done", "good", "ok"]},
         ["final_output", "rounds_taken", "accepted_by_user"]),

    _seg("hitl", "human_edit_before_execute",
         {"show_plan": True, "editable_fields": "${fields}",
          "proceed_without_edit": False},
         ["edited_plan", "was_edited", "execution_approved"]),

    _seg("hitl", "preference_learning",
         {"track_selections": True, "update_weights": True,
          "min_samples": 5},
         ["preferences_updated", "sample_count", "top_preference"]),

    _seg("hitl", "out_of_scope_escalation",
         {"scope_check": "${scope_rules}", "escalation_target": "human"},
         ["in_scope", "escalated_to", "reason"]),

    _seg("hitl", "review_and_publish",
         {"review_checklist": "${checklist}", "require_all": True,
          "auto_publish_on_pass": False},
         ["review_passed", "failed_checks", "published"]),

    # ── Optimization & Efficiency (10) ────────────────────────────────────────

    _seg("optimize", "semantic_cache_lookup",
         {"similarity_threshold": 0.92, "ttl_seconds": 3600,
          "cache_backend": "memory"},
         ["cache_hit", "cached_response", "similarity_score"]),

    _seg("optimize", "model_routing_by_complexity",
         {"simple_model": "${cheap_model}", "complex_model": "${smart_model}",
          "complexity_threshold": 0.6},
         ["model_selected", "complexity_score", "estimated_cost"]),

    _seg("optimize", "token_budget_management",
         {"max_input_tokens": 4000, "max_output_tokens": 1000,
          "compress_if_over": True},
         ["within_budget", "input_tokens", "output_tokens", "compressed"]),

    _seg("optimize", "prompt_compression",
         {"target_ratio": 0.5, "preserve_instructions": True,
          "method": "selective_removal"},
         ["compressed_prompt", "original_tokens", "compressed_tokens"]),

    _seg("optimize", "deduplicate_llm_calls",
         {"hash_by": "prompt_content", "window_seconds": 60},
         ["result", "was_duplicate", "original_call_id"]),

    _seg("optimize", "cost_aware_tool_selection",
         {"cost_per_tool": "${tool_costs}", "budget": "${budget}",
          "prefer_cheap": True},
         ["selected_tool", "estimated_cost", "budget_remaining"]),

    _seg("optimize", "output_format_optimization",
         {"target_format": "${format}", "minimize_tokens": True,
          "structured": True},
         ["formatted_output", "token_count", "format_used"]),

    _seg("optimize", "request_batching",
         {"batch_window_ms": 100, "max_batch_size": 20,
          "priority_queue": True},
         ["batch_results", "batch_size", "wait_time_ms"]),

    _seg("optimize", "prompt_template_reuse",
         {"template_id": "${template}", "fill_vars": "${variables}",
          "validate_filled": True},
         ["filled_prompt", "template_hit", "variables_used"]),

    _seg("optimize", "latency_aware_routing",
         {"latency_budget_ms": 2000, "prefer_cached": True,
          "parallel_if_slow": True},
         ["result", "latency_ms", "route_used"]),

]


def load_fragments(tenant_id: str) -> List[TemplateSegment]:
    """
    Load pre-warmed fragments as TemplateSegment objects.
    Called during Mnemon initialisation for cold-start value.
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from ..core.memory import SimpleEmbedder
    embedder = SimpleEmbedder()

    segments = []
    for i, frag in enumerate(CURATED_FRAGMENTS):
        content_str = json.dumps(frag["content"], sort_keys=True)
        sig = embedder.embed(content_str)

        seg = TemplateSegment(
            segment_id=f"prewarmed_{i:04d}_{frag['fingerprint']}",
            tenant_id=tenant_id,
            content=frag["content"],
            fingerprint=frag["fingerprint"],
            signature=sig,
            domain_tags=set(frag["domain_tags"]),
            outputs=frag["outputs"],
            success_rate=frag["success_rate"],
            confidence=frag["confidence"],
            is_generated=False,
            use_count=0,
        )
        segments.append(seg)

    return segments


FRAGMENT_COUNT = len(CURATED_FRAGMENTS)


# ─────────────────────────────────────────────
# META-VOCABULARY FOR SEMANTIC TAGGER
# ─────────────────────────────────────────────
# 150 domain concepts across 9 categories.
# Used by SemanticVocabularyTagger to assign tags
# via embedding similarity — no LLM on the write path.

META_VOCABULARY: List[str] = [
    # retrieval & rag (20)
    "retrieval augmented generation", "vector similarity search", "document chunking",
    "semantic retrieval", "hybrid search", "document embedding", "reranking",
    "context compression", "knowledge graph query", "multi query retrieval",
    "citation generation", "fact grounded answer", "parent document retrieval",
    "cross domain retrieval", "query expansion", "vector store upsert",
    "relevance filtering", "long context handling", "RAG pipeline", "dense retrieval",

    # reasoning & planning (20)
    "chain of thought", "react reasoning", "task decomposition", "execution plan",
    "tree of thought", "hypothesis verification", "conditional routing",
    "self critique", "constraint satisfaction", "root cause analysis",
    "option evaluation", "dynamic replanning", "goal decomposition",
    "step by step reasoning", "backward chaining", "multi step reasoning",
    "problem solving", "decision making", "planning agent", "reasoning loop",

    # multi-agent (20)
    "supervisor agent", "worker agent", "agent handoff", "agent delegation",
    "parallel agents", "swarm coordination", "agent consensus", "skill routing",
    "subagent tool", "agent broadcast", "multi agent system", "agent orchestration",
    "agent specialization", "fallback agent", "agent collaboration", "agent debate",
    "hierarchical agents", "peer to peer agents", "agent aggregation", "agent mesh",

    # tool use (20)
    "tool calling", "function calling", "tool selection", "tool chaining",
    "parallel tool execution", "tool error handling", "tool validation",
    "tool result caching", "code execution", "web search tool",
    "SQL generation", "tool composition", "dynamic tool loading",
    "tool sandboxing", "tool cost optimization", "tool instrumentation",
    "tool timeout", "tool fallback", "tool argument extraction", "tool grounding",

    # memory (20)
    "conversation memory", "long term memory", "memory injection",
    "context window management", "conversation summarization", "checkpoint save",
    "checkpoint restore", "entity tracking", "memory pruning", "thread isolation",
    "session memory", "cross session context", "episodic memory", "working memory",
    "memory retrieval", "memory compression", "memory update", "context trimming",
    "token budget", "context overflow handling",

    # structured output (20)
    "JSON extraction", "pydantic parsing", "named entity recognition",
    "action item extraction", "key fact extraction", "table extraction",
    "requirements parsing", "sentiment analysis", "intent classification",
    "relationship extraction", "document classification", "schema validation",
    "structured generation", "output formatting", "data normalization",
    "information extraction", "entity extraction", "attribute extraction",
    "hierarchical extraction", "multi label classification",

    # error handling (15)
    "exponential backoff", "model fallback", "self correction", "circuit breaker",
    "graceful degradation", "guardrail validation", "timeout handling",
    "error classification", "retry logic", "error recovery",
    "hallucination detection", "safety check", "output validation",
    "transient error", "permanent error",

    # streaming & async (15)
    "token streaming", "chunk streaming", "async execution", "event driven",
    "batch processing", "long running task", "streaming aggregation",
    "concurrent tasks", "parallel execution", "async tool call",
    "real time output", "progressive response", "status polling",
    "task queue", "background job",

    # human in the loop (15)
    "human approval", "human escalation", "human feedback", "interactive refinement",
    "confidence threshold", "human review", "preference learning",
    "out of scope handling", "human oversight", "manual review",
    "edit before execute", "human in the loop", "approval workflow",
    "user correction", "human validation",

    # optimization (15)
    "semantic caching", "model routing", "token optimization", "prompt compression",
    "deduplication", "cost optimization", "latency optimization", "request batching",
    "prompt template", "output compression", "efficiency", "cost aware routing",
    "smart caching", "call deduplication", "budget management",
]
