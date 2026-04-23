"""Top-level LangGraph workflow for the deep-research agent."""

from __future__ import annotations

import json
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from perplexity_at_home.agents.deep_research.answer_agent import build_answer_agent
from perplexity_at_home.agents.deep_research.context import DeepResearchContext
from perplexity_at_home.agents.deep_research.planner_agent import build_planner_agent
from perplexity_at_home.agents.deep_research.query_agent import build_query_agent
from perplexity_at_home.agents.deep_research.reflection_agent import (
    build_reflection_agent,
)
from perplexity_at_home.agents.deep_research.retrieval_agent import (
    build_retrieval_agent,
)
from perplexity_at_home.agents.deep_research.state import (
    DeepResearchState,
    EvidenceItemRecord,
    PlannedToolCallRecord,
    ReflectionDecisionRecord,
)

_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_ACTION_TO_STRATEGY = {
    "requery": "search",
    "extract": "search_then_extract",
    "map": "map_then_extract",
    "crawl": "crawl_domain",
    "research": "tavily_research",
}


class SupportsInvoke(Protocol):
    """Protocol for child agents invoked by the top-level graph."""

    def invoke(
        self,
        input: dict[str, Any],
        *,
        context: DeepResearchContext,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke the child agent."""


def _extract_latest_user_question(state: DeepResearchState) -> str:
    """Extract the best available user question from state."""
    original_question = state.get("original_question")
    if isinstance(original_question, str) and original_question.strip():
        return original_question

    for message in reversed(state.get("messages", [])):
        if isinstance(message, dict):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            continue

        msg_type = getattr(message, "type", None)
        content = getattr(message, "content", None)
        if msg_type == "human" and isinstance(content, str) and content.strip():
            return content

    raise ValueError("Could not extract a user question from the deep-research state.")


def _structured_response_to_dict(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a child-agent structured response into a JSON-friendly dict."""
    structured = result.get("structured_response")
    if structured is None:
        raise KeyError("Child agent did not return a structured_response payload.")

    model_dump = getattr(structured, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped

    if isinstance(structured, dict):
        return structured

    raise TypeError("Child agent structured_response must be a model or dictionary.")


def _deduplicate_strings(values: list[str]) -> list[str]:
    """Deduplicate strings while preserving order."""
    seen: set[str] = set()
    deduplicated: list[str] = []

    for value in values:
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        deduplicated.append(stripped)

    return deduplicated


def _deduplicate_evidence_items(
    evidence_items: list[EvidenceItemRecord],
) -> list[EvidenceItemRecord]:
    """Deduplicate evidence items while preserving first-seen order."""
    deduplicated: list[EvidenceItemRecord] = []
    seen: set[tuple[str, str, str, str]] = set()

    for item in evidence_items:
        url = str(item.get("url") or "")
        title = str(item.get("title") or "")
        content = str(item.get("content") or "")
        subquestion_id = str(item.get("subquestion_id") or "")
        key = (url, title, content[:200], subquestion_id)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)

    return deduplicated


def _tool_name_for_strategy(strategy: str) -> str:
    """Map a retrieval strategy to the main tool family it implies."""
    if strategy in {"search", "search_then_extract"}:
        return "search"
    if strategy == "extract_known_urls":
        return "extract"
    if strategy == "map_then_extract":
        return "map"
    if strategy == "crawl_domain":
        return "crawl"
    if strategy == "tavily_research":
        return "research"
    return "search"


def _build_planned_tool_calls(query_plans: list[dict[str, Any]]) -> list[PlannedToolCallRecord]:
    """Translate query plans into graph-visible planned tool calls."""
    planned_calls: list[PlannedToolCallRecord] = []

    for plan in query_plans:
        subquestion_id = str(plan.get("subquestion_id") or "followup")
        recommendation = plan.get("retrieval_recommendation", {})
        strategy = str(recommendation.get("strategy") or "search")
        tool_name = _tool_name_for_strategy(strategy)

        for query in plan.get("queries", []):
            priority = int(query.get("priority", 1))
            planned_calls.append(
                {
                    "tool_call_id": f"{subquestion_id}-{tool_name}-{priority}",
                    "tool_name": tool_name,
                    "subquestion_id": subquestion_id,
                    "query": str(query.get("query") or ""),
                    "url": "",
                    "priority": priority,
                    "rationale": str(query.get("rationale") or recommendation.get("rationale") or ""),
                    "metadata": {
                        "recommended_strategy": strategy,
                        "target_topic": query.get("target_topic"),
                        "preferred_domains": recommendation.get("preferred_domains", []),
                        "known_urls": recommendation.get("known_urls", []),
                    },
                }
            )

        for index, url in enumerate(recommendation.get("known_urls", []), start=1):
            planned_calls.append(
                {
                    "tool_call_id": f"{subquestion_id}-extract-url-{index}",
                    "tool_name": "extract",
                    "subquestion_id": subquestion_id,
                    "query": "",
                    "url": str(url),
                    "priority": index,
                    "rationale": str(recommendation.get("rationale") or "Known URL extraction."),
                    "metadata": {
                        "recommended_strategy": strategy,
                        "preferred_domains": recommendation.get("preferred_domains", []),
                    },
                }
            )

    return planned_calls


def _latest_reflection(state: DeepResearchState) -> ReflectionDecisionRecord | None:
    """Return the latest reflection decision if one exists."""
    reflection_history = state.get("reflection_history", [])
    if not reflection_history:
        return None
    latest = reflection_history[-1]
    return latest if isinstance(latest, dict) else None


def _subquestion_lookup(state: DeepResearchState) -> dict[str, dict[str, Any]]:
    """Index subquestions by identifier."""
    lookup: dict[str, dict[str, Any]] = {}
    for subquestion in state.get("subquestions", []):
        subquestion_id = subquestion.get("subquestion_id")
        if isinstance(subquestion_id, str) and subquestion_id:
            lookup[subquestion_id] = subquestion
    return lookup


def _collect_known_urls(state: DeepResearchState, subquestion_ids: list[str]) -> list[str]:
    """Collect known URLs for targeted follow-up extraction or verification."""
    target_ids = set(subquestion_ids)
    urls: list[str] = []

    for evidence_item in state.get("evidence_items", []):
        evidence_subquestion_id = evidence_item.get("subquestion_id")
        if target_ids and evidence_subquestion_id not in target_ids:
            continue
        url = evidence_item.get("url")
        if isinstance(url, str) and url.strip():
            urls.append(url.strip())

    return _deduplicate_strings(urls)


def _collect_preferred_domains(state: DeepResearchState, known_urls: list[str]) -> list[str]:
    """Collect preferred domains from known URLs and research brief hints."""
    domains: list[str] = []

    for url in known_urls:
        hostname = urlparse(url).hostname
        if hostname:
            domains.append(hostname)

    research_brief = state.get("research_brief", {})
    domains.extend(str(domain) for domain in research_brief.get("domain_hints", []))

    return _deduplicate_strings(domains)


def _build_followup_query_plans(
    state: DeepResearchState,
    *,
    action: str,
) -> list[dict[str, Any]]:
    """Build follow-up query plans from the latest reflection output."""
    latest_reflection = _latest_reflection(state) or {}
    followup_queries = sorted(
        latest_reflection.get("followup_queries", []),
        key=lambda item: int(item.get("priority", 1)),
    )
    max_branches = int(state.get("max_parallel_retrieval_branches_allowed", 1) or 1)
    subquestion_lookup = _subquestion_lookup(state)

    if not followup_queries:
        fallback_query = latest_reflection.get("rationale") or state.get("normalized_question") or state.get(
            "original_question"
        )
        followup_queries = [
            {
                "query": str(fallback_query),
                "rationale": "Fallback follow-up query based on the latest reflection.",
                "priority": 1,
                "target_subquestion_ids": state.get("active_subquestion_ids", []),
            }
        ]

    plans: list[dict[str, Any]] = []
    for index, followup in enumerate(followup_queries[:max_branches], start=1):
        target_subquestion_ids = [
            subquestion_id
            for subquestion_id in followup.get("target_subquestion_ids", [])
            if isinstance(subquestion_id, str) and subquestion_id
        ]
        fallback_subquestion_id = (
            target_subquestion_ids[0]
            if target_subquestion_ids
            else f"followup_{state.get('iteration_count', 0) + 1}_{index}"
        )
        subquestion = subquestion_lookup.get(fallback_subquestion_id, {})
        known_urls = _collect_known_urls(state, target_subquestion_ids)
        preferred_domains = _collect_preferred_domains(state, known_urls)
        strategy = _ACTION_TO_STRATEGY.get(action, "search")

        if action == "extract" and known_urls:
            strategy = "extract_known_urls"

        plans.append(
            {
                "subquestion_id": fallback_subquestion_id,
                "subquestion": str(subquestion.get("question") or followup.get("query") or ""),
                "research_focus": str(
                    followup.get("rationale")
                    or latest_reflection.get("rationale")
                    or "Target the remaining evidence gaps."
                ),
                "requires_freshness": True,
                "ambiguity_note": None,
                "target_queries": 1,
                "min_queries": 1,
                "max_queries": 1,
                "retrieval_recommendation": {
                    "strategy": strategy,
                    "rationale": str(
                        followup.get("rationale")
                        or latest_reflection.get("rationale")
                        or "Follow the reflection agent recommendation."
                    ),
                    "preferred_domains": preferred_domains,
                    "known_urls": known_urls,
                    "should_fan_out": False,
                    "recommended_max_branches": 1,
                },
                "queries": [
                    {
                        "query": str(followup.get("query") or state.get("normalized_question") or ""),
                        "rationale": str(
                            followup.get("rationale")
                            or latest_reflection.get("rationale")
                            or "Follow-up query for unresolved gaps."
                        ),
                        "priority": int(followup.get("priority", index)),
                        "intent": "verification",
                        "target_topic": "general",
                        "prefer_recent_sources": True,
                        "preferred_source_types": [],
                        "follow_up_of": None,
                    }
                ],
            }
        )

    return plans


def build_deep_research_graph(
    *,
    planner_agent: SupportsInvoke | None = None,
    query_agent: SupportsInvoke | None = None,
    retrieval_agent: SupportsInvoke | None = None,
    reflection_agent: SupportsInvoke | None = None,
    answer_agent: SupportsInvoke | None = None,
    checkpointer: Any = None,
    store: Any = None,
    debug: bool = False,
) -> Any:
    """Build the compiled top-level deep-research graph."""
    resolved_planner_agent = planner_agent or build_planner_agent()
    resolved_query_agent = query_agent or build_query_agent()
    resolved_retrieval_agent = retrieval_agent or build_retrieval_agent()
    resolved_reflection_agent = reflection_agent or build_reflection_agent()
    resolved_answer_agent = answer_agent or build_answer_agent()

    def plan_research(
        state: DeepResearchState,
        runtime: Runtime[DeepResearchContext],
    ) -> dict[str, Any]:
        context = runtime.context
        user_question = _extract_latest_user_question(state)
        result = resolved_planner_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_question,
                    }
                ]
            },
            context=context,
            config={
                "configurable": {
                    "thread_id": f"{context.thread_id}-planner-agent",
                }
            },
        )
        planning = _structured_response_to_dict(result)
        subquestions = planning.get("subquestions", [])
        sorted_subquestions = sorted(
            subquestions,
            key=lambda item: _PRIORITY_ORDER.get(str(item.get("priority") or "medium"), 1),
        )
        capped_subquestions = sorted_subquestions[: context.max_subquestions]
        active_subquestion_ids = [
            str(subquestion.get("subquestion_id"))
            for subquestion in capped_subquestions
            if isinstance(subquestion.get("subquestion_id"), str)
        ]

        return {
            "original_question": planning["original_question"],
            "normalized_question": planning["normalized_question"],
            "clarified_question": planning["normalized_question"],
            "clarification_needed": bool(planning.get("needs_clarification", False)),
            "clarification_question": str(planning.get("clarification_question") or ""),
            "research_brief": planning["research_brief"],
            "subquestions": capped_subquestions,
            "planning_notes": list(planning.get("planning_notes", [])),
            "active_subquestion_ids": active_subquestion_ids,
            "completed_subquestion_ids": [],
            "max_iterations_allowed": context.max_iterations,
            "max_parallel_retrieval_branches_allowed": context.max_parallel_retrieval_branches,
            "clarification_interrupts_allowed": context.allow_interrupts_for_clarification,
        }

    def request_clarification(state: DeepResearchState) -> dict[str, Any]:
        latest_reflection = _latest_reflection(state) or {}
        clarification_question = (
            str(state.get("clarification_question") or "").strip()
            or str(latest_reflection.get("rationale") or "").strip()
            or "The request needs clarification before deep research can continue."
        )

        return {
            "final_answer": {
                "status": "needs_clarification",
                "report_markdown": (
                    "## Clarification Needed\n\n"
                    "Deep research paused because the scope is still ambiguous.\n\n"
                    f"{clarification_question}"
                ),
                "executive_summary": clarification_question,
                "key_findings": [],
                "citations": [],
                "confidence": 0.0,
                "used_search": bool(state.get("evidence_items")),
                "evidence_count": len(state.get("evidence_items", [])),
                "uncertainty_note": "Clarification is required before reliable synthesis.",
                "unresolved_questions": [clarification_question],
            },
            "is_complete": True,
        }

    def generate_query_plans(
        state: DeepResearchState,
        runtime: Runtime[DeepResearchContext],
    ) -> dict[str, Any]:
        context = runtime.context
        query_input = {
            "original_question": state["original_question"],
            "normalized_question": state["normalized_question"],
            "research_brief": state.get("research_brief", {}),
            "subquestions": state.get("subquestions", []),
            "planning_notes": state.get("planning_notes", []),
        }
        result = resolved_query_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(query_input, indent=2, default=str),
                    }
                ]
            },
            context=context,
            config={
                "configurable": {
                    "thread_id": f"{context.thread_id}-query-agent",
                }
            },
        )
        query_planning = _structured_response_to_dict(result)
        plans = list(query_planning.get("plans", []))
        return {
            "query_plans": plans,
            "active_query_plans": plans,
            "query_plan_notes": list(query_planning.get("global_notes", [])),
            "planned_tool_calls": _build_planned_tool_calls(plans),
            "active_retrieval_action": "initial",
        }

    def run_retrieval(
        state: DeepResearchState,
        runtime: Runtime[DeepResearchContext],
    ) -> dict[str, Any]:
        context = runtime.context
        payload = {
            "original_question": state["original_question"],
            "normalized_question": state.get("normalized_question"),
            "research_brief": state.get("research_brief", {}),
            "subquestion_query_plans": state.get("active_query_plans", []),
            "prior_evidence": state.get("evidence_items", []),
            "open_gaps": state.get("open_gaps", []),
            "iteration_count": state.get("iteration_count", 0),
            "query_plan_notes": state.get("query_plan_notes", []),
            "latest_reflection": _latest_reflection(state),
        }
        next_iteration = int(state.get("iteration_count", 0)) + 1
        result = resolved_retrieval_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(payload, indent=2, default=str),
                    }
                ]
            },
            context=context,
            config={
                "configurable": {
                    "thread_id": f"{context.thread_id}-retrieval-agent-{next_iteration}",
                }
            },
        )
        retrieval = _structured_response_to_dict(result)
        raw_retrieval_results = list(state.get("raw_retrieval_results", []))
        raw_retrieval_results.append(retrieval)

        evidence_items = _deduplicate_evidence_items(
            list(state.get("evidence_items", [])) + list(retrieval.get("evidence_items", []))
        )
        open_gaps = _deduplicate_strings(
            list(state.get("open_gaps", [])) + list(retrieval.get("unresolved_gaps", []))
        )

        retrieval_router_decisions = list(state.get("retrieval_router_decisions", []))
        retrieval_router_decisions.append(
            {
                "iteration": next_iteration,
                "action": state.get("active_retrieval_action", "initial"),
                "recommended_strategy": retrieval.get("recommended_strategy"),
                "applied_strategy": retrieval.get("applied_strategy"),
                "recommended_next_action": retrieval.get("recommended_next_action"),
                "followed_recommended_strategy": retrieval.get("followed_recommended_strategy"),
                "confidence": retrieval.get("confidence"),
                "used_tools": retrieval.get("used_tools", []),
            }
        )

        return {
            "raw_retrieval_results": raw_retrieval_results,
            "evidence_items": evidence_items,
            "open_gaps": open_gaps,
            "retrieval_router_decisions": retrieval_router_decisions,
        }

    def reflect_on_evidence(
        state: DeepResearchState,
        runtime: Runtime[DeepResearchContext],
    ) -> dict[str, Any]:
        context = runtime.context
        payload = {
            "original_question": state["original_question"],
            "normalized_question": state.get("normalized_question"),
            "research_brief": state.get("research_brief", {}),
            "subquestions": state.get("subquestions", []),
            "query_plans": state.get("query_plans", []),
            "evidence_items": state.get("evidence_items", []),
            "open_gaps": state.get("open_gaps", []),
            "retrieval_router_decisions": state.get("retrieval_router_decisions", []),
            "iteration_count": state.get("iteration_count", 0),
        }
        result = resolved_reflection_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(payload, indent=2, default=str),
                    }
                ]
            },
            context=context,
            config={
                "configurable": {
                    "thread_id": f"{context.thread_id}-reflection-agent-{state.get('iteration_count', 0) + 1}",
                }
            },
        )
        reflection = _structured_response_to_dict(result)
        reflection_history = list(state.get("reflection_history", []))
        reflection_history.append(reflection)

        reflected_gaps = [
            str(gap.get("description") or "")
            for gap in reflection.get("open_gaps", [])
            if isinstance(gap, dict)
        ]
        verification_failures = [
            str(conflict.get("description") or "")
            for conflict in reflection.get("conflicting_claims", [])
            if isinstance(conflict, dict)
        ]

        return {
            "reflection_history": reflection_history,
            "open_gaps": _deduplicate_strings(list(state.get("open_gaps", [])) + reflected_gaps),
            "verification_failures": _deduplicate_strings(
                list(state.get("verification_failures", [])) + verification_failures
            ),
            "iteration_count": int(state.get("iteration_count", 0)) + 1,
        }

    def _prepare_followup(
        state: DeepResearchState,
        *,
        action: str,
    ) -> dict[str, Any]:
        followup_plans = _build_followup_query_plans(state, action=action)
        all_subquestion_ids = [
            str(subquestion.get("subquestion_id"))
            for subquestion in state.get("subquestions", [])
            if isinstance(subquestion.get("subquestion_id"), str)
        ]
        active_subquestion_ids = _deduplicate_strings(
            [
                str(plan.get("subquestion_id") or "")
                for plan in followup_plans
                if isinstance(plan.get("subquestion_id"), str)
            ]
        )
        completed_subquestion_ids = [
            subquestion_id
            for subquestion_id in all_subquestion_ids
            if subquestion_id not in set(active_subquestion_ids)
        ]

        return {
            "query_plans": list(state.get("query_plans", [])) + followup_plans,
            "active_query_plans": followup_plans,
            "planned_tool_calls": list(state.get("planned_tool_calls", []))
            + _build_planned_tool_calls(followup_plans),
            "active_subquestion_ids": active_subquestion_ids,
            "completed_subquestion_ids": completed_subquestion_ids,
            "active_retrieval_action": action,
        }

    def prepare_requery_followup(state: DeepResearchState) -> dict[str, Any]:
        return _prepare_followup(state, action="requery")

    def prepare_extract_followup(state: DeepResearchState) -> dict[str, Any]:
        return _prepare_followup(state, action="extract")

    def prepare_map_followup(state: DeepResearchState) -> dict[str, Any]:
        return _prepare_followup(state, action="map")

    def prepare_crawl_followup(state: DeepResearchState) -> dict[str, Any]:
        return _prepare_followup(state, action="crawl")

    def prepare_research_followup(state: DeepResearchState) -> dict[str, Any]:
        return _prepare_followup(state, action="research")

    def synthesize_answer(
        state: DeepResearchState,
        runtime: Runtime[DeepResearchContext],
    ) -> dict[str, Any]:
        context = runtime.context
        payload = {
            "original_question": state["original_question"],
            "normalized_question": state.get("normalized_question"),
            "clarified_question": state.get("clarified_question"),
            "research_brief": state.get("research_brief", {}),
            "subquestions": state.get("subquestions", []),
            "query_plans": state.get("query_plans", []),
            "evidence_items": state.get("evidence_items", []),
            "open_gaps": state.get("open_gaps", []),
            "reflection_history": state.get("reflection_history", []),
            "verification_failures": state.get("verification_failures", []),
            "retrieval_router_decisions": state.get("retrieval_router_decisions", []),
            "iteration_count": state.get("iteration_count", 0),
        }
        result = resolved_answer_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(payload, indent=2, default=str),
                    }
                ]
            },
            context=context,
            config={
                "configurable": {
                    "thread_id": f"{context.thread_id}-answer-agent",
                }
            },
        )
        answer = _structured_response_to_dict(result)
        return {
            "key_findings": list(answer.get("key_findings", [])),
            "completed_subquestion_ids": [
                str(subquestion.get("subquestion_id"))
                for subquestion in state.get("subquestions", [])
                if isinstance(subquestion.get("subquestion_id"), str)
            ],
            "final_answer": answer,
            "is_complete": True,
        }

    def route_after_planning(
        state: DeepResearchState,
    ) -> Literal["request_clarification", "generate_query_plans"]:
        needs_clarification = bool(state.get("clarification_needed", False))
        clarification_allowed = bool(state.get("clarification_interrupts_allowed", False))
        has_clarification_question = bool(str(state.get("clarification_question") or "").strip())
        if needs_clarification and clarification_allowed and has_clarification_question:
            return "request_clarification"
        return "generate_query_plans"

    def route_after_reflection(
        state: DeepResearchState,
    ) -> Literal[
        "request_clarification",
        "synthesize_answer",
        "prepare_requery_followup",
        "prepare_extract_followup",
        "prepare_map_followup",
        "prepare_crawl_followup",
        "prepare_research_followup",
    ]:
        latest_reflection = _latest_reflection(state) or {}
        if bool(latest_reflection.get("is_sufficient", False)):
            return "synthesize_answer"

        if int(state.get("iteration_count", 0)) >= int(state.get("max_iterations_allowed", 0)):
            return "synthesize_answer"

        if (
            latest_reflection.get("recommended_next_action") == "clarify"
            and bool(state.get("clarification_interrupts_allowed", False))
        ):
            return "request_clarification"

        action = str(latest_reflection.get("recommended_next_action") or "requery")
        if action == "extract":
            return "prepare_extract_followup"
        if action == "map":
            return "prepare_map_followup"
        if action == "crawl":
            return "prepare_crawl_followup"
        if action == "research":
            return "prepare_research_followup"
        return "prepare_requery_followup"

    graph = StateGraph(DeepResearchState)
    graph.add_node("plan_research", plan_research)
    graph.add_node("request_clarification", request_clarification)
    graph.add_node("generate_query_plans", generate_query_plans)
    graph.add_node("run_retrieval", run_retrieval)
    graph.add_node("reflect_on_evidence", reflect_on_evidence)
    graph.add_node("prepare_requery_followup", prepare_requery_followup)
    graph.add_node("prepare_extract_followup", prepare_extract_followup)
    graph.add_node("prepare_map_followup", prepare_map_followup)
    graph.add_node("prepare_crawl_followup", prepare_crawl_followup)
    graph.add_node("prepare_research_followup", prepare_research_followup)
    graph.add_node("synthesize_answer", synthesize_answer)

    graph.add_edge(START, "plan_research")
    graph.add_conditional_edges("plan_research", route_after_planning)
    graph.add_edge("generate_query_plans", "run_retrieval")
    graph.add_edge("run_retrieval", "reflect_on_evidence")
    graph.add_conditional_edges("reflect_on_evidence", route_after_reflection)
    graph.add_edge("prepare_requery_followup", "run_retrieval")
    graph.add_edge("prepare_extract_followup", "run_retrieval")
    graph.add_edge("prepare_map_followup", "run_retrieval")
    graph.add_edge("prepare_crawl_followup", "run_retrieval")
    graph.add_edge("prepare_research_followup", "run_retrieval")
    graph.add_edge("request_clarification", END)
    graph.add_edge("synthesize_answer", END)

    return graph.compile(
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name="deep_research",
    )
