from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from perplexity_at_home.agents.deep_research.context import DeepResearchContext
from perplexity_at_home.agents.deep_research.graph import build_deep_research_graph


@dataclass
class StubAgent:
    responses: list[dict[str, Any]]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def invoke(
        self,
        input: dict[str, Any],
        *,
        context: DeepResearchContext,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "input": input,
                "context": context,
                "config": config,
            }
        )
        if not self.responses:
            raise AssertionError("StubAgent received more calls than configured responses.")
        return {"structured_response": self.responses.pop(0)}


def _initial_state(question: str) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": question}],
        "original_question": question,
        "normalized_question": "",
        "clarified_question": "",
        "clarification_needed": False,
        "clarification_question": "",
        "research_brief": {},
        "subquestions": [],
        "planning_notes": [],
        "query_plans": [],
        "active_query_plans": [],
        "query_plan_notes": [],
        "planned_tool_calls": [],
        "raw_retrieval_results": [],
        "evidence_items": [],
        "key_findings": [],
        "open_gaps": [],
        "reflection_history": [],
        "verification_failures": [],
        "iteration_count": 0,
        "active_subquestion_ids": [],
        "completed_subquestion_ids": [],
        "active_retrieval_action": "initial",
        "retrieval_router_decisions": [],
        "max_iterations_allowed": 3,
        "max_parallel_retrieval_branches_allowed": 4,
        "clarification_interrupts_allowed": True,
        "is_complete": False,
    }


def test_deep_research_graph_routes_through_followup_and_synthesis() -> None:
    planner = StubAgent(
        responses=[
            {
                "original_question": "Research Tavily changes",
                "normalized_question": "Research recent Tavily LangChain changes",
                "needs_clarification": False,
                "clarification_question": None,
                "research_brief": {
                    "title": "Recent Tavily LangChain changes",
                    "research_goal": "Identify recent changes and why they matter.",
                    "scope": "broad",
                    "requires_freshness": True,
                    "expected_deliverable": "A cited markdown report.",
                    "objectives": [],
                    "constraints": [],
                    "domain_hints": ["docs.tavily.com", "github.com"],
                },
                "subquestions": [
                    {
                        "subquestion_id": "sq_1",
                        "question": "What changed?",
                        "rationale": "Core change tracking.",
                        "priority": "high",
                        "requires_freshness": True,
                        "preferred_source_types": ["official docs"],
                        "dependencies": [],
                    }
                ],
                "planning_notes": [],
            }
        ]
    )
    query = StubAgent(
        responses=[
            {
                "original_question": "Research Tavily changes",
                "normalized_question": "Research recent Tavily LangChain changes",
                "plan_count": 1,
                "global_notes": ["Start with official docs and repository sources."],
                "plans": [
                    {
                        "subquestion_id": "sq_1",
                        "subquestion": "What changed?",
                        "research_focus": "Identify recent integration changes.",
                        "requires_freshness": True,
                        "ambiguity_note": None,
                        "target_queries": 2,
                        "min_queries": 1,
                        "max_queries": 2,
                        "retrieval_recommendation": {
                            "strategy": "search_then_extract",
                            "rationale": "Search broadly, then deepen strong hits.",
                            "preferred_domains": ["docs.tavily.com", "github.com"],
                            "known_urls": [],
                            "should_fan_out": True,
                            "recommended_max_branches": 2,
                        },
                        "queries": [
                            {
                                "query": "Tavily LangChain integration recent changes",
                                "rationale": "Primary recent-changes query.",
                                "priority": 1,
                                "intent": "direct",
                                "target_topic": "general",
                                "prefer_recent_sources": True,
                                "preferred_source_types": ["official docs"],
                                "follow_up_of": None,
                            }
                        ],
                    }
                ],
            }
        ]
    )
    retrieval = StubAgent(
        responses=[
            {
                "retrieval_summary": "Collected official documentation but need site-structure context.",
                "recommended_strategy": "search_then_extract",
                "applied_strategy": "search_then_extract",
                "followed_recommended_strategy": True,
                "strategy_rationale": "The initial pass discovered relevant official docs.",
                "used_tools": [{"tool_name": "search", "rationale": "Initial discovery", "input_summary": "query", "output_summary": "docs", "subquestion_id": "sq_1"}],
                "unresolved_gaps": ["Need broader documentation map to verify adjacent pages."],
                "recommended_next_action": "reflect",
                "confidence": 0.62,
                "evidence_items": [
                    {
                        "source_type": "search",
                        "subquestion_id": "sq_1",
                        "url": "https://docs.tavily.com/documentation/integrations/langchain",
                        "title": "LangChain integration docs",
                        "content": "Current integration docs.",
                        "score": 0.95,
                        "raw_content": None,
                        "supports": "Current API surface",
                        "source_authority": "official_docs",
                        "is_primary_source": True,
                        "evidence_date": "2026-04-01",
                    }
                ],
            },
            {
                "retrieval_summary": "Mapped related docs and confirmed surrounding guidance.",
                "recommended_strategy": "map_then_extract",
                "applied_strategy": "map_then_extract",
                "followed_recommended_strategy": True,
                "strategy_rationale": "The follow-up used map to expand official coverage.",
                "used_tools": [{"tool_name": "map", "rationale": "Site exploration", "input_summary": "docs.tavily.com", "output_summary": "related pages", "subquestion_id": "sq_1"}],
                "unresolved_gaps": [],
                "recommended_next_action": "reflect",
                "confidence": 0.84,
                "evidence_items": [
                    {
                        "source_type": "map",
                        "subquestion_id": "sq_1",
                        "url": "https://docs.tavily.com/documentation/integrations/langchain/migration",
                        "title": "Migration guidance",
                        "content": "Migration docs confirm updated usage patterns.",
                        "score": 0.91,
                        "raw_content": None,
                        "supports": "Usage changes",
                        "source_authority": "official_docs",
                        "is_primary_source": True,
                        "evidence_date": "2026-04-02",
                    }
                ],
            },
        ]
    )
    reflection = StubAgent(
        responses=[
            {
                "is_sufficient": False,
                "recommended_next_action": "map",
                "rationale": "Need official site structure to confirm adjacent migration guidance.",
                "open_gaps": [
                    {
                        "gap_id": "gap_1",
                        "description": "Missing official migration guidance.",
                        "severity": "high",
                        "affected_subquestion_ids": ["sq_1"],
                    }
                ],
                "conflicting_claims": [],
                "confidence": 0.58,
                "followup_queries": [
                    {
                        "query": "Tavily LangChain integration migration documentation",
                        "rationale": "Targets missing official migration guidance.",
                        "priority": 1,
                        "target_subquestion_ids": ["sq_1"],
                        "recommended_action": "map",
                    }
                ],
                "notes": [],
            },
            {
                "is_sufficient": True,
                "recommended_next_action": "synthesize",
                "rationale": "The evidence now covers both current docs and migration guidance.",
                "open_gaps": [],
                "conflicting_claims": [],
                "confidence": 0.89,
                "followup_queries": [],
                "notes": [],
            },
        ]
    )
    answer = StubAgent(
        responses=[
            {
                "report_markdown": "# Report\n\nThe integration now has clearer migration guidance.",
                "executive_summary": "Current docs and migration guidance were both verified.",
                "key_findings": ["Official docs and migration guidance are both present."],
                "citations": [
                    {
                        "title": "LangChain integration docs",
                        "url": "https://docs.tavily.com/documentation/integrations/langchain",
                        "supports": "Current API surface",
                    }
                ],
                "confidence": 0.9,
                "used_search": True,
                "evidence_count": 2,
                "uncertainty_note": None,
                "unresolved_questions": [],
            }
        ]
    )

    graph = build_deep_research_graph(
        planner_agent=planner,
        query_agent=query,
        retrieval_agent=retrieval,
        reflection_agent=reflection,
        answer_agent=answer,
    )
    result = graph.invoke(
        _initial_state("Research Tavily changes"),
        context=DeepResearchContext(current_datetime="2026-04-23 10:00:00 EDT"),
        config={"configurable": {"thread_id": "test-deep-research"}},
    )

    assert result["is_complete"] is True
    assert result["iteration_count"] == 2
    assert result["final_answer"]["executive_summary"] == (
        "Current docs and migration guidance were both verified."
    )
    assert len(result["raw_retrieval_results"]) == 2
    assert [entry["action"] for entry in result["retrieval_router_decisions"]] == [
        "initial",
        "map",
    ]
    assert len(result["evidence_items"]) == 2


def test_deep_research_graph_can_stop_for_clarification() -> None:
    planner = StubAgent(
        responses=[
            {
                "original_question": "Research Tavily",
                "normalized_question": "Research Tavily",
                "needs_clarification": True,
                "clarification_question": "Which Tavily product or integration should the report focus on?",
                "research_brief": {
                    "title": "Clarify scope",
                    "research_goal": "Clarify the request.",
                    "scope": "moderate",
                    "requires_freshness": False,
                    "expected_deliverable": "A clarified request.",
                    "objectives": [],
                    "constraints": [],
                    "domain_hints": [],
                },
                "subquestions": [],
                "planning_notes": [],
            }
        ]
    )
    query = StubAgent(responses=[])
    retrieval = StubAgent(responses=[])
    reflection = StubAgent(responses=[])
    answer = StubAgent(responses=[])

    graph = build_deep_research_graph(
        planner_agent=planner,
        query_agent=query,
        retrieval_agent=retrieval,
        reflection_agent=reflection,
        answer_agent=answer,
    )
    result = graph.invoke(
        _initial_state("Research Tavily"),
        context=DeepResearchContext(current_datetime="2026-04-23 10:00:00 EDT"),
        config={"configurable": {"thread_id": "test-clarification"}},
    )

    assert result["is_complete"] is True
    assert result["final_answer"]["status"] == "needs_clarification"
    assert "Which Tavily product or integration" in result["final_answer"]["executive_summary"]
    assert query.calls == []
