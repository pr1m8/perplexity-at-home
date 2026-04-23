from __future__ import annotations

from perplexity_at_home.core.serde import (
    CHECKPOINTER_MSGPACK_ALLOWLIST,
    build_checkpointer_serde,
)


def test_checkpointer_allowlist_covers_expected_structured_models() -> None:
    assert set(CHECKPOINTER_MSGPACK_ALLOWLIST) == {
        ("perplexity_at_home.agents.quick_search.models", "QuickSearchAnswer"),
        ("perplexity_at_home.agents.pro_search.query_agent.models", "ProSearchQueryPlan"),
        ("perplexity_at_home.agents.pro_search.answer_agent.models", "ProSearchAnswer"),
        ("perplexity_at_home.agents.deep_research.planner_agent.models", "PlannerOutput"),
        ("perplexity_at_home.agents.deep_research.query_agent.models", "DeepResearchQueryPlans"),
        ("perplexity_at_home.agents.deep_research.retrieval_agent.models", "RetrievalAgentResult"),
        ("perplexity_at_home.agents.deep_research.reflection_agent.models", "ReflectionDecision"),
        ("perplexity_at_home.agents.deep_research.answer_agent.models", "DeepResearchAnswer"),
    }


def test_build_checkpointer_serde_configures_a_msgpack_allowlist() -> None:
    serializer = build_checkpointer_serde()

    assert serializer._allowed_msgpack_modules is not None
    assert serializer._allowed_msgpack_modules is not True
