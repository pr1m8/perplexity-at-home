"""Serializer helpers for LangGraph persistence.

Purpose:
    Provide a strict-safe serializer configuration for the Postgres
    checkpointer used across the packaged workflows.

Design:
    Child agents in quick-search, pro-search, and deep-research use structured
    Pydantic outputs. When those child agents share a Postgres checkpointer,
    LangGraph must be allowed to deserialize the exact structured response
    types that the package intentionally persists.

    This module keeps that allowlist in one place so the runtime can enable
    ``LANGGRAPH_STRICT_MSGPACK=true`` without breaking nested workflow runs.
"""

from __future__ import annotations

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

CHECKPOINTER_MSGPACK_ALLOWLIST = (
    ("perplexity_at_home.agents.quick_search.models", "QuickSearchAnswer"),
    ("perplexity_at_home.agents.pro_search.query_agent.models", "ProSearchQueryPlan"),
    ("perplexity_at_home.agents.pro_search.answer_agent.models", "ProSearchAnswer"),
    ("perplexity_at_home.agents.deep_research.planner_agent.models", "PlannerOutput"),
    ("perplexity_at_home.agents.deep_research.query_agent.models", "DeepResearchQueryPlans"),
    ("perplexity_at_home.agents.deep_research.retrieval_agent.models", "RetrievalAgentResult"),
    ("perplexity_at_home.agents.deep_research.reflection_agent.models", "ReflectionDecision"),
    ("perplexity_at_home.agents.deep_research.answer_agent.models", "DeepResearchAnswer"),
)

__all__ = [
    "CHECKPOINTER_MSGPACK_ALLOWLIST",
    "build_checkpointer_serde",
]


def build_checkpointer_serde() -> JsonPlusSerializer:
    """Return the checkpointer serializer used by packaged runtimes.

    Returns:
        A ``JsonPlusSerializer`` configured with the exact structured model
        types persisted by this repository's LangGraph workflows.
    """
    return JsonPlusSerializer(allowed_msgpack_modules=()).with_msgpack_allowlist(
        CHECKPOINTER_MSGPACK_ALLOWLIST
    )
